// @zakkster/lite-delaunay -- correctness suite for createSpatialIndex
// -----------------------------------------------------------------------------
// node:test only (no bespoke harness, no deps). Run with:
//   node --expose-gc --test test/SpatialIndex.test.js
//
// Covers the planner's five top-level assertions:
//   A1 EQUIVALENCE -- brute-force-vs-index sweep across seeded input classes.
//   A2 BOUNDARY    -- maxDistSq inclusivity, k clamp, far/negative queries.
//   A3 FAIL-CLOSED -- constructor/build/query validation, NaN handling.
//   A4 LIFECYCLE   -- dispose, double-dispose, stale-handle isolation, 4
//                     concurrent handles.
//   A5 REUSE       -- 1000x coherent-drift queries on one handle stay correct.
//
// Plus explicit boundary-matrix entries the planner's five assertions don't
// individually name: maxPoints=0, N-1/N/N+1, empty/null/undefined inputs,
// -0, k=NaN, non-integer k, dispose-during-iteration, re-entrant buffer
// reuse, and one adversarial case (post-build source-array mutation).
//
// The brute-force reference below implements the EXACT semantics the
// implementation documents: skip non-finite points, d <= maxDistSq
// (inclusive), k nearest by squared distance, sorted nearest-first.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createSpatialIndex } from "../Delaunay.js";

// -----------------------------------------------------------------------------
// seeded PRNG (mulberry32) -- deterministic, zero-dependency
// -----------------------------------------------------------------------------
function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
        a |= 0; a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// -----------------------------------------------------------------------------
// data-class generators -- all return { pxs: Float32Array, pys: Float32Array }
// -----------------------------------------------------------------------------
function genUniform(rng, n, lo, hi) {
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    const span = hi - lo;
    for (let i = 0; i < n; i++) {
        pxs[i] = lo + rng() * span;
        pys[i] = lo + rng() * span;
    }
    return { pxs, pys };
}

// ~10^4:1 density skew: dense sub-pixel clusters plus rare far outliers.
function genClustered(rng, n) {
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const c = i % 8;
        let x = c * 1.3 + rng() * 2.0;
        let y = c * 1.1 + rng() * 2.0;
        if (rng() < 0.0001) { x = rng() * 2e4; y = rng() * 2e4; }
        pxs[i] = x; pys[i] = y;
    }
    return { pxs, pys };
}

function genCollinear(rng, n) {
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const t = rng() * 1000 - 500;
        pxs[i] = t;
        pys[i] = 2 * t + 3;
    }
    return { pxs, pys };
}

function genCoincident(rng, n) {
    const pxs = new Float32Array(n).fill(12.5);
    const pys = new Float32Array(n).fill(-7.25);
    return { pxs, pys };
}

// ~10% NaN, forced at index 0 and n-1.
function genNaNBearing(rng, n) {
    const { pxs, pys } = genUniform(rng, n, -1000, 1000);
    for (let i = 0; i < n; i++) {
        if (i === 0 || i === n - 1 || rng() < 0.1) pxs[i] = NaN;
    }
    return { pxs, pys };
}

function genMixedClusterNaN(rng, n) {
    const { pxs, pys } = genClustered(rng, n);
    for (let i = 0; i < n; i++) {
        if (i === 0 || i === n - 1 || rng() < 0.1) pxs[i] = NaN;
    }
    return { pxs, pys };
}

const CLASSES = {
    uniform: (rng, n) => genUniform(rng, n, -1000, 1000),
    clustered: genClustered,
    collinear: genCollinear,
    coincident: genCoincident,
    "nan-bearing": genNaNBearing,
    "mixed-cluster-nan": genMixedClusterNaN,
};

const SIZES = [3, 7, 64, 1000];
const SEEDS = [0x1234, 0xC0FFEE, 0xFEED5EED];

// -----------------------------------------------------------------------------
// brute-force reference -- charts' exact linear-scan semantics
// -----------------------------------------------------------------------------
function isFinitePoint(px, py) {
    return px === px && py === py &&
        px !== Infinity && px !== -Infinity &&
        py !== Infinity && py !== -Infinity;
}

function bruteCandidates(pxs, pys, n, qx, qy, maxDistSq) {
    const out = [];
    for (let i = 0; i < n; i++) {
        const px = pxs[i], py = pys[i];
        if (!isFinitePoint(px, py)) continue;
        const dx = qx - px, dy = qy - py;
        const d = dx * dx + dy * dy;
        if (d <= maxDistSq) out.push({ idx: i, d });
    }
    out.sort((a, b) => a.d - b.d);
    return out;
}

function clampK(k) {
    if (!(k > 0)) return 0; // covers k <= 0 and NaN
    return k > 8 ? 8 : k;
}

function bboxOfFinite(pxs, pys, n) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, count = 0;
    for (let i = 0; i < n; i++) {
        const px = pxs[i], py = pys[i];
        if (!isFinitePoint(px, py)) continue;
        count++;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
    }
    return { minX, minY, maxX, maxY, count };
}

// -----------------------------------------------------------------------------
// generic equivalence assertion -- proves count, sort order, inclusive bound,
// finite-point mapping, f32 lockstep, distance multiset, and tie-safe index
// agreement all in one place. Used by A1, A2, A4, A5.
// -----------------------------------------------------------------------------
function verifyEquivalence(pxs, pys, n, qx, qy, k, maxDistSq, outIndices, outDistSq, count, label) {
    const kClamped = clampK(k);
    const candidates = bruteCandidates(pxs, pys, n, qx, qy, maxDistSq);
    const expectedCount = Math.min(kClamped, candidates.length);
    assert.equal(count, expectedCount, `${label}: count mismatch`);

    const seen = new Set();
    for (let i = 0; i < count; i++) {
        if (i > 0) {
            assert.ok(outDistSq[i] >= outDistSq[i - 1], `${label}: result not sorted nearest-first at rank ${i}`);
        }
        const idx = outIndices[i];
        assert.ok(!seen.has(idx), `${label}: duplicate index ${idx} in result`);
        seen.add(idx);

        const px = pxs[idx], py = pys[idx];
        assert.ok(isFinitePoint(px, py), `${label}: returned index ${idx} maps to a non-finite point`);

        const dx = qx - px, dy = qy - py;
        const d64 = dx * dx + dy * dy;
        assert.ok(d64 <= maxDistSq, `${label}: returned point d=${d64} exceeds maxDistSq=${maxDistSq}`);
        assert.equal(outDistSq[i], Math.fround(d64), `${label}: f32 lockstep mismatch at rank ${i}`);
    }

    // distance MULTISET equivalence (f32 tolerance via fround), rank-by-rank
    const expTopK = candidates.slice(0, expectedCount);
    for (let i = 0; i < count; i++) {
        assert.equal(outDistSq[i], Math.fround(expTopK[i].d), `${label}: distance multiset mismatch at rank ${i}`);
    }

    // index agreement everywhere EXCEPT inside equal-distance tie groups
    const freq = new Map();
    for (const c of candidates) {
        const key = Math.fround(c.d);
        freq.set(key, (freq.get(key) || 0) + 1);
    }
    for (let i = 0; i < count; i++) {
        const key = Math.fround(expTopK[i].d);
        if (freq.get(key) === 1) {
            assert.equal(outIndices[i], expTopK[i].idx, `${label}: index mismatch at unique-distance rank ${i}`);
        }
    }
}

// =============================================================================
// A1 EQUIVALENCE
// =============================================================================
test("A1 EQUIVALENCE: brute-force-vs-index sweep across seeded input classes", () => {
    const outI = new Int32Array(8);
    const outD = new Float32Array(8);
    let totalQueries = 0;

    for (const [className, gen] of Object.entries(CLASSES)) {
        for (const size of SIZES) {
            for (const seed of SEEDS) {
                const rng = mulberry32(seed ^ size ^ (className.length * 7919));
                const { pxs, pys } = gen(rng, size);
                const factory = createSpatialIndex(size);
                const handle = factory(pxs, pys, size);

                const { minX, minY, maxX, maxY, count: finiteCount } = bboxOfFinite(pxs, pys, size);
                const spanX = (isFinite(maxX - minX) ? maxX - minX : 1) || 1;
                const spanY = (isFinite(maxY - minY) ? maxY - minY : 1) || 1;
                const mid = Math.pow(0.15 * Math.max(spanX, spanY, 1), 2);

                const QUERIES_PER_COMBO = 50;
                const kCycle = [1, 3, 5, 8];
                for (let q = 0; q < QUERIES_PER_COMBO; q++) {
                    const strategy = q % 4;
                    let qx, qy, maxDistSq;

                    if (strategy === 3 && finiteCount > 0) {
                        // exact d*d of a known finite point -- inclusive boundary
                        let targetIdx = -1;
                        while (targetIdx < 0) {
                            const cand = (rng() * size) | 0;
                            if (isFinitePoint(pxs[cand], pys[cand])) targetIdx = cand;
                        }
                        qx = minX - spanX * 0.5 + rng() * spanX * 2;
                        qy = minY - spanY * 0.5 + rng() * spanY * 2;
                        const dx = qx - pxs[targetIdx], dy = qy - pys[targetIdx];
                        maxDistSq = dx * dx + dy * dy;
                    } else {
                        qx = minX - spanX * 0.5 + rng() * spanX * 2;
                        qy = minY - spanY * 0.5 + rng() * spanY * 2;
                        maxDistSq = strategy === 0 ? Infinity : strategy === 1 ? mid : 0;
                    }

                    const k = kCycle[q % kCycle.length];
                    const label = `${className} n=${size} seed=${seed} q=${q} k=${k} maxDistSq=${maxDistSq}`;
                    const c = handle.findNearest(qx, qy, k, maxDistSq, outI, outD);
                    verifyEquivalence(pxs, pys, size, qx, qy, k, maxDistSq, outI, outD, c, label);
                    totalQueries++;
                }
                handle.dispose();
            }
        }
    }

    // sanity on the sweep itself: >= 500 queries per class, 6 classes.
    assert.ok(totalQueries >= 500 * Object.keys(CLASSES).length,
        `sweep too small: ${totalQueries} queries`);
});

// =============================================================================
// A2 BOUNDARY
// =============================================================================
test("A2 BOUNDARY: maxDistSq=0 returns only exact-coordinate hits", () => {
    const pxs = new Float32Array([0, 0, 5, -3]);
    const pys = new Float32Array([0, 0, 5, 4]);
    const factory = createSpatialIndex(4);
    const handle = factory(pxs, pys, 4);
    const outI = new Int32Array(8), outD = new Float32Array(8);
    const c = handle.findNearest(0, 0, 8, 0, outI, outD);
    assert.equal(c, 2, "expected both coincident-at-origin points");
    assert.equal(outD[0], 0);
    assert.equal(outD[1], 0);
    const idxSet = new Set([outI[0], outI[1]]);
    assert.deepEqual(idxSet, new Set([0, 1]));
    handle.dispose();
});

test("A2 BOUNDARY: maxDistSq = d*d includes the point at exactly d (inclusive)", () => {
    const pxs = new Float32Array([10, 100]);
    const pys = new Float32Array([0, 0]);
    const factory = createSpatialIndex(2);
    const handle = factory(pxs, pys, 2);
    const outI = new Int32Array(8), outD = new Float32Array(8);
    // distance from (0,0) to (10,0) is exactly 10, so maxDistSq = 100
    const c = handle.findNearest(0, 0, 8, 100, outI, outD);
    assert.equal(c, 1, "boundary point must be INCLUDED, not excluded");
    assert.equal(outI[0], 0);
    assert.equal(outD[0], 100);
    handle.dispose();
});

test("A2 BOUNDARY: k=0 returns 0 (works even with zero-length out buffers)", () => {
    const pxs = new Float32Array([1, 2, 3]);
    const pys = new Float32Array([1, 2, 3]);
    const factory = createSpatialIndex(3);
    const handle = factory(pxs, pys, 3);
    const c = handle.findNearest(0, 0, 0, Infinity, new Int32Array(0), new Float32Array(0));
    assert.equal(c, 0);
    handle.dispose();
});

test("A2 BOUNDARY: k>8 clamps to 8", () => {
    const rng = mulberry32(42);
    const { pxs, pys } = genUniform(rng, 64, -100, 100);
    const factory = createSpatialIndex(64);
    const handle = factory(pxs, pys, 64);
    const outI = new Int32Array(8), outD = new Float32Array(8);
    const c = handle.findNearest(0, 0, 100, Infinity, outI, outD);
    assert.ok(c <= 8, `count ${c} exceeds clamp of 8`);
    verifyEquivalence(pxs, pys, 64, 0, 0, 100, Infinity, outI, outD, c, "k=100 clamp");
    handle.dispose();
});

test("A2 BOUNDARY: k=1 returns exactly the nearest point", () => {
    const pxs = new Float32Array([50, 1, -30]);
    const pys = new Float32Array([50, 1, -30]);
    const factory = createSpatialIndex(3);
    const handle = factory(pxs, pys, 3);
    const outI = new Int32Array(8), outD = new Float32Array(8);
    const c = handle.findNearest(0, 0, 1, Infinity, outI, outD);
    assert.equal(c, 1);
    assert.equal(outI[0], 1); // (1,1) is nearest to origin
    handle.dispose();
});

test("A2 BOUNDARY: query far outside bbox (1e9 away) is correct, no crash/overflow", () => {
    const rng = mulberry32(7);
    const { pxs, pys } = genUniform(rng, 64, -50, 50);
    const factory = createSpatialIndex(64);
    const handle = factory(pxs, pys, 64);
    const outI = new Int32Array(8), outD = new Float32Array(8);

    // huge maxDistSq -> must find the globally nearest point despite the
    // query being 1e9 away from the whole point cloud.
    let c = handle.findNearest(1e9, 1e9, 8, Infinity, outI, outD);
    verifyEquivalence(pxs, pys, 64, 1e9, 1e9, 8, Infinity, outI, outD, c, "far query, Infinity radius");
    assert.ok(c > 0);

    // tight maxDistSq at that distance -> nothing within range.
    c = handle.findNearest(1e9, 1e9, 8, 1000 * 1000, outI, outD);
    assert.equal(c, 0);
    handle.dispose();
});

test("A2 BOUNDARY: negative coordinates behave identically to positive", () => {
    const rng = mulberry32(99);
    const { pxs, pys } = genUniform(rng, 128, -5000, -1000);
    const factory = createSpatialIndex(128);
    const handle = factory(pxs, pys, 128);
    const outI = new Int32Array(8), outD = new Float32Array(8);
    for (let q = 0; q < 20; q++) {
        const qx = -5000 + rng() * 4000;
        const qy = -5000 + rng() * 4000;
        const c = handle.findNearest(qx, qy, 5, Infinity, outI, outD);
        verifyEquivalence(pxs, pys, 128, qx, qy, 5, Infinity, outI, outD, c, `negative coords q=${q}`);
    }
    handle.dispose();
});

test("A2 BOUNDARY: -0 query and -0 maxDistSq behave as +0", () => {
    const pxs = new Float32Array([0, 1, -1]);
    const pys = new Float32Array([0, 1, -1]);
    const factory = createSpatialIndex(3);
    const handle = factory(pxs, pys, 3);
    const outI = new Int32Array(8), outD = new Float32Array(8);
    const cPos = handle.findNearest(0, 0, 8, 0, outI, outD);
    const outI2 = new Int32Array(8), outD2 = new Float32Array(8);
    const cNeg = handle.findNearest(-0, -0, 8, -0, outI2, outD2);
    assert.equal(cNeg, cPos);
    for (let i = 0; i < cPos; i++) {
        assert.equal(outI2[i], outI[i]);
        assert.equal(outD2[i], outD[i]);
    }
    handle.dispose();
});

// =============================================================================
// A3 FAIL-CLOSED
// =============================================================================
test("A3 FAIL-CLOSED: createSpatialIndex rejects invalid maxPoints", () => {
    assert.throws(() => createSpatialIndex(-1));
    assert.throws(() => createSpatialIndex(1.5));
    assert.throws(() => createSpatialIndex(NaN));
    assert.throws(() => createSpatialIndex(undefined));
    assert.throws(() => createSpatialIndex(null));
    assert.throws(() => createSpatialIndex("5"));
    assert.throws(() => createSpatialIndex(Infinity));
    // valid boundary: 0 is a non-negative integer, must NOT throw.
    assert.doesNotThrow(() => createSpatialIndex(0));
});

test("A3 FAIL-CLOSED: build throws when n > maxPoints (N+1), accepts N and N-1", () => {
    const factory = createSpatialIndex(5);
    const pxs = new Float32Array(6).fill(1);
    const pys = new Float32Array(6).fill(1);
    assert.throws(() => factory(pxs, pys, 6), "n = maxPoints + 1 must throw");
    assert.doesNotThrow(() => factory(pxs, pys, 5).dispose(), "n = maxPoints must succeed");
    assert.doesNotThrow(() => factory(pxs, pys, 4).dispose(), "n = maxPoints - 1 must succeed");
});

test("A3 FAIL-CLOSED: build rejects null/undefined/empty inputs", () => {
    const factory = createSpatialIndex(4);
    assert.throws(() => factory(null, new Float32Array(4), 4));
    assert.throws(() => factory(new Float32Array(4), null, 4));
    assert.throws(() => factory(undefined, undefined, 4));
    assert.throws(() => factory(new Float32Array(2), new Float32Array(2), 3), "pxs shorter than n");
    // n=0 with genuinely empty arrays must NOT throw.
    assert.doesNotThrow(() => factory([], [], 0).dispose());
});

test("A3 FAIL-CLOSED: findNearest throws on short out-buffers (post-clamp)", () => {
    const factory = createSpatialIndex(10);
    const rng = mulberry32(3);
    const { pxs, pys } = genUniform(rng, 10, -10, 10);
    const handle = factory(pxs, pys, 10);
    assert.throws(() => handle.findNearest(0, 0, 8, Infinity, new Int32Array(4), new Float32Array(8)));
    assert.throws(() => handle.findNearest(0, 0, 8, Infinity, new Int32Array(8), new Float32Array(4)));
    // k clamped from 100 to 8 -- length-4 buffers must still throw post-clamp.
    assert.throws(() => handle.findNearest(0, 0, 100, Infinity, new Int32Array(4), new Float32Array(4)));
    // exactly length 8 must NOT throw.
    assert.doesNotThrow(() => handle.findNearest(0, 0, 8, Infinity, new Int32Array(8), new Float32Array(8)));
    handle.dispose();
});

test("A3 FAIL-CLOSED: non-finite query returns 0 (never throws)", () => {
    const rng = mulberry32(4);
    const { pxs, pys } = genUniform(rng, 20, -20, 20);
    const factory = createSpatialIndex(20);
    const handle = factory(pxs, pys, 20);
    const outI = new Int32Array(8), outD = new Float32Array(8);
    for (const [qx, qy] of [
        [NaN, 0], [0, NaN], [NaN, NaN],
        [Infinity, 0], [0, Infinity], [-Infinity, 0], [0, -Infinity],
        [Infinity, NaN],
    ]) {
        const c = handle.findNearest(qx, qy, 8, Infinity, outI, outD);
        assert.equal(c, 0, `query (${qx}, ${qy}) must return 0`);
    }
    handle.dispose();
});

test("A3 FAIL-CLOSED: all-NaN input builds successfully; every query returns 0", () => {
    const n = 50;
    const pxs = new Float32Array(n).fill(NaN);
    const pys = new Float32Array(n).fill(NaN);
    const factory = createSpatialIndex(n);
    let handle;
    assert.doesNotThrow(() => { handle = factory(pxs, pys, n); });
    const outI = new Int32Array(8), outD = new Float32Array(8);
    const rng = mulberry32(5);
    for (let i = 0; i < 20; i++) {
        const c = handle.findNearest(rng() * 100 - 50, rng() * 100 - 50, 8, Infinity, outI, outD);
        assert.equal(c, 0);
    }
    handle.dispose();
});

test("A3 FAIL-CLOSED: n=0,1,2 builds answer correctly (linear-degenerate)", () => {
    for (const n of [0, 1, 2]) {
        const rng = mulberry32(100 + n);
        const { pxs, pys } = genUniform(rng, Math.max(n, 1), -10, 10);
        const factory = createSpatialIndex(Math.max(n, 2));
        const handle = factory(pxs, pys, n);
        const outI = new Int32Array(8), outD = new Float32Array(8);
        for (let q = 0; q < 10; q++) {
            const qx = rng() * 20 - 10, qy = rng() * 20 - 10;
            const c = handle.findNearest(qx, qy, 8, Infinity, outI, outD);
            verifyEquivalence(pxs, pys, n, qx, qy, 8, Infinity, outI, outD, c, `n=${n} q=${q}`);
        }
        handle.dispose();
    }
});

// =============================================================================
// A4 LIFECYCLE
// =============================================================================
test("A4 LIFECYCLE: dispose then findNearest throws", () => {
    const factory = createSpatialIndex(4);
    const handle = factory(new Float32Array([1, 2]), new Float32Array([1, 2]), 2);
    handle.dispose();
    assert.throws(() => handle.findNearest(0, 0, 8, Infinity, new Int32Array(8), new Float32Array(8)));
});

test("A4 LIFECYCLE: double-dispose throws", () => {
    const factory = createSpatialIndex(4);
    const handle = factory(new Float32Array([1, 2]), new Float32Array([1, 2]), 2);
    handle.dispose();
    assert.throws(() => handle.dispose());
});

test("A4 LIFECYCLE: stale handle cannot kill a live handle built after it (slot reuse isolation)", () => {
    const factory = createSpatialIndex(8);
    const rngA = mulberry32(11);
    const dataA = genUniform(rngA, 8, -10, 10);
    const rngB = mulberry32(22);
    const dataB = genUniform(rngB, 8, 100, 200);

    const handleA = factory(dataA.pxs, dataA.pys, 8);
    handleA.dispose(); // frees the slot
    const handleB = factory(dataB.pxs, dataB.pys, 8); // very likely reuses A's slot

    const outI = new Int32Array(8), outD = new Float32Array(8);

    // B answers correctly BEFORE the stale attempts.
    let c = handleB.findNearest(150, 150, 8, Infinity, outI, outD);
    verifyEquivalence(dataB.pxs, dataB.pys, 8, 150, 150, 8, Infinity, outI, outD, c, "B before stale ops");

    // stale A must throw on both entry points.
    assert.throws(() => handleA.findNearest(0, 0, 8, Infinity, new Int32Array(8), new Float32Array(8)));
    assert.throws(() => handleA.dispose());

    // B still answers correctly AFTER the stale attempts (no corruption).
    c = handleB.findNearest(150, 150, 8, Infinity, outI, outD);
    verifyEquivalence(dataB.pxs, dataB.pys, 8, 150, 150, 8, Infinity, outI, outD, c, "B after stale ops");

    handleB.dispose();
});

test("A4 LIFECYCLE: 4 concurrent handles from one factory answer independently, disposed in shuffled order", () => {
    const factory = createSpatialIndex(32);
    const datasets = [];
    const handles = [];
    for (let i = 0; i < 4; i++) {
        const rng = mulberry32(1000 + i);
        const n = 8 + i * 6;
        const data = genUniform(rng, n, i * 1000, i * 1000 + 200);
        datasets.push({ data, n });
        handles.push(factory(data.pxs, data.pys, n));
    }

    const outI = new Int32Array(8), outD = new Float32Array(8);
    function checkAll(liveMask) {
        for (let i = 0; i < 4; i++) {
            if (!liveMask[i]) continue;
            const { data, n } = datasets[i];
            const rng = mulberry32(2000 + i);
            for (let q = 0; q < 10; q++) {
                const qx = i * 1000 + rng() * 200;
                const qy = i * 1000 + rng() * 200;
                const c = handles[i].findNearest(qx, qy, 5, Infinity, outI, outD);
                verifyEquivalence(data.pxs, data.pys, n, qx, qy, 5, Infinity, outI, outD, c, `concurrent handle ${i} q=${q}`);
            }
        }
    }

    const liveMask = [true, true, true, true];
    checkAll(liveMask);

    // shuffled dispose order
    const disposeOrder = [2, 0, 3, 1];
    for (const i of disposeOrder) {
        handles[i].dispose();
        liveMask[i] = false;
        checkAll(liveMask); // remaining live handles must remain correct
        assert.throws(() => handles[i].findNearest(0, 0, 8, Infinity, outI, outD), `handle ${i} must throw after dispose`);
    }
});

test("A4 LIFECYCLE / BOUNDARY: dispose-during-iteration does not corrupt prior results or sibling handles", () => {
    const factory = createSpatialIndex(16);
    const rng = mulberry32(321);
    const { pxs, pys } = genUniform(rng, 16, -50, 50);
    const handle = factory(pxs, pys, 16);
    const sibling = factory(pxs, pys, 16); // built after `handle`, must be unaffected

    const outI = new Int32Array(8), outD = new Float32Array(8);
    const queries = [];
    for (let i = 0; i < 10; i++) queries.push([rng() * 100 - 50, rng() * 100 - 50]);

    const collected = [];
    for (let i = 0; i < queries.length; i++) {
        if (i === 5) {
            handle.dispose(); // dispose MID-ITERATION
            continue;
        }
        if (i < 5) {
            const [qx, qy] = queries[i];
            const c = handle.findNearest(qx, qy, 8, Infinity, outI, outD);
            collected.push({ qx, qy, c, idx: outI.slice(0, c), dist: outD.slice(0, c) });
        } else {
            // handle is disposed; every subsequent use must fail closed.
            const [qx, qy] = queries[i];
            assert.throws(() => handle.findNearest(qx, qy, 8, Infinity, outI, outD));
        }
    }

    // results collected BEFORE dispose remain internally valid.
    for (const r of collected) {
        verifyEquivalence(pxs, pys, 16, r.qx, r.qy, 8, Infinity, r.idx, r.dist, r.c, "pre-dispose result");
    }

    // sibling handle (built earlier, independent slot) is completely unaffected.
    const [qx, qy] = queries[0];
    const c = sibling.findNearest(qx, qy, 8, Infinity, outI, outD);
    verifyEquivalence(pxs, pys, 16, qx, qy, 8, Infinity, outI, outD, c, "sibling after handle disposed mid-iteration");
    sibling.dispose();
});

// =============================================================================
// A5 REUSE
// =============================================================================
test("A5 REUSE: same handle queried 1000x with coherent drift stays equivalent to brute force", () => {
    const N = 200;
    const rngData = mulberry32(0xABCD);
    const { pxs, pys } = genClustered(rngData, N);
    const factory = createSpatialIndex(N);
    const handle = factory(pxs, pys, N);

    const outI = new Int32Array(8), outD = new Float32Array(8);
    const rng = mulberry32(0xBEEF);
    const kCycle = [1, 2, 3, 5, 8, 4, 6, 7];
    let mx = 5, my = 5;
    for (let i = 0; i < 1000; i++) {
        mx += (rng() - 0.5) * 3;
        my += (rng() - 0.5) * 3;
        if ((i & 63) === 0) { mx = rng() * 40; my = rng() * 40; } // periodic jumps
        const k = kCycle[i % kCycle.length];
        const maxDistSq = (i % 5 === 0) ? 400 : Infinity;
        const c = handle.findNearest(mx, my, k, maxDistSq, outI, outD);
        verifyEquivalence(pxs, pys, N, mx, my, k, maxDistSq, outI, outD, c, `drift step ${i}`);
    }
    handle.dispose();
});

// =============================================================================
// BOUNDARY MATRIX extras (k=NaN, non-integer k, re-entrant buffer reuse)
// =============================================================================
test("BOUNDARY: k = NaN returns 0 without throwing (double-NaN-compare fail path)", () => {
    const rng = mulberry32(55);
    const { pxs, pys } = genUniform(rng, 20, -20, 20);
    const factory = createSpatialIndex(20);
    const handle = factory(pxs, pys, 20);
    const outI = new Int32Array(8), outD = new Float32Array(8);
    let c;
    assert.doesNotThrow(() => { c = handle.findNearest(0, 0, NaN, Infinity, outI, outD); });
    assert.equal(c, 0);
    handle.dispose();
});

test("ADVERSARIAL: non-integer k does not overflow a buffer sized to ceil(k)", () => {
    const rng = mulberry32(66);
    const { pxs, pys } = genUniform(rng, 30, -30, 30);
    const factory = createSpatialIndex(30);
    const handle = factory(pxs, pys, 30);
    const outI = new Int32Array(3), outD = new Float32Array(3); // sized to ceil(2.5)
    let c;
    assert.doesNotThrow(() => { c = handle.findNearest(0, 0, 2.5, Infinity, outI, outD); });
    assert.ok(c <= 3, `count ${c} overflowed the ceil(k)-sized buffer`);
    // every written slot must be a genuinely finite, in-range result.
    for (let i = 0; i < c; i++) {
        const idx = outI[i];
        assert.ok(idx >= 0 && idx < 30);
        assert.ok(isFinitePoint(pxs[idx], pys[idx]));
    }
    handle.dispose();
});

test("ADVERSARIAL: mutating source pxs/pys after build does not affect a built index (copy semantics)", () => {
    const n = 20;
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    const rng = mulberry32(77);
    for (let i = 0; i < n; i++) { pxs[i] = rng() * 100; pys[i] = rng() * 100; }
    const snapshotPxs = pxs.slice();
    const snapshotPys = pys.slice();

    const factory = createSpatialIndex(n);
    const handle = factory(pxs, pys, n);

    // mutate the SOURCE arrays in place after build.
    for (let i = 0; i < n; i++) { pxs[i] = 9999; pys[i] = 9999; }

    const outI = new Int32Array(8), outD = new Float32Array(8);
    const c = handle.findNearest(50, 50, 8, Infinity, outI, outD);
    // must match the ORIGINAL (pre-mutation) snapshot, not the mutated live arrays.
    verifyEquivalence(snapshotPxs, snapshotPys, n, 50, 50, 8, Infinity, outI, outD, c, "post-mutation build snapshot");

    // sanity: if it had read the mutated live arrays instead, every point
    // would now be at (9999,9999) and every result distSq would be identical.
    const distinctDist = new Set();
    for (let i = 0; i < c; i++) distinctDist.add(outD[i]);
    assert.ok(distinctDist.size > 1 || c <= 1, "index appears to have re-read mutated source arrays");

    handle.dispose();
});

test("BOUNDARY: re-entrant buffer reuse across differing k leaves no stale count/index leakage", () => {
    const rng = mulberry32(88);
    const { pxs, pys } = genUniform(rng, 40, -40, 40);
    const factory = createSpatialIndex(40);
    const handle = factory(pxs, pys, 40);
    const outI = new Int32Array(8), outD = new Float32Array(8);

    // first call: large k, fills the whole buffer.
    const c1 = handle.findNearest(0, 0, 8, Infinity, outI, outD);
    assert.ok(c1 > 0);
    const staleIdx = outI[c1 - 1];
    const staleDist = outD[c1 - 1];

    // second call on the SAME buffers: small k, must report the smaller count
    // and must not let the previous call's tail entries leak into the
    // reported result window (verified by the generic equivalence checker,
    // which only inspects indices [0, count)).
    const c2 = handle.findNearest(0, 0, 2, Infinity, outI, outD);
    assert.equal(c2, Math.min(2, c1));
    verifyEquivalence(pxs, pys, 40, 0, 0, 2, Infinity, outI, outD, c2, "reentrant reuse k=2 after k=8");

    // bytes beyond the new count are explicitly UNDEFINED by contract, but
    // must still be whatever the previous call left (no wild pointer writes,
    // no NaN corruption of adjacent slots).
    if (c1 > c2) {
        assert.ok(Number.isFinite(outD[c1 - 1]) || outD[c1 - 1] === staleDist);
        void staleIdx;
    }
    handle.dispose();
});
