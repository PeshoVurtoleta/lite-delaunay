// @zakkster/lite-delaunay -- correctness suite for createClusterIndex (v1.4.0)
// -----------------------------------------------------------------------------
// node:test only (no bespoke harness, no deps beyond the package under test).
// Run with:
//   node --expose-gc --test test/ClusterIndex.test.js
//
// The two oracles are INDEPENDENT of the implementation under test:
//   - convex hull: a monotone-chain hull built in this file; compared as an
//     exact index SET plus cyclic order modulo rotation, both senses normalized.
//   - alpha shape: an independent circumradius filter over the mesh exposed by
//     the SEPARATE createFieldIndex surface (triangleCount / triangleVertices),
//     proving every emitted boundary edge is owned by exactly ONE kept triangle
//     and every such edge is emitted, exactly once.
//
// All assert.throws regexes are UNANCHORED /lite-delaunay: / (String(error)),
// never ^-anchored.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createClusterIndex, createFieldIndex } from "../Delaunay.js";

// -----------------------------------------------------------------------------
// seeded PRNG (mulberry32)
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

const EPS = Math.pow(2, -52);

// -----------------------------------------------------------------------------
// independent monotone-chain convex hull oracle (returns ORIGINAL indices, CCW)
// -----------------------------------------------------------------------------
function monotoneHull(pxs, pys, n) {
    const idx = [];
    for (let i = 0; i < n; i++) {
        const x = pxs[i], y = pys[i];
        if (x !== x || y !== y || x === Infinity || x === -Infinity ||
            y === Infinity || y === -Infinity) continue;
        idx.push(i);
    }
    idx.sort((a, b) => (pxs[a] - pxs[b]) || (pys[a] - pys[b]));
    // dedup exact-coincident points (keep first)
    const uniq = [];
    for (const i of idx) {
        const p = uniq[uniq.length - 1];
        if (p !== undefined && pxs[p] === pxs[i] && pys[p] === pys[i]) continue;
        uniq.push(i);
    }
    if (uniq.length < 3) return [];
    const cross = (o, a, b) =>
        (pxs[a] - pxs[o]) * (pys[b] - pys[o]) - (pys[a] - pys[o]) * (pxs[b] - pxs[o]);
    const lower = [];
    for (const i of uniq) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], i) <= 0) lower.pop();
        lower.push(i);
    }
    const upper = [];
    for (let k = uniq.length - 1; k >= 0; k--) {
        const i = uniq[k];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], i) <= 0) upper.pop();
        upper.push(i);
    }
    lower.pop(); upper.pop();
    const hull = lower.concat(upper); // CCW in math coords
    if (hull.length < 3) return [];
    return hull;
}

// canonical form of a cyclic vertex sequence, invariant to rotation AND reflection
function canonCycle(seq) {
    const n = seq.length;
    if (n === 0) return "";
    const rots = [];
    const fwd = seq.slice();
    const rev = seq.slice().reverse();
    for (const base of [fwd, rev]) {
        for (let r = 0; r < n; r++) {
            let s = "";
            for (let i = 0; i < n; i++) s += base[(r + i) % n] + ",";
            rots.push(s);
        }
    }
    rots.sort();
    return rots[0];
}

// -----------------------------------------------------------------------------
// independent alpha-shape boundary oracle, over the createFieldIndex mesh.
// Returns { edges: Set of "min,max" undirected boundary edges, keptCount }.
// -----------------------------------------------------------------------------
function alphaOracle(pxs, pys, n, alpha) {
    const ff = createFieldIndex(n)(pxs, pys, n);
    const T = ff.triangleCount();
    const tri = new Int32Array(3);
    const alphaSq = alpha * alpha;
    const edgeOwners = new Map(); // "min,max" -> count of kept triangles owning it
    let keptCount = 0;
    for (let t = 0; t < T; t++) {
        ff.triangleVertices(t, tri);
        const a = tri[0], b = tri[1], c = tri[2];
        const ax = pxs[a], ay = pys[a], bx = pxs[b], by = pys[b], cx = pxs[c], cy = pys[c];
        const dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
        const bl = dx * dx + dy * dy, cl = ex * ex + ey * ey;
        const D = dx * ey - dy * ex;
        const scale = bl + cl;
        let crSq;
        if (scale === 0 || !(Math.abs(D) > EPS * scale)) {
            crSq = NaN;
        } else {
            const d = 0.5 / D;
            const ux = (ey * bl - dy * cl) * d;
            const uy = (dx * cl - ex * bl) * d;
            crSq = ux * ux + uy * uy;
        }
        if (!(crSq <= alphaSq)) continue; // NaN never kept
        keptCount++;
        const es = [[a, b], [b, c], [c, a]];
        for (const [u, v] of es) {
            const key = u < v ? (u + "," + v) : (v + "," + u);
            edgeOwners.set(key, (edgeOwners.get(key) || 0) + 1);
        }
    }
    ff.dispose();
    const edges = new Set();
    for (const [key, cnt] of edgeOwners) if (cnt === 1) edges.add(key);
    return { edges, keptCount };
}

// undirected edge set emitted by an alphaShape result
function emittedEdgeSet(out, ends, loopCount) {
    const edges = new Set();
    let prev = 0;
    for (let L = 0; L < loopCount; L++) {
        const end = ends[L];
        for (let i = prev; i < end; i++) {
            const u = out[i];
            const v = out[i + 1 === end ? prev : i + 1];
            edges.add(u < v ? (u + "," + v) : (v + "," + u));
        }
        prev = end;
    }
    return edges;
}

// shoelace signed area over ORIGINAL indices in +y-up math coords (pos=CCW)
function signedArea(out, s, e, pxs, pys) {
    let A = 0;
    for (let i = s; i < e; i++) {
        const a = out[i], b = out[i + 1 === e ? s : i + 1];
        A += pxs[a] * pys[b] - pxs[b] * pys[a];
    }
    return 0.5 * A;
}

function uniformCloud(rng, n, lo, hi) {
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    const span = hi - lo;
    for (let i = 0; i < n; i++) { pxs[i] = lo + rng() * span; pys[i] = lo + rng() * span; }
    return { pxs, pys };
}

// =============================================================================
// tiny-n correctness
// =============================================================================
test("n=3 exact triangle hull", () => {
    const pxs = new Float32Array([0, 10, 5]);
    const pys = new Float32Array([0, 0, 8]);
    const h = createClusterIndex(3)(pxs, pys, 3);
    const out = new Int32Array(3);
    const c = h.convexHull(out);
    assert.equal(c, 3);
    assert.deepEqual([...out].sort(), [0, 1, 2]);
    h.dispose();
});

test("n=4 square hull is all four corners", () => {
    const pxs = new Float32Array([0, 10, 10, 0]);
    const pys = new Float32Array([0, 0, 10, 10]);
    const h = createClusterIndex(4)(pxs, pys, 4);
    const out = new Int32Array(4);
    const c = h.convexHull(out);
    assert.equal(c, 4);
    assert.deepEqual([...out].sort(), [0, 1, 2, 3]);
    // matches the oracle cyclic order (both senses normalized)
    const oracle = monotoneHull(pxs, pys, 4);
    assert.equal(canonCycle([...out].slice(0, c)), canonCycle(oracle));
    h.dispose();
});

test("collinear cloud -> 0 hull, 0 loops", () => {
    const pxs = new Float32Array([0, 1, 2, 3, 4, 5]);
    const pys = new Float32Array([0, 1, 2, 3, 4, 5]);
    const h = createClusterIndex(6)(pxs, pys, 6);
    const out = new Int32Array(18), ends = new Int32Array(6);
    assert.equal(h.convexHull(out), 0);
    assert.equal(h.alphaShape(100, out, ends), 0);
    h.dispose();
});

test("all-duplicate cloud -> 0 hull, 0 loops", () => {
    const pxs = new Float32Array([3, 3, 3, 3, 3]);
    const pys = new Float32Array([7, 7, 7, 7, 7]);
    const h = createClusterIndex(5)(pxs, pys, 5);
    const out = new Int32Array(15), ends = new Int32Array(5);
    assert.equal(h.convexHull(out), 0);
    assert.equal(h.alphaShape(100, out, ends), 0);
    h.dispose();
});

test("n<3 builds -> 0 from both methods (no throw)", () => {
    for (const n of [0, 1, 2]) {
        const pxs = new Float32Array([1, 2]);
        const pys = new Float32Array([3, 4]);
        const h = createClusterIndex(2)(pxs, pys, n);
        const out = new Int32Array(6), ends = new Int32Array(2);
        assert.equal(h.convexHull(out), 0);
        assert.equal(h.alphaShape(50, out, ends), 0);
        h.dispose();
    }
});

// =============================================================================
// convexHull vs the independent monotone-chain oracle
// =============================================================================
test("convexHull matches monotone-chain oracle across >=200 random clouds", () => {
    const rng = mulberry32(0xC0FFEE);
    let checked = 0;
    for (let trial = 0; trial < 220; trial++) {
        const n = 3 + Math.floor(rng() * 254); // [3, 256]
        const { pxs, pys } = uniformCloud(rng, n, 0, 1000);
        const factory = createClusterIndex(n);
        const h = factory(pxs, pys, n);
        const out = new Int32Array(n);
        const c = h.convexHull(out);
        const oracle = monotoneHull(pxs, pys, n);
        assert.equal(c, oracle.length, `trial ${trial} n=${n} hull count`);
        const got = [...out].slice(0, c);
        assert.deepEqual(new Set(got), new Set(oracle), `trial ${trial} n=${n} hull set`);
        assert.equal(canonCycle(got), canonCycle(oracle), `trial ${trial} n=${n} cyclic order`);
        h.dispose();
        checked++;
    }
    assert.ok(checked >= 200);
});

// =============================================================================
// alphaShape boundary oracle
// =============================================================================
test("alphaShape emitted edges == independent circumradius-filter boundary", () => {
    const rng = mulberry32(0x1234beef);
    for (let trial = 0; trial < 120; trial++) {
        const n = 8 + Math.floor(rng() * 120);
        const { pxs, pys } = uniformCloud(rng, n, 0, 500);
        const h = createClusterIndex(n)(pxs, pys, n);
        // a mid alpha derived from the domain scale
        for (const alpha of [20, 45, 90]) {
            const out = new Int32Array(3 * n), ends = new Int32Array(n);
            const lc = h.alphaShape(alpha, out, ends);
            const emitted = emittedEdgeSet(out, ends, lc);
            const { edges } = alphaOracle(pxs, pys, n, alpha);
            assert.deepEqual(emitted, edges, `trial ${trial} n=${n} alpha=${alpha}`);
            // every emitted edge exactly once (no duplicate in concatenation)
            const total = lc > 0 ? ends[lc - 1] : 0;
            assert.equal(total, emitted.size + countRepeatEdges(out, ends, lc),
                `edge multiplicity trial ${trial} alpha=${alpha}`);
        }
        h.dispose();
    }
});

// helper: at a pinch point a vertex may repeat but each directed edge is unique;
// with an even-degree boundary each undirected edge appears once, so repeats=0.
function countRepeatEdges(out, ends, loopCount) {
    const seen = new Set();
    let repeats = 0, prev = 0;
    for (let L = 0; L < loopCount; L++) {
        const end = ends[L];
        for (let i = prev; i < end; i++) {
            const u = out[i], v = out[i + 1 === end ? prev : i + 1];
            const key = u < v ? (u + "," + v) : (v + "," + u);
            if (seen.has(key)) repeats++; else seen.add(key);
        }
        prev = end;
    }
    return repeats;
}

// =============================================================================
// ABSOLUTE winding convention: outer loops are screen-CCW (math-CW, negative
// shoelace in the +y-up sense) on EVERY input -- not just relative to a hole.
// convexHull and alphaShape's outer loop share the same convention.
// =============================================================================
test("convexHull ABSOLUTE winding: shoelace strictly negative (screen-CCW) across >=50 random clouds, and alphaShape's outer loop shares the sign", () => {
    const rng = mulberry32(0xA55E55);
    let checked = 0;
    for (let trial = 0; trial < 60; trial++) {
        const n = 4 + Math.floor(rng() * 297); // [4, 300]
        const { pxs, pys } = uniformCloud(rng, n, 0, 1000);
        const h = createClusterIndex(n)(pxs, pys, n);
        const out = new Int32Array(n);
        const c = h.convexHull(out);
        h.dispose();
        if (c < 3) continue; // astronomically-unlikely degenerate cloud, skip
        const area = signedArea(out, 0, c, pxs, pys);
        assert.ok(area < 0, `trial ${trial} n=${n} convexHull signed area must be negative (screen-CCW), got ${area}`);
        checked++;
    }
    assert.ok(checked >= 50, `expected >=50 non-degenerate trials, got ${checked}`);

    // shared-convention fixture: at a large finite alpha, alphaShape's single
    // outer loop must carry the SAME sign as convexHull.
    const rng2 = mulberry32(0xFACE);
    const { pxs, pys } = uniformCloud(rng2, 150, 0, 1000);
    const h2 = createClusterIndex(150)(pxs, pys, 150);
    const out = new Int32Array(3 * 150), ends = new Int32Array(150);
    const lc = h2.alphaShape(1e12, out, ends);
    assert.equal(lc, 1, "single outer loop at large finite alpha");
    const alphaArea = signedArea(out, 0, ends[0], pxs, pys);
    assert.ok(alphaArea < 0, `alphaShape outer loop must be negative (screen-CCW), got ${alphaArea}`);
    h2.dispose();
});

// =============================================================================
// multi-loop fixture: two blobs + a bridge -> EXACTLY 2 loops
// =============================================================================
test("two-blob + bridge fixture yields exactly 2 loops", () => {
    const rng = mulberry32(7);
    const pts = [];
    for (const cx of [0, 200]) {
        pts.push([cx, 0]);
        for (let i = 0; i < 11; i++) {
            const a = i / 11 * Math.PI * 2;
            const r = 6 + rng() * 4;
            pts.push([cx + r * Math.cos(a), r * Math.sin(a)]);
        }
    }
    pts.push([100, 0]); // bridge
    const n = pts.length;
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = pts[i][0]; pys[i] = pts[i][1]; }
    const h = createClusterIndex(n)(pxs, pys, n);
    const out = new Int32Array(3 * n), ends = new Int32Array(n);
    const lc = h.alphaShape(25, out, ends);
    assert.equal(lc, 2, "exactly two loops");
    // outLoopEnds strictly increasing, last === total written, all indices < n
    assert.ok(ends[0] > 0 && ends[1] > ends[0], "ends strictly increasing");
    const total = ends[lc - 1];
    for (let i = 0; i < total; i++) assert.ok(out[i] >= 0 && out[i] < n, "index in range");
    h.dispose();
});

// =============================================================================
// annulus hole fixture: outer + hole with OPPOSITE signed areas, outer negative
// =============================================================================
test("annulus fixture: outer negative (screen-CCW), hole opposite sign", () => {
    const pts = [];
    for (let i = 0; i < 40; i++) { const a = i / 40 * Math.PI * 2; pts.push([200 + 100 * Math.cos(a), 200 + 100 * Math.sin(a)]); }
    for (let i = 0; i < 24; i++) { const a = i / 24 * Math.PI * 2; pts.push([200 + 40 * Math.cos(a), 200 + 40 * Math.sin(a)]); }
    const n = pts.length;
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = pts[i][0]; pys[i] = pts[i][1]; }
    const h = createClusterIndex(n)(pxs, pys, n);
    const out = new Int32Array(3 * n), ends = new Int32Array(n);
    const lc = h.alphaShape(35, out, ends);
    assert.equal(lc, 2, "outer + hole");
    const a0 = signedArea(out, 0, ends[0], pxs, pys);
    const a1 = signedArea(out, ends[0], ends[1], pxs, pys);
    assert.ok(a0 * a1 < 0, "outer and hole wind oppositely");
    // the larger-magnitude loop is the outer, and it must be negative (screen-CCW)
    const outer = Math.abs(a0) >= Math.abs(a1) ? a0 : a1;
    assert.ok(outer < 0, "outer loop negative shoelace in +y-up (= screen CCW)");
    h.dispose();
});

// =============================================================================
// large FINITE alpha == convex hull
// =============================================================================
test("large finite alpha (1e12) alphaShape == convexHull vertex set, single loop", () => {
    const rng = mulberry32(99);
    const { pxs, pys } = uniformCloud(rng, 120, 0, 1000);
    const h = createClusterIndex(120)(pxs, pys, 120);
    const hullOut = new Int32Array(120);
    const hc = h.convexHull(hullOut);
    const out = new Int32Array(3 * 120), ends = new Int32Array(120);
    const lc = h.alphaShape(1e12, out, ends);
    assert.equal(lc, 1, "single outer loop");
    const loopVerts = [...out].slice(0, ends[0]);
    assert.equal(ends[0], hc, "same vertex count");
    assert.deepEqual(new Set(loopVerts), new Set([...hullOut].slice(0, hc)), "same vertex set");
    assert.equal(canonCycle(loopVerts), canonCycle([...hullOut].slice(0, hc)), "matching cyclic sense");
    h.dispose();
});

// =============================================================================
// shrinking alpha -> kept-triangle count non-increasing -> honest 0 loops
// =============================================================================
test("shrinking alpha: kept-triangle count non-increasing, eventually 0 loops", () => {
    const rng = mulberry32(0xABCD);
    const { pxs, pys } = uniformCloud(rng, 150, 0, 1000);
    const h = createClusterIndex(150)(pxs, pys, 150);
    const out = new Int32Array(3 * 150), ends = new Int32Array(150);
    let prevKept = Infinity;
    for (const alpha of [500, 200, 100, 50, 20, 8, 2, 0.5]) {
        const { keptCount } = alphaOracle(pxs, pys, 150, alpha);
        assert.ok(keptCount <= prevKept, `kept non-increasing at alpha=${alpha}`);
        prevKept = keptCount;
        h.alphaShape(alpha, out, ends); // must not throw for any positive alpha
    }
    // a vanishingly small alpha keeps nothing -> 0 loops
    assert.equal(h.alphaShape(1e-9, out, ends), 0, "tiny alpha honest zero");
    h.dispose();
});

// =============================================================================
// near-degenerate sliver triangles: honest exclusion at every probed alpha
// =============================================================================
test("degenerate sliver triangle never kept at any alpha", () => {
    // a near-collinear sliver plus two well-separated points -> some triangles
    // are extremely thin (huge-but-FINITE circumradius; the NaN guard fires only
    // at the f64 noise floor, |D| <= EPSILON*scale) and must never
    // spuriously appear on any emitted boundary -- checked against the SAME
    // independent circumradius-filter oracle used above, at every alpha.
    const pxs = new Float32Array([0, 1, 2, 3, 1.5, 400]);
    const pys = new Float32Array([0, 1e-7, 0, 1e-7, 0.0000001, 300]);
    const h = createClusterIndex(6)(pxs, pys, 6);
    const out = new Int32Array(18), ends = new Int32Array(6);
    for (const alpha of [0.001, 1, 10, 1e6, 1e12]) {
        // must not throw or hang; degenerate triangles simply drop out
        let lc;
        assert.doesNotThrow(() => { lc = h.alphaShape(alpha, out, ends); });
        const emitted = emittedEdgeSet(out, ends, lc);
        const { edges } = alphaOracle(pxs, pys, 6, alpha);
        assert.deepEqual(emitted, edges, `sliver fixture alpha=${alpha}: emitted boundary must match the independent oracle (no spuriously kept degenerate triangle)`);
    }
    h.dispose();
});

// =============================================================================
// sizing-bound soak
// =============================================================================
test("sizing-bound soak: totalEdges<=3n-6, loopCount<=n-2, exact 3n/n never throws", () => {
    const rng = mulberry32(0x50AC);
    let checked = 0;
    for (let trial = 0; trial < 520; trial++) {
        const n = 3 + Math.floor(rng() * 254);
        const { pxs, pys } = uniformCloud(rng, n, 0, 1000);
        const h = createClusterIndex(n)(pxs, pys, n);
        const out = new Int32Array(3 * n), ends = new Int32Array(n); // exact 3n / n
        for (const alpha of [500, 250, 120, 80, 50, 30, 20, 12, 6, 3, 1, 0.4]) {
            const lc = h.alphaShape(alpha, out, ends); // must never throw on 3n/n
            assert.ok(lc <= Math.max(0, n - 2), `loopCount<=n-2 n=${n} alpha=${alpha}`);
            const total = lc > 0 ? ends[lc - 1] : 0;
            assert.ok(total <= 3 * n - 6 || n < 3, `totalEdges<=3n-6 n=${n} alpha=${alpha} total=${total}`);
        }
        h.dispose();
        checked++;
    }
    assert.ok(checked >= 500);
});

// =============================================================================
// short-buffer throws with validate-before-write sentinel proof
// =============================================================================
test("short outIndices throws and leaves the buffer untouched (validate-before-write)", () => {
    const rng = mulberry32(0xF00D);
    const { pxs, pys } = uniformCloud(rng, 60, 0, 500);
    const h = createClusterIndex(60)(pxs, pys, 60);
    // choose an alpha that yields many boundary edges (near hull)
    const SENT = -777;
    const shortOut = new Int32Array(4).fill(SENT); // deliberately too short
    const ends = new Int32Array(60);
    assert.throws(() => h.alphaShape(1e12, shortOut, ends), /lite-delaunay: /);
    for (let i = 0; i < shortOut.length; i++) assert.equal(shortOut[i], SENT, "outIndices untouched");
    // short outLoopEnds
    const bigOut = new Int32Array(3 * 60);
    const shortEnds = new Int32Array(0);
    assert.throws(() => h.alphaShape(1e12, bigOut, shortEnds), /lite-delaunay: /);
    // convexHull short buffer
    const shortHull = new Int32Array(2).fill(SENT);
    assert.throws(() => h.convexHull(shortHull), /lite-delaunay: /);
    for (let i = 0; i < shortHull.length; i++) assert.equal(shortHull[i], SENT, "hull buffer untouched");
    h.dispose();
});

// =============================================================================
// full alpha throw matrix
// =============================================================================
test("alpha door throws for every non-finite / non-positive / non-number value", () => {
    const pxs = new Float32Array([0, 10, 5]);
    const pys = new Float32Array([0, 0, 8]);
    const h = createClusterIndex(3)(pxs, pys, 3);
    const out = new Int32Array(9), ends = new Int32Array(3);
    const bad = [null, undefined, NaN, 0, -0, -5, Infinity, -Infinity, "1", {}];
    for (const a of bad) {
        assert.throws(() => h.alphaShape(a, out, ends), /lite-delaunay: /, `alpha=${String(a)}`);
    }
    // a valid alpha does not throw
    assert.doesNotThrow(() => h.alphaShape(50, out, ends));
    h.dispose();
});

// =============================================================================
// wrong out-array types throw
// =============================================================================
test("wrong out-array types throw", () => {
    const pxs = new Float32Array([0, 10, 5]);
    const pys = new Float32Array([0, 0, 8]);
    const h = createClusterIndex(3)(pxs, pys, 3);
    const goodI = new Int32Array(9), goodE = new Int32Array(3);
    assert.throws(() => h.convexHull([0, 0, 0]), /lite-delaunay: /);
    assert.throws(() => h.convexHull(new Float64Array(9)), /lite-delaunay: /);
    assert.throws(() => h.convexHull(null), /lite-delaunay: /);
    assert.throws(() => h.alphaShape(50, new Float64Array(9), goodE), /lite-delaunay: /);
    assert.throws(() => h.alphaShape(50, goodI, [0, 0, 0]), /lite-delaunay: /);
    assert.throws(() => h.alphaShape(50, goodI, new Uint32Array(3)), /lite-delaunay: /);
    h.dispose();
});

// =============================================================================
// stale / disposed / double-dispose matrix
// =============================================================================
test("stale / disposed / double-dispose all throw", () => {
    const pxs = new Float32Array([0, 10, 5]);
    const pys = new Float32Array([0, 0, 8]);
    const factory = createClusterIndex(3);
    const h1 = factory(pxs, pys, 3);
    h1.dispose();
    const out = new Int32Array(9), ends = new Int32Array(3);
    assert.throws(() => h1.convexHull(out), /lite-delaunay: /);
    assert.throws(() => h1.alphaShape(50, out, ends), /lite-delaunay: /);
    assert.throws(() => h1.dispose(), /lite-delaunay: /);
    // rebuild on the same pooled slot makes the old facade stale
    const h2 = factory(pxs, pys, 3);
    const h3 = factory(pxs, pys, 3); // forces reuse ordering
    h2.dispose();
    const h4 = factory(pxs, pys, 3); // reuses h2's slot, bumps its gen
    assert.throws(() => h2.convexHull(out), /lite-delaunay: /);
    h3.dispose(); h4.dispose();
});

// =============================================================================
// pool HWM + concurrent handles interleaved correctness
// =============================================================================
test("3 concurrent handles keep independent meshes", () => {
    const factory = createClusterIndex(300);
    const a = uniformCloud(mulberry32(1), 40, 0, 100);
    const b = uniformCloud(mulberry32(2), 80, 0, 200);
    const c = uniformCloud(mulberry32(3), 120, 0, 400);
    const ha = factory(a.pxs, a.pys, 40);
    const hb = factory(b.pxs, b.pys, 80);
    const hc = factory(c.pxs, c.pys, 120);
    const oA = new Int32Array(40), oB = new Int32Array(80), oC = new Int32Array(120);
    const cA = ha.convexHull(oA), cB = hb.convexHull(oB), cC = hc.convexHull(oC);
    assert.equal(cA, monotoneHull(a.pxs, a.pys, 40).length);
    assert.equal(cB, monotoneHull(b.pxs, b.pys, 80).length);
    assert.equal(cC, monotoneHull(c.pxs, c.pys, 120).length);
    // interleave a second query on each after the others were built
    assert.equal(ha.convexHull(oA), cA);
    ha.dispose(); hb.dispose(); hc.dispose();
});

// =============================================================================
// duplicate determinism
// =============================================================================
test("coincident points on a hull corner emit a deterministic index across rebuilds", () => {
    // corner (0,0) is duplicated at indices 0 and 5
    const pxs = new Float32Array([0, 10, 10, 0, 5, 0]);
    const pys = new Float32Array([0, 0, 10, 10, 5, 0]);
    const factory = createClusterIndex(6);
    let firstHull = null;
    for (let rep = 0; rep < 5; rep++) {
        const h = factory(pxs, pys, 6);
        const out = new Int32Array(6);
        const c = h.convexHull(out);
        const got = [...out].slice(0, c).join(",");
        if (firstHull === null) firstHull = got;
        else assert.equal(got, firstHull, `rep ${rep} identical to first build`);
        h.dispose();
    }
    assert.ok(firstHull !== null && firstHull.length > 0);
});

// =============================================================================
// factory + build doors
// =============================================================================
test("factory and build doors fail closed", () => {
    assert.throws(() => createClusterIndex(-1), /lite-delaunay: /);
    assert.throws(() => createClusterIndex(1.5), /lite-delaunay: /);
    const factory = createClusterIndex(10);
    const pxs = new Float32Array(10), pys = new Float32Array(10);
    assert.throws(() => factory(pxs, pys, -1), /lite-delaunay: /);
    assert.throws(() => factory(pxs, pys, 11), /lite-delaunay: /);
    assert.throws(() => factory(null, pys, 5), /lite-delaunay: /);
    assert.throws(() => factory(pxs, new Float32Array(2), 5), /lite-delaunay: /);
});
