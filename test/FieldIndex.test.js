// @zakkster/lite-delaunay -- correctness suite for createFieldIndex (v1.3.0)
// -----------------------------------------------------------------------------
// node:test only (no bespoke harness, no deps beyond the package under test).
// Run with:
//   node --expose-gc --test test/FieldIndex.test.js
//
// Covers the planner's assertions:
//   A1  PLANAR EXACTNESS    -- interpolate() and sampleField() reproduce an
//                               exact plane z=3+0.75x-0.5y to float tolerance.
//   A2  LOCATE VS BRUTE FORCE -- locate() agrees with an O(T) linear scan (via
//                               triangleVertices + original coords) across 3
//                               cloud shapes x 5000 queries, inside and outside
//                               the hull.
//   A3  BARYCENTRIC          -- weights sum to 1, are non-negative, and
//                               reconstruct q from ORIGINAL triangle coords;
//                               a different triangle returns false but still
//                               writes 3 finite weights.
//   A4  SAMPLEFIELD REFERENCE -- every cell of a straddling-bbox rasterization
//                               matches a per-cell locate+interpolate walk,
//                               NaN mask included, return value = finite count.
//   A5  GRID ORIENTATION     -- row 0 = yMin (+y-up), col 0 = xMin; explicit
//                               index = row*gridW + col.
//   A6  FAIL-CLOSED VALUES   -- degenerate matrix (n=0,1,2, all-NaN, 64
//                               collinear, 64 coincident); NaN-z confinement;
//                               Infinity-z exclusion from the finite count.
//   A7  THROW MATRIX         -- every documented throw case, message-checked,
//                               plus validate-before-write on sampleField.
//   A8  LIFECYCLE + POOL     -- dispose/double-dispose/stale-handle/slot-reuse,
//                               2 concurrent handles with independent cursors,
//                               zs-swap on one mesh.
//
// Plus one ADVERSARIAL case the planner did not enumerate (sampleField called
// with outGrid aliasing the SAME array object as zs -- proves the gather-
// before-write ordering survives) and one BOUNDARY case (dispose-during-
// iteration, mirroring test/CellIndex.test.js's pattern).

import { test } from "node:test";
import assert from "node:assert/strict";
import { createFieldIndex } from "../Delaunay.js";

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
// cloud generators -- copied/adapted from test/CellIndex.test.js's pattern
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

// Jittered grid -- breaks exact cocircularity while keeping a "small grid"
// point pattern.
function genGridJittered(rng, side, cellSize) {
    const n = side * side;
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    const jitter = cellSize * 0.15;
    let k = 0;
    for (let r = 0; r < side; r++) {
        for (let c = 0; c < side; c++) {
            pxs[k] = c * cellSize + (rng() - 0.5) * 2 * jitter;
            pys[k] = r * cellSize + (rng() - 0.5) * 2 * jitter;
            k++;
        }
    }
    return { pxs, pys };
}

// -----------------------------------------------------------------------------
// geometry helpers
// -----------------------------------------------------------------------------
function isFinitePoint(px, py) {
    return px === px && py === py &&
        px !== Infinity && px !== -Infinity &&
        py !== Infinity && py !== -Infinity;
}

// Bbox of the finite subset, expanded by marginFrac * span on every side (so
// a query sweep over the result covers points both inside AND outside the
// convex hull).
function expandedBBox(pxs, pys, n, marginFrac) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
        const px = pxs[i], py = pys[i];
        if (!isFinitePoint(px, py)) continue;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
    }
    const spanX = (maxX - minX) || 1, spanY = (maxY - minY) || 1;
    return {
        x0: minX - spanX * marginFrac, y0: minY - spanY * marginFrac,
        x1: maxX + spanX * marginFrac, y1: maxY + spanY * marginFrac,
    };
}

// Barycentric weights of (qx,qy) in triangle (ax,ay)-(bx,by)-(cx,cy), computed
// independently of the module under test (same formula family the field index
// documents, but re-derived here rather than imported).
function baryOf(ax, ay, bx, by, cx, cy, qx, qy) {
    const v0x = bx - ax, v0y = by - ay, v1x = cx - ax, v1y = cy - ay;
    const d = v0x * v1y - v0y * v1x;
    const qx0 = qx - ax, qy0 = qy - ay;
    const w1 = (qx0 * v1y - qy0 * v1x) / d;
    const w2 = (v0x * qy0 - v0y * qx0) / d;
    const w0 = 1 - w1 - w2;
    return { w0, w1, w2, d };
}

// O(T) brute-force locate: scan every triangle (via triangleVertices + the
// ORIGINAL pxs/pys used to build) for one that contains (qx,qy). Independent
// of the walk under test (a straight linear scan, not a remembering walk).
function bruteLocate(handle, T, pxs, pys, tmpI3, qx, qy, tol) {
    for (let t = 0; t < T; t++) {
        handle.triangleVertices(t, tmpI3);
        const ia = tmpI3[0], ib = tmpI3[1], ic = tmpI3[2];
        const { w0, w1, w2, d } = baryOf(
            pxs[ia], pys[ia], pxs[ib], pys[ib], pxs[ic], pys[ic], qx, qy);
        if (d === 0) continue;
        if (w0 >= -tol && w1 >= -tol && w2 >= -tol) return t;
    }
    return -1;
}

// Does triangle t (via triangleVertices + original coords) contain (qx,qy)?
function triContainsQ(handle, t, pxs, pys, tmpI3, qx, qy, tol) {
    handle.triangleVertices(t, tmpI3);
    const ia = tmpI3[0], ib = tmpI3[1], ic = tmpI3[2];
    const { w0, w1, w2 } = baryOf(
        pxs[ia], pys[ia], pxs[ib], pys[ib], pxs[ic], pys[ic], qx, qy);
    return (w0 >= -tol && w1 >= -tol && w2 >= -tol);
}

// =============================================================================
// A1 PLANAR EXACTNESS
// =============================================================================
test("A1 PLANAR EXACTNESS: interpolate() matches z=3+0.75x-0.5y on 5000 rejection-sampled interior queries", () => {
    const N = 512;
    const rng = mulberry32(0xA100);
    const { pxs, pys } = genUniform(rng, N, 0, 1000); // Float32Array
    const factory = createFieldIndex(N);
    const handle = factory(pxs, pys, N);

    const zs = new Float64Array(N);
    for (let i = 0; i < N; i++) zs[i] = 3 + 0.75 * pxs[i] - 0.5 * pys[i];

    const qrng = mulberry32(0xA101);
    let checked = 0, attempts = 0, failures = 0;
    const MAX_ATTEMPTS = 300000;
    while (checked < 5000 && attempts < MAX_ATTEMPTS) {
        attempts++;
        const qx = qrng() * 1000, qy = qrng() * 1000;
        const t = handle.locate(qx, qy);
        if (t < 0) continue;
        const val = handle.interpolate(zs, qx, qy);
        const exact = 3 + 0.75 * qx - 0.5 * qy;
        const tol = 1e-3 * (1 + Math.abs(exact));
        if (Math.abs(val - exact) > tol) failures++;
        checked++;
    }
    assert.ok(checked >= 5000, `too few interior samples: ${checked} (attempts=${attempts})`);
    assert.equal(failures, 0, `${failures}/${checked} planar-interpolate mismatches`);
    handle.dispose();
    console.log(`A1 PLANAR EXACTNESS interpolate checked=${checked} attempts=${attempts} failures=${failures}`);
});

test("A1 PLANAR EXACTNESS: sampleField() 64x64 matches z=3+0.75x-0.5y at every FINITE cell", () => {
    const N = 512;
    const rng = mulberry32(0xA110);
    const { pxs, pys } = genUniform(rng, N, 0, 1000);
    const factory = createFieldIndex(N);
    const handle = factory(pxs, pys, N);

    const zs = new Float64Array(N);
    for (let i = 0; i < N; i++) zs[i] = 3 + 0.75 * pxs[i] - 0.5 * pys[i];

    const gridW = 64, gridH = 64;
    const bx0 = 200, by0 = 200, bx1 = 800, by1 = 800; // interior of [0,1000]^2
    const outGrid = new Float64Array(gridW * gridH);
    const count = handle.sampleField(zs, gridW, gridH, bx0, by0, bx1, by1, outGrid);

    const dx = (bx1 - bx0) / gridW, dy = (by1 - by0) / gridH;
    let finiteChecked = 0, failures = 0;
    for (let row = 0; row < gridH; row++) {
        const qy = by0 + (row + 0.5) * dy;
        for (let col = 0; col < gridW; col++) {
            const qx = bx0 + (col + 0.5) * dx;
            const v = outGrid[row * gridW + col];
            if (!Number.isFinite(v)) continue;
            const exact = 3 + 0.75 * qx - 0.5 * qy;
            const tol = 1e-3 * (1 + Math.abs(exact));
            if (Math.abs(v - exact) > tol) failures++;
            finiteChecked++;
        }
    }
    assert.equal(failures, 0, `${failures}/${finiteChecked} planar-sampleField mismatches`);
    assert.equal(finiteChecked, count, `returned count ${count} != finite cells checked ${finiteChecked}`);
    assert.ok(finiteChecked > gridW * gridH * 0.5, `too few finite cells: ${finiteChecked}`);
    handle.dispose();
    console.log(`A1 PLANAR EXACTNESS sampleField finiteChecked=${finiteChecked} of ${gridW * gridH} failures=${failures}`);
});

// =============================================================================
// A2 LOCATE VS BRUTE FORCE
// =============================================================================
function runLocateVsBrute(name, pxs, pys, n, marginFrac, seed) {
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);
    const T = handle.triangleCount();
    assert.ok(T > 0, `${name}: test setup needs a real mesh`);
    const tmpI3 = new Int32Array(3);
    const bbox = expandedBBox(pxs, pys, n, marginFrac);
    const qrng = mulberry32(seed);
    const SAMPLES = 5000;
    let mismatches = 0, insideChecked = 0, outsideChecked = 0;
    for (let s = 0; s < SAMPLES; s++) {
        const qx = bbox.x0 + qrng() * (bbox.x1 - bbox.x0);
        const qy = bbox.y0 + qrng() * (bbox.y1 - bbox.y0);
        const bruteT = bruteLocate(handle, T, pxs, pys, tmpI3, qx, qy, 1e-9);
        const locT = handle.locate(qx, qy);
        if (bruteT >= 0) {
            if (locT < 0 || !triContainsQ(handle, locT, pxs, pys, tmpI3, qx, qy, 1e-9)) {
                mismatches++;
            } else {
                insideChecked++;
            }
        } else {
            if (locT !== -1) mismatches++; else outsideChecked++;
        }
    }
    assert.equal(mismatches, 0, `${name}: locate-vs-brute mismatches=${mismatches}/${SAMPLES}`);
    handle.dispose();
    console.log(`A2 LOCATE VS BRUTE FORCE ${name} inside=${insideChecked} outside=${outsideChecked} mismatches=${mismatches}`);
    return { insideChecked, outsideChecked, mismatches };
}

test("A2 LOCATE VS BRUTE FORCE: uniform cloud, 5000 queries incl. outside the hull", () => {
    const rng = mulberry32(0xA200);
    const { pxs, pys } = genUniform(rng, 300, -200, 200);
    const r = runLocateVsBrute("uniform", pxs, pys, 300, 0.4, 0xA201);
    assert.ok(r.insideChecked > 0 && r.outsideChecked > 0, "expected both inside and outside samples");
});

test("A2 LOCATE VS BRUTE FORCE: clustered cloud, 5000 queries incl. outside the hull", () => {
    const rng = mulberry32(0xA210);
    const { pxs, pys } = genClustered(rng, 300);
    const r = runLocateVsBrute("clustered", pxs, pys, 300, 0.4, 0xA211);
    assert.ok(r.insideChecked > 0 && r.outsideChecked > 0, "expected both inside and outside samples");
});

test("A2 LOCATE VS BRUTE FORCE: jittered-grid cloud, 5000 queries incl. outside the hull", () => {
    const rng = mulberry32(0xA220);
    const { pxs, pys } = genGridJittered(rng, 18, 10); // n=324
    const r = runLocateVsBrute("jittered-grid", pxs, pys, 324, 0.35, 0xA221);
    assert.ok(r.insideChecked > 0 && r.outsideChecked > 0, "expected both inside and outside samples");
});

// =============================================================================
// A3 BARYCENTRIC
// =============================================================================
test("A3 BARYCENTRIC: weights sum to 1, are non-negative, and reconstruct q from ORIGINAL triangle coords", () => {
    const N = 400;
    const rng = mulberry32(0xA300);
    const { pxs, pys } = genUniform(rng, N, -500, 500); // bbox diagonal ~1414
    const factory = createFieldIndex(N);
    const handle = factory(pxs, pys, N);
    const outW3 = new Float64Array(3);
    const tmpI3 = new Int32Array(3);

    const qrng = mulberry32(0xA301);
    let checked = 0, attempts = 0;
    const DIAG = 1414;
    while (checked < 2000 && attempts < 300000) {
        attempts++;
        const qx = qrng() * 1000 - 500, qy = qrng() * 1000 - 500;
        const t = handle.locate(qx, qy);
        if (t < 0) continue;
        const isInside = handle.barycentric(t, qx, qy, outW3);
        assert.ok(isInside, `locate()'s own triangle t=${t} must be inside-or-on per barycentric()`);
        const w0 = outW3[0], w1 = outW3[1], w2 = outW3[2];
        assert.ok(Math.abs(w0 + w1 + w2 - 1) <= 1e-9, `weights sum ${w0 + w1 + w2} != 1`);
        assert.ok(w0 >= -1e-9 && w1 >= -1e-9 && w2 >= -1e-9, `negative weight (${w0},${w1},${w2})`);
        handle.triangleVertices(t, tmpI3);
        const ax = pxs[tmpI3[0]], ay = pys[tmpI3[0]];
        const bx = pxs[tmpI3[1]], by = pys[tmpI3[1]];
        const cx = pxs[tmpI3[2]], cy = pys[tmpI3[2]];
        const rx = w0 * ax + w1 * bx + w2 * cx, ry = w0 * ay + w1 * by + w2 * cy;
        const tol = 1e-6 * DIAG;
        assert.ok(Math.abs(rx - qx) <= tol && Math.abs(ry - qy) <= tol,
            `reconstruction mismatch: got (${rx},${ry}) want (${qx},${qy})`);
        checked++;
    }
    assert.ok(checked >= 2000, `too few checked: ${checked}`);
    handle.dispose();
    console.log(`A3 BARYCENTRIC reconstruct checked=${checked}`);
});

test("A3 BARYCENTRIC: a query in a DIFFERENT triangle returns false but still writes 3 finite weights", () => {
    const N = 400;
    const rng = mulberry32(0xA310);
    const { pxs, pys } = genUniform(rng, N, -500, 500);
    const factory = createFieldIndex(N);
    const handle = factory(pxs, pys, N);
    const T = handle.triangleCount();
    const outW3 = new Float64Array(3);
    const tmpI3 = new Int32Array(3);

    const qrng = mulberry32(0xA311);
    let checked = 0, attempts = 0;
    while (checked < 300 && attempts < 100000) {
        attempts++;
        const qx = qrng() * 1000 - 500, qy = qrng() * 1000 - 500;
        const t = handle.locate(qx, qy);
        if (t < 0) continue;
        // Find a triangle GUARANTEED to not contain q (independent scan).
        let other = -1;
        for (let k = 0; k < T; k++) {
            const cand = (t + 1 + k) % T;
            if (!triContainsQ(handle, cand, pxs, pys, tmpI3, qx, qy, 1e-9)) { other = cand; break; }
        }
        if (other < 0) continue; // pathological: skip (mesh too small)
        const isInside = handle.barycentric(other, qx, qy, outW3);
        assert.equal(isInside, false, `triangle ${other} (!= located ${t}) must report false for q`);
        assert.ok(Number.isFinite(outW3[0]) && Number.isFinite(outW3[1]) && Number.isFinite(outW3[2]),
            `barycentric() on a non-containing triangle must still write 3 finite weights, got (${outW3[0]},${outW3[1]},${outW3[2]})`);
        checked++;
    }
    assert.ok(checked >= 300, `too few checked: ${checked}`);
    handle.dispose();
    console.log(`A3 BARYCENTRIC different-triangle checked=${checked}`);
});

// =============================================================================
// A4 SAMPLEFIELD REFERENCE
// =============================================================================
test("A4 SAMPLEFIELD REFERENCE: 64x64 over a hull-straddling bbox matches per-cell locate+interpolate exactly", () => {
    const rng = mulberry32(0xA400);
    const { pxs, pys } = genGridJittered(rng, 19, 32); // n=361, extent ~[0,608]^2
    const n = 361;
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);

    const zs = new Float64Array(n);
    for (let i = 0; i < n; i++) zs[i] = Math.sin(pxs[i] * 0.02) + Math.cos(pys[i] * 0.015) * 3;

    const gridW = 64, gridH = 64;
    const bx0 = -100, by0 = -100, bx1 = 700, by1 = 700; // straddles the hull
    const outGrid = new Float64Array(gridW * gridH);
    const count = handle.sampleField(zs, gridW, gridH, bx0, by0, bx1, by1, outGrid);

    const dx = (bx1 - bx0) / gridW, dy = (by1 - by0) / gridH;
    let finiteMatches = 0, nanMatches = 0, mismatches = 0;
    for (let row = 0; row < gridH; row++) {
        const qy = by0 + (row + 0.5) * dy;
        for (let col = 0; col < gridW; col++) {
            const qx = bx0 + (col + 0.5) * dx;
            const t = handle.locate(qx, qy);
            const ref = t < 0 ? NaN : handle.interpolate(zs, qx, qy);
            const v = outGrid[row * gridW + col];
            const refNaN = Number.isNaN(ref), vNaN = Number.isNaN(v);
            if (refNaN !== vNaN) { mismatches++; continue; }
            if (refNaN) { nanMatches++; continue; }
            if (Math.abs(v - ref) > 1e-12) mismatches++; else finiteMatches++;
        }
    }
    assert.equal(mismatches, 0, `${mismatches}/${gridW * gridH} sampleField-vs-reference mismatches`);
    assert.equal(finiteMatches, count, `returned count ${count} != finite-matched cells ${finiteMatches}`);
    assert.ok(nanMatches > 0, "expected at least one out-of-hull NaN cell (bbox must straddle the hull)");
    handle.dispose();
    console.log(`A4 SAMPLEFIELD REFERENCE finite=${finiteMatches} nan=${nanMatches} mismatches=${mismatches} returned=${count}`);
});

// =============================================================================
// A5 GRID ORIENTATION
// =============================================================================
test("A5 GRID ORIENTATION: row 0 = yMin (+y-up) for z=y; explicit index = row*gridW+col", () => {
    // Square + center: (0,0),(10,0),(10,10),(0,10),(5,5) -- a non-degenerate
    // 4-triangle fan, no exact cocircularity issue (center breaks the tie).
    const pxs = new Float32Array([0, 10, 10, 0, 5]);
    const pys = new Float32Array([0, 0, 10, 10, 5]);
    const n = 5;
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);
    assert.ok(handle.triangleCount() >= 3, "test setup: fixture must triangulate to a real mesh");

    const zsY = new Float64Array(pys); // z = y
    const gridW = 4, gridH = 4;
    const outGrid = new Float64Array(gridW * gridH);
    const count = handle.sampleField(zsY, gridW, gridH, 0, 0, 10, 10, outGrid);
    assert.ok(count > 0, "expected finite cells");

    let colsChecked = 0;
    for (let col = 0; col < gridW; col++) {
        const rowLo = outGrid[0 * gridW + col];
        const rowHi = outGrid[3 * gridW + col];
        if (!Number.isFinite(rowLo) || !Number.isFinite(rowHi)) continue;
        assert.ok(rowLo < rowHi, `col ${col}: row0 (${rowLo}) must be < row3 (${rowHi}) for z=y`);
        colsChecked++;
    }
    assert.ok(colsChecked > 0, "no finite row0/row3 pair to compare -- fixture too small");

    // Explicit index formula check at an arbitrary interior cell.
    const row = 2, col = 1;
    const dx = 10 / gridW, dy = 10 / gridH;
    const qx = (col + 0.5) * dx, qy = (row + 0.5) * dy;
    const ref = handle.interpolate(zsY, qx, qy);
    assert.ok(Math.abs(outGrid[row * gridW + col] - ref) <= 1e-9,
        `index formula: outGrid[row*gridW+col] must equal the (row,col) cell-center interpolation`);

    handle.dispose();
    console.log(`A5 GRID ORIENTATION row-test colsChecked=${colsChecked}/${gridW} count=${count}`);
});

test("A5 GRID ORIENTATION: col 0 = xMin for z=x", () => {
    const pxs = new Float32Array([0, 10, 10, 0, 5]);
    const pys = new Float32Array([0, 0, 10, 10, 5]);
    const n = 5;
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);

    const zsX = new Float64Array(pxs); // z = x
    const gridW = 4, gridH = 4;
    const outGrid = new Float64Array(gridW * gridH);
    const count = handle.sampleField(zsX, gridW, gridH, 0, 0, 10, 10, outGrid);
    assert.ok(count > 0, "expected finite cells");

    let rowsChecked = 0;
    for (let row = 0; row < gridH; row++) {
        const colLo = outGrid[row * gridW + 0];
        const colHi = outGrid[row * gridW + 3];
        if (!Number.isFinite(colLo) || !Number.isFinite(colHi)) continue;
        assert.ok(colLo < colHi, `row ${row}: col0 (${colLo}) must be < col3 (${colHi}) for z=x`);
        rowsChecked++;
    }
    assert.ok(rowsChecked > 0, "no finite col0/col3 pair to compare -- fixture too small");
    handle.dispose();
    console.log(`A5 GRID ORIENTATION col-test rowsChecked=${rowsChecked}/${gridH} count=${count}`);
});

// =============================================================================
// A6 FAIL-CLOSED VALUES
// =============================================================================
function assertDegenerate(name, pxs, pys, n, maxPoints) {
    const factory = createFieldIndex(maxPoints);
    const handle = factory(pxs, pys, n);
    assert.equal(handle.triangleCount(), 0, `${name}: triangleCount must be 0`);

    const qrng = mulberry32(0xDE9E ^ (n + 1));
    let locateChecked = 0;
    for (let i = 0; i < 10; i++) {
        const qx = qrng() * 200 - 100, qy = qrng() * 200 - 100;
        assert.equal(handle.locate(qx, qy), -1, `${name}: locate(${qx},${qy}) must be -1`);
        locateChecked++;
    }

    const gridW = 8, gridH = 8, total = gridW * gridH;
    const SENTINEL = -987654.321;
    const outGrid = new Float64Array(total + 4).fill(SENTINEL);
    const zs = new Float64Array(Math.max(n, 1)).fill(2.5);
    const count = handle.sampleField(zs, gridW, gridH, -50, -50, 50, 50, outGrid);
    assert.equal(count, 0, `${name}: sampleField must return 0`);
    let nanCount = 0;
    for (let k = 0; k < total; k++) {
        assert.ok(Number.isNaN(outGrid[k]), `${name}: cell ${k} must be NaN`);
        nanCount++;
    }
    for (let k = total; k < outGrid.length; k++) {
        assert.equal(outGrid[k], SENTINEL, `${name}: beyond-prefix cell ${k} must be untouched`);
    }
    handle.dispose();
    console.log(`A6 DEGENERATE ${name} locate=${locateChecked}/10 nanCells=${nanCount}/${total}`);
}

test("A6 FAIL-CLOSED VALUES: degenerate build n=0", () => {
    assertDegenerate("n=0", new Float32Array(0), new Float32Array(0), 0, 5);
});

test("A6 FAIL-CLOSED VALUES: degenerate build n=1", () => {
    const rng = mulberry32(0x0601);
    const { pxs, pys } = genUniform(rng, 1, -10, 10);
    assertDegenerate("n=1", pxs, pys, 1, 5);
});

test("A6 FAIL-CLOSED VALUES: degenerate build n=2", () => {
    const rng = mulberry32(0x0602);
    const { pxs, pys } = genUniform(rng, 2, -10, 10);
    assertDegenerate("n=2", pxs, pys, 2, 5);
});

test("A6 FAIL-CLOSED VALUES: degenerate build all-NaN (n=50)", () => {
    const n = 50;
    assertDegenerate("all-NaN", new Float32Array(n).fill(NaN), new Float32Array(n).fill(NaN), n, n);
});

test("A6 FAIL-CLOSED VALUES: degenerate build 64 collinear points", () => {
    const n = 64;
    const rng = mulberry32(0x0603);
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const t = rng() * 1000 - 500;
        pxs[i] = t; pys[i] = 2 * t + 3;
    }
    assertDegenerate("64-collinear", pxs, pys, n, n);
});

test("A6 FAIL-CLOSED VALUES: degenerate build 64 coincident points", () => {
    const n = 64;
    const pxs = new Float32Array(n).fill(3.5);
    const pys = new Float32Array(n).fill(-2.25);
    assertDegenerate("64-coincident", pxs, pys, n, n);
});

test("A6 FAIL-CLOSED VALUES: a NaN z corner confines NaN to exactly its incident triangles", () => {
    const rng = mulberry32(0xA6A6);
    const { pxs, pys } = genGridJittered(rng, 10, 40); // n=100
    const n = 100;
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);
    const T = handle.triangleCount();
    assert.ok(T > 0, "test setup: need a real mesh");

    const nanSite = 45;
    const zs = new Float64Array(n);
    for (let i = 0; i < n; i++) zs[i] = Math.sin(pxs[i] * 0.02) + Math.cos(pys[i] * 0.02);
    zs[nanSite] = NaN;

    const tmpI3 = new Int32Array(3);
    const incidentCentroids = [];
    const incidentTriSet = new Set();
    for (let t = 0; t < T; t++) {
        handle.triangleVertices(t, tmpI3);
        if (tmpI3[0] === nanSite || tmpI3[1] === nanSite || tmpI3[2] === nanSite) {
            incidentTriSet.add(t);
            const ax = pxs[tmpI3[0]], ay = pys[tmpI3[0]];
            const bx = pxs[tmpI3[1]], by = pys[tmpI3[1]];
            const cx = pxs[tmpI3[2]], cy = pys[tmpI3[2]];
            incidentCentroids.push([(ax + bx + cx) / 3, (ay + by + cy) / 3]);
        }
    }
    assert.ok(incidentTriSet.size > 0, "test setup: nanSite must be part of the mesh");

    let nanChecked = 0;
    for (const [cx, cy] of incidentCentroids) {
        const val = handle.interpolate(zs, cx, cy);
        assert.ok(Number.isNaN(val), `incident centroid (${cx},${cy}) must be NaN`);
        nanChecked++;
    }

    const bbox = expandedBBox(pxs, pys, n, 0.0);
    const qrng = mulberry32(0xA6A7);
    let elsewhereChecked = 0, attempts = 0;
    while (elsewhereChecked < 200 && attempts < 300000) {
        attempts++;
        const qx = bbox.x0 + qrng() * (bbox.x1 - bbox.x0);
        const qy = bbox.y0 + qrng() * (bbox.y1 - bbox.y0);
        const t = handle.locate(qx, qy);
        if (t < 0 || incidentTriSet.has(t)) continue;
        const val = handle.interpolate(zs, qx, qy);
        assert.ok(Number.isFinite(val), `non-incident query (${qx},${qy}) triangle ${t} should be finite, got ${val}`);
        elsewhereChecked++;
    }
    assert.ok(elsewhereChecked >= 200, `too few elsewhere samples: ${elsewhereChecked}`);
    handle.dispose();
    console.log(`A6 NaN-confinement incident=${incidentCentroids.length} elsewhereChecked=${elsewhereChecked}`);
});

test("A6 FAIL-CLOSED VALUES: an Infinity z produces non-finite cells excluded from the returned count", () => {
    const rng = mulberry32(0xA6B0);
    const { pxs, pys } = genGridJittered(rng, 10, 40); // n=100
    const n = 100;
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);

    const infSite = 60;
    const zs = new Float64Array(n);
    for (let i = 0; i < n; i++) zs[i] = Math.sin(pxs[i] * 0.02) + Math.cos(pys[i] * 0.02);
    zs[infSite] = Infinity;

    const bbox = expandedBBox(pxs, pys, n, 0.05);
    const gridW = 32, gridH = 32, total = gridW * gridH;
    const outGrid = new Float64Array(total);
    const count = handle.sampleField(zs, gridW, gridH, bbox.x0, bbox.y0, bbox.x1, bbox.y1, outGrid);

    let finiteCount = 0, nonFiniteCount = 0;
    for (let k = 0; k < total; k++) {
        if (Number.isFinite(outGrid[k])) finiteCount++; else nonFiniteCount++;
    }
    assert.equal(finiteCount, count, `returned count ${count} must equal actual finite cells ${finiteCount}`);
    assert.ok(nonFiniteCount > 0, "expected at least one non-finite cell from the Infinity z site");
    handle.dispose();
    console.log(`A6 Infinity-exclusion finite=${finiteCount} nonFinite=${nonFiniteCount} returned=${count}`);
});

// =============================================================================
// A7 THROW MATRIX
// =============================================================================
test("A7 THROW MATRIX: createFieldIndex(maxPoints) validation matrix", () => {
    for (const bad of [-1, 1.5, NaN, undefined, null, "5", Infinity, -0.5]) {
        assert.throws(() => createFieldIndex(bad), /lite-delaunay: /, `maxPoints=${String(bad)} must throw`);
    }
    assert.doesNotThrow(() => createFieldIndex(0), "maxPoints=0 must NOT throw");
    assert.doesNotThrow(() => createFieldIndex(-0), "maxPoints=-0 must NOT throw (integer zero)");
});

test("A7 THROW MATRIX: factory(pxs,pys,n) build validation matrix", () => {
    const MAXP = 5;
    const factory = createFieldIndex(MAXP);
    const pxs = new Float32Array(6).fill(1);
    const pys = new Float32Array(6).fill(1);

    for (const bad of [-1, 1.5, NaN, undefined, null, "3"]) {
        assert.throws(() => factory(pxs, pys, bad), /lite-delaunay: /, `n=${String(bad)} must throw`);
    }
    assert.throws(() => factory(pxs, pys, MAXP + 1), /lite-delaunay: /, "n = maxPoints + 1 must throw");
    assert.doesNotThrow(() => factory(pxs, pys, MAXP).dispose(), "n = maxPoints must succeed");
    assert.doesNotThrow(() => factory(pxs, pys, MAXP - 1).dispose(), "n = maxPoints - 1 must succeed");
    assert.doesNotThrow(() => factory([], [], -0).dispose(), "n = -0 must NOT throw");

    assert.throws(() => factory(null, pys, 4), /lite-delaunay: /);
    assert.throws(() => factory(pxs, null, 4), /lite-delaunay: /);
    assert.throws(() => factory(undefined, undefined, 4), /lite-delaunay: /);
    assert.throws(() => factory(new Float32Array(2), new Float32Array(2), 3), /lite-delaunay: /, "pxs shorter than n");

    assert.doesNotThrow(() => factory([], [], 0).dispose());
});

test("A7 THROW MATRIX: barycentric()/triangleVertices() t-range + out-array length (incl. degenerate build)", () => {
    // Degenerate build: triangleCount()=0, so [0,0) is empty -- ANY t throws.
    const degFactory = createFieldIndex(5);
    const degHandle = degFactory(new Float32Array(5).fill(NaN), new Float32Array(5).fill(NaN), 5);
    const outW3 = new Float64Array(3), outI3 = new Int32Array(3);
    for (const t of [0, -1, 0.5, NaN, 1]) {
        assert.throws(() => degHandle.barycentric(t, 0, 0, outW3), /lite-delaunay: /, `degenerate t=${String(t)} barycentric must throw`);
        assert.throws(() => degHandle.triangleVertices(t, outI3), /lite-delaunay: /, `degenerate t=${String(t)} triangleVertices must throw`);
    }
    degHandle.dispose();

    // Real mesh.
    const n = 40;
    const rng = mulberry32(0xA710);
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = rng() * 100; pys[i] = rng() * 100; }
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);
    const T = handle.triangleCount();
    assert.ok(T > 0, "test setup: need a real mesh");

    for (const t of [-1, T, T + 1, 0.5, NaN, -0.5]) {
        assert.throws(() => handle.barycentric(t, 0, 0, outW3), /lite-delaunay: /, `t=${String(t)} barycentric range must throw`);
        assert.throws(() => handle.triangleVertices(t, outI3), /lite-delaunay: /, `t=${String(t)} triangleVertices range must throw`);
    }
    // Boundary: 0, T-1, and -0 (integer zero, must NOT throw) are all valid.
    assert.doesNotThrow(() => handle.barycentric(0, pxs[0], pys[0], outW3));
    assert.doesNotThrow(() => handle.triangleVertices(0, outI3));
    assert.doesNotThrow(() => handle.barycentric(T - 1, 0, 0, outW3));
    assert.doesNotThrow(() => handle.triangleVertices(T - 1, outI3));
    assert.doesNotThrow(() => handle.barycentric(-0, pxs[0], pys[0], outW3), "t=-0 must behave like t=0");

    // Out-array too short / wrong.
    assert.throws(() => handle.barycentric(0, 0, 0, new Float64Array(2)), /lite-delaunay: /);
    assert.throws(() => handle.barycentric(0, 0, 0, [1, 2]), /lite-delaunay: /);
    assert.throws(() => handle.barycentric(0, 0, 0, null), /lite-delaunay: /);
    assert.throws(() => handle.triangleVertices(0, new Int32Array(2)), /lite-delaunay: /);
    assert.throws(() => handle.triangleVertices(0, [1, 2]), /lite-delaunay: /);
    assert.throws(() => handle.triangleVertices(0, null), /lite-delaunay: /);
    handle.dispose();
});

test("A7 THROW MATRIX: interpolate()/sampleField() zs missing/short (Float32Array/Float64Array/number[] all accepted)", () => {
    const n = 30;
    const rng = mulberry32(0xA720);
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = rng() * 50; pys[i] = rng() * 50; }
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);

    assert.throws(() => handle.interpolate(null, 0, 0), /lite-delaunay: /);
    assert.throws(() => handle.interpolate(undefined, 0, 0), /lite-delaunay: /);
    assert.throws(() => handle.interpolate(new Float64Array(n - 1), 0, 0), /lite-delaunay: /);
    assert.doesNotThrow(() => handle.interpolate(new Float64Array(n), 0, 0));
    assert.doesNotThrow(() => handle.interpolate(new Array(n).fill(1), 0, 0));
    assert.doesNotThrow(() => handle.interpolate(new Float32Array(n), 0, 0));

    const outGrid = new Float64Array(16);
    assert.throws(() => handle.sampleField(null, 4, 4, 0, 0, 10, 10, outGrid), /lite-delaunay: /);
    assert.throws(() => handle.sampleField(new Float64Array(n - 1), 4, 4, 0, 0, 10, 10, outGrid), /lite-delaunay: /);
    assert.doesNotThrow(() => handle.sampleField(new Float64Array(n), 4, 4, 0, 0, 10, 10, outGrid));
    assert.doesNotThrow(() => handle.sampleField(new Array(n).fill(1), 4, 4, 0, 0, 10, 10, outGrid));
    handle.dispose();
});

test("A7 THROW MATRIX: sampleField() gridW/gridH, outGrid type/length, bbox validation", () => {
    const n = 30;
    const rng = mulberry32(0xA730);
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = rng() * 50; pys[i] = rng() * 50; }
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);
    const zs = new Float64Array(n).fill(1);
    const outGrid = new Float64Array(64);

    for (const bad of [0, -1, 1.5, NaN, Infinity, "4", null, undefined]) {
        assert.throws(() => handle.sampleField(zs, bad, 4, 0, 0, 10, 10, outGrid), /lite-delaunay: /, `gridW=${String(bad)}`);
        assert.throws(() => handle.sampleField(zs, 4, bad, 0, 0, 10, 10, outGrid), /lite-delaunay: /, `gridH=${String(bad)}`);
    }
    assert.throws(() => handle.sampleField(zs, 4, 4, 0, 0, 10, 10, null), /lite-delaunay: /);
    assert.throws(() => handle.sampleField(zs, 4, 4, 0, 0, 10, 10, new Array(16).fill(0)), /lite-delaunay: /, "number[] outGrid must throw");
    assert.throws(() => handle.sampleField(zs, 4, 4, 0, 0, 10, 10, new Float64Array(15)), /lite-delaunay: /, "outGrid too short");
    assert.doesNotThrow(() => handle.sampleField(zs, 4, 4, 0, 0, 10, 10, new Float64Array(16)));
    assert.doesNotThrow(() => handle.sampleField(zs, 4, 4, 0, 0, 10, 10, new Float32Array(16)));

    const badBoxes = [
        [NaN, -10, 10, 10], [-10, NaN, 10, 10], [-10, -10, NaN, 10], [-10, -10, 10, NaN],
        [Infinity, -10, 10, 10], [-Infinity, -10, 10, 10],
        [-10, Infinity, 10, 10], [-10, -Infinity, 10, 10],
        [-10, -10, Infinity, 10], [-10, -10, -Infinity, 10],
        [-10, -10, 10, Infinity], [-10, -10, 10, -Infinity],
        [5, -10, 5, 10], [10, -10, 5, 10],
        [-10, 5, 10, 5], [-10, 10, 10, -10],
    ];
    for (const [bx0, by0, bx1, by1] of badBoxes) {
        assert.throws(() => handle.sampleField(zs, 4, 4, bx0, by0, bx1, by1, outGrid), /lite-delaunay: /,
            `bbox (${bx0},${by0},${bx1},${by1}) must throw`);
    }
    assert.doesNotThrow(() => handle.sampleField(zs, 4, 4, -10, -10, 10, 10, outGrid));
    handle.dispose();
});

test("A7 THROW MATRIX: a throwing sampleField() call does NOT write outGrid (validate-before-write)", () => {
    const n = 30;
    const rng = mulberry32(0xA740);
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = rng() * 50; pys[i] = rng() * 50; }
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);
    const zs = new Float64Array(n).fill(1);
    const gridW = 4, gridH = 4, total = 16;
    const SENTINEL = -424242.5;
    function freshBuf() { return new Float64Array(total).fill(SENTINEL); }

    let buf = freshBuf();
    assert.throws(() => handle.sampleField(zs, 0, gridH, 0, 0, 10, 10, buf), /lite-delaunay: /);
    for (let k = 0; k < total; k++) assert.equal(buf[k], SENTINEL, `bad-grid: outGrid[${k}] written before throw`);

    buf = freshBuf();
    assert.throws(() => handle.sampleField(null, gridW, gridH, 0, 0, 10, 10, buf), /lite-delaunay: /);
    for (let k = 0; k < total; k++) assert.equal(buf[k], SENTINEL, `bad-zs: outGrid[${k}] written before throw`);

    buf = freshBuf();
    assert.throws(() => handle.sampleField(zs, gridW, gridH, 10, 0, 0, 10, buf), /lite-delaunay: /);
    for (let k = 0; k < total; k++) assert.equal(buf[k], SENTINEL, `bad-bbox: outGrid[${k}] written before throw`);

    handle.dispose();
});

// =============================================================================
// A8 LIFECYCLE + POOL
// =============================================================================
test("A8 LIFECYCLE: disposed handle throws on every method; double-dispose throws", () => {
    const n = 20;
    const rng = mulberry32(0xA800);
    const { pxs, pys } = genUniform(rng, n, -30, 30);
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);
    handle.dispose();

    const outW3 = new Float64Array(3), outI3 = new Int32Array(3);
    const zs = new Float64Array(n).fill(1), outGrid = new Float64Array(16);
    assert.throws(() => handle.locate(0, 0), /lite-delaunay: /);
    assert.throws(() => handle.barycentric(0, 0, 0, outW3), /lite-delaunay: /);
    assert.throws(() => handle.triangleVertices(0, outI3), /lite-delaunay: /);
    assert.throws(() => handle.triangleCount(), /lite-delaunay: /);
    assert.throws(() => handle.interpolate(zs, 0, 0), /lite-delaunay: /);
    assert.throws(() => handle.sampleField(zs, 4, 4, 0, 0, 10, 10, outGrid), /lite-delaunay: /);
    assert.throws(() => handle.dispose(), /lite-delaunay: /, "double-dispose must throw");
});

test("A8 LIFECYCLE: stale handle after slot rebuild throws every method; new handle works; slot identity is reused", () => {
    const factory = createFieldIndex(20);
    const dataA = genUniform(mulberry32(0xA8A0), 20, -50, -10);
    const dataB = genUniform(mulberry32(0xA8A1), 20, 100, 200);

    const handleA = factory(dataA.pxs, dataA.pys, 20);
    const slotA = handleA._slot; // captured BEFORE dispose -- dispose() nulls handleA._slot
    handleA.dispose();
    const handleB = factory(dataB.pxs, dataB.pys, 20);
    assert.equal(handleB._slot, slotA, "expected pooled slot reuse (LIFO free stack)");

    const outW3 = new Float64Array(3), outI3 = new Int32Array(3);
    const zsA = new Float64Array(20).fill(1), outGrid = new Float64Array(16);
    assert.throws(() => handleA.locate(0, 0), /lite-delaunay: /);
    assert.throws(() => handleA.barycentric(0, 0, 0, outW3), /lite-delaunay: /);
    assert.throws(() => handleA.triangleVertices(0, outI3), /lite-delaunay: /);
    assert.throws(() => handleA.triangleCount(), /lite-delaunay: /);
    assert.throws(() => handleA.interpolate(zsA, 0, 0), /lite-delaunay: /);
    assert.throws(() => handleA.sampleField(zsA, 4, 4, 0, 0, 10, 10, outGrid), /lite-delaunay: /);
    assert.throws(() => handleA.dispose(), /lite-delaunay: /);

    let t;
    assert.doesNotThrow(() => { t = handleB.locate(dataB.pxs[0], dataB.pys[0]); });
    assert.ok(t === -1 || t >= 0);
    handleB.dispose();
});

test("A8 LIFECYCLE: two concurrent handles over DIFFERENT clouds answer independently across 2000 interleaved queries", () => {
    const factory = createFieldIndex(200);
    const dataA = genUniform(mulberry32(0xA8B0), 150, 0, 500);
    const dataB = genGridJittered(mulberry32(0xA8B1), 12, 15); // n=144
    const nA = 150, nB = 144;
    const handleA = factory(dataA.pxs, dataA.pys, nA);
    const handleB = factory(dataB.pxs, dataB.pys, nB);
    assert.notEqual(handleA._slot, handleB._slot, "test setup: expected two DIFFERENT concurrent pool slots");

    const zsA = new Float64Array(nA); for (let i = 0; i < nA; i++) zsA[i] = dataA.pxs[i] * 0.3 + dataA.pys[i] * 0.7;
    const zsB = new Float64Array(nB); for (let i = 0; i < nB; i++) zsB[i] = dataB.pxs[i] * 0.5 - dataB.pys[i] * 0.2;

    const qrngA = mulberry32(0xA8C0), qrngB = mulberry32(0xA8C1);
    const QN = 2000;
    const qA = [], qB = [], refA = [], refB = [];
    for (let i = 0; i < QN; i++) {
        const qx = qrngA() * 600 - 50, qy = qrngA() * 600 - 50;
        qA.push([qx, qy]);
        refA.push({ loc: handleA.locate(qx, qy), val: handleA.interpolate(zsA, qx, qy) });
    }
    for (let i = 0; i < QN; i++) {
        const qx = qrngB() * 250 - 20, qy = qrngB() * 250 - 20;
        qB.push([qx, qy]);
        refB.push({ loc: handleB.locate(qx, qy), val: handleB.interpolate(zsB, qx, qy) });
    }

    let mismatches = 0;
    for (let i = 0; i < QN; i++) {
        const [qxA, qyA] = qA[i], [qxB, qyB] = qB[i];
        const locA = handleA.locate(qxA, qyA);
        const locB = handleB.locate(qxB, qyB);
        const valA = handleA.interpolate(zsA, qxA, qyA);
        const valB = handleB.interpolate(zsB, qxB, qyB);
        if (locA !== refA[i].loc) mismatches++;
        if (locB !== refB[i].loc) mismatches++;
        if (!(Number.isNaN(valA) && Number.isNaN(refA[i].val)) && valA !== refA[i].val) mismatches++;
        if (!(Number.isNaN(valB) && Number.isNaN(refB[i].val)) && valB !== refB[i].val) mismatches++;
    }
    assert.equal(mismatches, 0, `interleaved concurrent-handle mismatches=${mismatches}/${QN * 4}`);
    handleA.dispose(); handleB.dispose();
    console.log(`A8 CONCURRENT interleaved queries=${QN} mismatches=${mismatches}`);
});

test("A8 LIFECYCLE: interpolate() on the same mesh with two different zs arrays yields two different closed-form values", () => {
    const n = 60;
    const rng = mulberry32(0xA8D0);
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = rng() * 300; pys[i] = rng() * 300; }
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);

    const zs1 = new Float64Array(n); for (let i = 0; i < n; i++) zs1[i] = 2 + 0.5 * pxs[i] - 0.25 * pys[i];
    const zs2 = new Float32Array(n); for (let i = 0; i < n; i++) zs2[i] = -3 + 1.1 * pxs[i] + 0.9 * pys[i];

    const qrng = mulberry32(0xA8D1);
    let checked = 0, attempts = 0;
    while (checked < 300 && attempts < 50000) {
        attempts++;
        const qx = qrng() * 300, qy = qrng() * 300;
        const t = handle.locate(qx, qy);
        if (t < 0) continue;
        const v1 = handle.interpolate(zs1, qx, qy);
        const v2 = handle.interpolate(zs2, qx, qy);
        const e1 = 2 + 0.5 * qx - 0.25 * qy;
        const e2 = -3 + 1.1 * qx + 0.9 * qy;
        assert.ok(Math.abs(v1 - e1) <= 1e-3 * (1 + Math.abs(e1)), `zs1: got ${v1} want ~${e1}`);
        // zs2 is a Float32Array: input precision, not interpolation, sets the tolerance floor.
        assert.ok(Math.abs(v2 - e2) <= 5e-2 * (1 + Math.abs(e2)), `zs2: got ${v2} want ~${e2}`);
        assert.notEqual(v1, v2, "different zs arrays must produce different interpolated values");
        checked++;
    }
    assert.ok(checked >= 300, `too few checked: ${checked}`);
    handle.dispose();
    console.log(`A8 zs-swap checked=${checked}`);
});

// =============================================================================
// ADVERSARIAL (not planner-enumerated)
// =============================================================================
test("ADVERSARIAL: sampleField() with outGrid ALIASING the same array object as zs matches a non-aliased run (gather-before-write ordering)", () => {
    const n = 40;
    const rng = mulberry32(0xADAD);
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = rng() * 200; pys[i] = rng() * 200; }
    const factory = createFieldIndex(n);
    const handle = factory(pxs, pys, n);
    const gridW = 6, gridH = 6, total = 36;
    const size = Math.max(n, total);

    const zsClean = new Float64Array(size);
    for (let i = 0; i < n; i++) zsClean[i] = 5 + 0.4 * pxs[i] - 0.6 * pys[i];
    const outClean = new Float64Array(total);
    const countClean = handle.sampleField(zsClean, gridW, gridH, 0, 0, 200, 200, outClean);

    const shared = new Float64Array(size);
    for (let i = 0; i < n; i++) shared[i] = 5 + 0.4 * pxs[i] - 0.6 * pys[i]; // identical zs content
    const countShared = handle.sampleField(shared, gridW, gridH, 0, 0, 200, 200, shared); // zs === outGrid

    assert.equal(countShared, countClean, "aliased call must return the same finite count");
    let mismatches = 0;
    for (let k = 0; k < total; k++) {
        const a = outClean[k], b = shared[k];
        const aNaN = Number.isNaN(a), bNaN = Number.isNaN(b);
        if (aNaN !== bNaN) { mismatches++; continue; }
        if (!aNaN && a !== b) mismatches++;
    }
    assert.equal(mismatches, 0, `aliased zs===outGrid produced ${mismatches}/${total} divergent cells vs a clean run`);
    handle.dispose();
    console.log(`ADVERSARIAL alias-zs-outGrid mismatches=${mismatches}/${total} count=${countShared}`);
});

// =============================================================================
// BOUNDARY
// =============================================================================
test("BOUNDARY: dispose-during-iteration does not corrupt prior results or a sibling handle", () => {
    const factory = createFieldIndex(20);
    const rng = mulberry32(0xB0B8);
    const { pxs, pys } = genUniform(rng, 20, -50, 50);
    const handle = factory(pxs, pys, 20);
    const sibling = factory(pxs, pys, 20);
    const zs = new Float64Array(20); for (let i = 0; i < 20; i++) zs[i] = pxs[i] + pys[i];

    const qrng = mulberry32(0xB0B9);
    const qs = [];
    for (let i = 0; i < 16; i++) qs.push([qrng() * 80 - 40, qrng() * 80 - 40]);

    const collected = [];
    for (let i = 0; i < qs.length; i++) {
        const [qx, qy] = qs[i];
        if (i === 8) { handle.dispose(); continue; }
        if (i < 8) {
            const val = handle.interpolate(zs, qx, qy);
            collected.push({ i, qx, qy, val });
        } else {
            assert.throws(() => handle.interpolate(zs, qx, qy), /lite-delaunay: /);
        }
    }

    const [sqx, sqy] = qs[0];
    assert.doesNotThrow(() => sibling.interpolate(zs, sqx, sqy));

    const verifyFactory = createFieldIndex(20);
    const verifyHandle = verifyFactory(pxs, pys, 20);
    for (const r of collected) {
        const v = verifyHandle.interpolate(zs, r.qx, r.qy);
        const same = (Number.isNaN(v) && Number.isNaN(r.val)) || v === r.val;
        assert.ok(same, `query ${r.i}: pre-dispose value doesn't match a fresh rebuild (got ${v} want ${r.val})`);
    }
    verifyHandle.dispose();
    sibling.dispose();
    console.log(`BOUNDARY dispose-during-iteration prior-results-verified=${collected.length}`);
});
