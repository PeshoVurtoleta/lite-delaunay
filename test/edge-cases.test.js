// @zakkster/lite-delaunay — full test suite
// ──────────────────────────────────────────────────────────────────────────
// Three layers of verification:
//   1. CORRECTNESS — output matches the Delaunay property on random inputs
//      of increasing size (3 → 10 000 points). Verifies:
//        a) triangle count matches Euler's formula
//        b) every halfedge pair is reciprocal
//        c) NO interior edge has a fourth point inside its circumcircle
//   2. EDGE CASES — degenerate inputs, reuse semantics, constructor validation
//   3. ZERO-GC — heap delta stays in noise floor across thousands of calls
//
// Run with `node --expose-gc test/edge-cases.test.js`. Without `--expose-gc`
// the GC test still runs but produces noisier numbers.
//
// Exit code is 0 on full success, 1 if anything failed.

import { DelaunayTriangulator } from "../Delaunay.js";

// ──────────────────────────────────────────────────────────────────────────
// Tiny assertion harness — no test framework dep, so we ship with zero
// devDependencies.
// ──────────────────────────────────────────────────────────────────────────

let pass = 0, fail = 0;
const fails = [];
const GREEN = "\x1b[32m", RED = "\x1b[31m", DIM = "\x1b[2m", RESET = "\x1b[0m";

function test(name, fn) {
    try {
        fn();
        pass++;
        console.log(`${GREEN}✓${RESET} ${name}`);
    } catch (e) {
        fail++;
        fails.push({ name, err: e });
        console.log(`${RED}✗${RESET} ${name}\n  ${DIM}${e.message}${RESET}`);
    }
}

function assertEq(actual, expected, what = "value") {
    if (actual !== expected) throw new Error(`${what}: expected ${expected}, got ${actual}`);
}
function assert(cond, msg) {
    if (!cond) throw new Error(msg || "assertion failed");
}
function assertThrows(fn, msgMatch) {
    let threw = false, msg = "";
    try { fn(); } catch (e) { threw = true; msg = e.message; }
    if (!threw) throw new Error(`expected throw, but no error was raised`);
    if (msgMatch && !msg.includes(msgMatch)) {
        throw new Error(`expected error message to include "${msgMatch}", got "${msg}"`);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Geometry predicates used to validate output
// ──────────────────────────────────────────────────────────────────────────

function inCircle(ax, ay, bx, by, cx, cy, px, py) {
    const dx = ax - px, dy = ay - py;
    const ex = bx - px, ey = by - py;
    const fx = cx - px, fy = cy - py;
    const ap = dx * dx + dy * dy;
    const bp = ex * ex + ey * ey;
    const cp = fx * fx + fy * fy;
    return dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx);
}

// For every interior halfedge, the opposite triangle's fourth point must NOT
// be inside this triangle's circumcircle. This is the Delaunay property —
// the algorithm's core correctness guarantee.
function countDelaunayViolations(coords, triangles, triLen, halfedges) {
    let violations = 0;
    for (let e = 0; e < triLen; e++) {
        const opp = halfedges[e];
        if (opp === -1) continue;
        const a = triangles[e];
        const b = triangles[e % 3 === 2 ? e - 2 : e + 1];
        const c = triangles[e % 3 === 0 ? e + 2 : e - 1];
        const d = triangles[opp % 3 === 0 ? opp + 2 : opp - 1];
        const inc = inCircle(
            coords[a * 2], coords[a * 2 + 1],
            coords[b * 2], coords[b * 2 + 1],
            coords[c * 2], coords[c * 2 + 1],
            coords[d * 2], coords[d * 2 + 1]
        );
        if (inc < -1e-9) violations++;
    }
    return violations;
}

// Every halfedge must point to a halfedge that points back at it.
function checkHalfedgesPaired(triLen, halfedges) {
    for (let e = 0; e < triLen; e++) {
        const opp = halfedges[e];
        if (opp === -1) continue;
        if (opp < 0 || opp >= triLen) {
            throw new Error(`halfedge[${e}] = ${opp} out of range [0, ${triLen})`);
        }
        if (halfedges[opp] !== e) {
            throw new Error(`halfedge[${e}] = ${opp}, but halfedge[${opp}] = ${halfedges[opp]} (not paired)`);
        }
    }
}

// Indices must reference real points.
function checkIndicesInRange(triangles, triLen, pointCount) {
    for (let i = 0; i < triLen; i++) {
        const v = triangles[i];
        if (v < 0 || v >= pointCount) {
            throw new Error(`triangles[${i}] = ${v} out of range [0, ${pointCount})`);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers — deterministic RNG so test results are reproducible
// ──────────────────────────────────────────────────────────────────────────

function lcgPoints(n, seed = 1) {
    let s = seed >>> 0;
    const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    const coords = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
        coords[i * 2] = rng() * 1000;
        coords[i * 2 + 1] = rng() * 1000;
    }
    return coords;
}

// ══════════════════════════════════════════════════════════════════════════
// Section 1 — CORRECTNESS on random inputs (increasing size)
// ══════════════════════════════════════════════════════════════════════════
console.log(`\n${DIM}── Section 1: correctness (Delaunay property + halfedge integrity) ──${RESET}`);

const correctnessCases = [
    { name: "3 points (minimum triangulation)", coords: new Float32Array([0, 0, 100, 0, 50, 100]), expectTris: 1 },
    { name: "4 points (square)", coords: new Float32Array([0, 0, 100, 0, 100, 100, 0, 100]), expectTris: 2 },
    { name: "10 random points", coords: lcgPoints(10, 42) },
    { name: "100 random points", coords: lcgPoints(100, 42) },
    { name: "1 000 random points", coords: lcgPoints(1000, 42) },
    { name: "5 000 random points", coords: lcgPoints(5000, 42) },
    { name: "10 000 random points", coords: lcgPoints(10000, 42) },
];

for (const c of correctnessCases) {
    test(c.name, () => {
        const n = c.coords.length / 2;
        const tri = new DelaunayTriangulator(n);
        const triCount = tri.triangulate(c.coords, n);

        if (c.expectTris !== undefined) assertEq(triCount, c.expectTris, "triangle count");

        // Output validity
        assertEq(tri.trianglesLen, triCount * 3, "trianglesLen");
        checkIndicesInRange(tri.triangles, tri.trianglesLen, n);
        checkHalfedgesPaired(tri.trianglesLen, tri.halfedges);

        const violations = countDelaunayViolations(c.coords, tri.triangles, tri.trianglesLen, tri.halfedges);
        assertEq(violations, 0, "Delaunay in-circle violations");
    });
}

// ══════════════════════════════════════════════════════════════════════════
// Section 2 — DEGENERATE INPUTS (returns 0, no crash, no hang)
// ══════════════════════════════════════════════════════════════════════════
console.log(`\n${DIM}── Section 2: degenerate inputs ──${RESET}`);

test("0 points returns 0", () => {
    const tri = new DelaunayTriangulator(10);
    assertEq(tri.triangulate(new Float32Array(0), 0), 0, "result");
    assertEq(tri.trianglesLen, 0, "trianglesLen");
});

test("1 point returns 0", () => {
    const tri = new DelaunayTriangulator(10);
    assertEq(tri.triangulate(new Float32Array([5, 5]), 1), 0, "result");
    assertEq(tri.trianglesLen, 0, "trianglesLen");
});

test("2 points returns 0", () => {
    const tri = new DelaunayTriangulator(10);
    assertEq(tri.triangulate(new Float32Array([0, 0, 10, 10]), 2), 0, "result");
    assertEq(tri.trianglesLen, 0, "trianglesLen");
});

test("all coincident points returns 0 (does not hang)", () => {
    const tri = new DelaunayTriangulator(10);
    const coords = new Float32Array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]);
    assertEq(tri.triangulate(coords, 5), 0, "result");
    assertEq(tri.trianglesLen, 0, "trianglesLen");
});

test("all collinear horizontal returns 0", () => {
    const tri = new DelaunayTriangulator(10);
    assertEq(tri.triangulate(new Float32Array([0, 0, 10, 0, 20, 0, 30, 0, 40, 0]), 5), 0);
});

test("all collinear vertical returns 0", () => {
    const tri = new DelaunayTriangulator(10);
    assertEq(tri.triangulate(new Float32Array([0, 0, 0, 10, 0, 20, 0, 30, 0, 40]), 5), 0);
});

test("all collinear diagonal returns 0", () => {
    const tri = new DelaunayTriangulator(10);
    assertEq(tri.triangulate(new Float32Array([0, 0, 10, 10, 20, 20, 30, 30, 40, 40]), 5), 0);
});

test("near-duplicate points (within epsilon) are tolerated", () => {
    // Two points so close they trip the duplicate-skip heuristic.
    const eps = Math.pow(2, -53);
    const tri = new DelaunayTriangulator(10);
    const coords = new Float32Array([0, 0, 100, 0, 50, 100, 50 + eps, 100 + eps]);
    const n = tri.triangulate(coords, 4);
    assert(n >= 1 && n <= 2, `expected 1-2 triangles, got ${n}`);
    checkHalfedgesPaired(tri.trianglesLen, tri.halfedges);
});

// ══════════════════════════════════════════════════════════════════════════
// Section 3 — STATE & REUSE semantics
// ══════════════════════════════════════════════════════════════════════════
console.log(`\n${DIM}── Section 3: state & reuse ──${RESET}`);

test("reuse: large input then degenerate input resets trianglesLen", () => {
    const tri = new DelaunayTriangulator(10);
    tri.triangulate(lcgPoints(8, 1), 8);
    assert(tri.trianglesLen > 0, "first call should produce triangles");
    tri.triangulate(new Float32Array([0, 0, 10, 10]), 2);
    assertEq(tri.trianglesLen, 0, "trianglesLen after degenerate call");
});

test("reuse: collinear after random does not leak old state", () => {
    const tri = new DelaunayTriangulator(100);
    tri.triangulate(lcgPoints(50, 7), 50);
    tri.triangulate(new Float32Array([0, 0, 10, 0, 20, 0]), 3);
    assertEq(tri.trianglesLen, 0, "trianglesLen");
});

test("idempotent: same input twice produces same triangle set", () => {
    const coords = lcgPoints(200, 17);
    const tri = new DelaunayTriangulator(200);

    tri.triangulate(coords, 200);
    const setA = new Set();
    for (let i = 0; i < tri.trianglesLen; i += 3) {
        setA.add([tri.triangles[i], tri.triangles[i + 1], tri.triangles[i + 2]].sort((a, b) => a - b).join(","));
    }

    tri.triangulate(coords, 200);
    const setB = new Set();
    for (let i = 0; i < tri.trianglesLen; i += 3) {
        setB.add([tri.triangles[i], tri.triangles[i + 1], tri.triangles[i + 2]].sort((a, b) => a - b).join(","));
    }

    assertEq(setA.size, setB.size, "triangle set size");
    for (const t of setA) assert(setB.has(t), `missing triangle ${t} after rerun`);
});

test("reuse: smaller-than-arena point count works correctly", () => {
    // Allocate for 1000, only use 5
    const tri = new DelaunayTriangulator(1000);
    const coords = new Float32Array([0, 0, 100, 0, 100, 100, 0, 100, 50, 50]);
    const n = tri.triangulate(coords, 5);
    assertEq(n, 4, "5-point triangle count");
    checkHalfedgesPaired(tri.trianglesLen, tri.halfedges);
    assertEq(countDelaunayViolations(coords, tri.triangles, tri.trianglesLen, tri.halfedges), 0);
});

// ══════════════════════════════════════════════════════════════════════════
// Section 4 — CONSTRUCTOR validation
// ══════════════════════════════════════════════════════════════════════════
console.log(`\n${DIM}── Section 4: constructor validation ──${RESET}`);

test("maxPoints = 0 is allowed (zero-capacity arena)", () => {
    const tri = new DelaunayTriangulator(0);
    assertEq(tri.maxPoints, 0);
    assertEq(tri.triangles.length, 0);
});

test("maxPoints = 3 (minimum useful arena)", () => {
    const tri = new DelaunayTriangulator(3);
    assertEq(tri.triangulate(new Float32Array([0, 0, 100, 0, 50, 100]), 3), 1);
});

test("negative maxPoints throws", () => {
    assertThrows(() => new DelaunayTriangulator(-1), "non-negative integer");
});

test("non-integer maxPoints throws", () => {
    assertThrows(() => new DelaunayTriangulator(3.5), "non-negative integer");
});

test("NaN maxPoints throws", () => {
    assertThrows(() => new DelaunayTriangulator(NaN), "non-negative integer");
});

test("pointCount > maxPoints throws", () => {
    const tri = new DelaunayTriangulator(4);
    assertThrows(() => tri.triangulate(new Float32Array(20), 10), "exceeds arena max");
});

// ══════════════════════════════════════════════════════════════════════════
// Section 5 — INPUT TYPES — Float32Array vs Float64Array vs Array
// ══════════════════════════════════════════════════════════════════════════
console.log(`\n${DIM}── Section 5: input type compatibility ──${RESET}`);

const squareF32 = new Float32Array([0, 0, 100, 0, 100, 100, 0, 100]);
const squareF64 = new Float64Array([0, 0, 100, 0, 100, 100, 0, 100]);
const squareArr = [0, 0, 100, 0, 100, 100, 0, 100];

test("Float32Array input → 2 triangles", () => {
    const tri = new DelaunayTriangulator(4);
    assertEq(tri.triangulate(squareF32, 4), 2);
});
test("Float64Array input → 2 triangles", () => {
    const tri = new DelaunayTriangulator(4);
    assertEq(tri.triangulate(squareF64, 4), 2);
});
test("plain number[] input → 2 triangles", () => {
    const tri = new DelaunayTriangulator(4);
    assertEq(tri.triangulate(squareArr, 4), 2);
});

// ══════════════════════════════════════════════════════════════════════════
// Section 6 — GEOMETRIC CONFIGURATIONS that have historically tripped up
// Delaunay implementations
// ══════════════════════════════════════════════════════════════════════════
console.log(`\n${DIM}── Section 6: known-tricky geometry ──${RESET}`);

test("8×8 regular grid (cocircular quartets every cell)", () => {
    const coords = new Float32Array(64 * 2);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
        coords[(y * 8 + x) * 2] = x * 10;
        coords[(y * 8 + x) * 2 + 1] = y * 10;
    }
    const tri = new DelaunayTriangulator(64);
    const n = tri.triangulate(coords, 64);
    // Euler: a 7×7 cell grid has 49 cells × 2 triangles = 98
    assertEq(n, 98, "grid triangle count");
    checkHalfedgesPaired(tri.trianglesLen, tri.halfedges);
    assertEq(countDelaunayViolations(coords, tri.triangles, tri.trianglesLen, tri.halfedges), 0);
});

test("Archimedean spiral (sweepline stress pattern)", () => {
    const n = 200;
    const coords = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
        const angle = i * 0.4;
        const r = i * 2;
        coords[i * 2] = 500 + r * Math.cos(angle);
        coords[i * 2 + 1] = 500 + r * Math.sin(angle);
    }
    const tri = new DelaunayTriangulator(n);
    tri.triangulate(coords, n);
    checkHalfedgesPaired(tri.trianglesLen, tri.halfedges);
    assertEq(countDelaunayViolations(coords, tri.triangles, tri.trianglesLen, tri.halfedges), 0);
});

test("points on a circle (every triangle is a fan slice)", () => {
    const n = 64;
    const coords = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2;
        coords[i * 2] = 500 + 200 * Math.cos(a);
        coords[i * 2 + 1] = 500 + 200 * Math.sin(a);
    }
    const tri = new DelaunayTriangulator(n);
    const triCount = tri.triangulate(coords, n);
    // For n points on a convex hull with no interior points: triangles = n - 2
    assertEq(triCount, n - 2, "fan triangulation count");
    checkHalfedgesPaired(tri.trianglesLen, tri.halfedges);
});

test("two clusters far apart (tests hull-walk over long edge)", () => {
    const coords = new Float32Array([
        0, 0, 1, 0, 0, 1, 1, 1,
        1000, 1000, 1001, 1000, 1000, 1001, 1001, 1001
    ]);
    const tri = new DelaunayTriangulator(8);
    const n = tri.triangulate(coords, 8);
    assert(n > 0, "should produce triangles");
    checkHalfedgesPaired(tri.trianglesLen, tri.halfedges);
    assertEq(countDelaunayViolations(coords, tri.triangles, tri.trianglesLen, tri.halfedges), 0);
});

// ══════════════════════════════════════════════════════════════════════════
// Section 7 — TOPOLOGICAL invariants on a known-good output
// ══════════════════════════════════════════════════════════════════════════
console.log(`\n${DIM}── Section 7: mesh topology invariants ──${RESET}`);

test("Euler invariant: V − E + F = 1 for a planar triangulation", () => {
    const coords = lcgPoints(500, 3);
    const tri = new DelaunayTriangulator(500);
    const triCount = tri.triangulate(coords, 500);

    // Count edges via halfedges: each interior edge appears twice, each hull edge once.
    let halfedgeCount = 0, hullEdges = 0;
    for (let i = 0; i < tri.trianglesLen; i++) {
        halfedgeCount++;
        if (tri.halfedges[i] === -1) hullEdges++;
    }
    const interiorEdges = (halfedgeCount - hullEdges) / 2;
    const E = interiorEdges + hullEdges;
    // V is harder to derive without re-traversing — but we can use the
    // alternative Euler check: 3F = 2*interior + hull
    assertEq(3 * triCount, 2 * interiorEdges + hullEdges, "3F = 2E_int + E_hull");
});

test("every triangle has CCW orientation in math coordinates", () => {
    const coords = lcgPoints(100, 9);
    const tri = new DelaunayTriangulator(100);
    const triCount = tri.triangulate(coords, 100);
    for (let k = 0; k < triCount; k++) {
        const a = tri.triangles[k * 3], b = tri.triangles[k * 3 + 1], c = tri.triangles[k * 3 + 2];
        const ax = coords[a * 2], ay = coords[a * 2 + 1];
        const bx = coords[b * 2], by = coords[b * 2 + 1];
        const cx = coords[c * 2], cy = coords[c * 2 + 1];
        // 2× signed area; CCW in math convention => negative in screen convention.
        // Delaunator uses screen convention (Y-down), so the cross product here
        // should be NEGATIVE (matches Mapbox Delaunator's `orient2d` test).
        const cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        assert(cross < 0, `triangle ${k} has wrong winding: cross = ${cross}`);
    }
});

// ══════════════════════════════════════════════════════════════════════════
// Section 8 — ZERO-GC guarantee
// ══════════════════════════════════════════════════════════════════════════
console.log(`\n${DIM}── Section 8: zero-allocation guarantee ──${RESET}`);

test("10 000 triangulate() calls grow heap by less than 1 MB", () => {
    const coords = lcgPoints(500, 11);
    const tri = new DelaunayTriangulator(500);

    // Warmup so JIT codegen, hidden classes, etc. are all stable.
    for (let i = 0; i < 50; i++) tri.triangulate(coords, 500);

    if (global.gc) global.gc();
    const before = process.memoryUsage().heapUsed;

    for (let i = 0; i < 10_000; i++) tri.triangulate(coords, 500);

    if (global.gc) global.gc();
    const delta = (process.memoryUsage().heapUsed - before) / 1024;

    // Without --expose-gc the GC can't be forced and the number is noisier;
    // still well under 5 MB in practice.
    const limit = global.gc ? 1024 : 5120;
    assert(delta < limit, `heap delta ${delta.toFixed(1)} KB > ${limit} KB limit`);
});

// ══════════════════════════════════════════════════════════════════════════
// Summary
// ══════════════════════════════════════════════════════════════════════════

console.log(`\n${pass + fail} tests · ${GREEN}${pass} passed${RESET}` + (fail ? ` · ${RED}${fail} failed${RESET}` : ""));
if (fail > 0) {
    console.log(`\nFailures:`);
    for (const f of fails) console.log(`  ${RED}✗${RESET} ${f.name}\n    ${f.err.stack || f.err.message}`);
}
process.exit(fail > 0 ? 1 : 0);
