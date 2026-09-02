// test/DelaunayProperty.test.js -- mathematical correctness of the
// triangulation kernel itself: halfedge twin pairing and the Delaunay
// in-circle property, plus exact triangle counts on known inputs.
// Ported (node:test, ASCII) from the pre-1.0 root-level Delaunay.test.js probe.
import { test } from "node:test";
import assert from "node:assert/strict";

import { DelaunayTriangulator } from "../Delaunay.js";

// Sign convention: a negative return means (px, py) lies INSIDE the
// circumcircle of triangle (a, b, c) -- an illegal (non-Delaunay) pair.
function inCircle(ax, ay, bx, by, cx, cy, px, py) {
    const dx = ax - px, dy = ay - py;
    const ex = bx - px, ey = by - py;
    const fx = cx - px, fy = cy - py;
    const ap = dx * dx + dy * dy;
    const bp = ex * ex + ey * ey;
    const cp = fx * fx + fy * fy;
    return dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx);
}

// Every interior half-edge must be twinned symmetrically:
// halfedges[halfedges[e]] === e. Returns a description of the first
// violation, or null when the pairing is sound.
function halfedgePairingError(triLen, halfedges) {
    for (let e = 0; e < triLen; e++) {
        const opp = halfedges[e];
        if (opp !== -1 && halfedges[opp] !== e) {
            return "halfedge[" + e + "]=" + opp +
                " but halfedge[" + opp + "]=" + halfedges[opp] + " (not paired)";
        }
    }
    return null;
}

// For every interior halfedge, the opposite triangle's fourth point must
// NOT be inside the circumcircle of this triangle. Returns the violation
// count (must be 0). The -1e-9 slack absorbs f64 predicate noise on
// f32-quantized inputs, matching the original probe.
function delaunayViolations(coords, triangles, triLen, halfedges) {
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

// sfc32 -- deterministic inputs, repeatable across runs.
function makeRandomPoints(n, seed = 1) {
    let a = seed | 0, b = (seed * 7919) | 0, c = (seed * 4937) | 0, d = (seed * 1013) | 0;
    const rng = () => {
        a >>>= 0;
        b >>>= 0;
        c >>>= 0;
        d >>>= 0;
        let t = (a + b) | 0;
        a = b ^ b >>> 9;
        b = c + (c << 3) | 0;
        c = (c << 21 | c >>> 11);
        d = d + 1 | 0;
        t = t + d | 0;
        c = c + t | 0;
        return (t >>> 0) / 4294967296;
    };
    const coords = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
        coords[i * 2] = rng() * 1000;
        coords[i * 2 + 1] = rng() * 1000;
    }
    return coords;
}

const CASES = [
    { name: "3 points (smallest valid)", coords: new Float32Array([0, 0, 100, 0, 50, 100]), expectTris: 1 },
    { name: "4 points (square)", coords: new Float32Array([0, 0, 100, 0, 100, 100, 0, 100]), expectTris: 2 },
    { name: "10 random points", coords: makeRandomPoints(10), expectTris: null },
    { name: "100 random points", coords: makeRandomPoints(100), expectTris: null },
    { name: "1000 random points", coords: makeRandomPoints(1000), expectTris: null },
    { name: "5000 random points", coords: makeRandomPoints(5000), expectTris: null },
    { name: "10000 random points", coords: makeRandomPoints(10000), expectTris: null },
];

for (const c of CASES) {
    test("delaunay property: " + c.name, () => {
        const n = c.coords.length / 2;
        const tri = new DelaunayTriangulator(n);
        const numTris = tri.triangulate(c.coords, n);
        if (c.expectTris !== null) {
            assert.equal(numTris, c.expectTris, "triangle count");
        }
        assert.equal(tri.trianglesLen, numTris * 3, "trianglesLen cursor");
        assert.equal(halfedgePairingError(tri.trianglesLen, tri.halfedges), null);
        assert.equal(
            delaunayViolations(c.coords, tri.triangles, tri.trianglesLen, tri.halfedges),
            0,
            "in-circle violations"
        );
    });
}
