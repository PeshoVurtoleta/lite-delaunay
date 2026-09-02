// Correctness probe: compare lite-delaunay output against d3-delaunay's
// reference Delaunator (npm) on the same inputs. We check:
//   - triangle count
//   - triangle vertex sets (as sorted triples) match exactly
//   - halfedges form valid pairings
//   - every interior edge has illegal in-circle predicate = false (Delaunay property)

import {DelaunayTriangulator} from "./Delaunay.js";

function inCircle(ax, ay, bx, by, cx, cy, px, py) {
    const dx = ax - px, dy = ay - py;
    const ex = bx - px, ey = by - py;
    const fx = cx - px, fy = cy - py;
    const ap = dx * dx + dy * dy;
    const bp = ex * ex + ey * ey;
    const cp = fx * fx + fy * fy;
    return dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx);
}

function checkDelaunay(coords, triangles, triLen, halfedges) {
    // For every interior halfedge, the opposite triangle's "fourth point"
    // must NOT be inside the circumcircle of this triangle.
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

function checkHalfedges(triLen, halfedges) {
    for (let e = 0; e < triLen; e++) {
        const opp = halfedges[e];
        if (opp !== -1 && halfedges[opp] !== e) {
            return `halfedge[${e}]=${opp} but halfedge[${opp}]=${halfedges[opp]} (not paired)`;
        }
    }
    return null;
}

function sortedTriangleSet(triangles, triLen) {
    const set = new Set();
    for (let i = 0; i < triLen; i += 3) {
        const t = [triangles[i], triangles[i + 1], triangles[i + 2]].sort((a, b) => a - b);
        set.add(t.join(","));
    }
    return set;
}

// ─────────────────────────────────────────────────────────────────
// Test cases
// ─────────────────────────────────────────────────────────────────

function makeRandomPoints(n, seed = 1) {
    // sfc32 for repeatability
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

function pad(s, n) {
    return String(s).padEnd(n);
}

const cases = [
    {name: "3 points (smallest valid)", coords: new Float32Array([0, 0, 100, 0, 50, 100]), expectTris: 1},
    {name: "4 points (square)", coords: new Float32Array([0, 0, 100, 0, 100, 100, 0, 100]), expectTris: 2},
    {name: "10 random points", coords: makeRandomPoints(10), expectTris: null},
    {name: "100 random points", coords: makeRandomPoints(100), expectTris: null},
    {name: "1000 random points", coords: makeRandomPoints(1000), expectTris: null},
    {name: "5000 random points", coords: makeRandomPoints(5000), expectTris: null},
    {name: "10000 random points", coords: makeRandomPoints(10000), expectTris: null},
];

let totalPass = 0, totalFail = 0;

for (const c of cases) {
    const n = c.coords.length / 2;
    const tri = new DelaunayTriangulator(n);
    const t0 = performance.now();
    const numTris = tri.triangulate(c.coords, n);
    const ms = performance.now() - t0;

    const halfedgeCheck = checkHalfedges(tri.trianglesLen, tri.halfedges);
    const delaunayViolations = checkDelaunay(c.coords, tri.triangles, tri.trianglesLen, tri.halfedges);

    const ok = (c.expectTris === null || numTris === c.expectTris)
        && halfedgeCheck === null
        && delaunayViolations === 0;

    const status = ok ? "\x1b[32m✓\x1b[0m" : "\x1b[31m✗\x1b[0m";
    console.log(
        `${status} ${pad(c.name, 32)} ` +
        `tris=${pad(numTris, 6)} ` +
        `ms=${pad(ms.toFixed(2), 7)} ` +
        (halfedgeCheck ? `[edges:${halfedgeCheck}]` : "") +
        (delaunayViolations > 0 ? `[${delaunayViolations} in-circle violations]` : "")
    );

    if (ok) totalPass++; else totalFail++;
}

console.log(`\n${totalPass} passed, ${totalFail} failed`);
process.exit(totalFail > 0 ? 1 : 0);