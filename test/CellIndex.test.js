// @zakkster/lite-delaunay -- correctness suite for createCellIndex (v1.2.0)
// -----------------------------------------------------------------------------
// node:test only (no bespoke harness, no deps beyond the package under test).
// Run with:
//   node --expose-gc --test test/CellIndex.test.js
//
// Covers the planner's assertions:
//   A1  TILING       -- shoelace area of ALL clipped cells == bbox area, per
//                        cloud x bbox sweep, including an exact-duplicate cloud.
//   A1b CONTAINMENT  -- brute-force nearest site's cell contains a random
//                        interior query point; other sites' cells do not.
//   A2  SIZING       -- vertexCount <= degree + 4 (interior) / + 5 (hull);
//                        huge-bbox exactness against independently computed
//                        circumcenters.
//   A3  DEGENERATE MATRIX -- n=0,1,2, all-coincident, all-collinear, all-NaN,
//                        mixed-NaN.
//   A4  FAIL-CLOSED MATRIX -- every documented throw case, message-checked.
//   A5  LIFECYCLE    -- dispose, double-dispose, stale-after-slot-reuse,
//                        3 concurrent handles, slot reuse identity,
//                        dispose-during-iteration.
//   A6  GEOMETRY SANITY -- convexity, finiteness, inside-or-on-bbox.
//   A7  COPY SEMANTICS -- post-build source mutation does not change answers.
//
// Plus explicit boundary-matrix entries (0, 1, N-1, N, N+1, empty, null,
// undefined, NaN, -0, duplicate dispose, dispose-during-iteration, re-entrant
// write, Float32Array output) and one adversarial case: two concurrent
// handles (different pool slots) interleaving cell() calls into the SAME
// caller-owned outXY buffer must not cross-contaminate each other's answers.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createCellIndex, DelaunayTriangulator } from "../Delaunay.js";

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
// cloud generators -- all return { pxs: Float32Array, pys: Float32Array }
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

// Jittered grid -- breaks exact cocircularity (which a perfect grid triggers)
// while keeping the "small grid" pattern the planner asked for.
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

// Needle: sharp hull vertex at the apex of (0,0),(10,0),(5,100) scaled up,
// filled with interior points via barycentric sampling so a real mesh forms.
function genNeedle(rng, n) {
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    const SCALE = 1000;
    pxs[0] = 0; pys[0] = 0;
    pxs[1] = 10 * SCALE; pys[1] = 0;
    pxs[2] = 5 * SCALE; pys[2] = 100 * SCALE;
    for (let i = 3; i < n; i++) {
        let a = rng(), b = rng();
        if (a + b > 1) { a = 1 - a; b = 1 - b; }
        const cw = 1 - a - b;
        pxs[i] = cw * pxs[0] + a * pxs[1] + b * pxs[2];
        pys[i] = cw * pys[0] + a * pys[1] + b * pys[2];
    }
    return { pxs, pys };
}

// Uniform cloud plus `dupPairCount` points that exactly duplicate an earlier
// point's coordinates -- the EPSILON-dedup-lost-duplicate stress case.
function genCloudWithDuplicates(rng, uniqueCount, dupPairCount) {
    const n = uniqueCount + dupPairCount;
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < uniqueCount; i++) {
        pxs[i] = rng() * 400 - 200;
        pys[i] = rng() * 400 - 200;
    }
    for (let k = 0; k < dupPairCount; k++) {
        const srcIdx = (rng() * uniqueCount) | 0;
        const dstIdx = uniqueCount + k;
        pxs[dstIdx] = pxs[srcIdx];
        pys[dstIdx] = pys[srcIdx];
    }
    return { pxs, pys, n };
}

// -----------------------------------------------------------------------------
// geometry helpers -- shoelace area, convexity, point-in-polygon, bbox tol
// -----------------------------------------------------------------------------
function isFinitePoint(px, py) {
    return px === px && py === py &&
        px !== Infinity && px !== -Infinity &&
        py !== Infinity && py !== -Infinity;
}

function bboxOfFiniteLocal(pxs, pys, n) {
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

// Three bboxes per cloud: strictly inside, containing the whole cloud, and
// offset so it misses the cloud's convex hull entirely (tiling must still
// hold, since hull cells extend to infinity and get clipped by ANY bbox).
function sweepBBoxesFor(pxs, pys, n) {
    const { minX, minY, maxX, maxY } = bboxOfFiniteLocal(pxs, pys, n);
    const spanX = (maxX - minX) || 1;
    const spanY = (maxY - minY) || 1;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    return [
        { label: "inside", x0: cx - spanX * 0.15, y0: cy - spanY * 0.15, x1: cx + spanX * 0.15, y1: cy + spanY * 0.15 },
        { label: "containing", x0: minX - spanX * 0.5, y0: minY - spanY * 0.5, x1: maxX + spanX * 0.5, y1: maxY + spanY * 0.5 },
        { label: "offset-miss", x0: maxX + spanX * 2, y0: maxY + spanY * 2, x1: maxX + spanX * 2 + spanX * 0.3, y1: maxY + spanY * 2 + spanY * 0.3 },
    ];
}

function mkBuf(maxPoints) {
    return new Float64Array(2 * (maxPoints + 8));
}

function polySignedArea(poly, count) {
    if (count < 3) return 0;
    let sum = 0;
    for (let k = 0; k < count; k++) {
        const j = (k + 1) % count;
        sum += poly[2 * k] * poly[2 * j + 1] - poly[2 * j] * poly[2 * k + 1];
    }
    return sum * 0.5;
}

function polyArea(poly, count) {
    return Math.abs(polySignedArea(poly, count));
}

function isConvex(poly, count, tol) {
    tol = tol === undefined ? 1e-9 : tol;
    if (count < 3) return false;
    let sign = 0;
    for (let k = 0; k < count; k++) {
        const x0 = poly[2 * k], y0 = poly[2 * k + 1];
        const j1 = (k + 1) % count, j2 = (k + 2) % count;
        const x1 = poly[2 * j1], y1 = poly[2 * j1 + 1];
        const x2 = poly[2 * j2], y2 = poly[2 * j2 + 1];
        const cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1);
        if (Math.abs(cross) < tol) continue;
        const s = cross > 0 ? 1 : -1;
        if (sign === 0) sign = s;
        else if (s !== sign) return false;
    }
    return true;
}

// Inclusive (inside-or-on) containment for a convex polygon of unknown winding.
function pointInPolygon(qx, qy, poly, count, tol) {
    tol = tol === undefined ? 1e-9 : tol;
    if (count < 3) return false;
    const area = polySignedArea(poly, count);
    const wind = area >= 0 ? 1 : -1;
    for (let k = 0; k < count; k++) {
        const j = (k + 1) % count;
        const x0 = poly[2 * k], y0 = poly[2 * k + 1];
        const x1 = poly[2 * j], y1 = poly[2 * j + 1];
        const cross = (x1 - x0) * (qy - y0) - (y1 - y0) * (qx - x0);
        if (wind * cross < -tol) return false;
    }
    return true;
}

// Strict (interior-only) containment -- boundary does NOT count.
function strictlyInside(qx, qy, poly, count, tol) {
    tol = tol === undefined ? 1e-9 : tol;
    if (count < 3) return false;
    const area = polySignedArea(poly, count);
    const wind = area >= 0 ? 1 : -1;
    for (let k = 0; k < count; k++) {
        const j = (k + 1) % count;
        const x0 = poly[2 * k], y0 = poly[2 * k + 1];
        const x1 = poly[2 * j], y1 = poly[2 * j + 1];
        const cross = (x1 - x0) * (qy - y0) - (y1 - y0) * (qx - x0);
        if (wind * cross <= tol) return false;
    }
    return true;
}

function bboxTol(a, b) {
    return 1e-9 * Math.max(1, Math.abs(a), Math.abs(b));
}

function insideOrOnBBox(x, y, bx0, by0, bx1, by1) {
    const tx = bboxTol(bx0, bx1), ty = bboxTol(by0, by1);
    return x >= bx0 - tx && x <= bx1 + tx && y >= by0 - ty && y <= by1 + ty;
}

// -----------------------------------------------------------------------------
// independent mesh info -- degree / hull-membership / circumcenters, computed
// via a FRESH DelaunayTriangulator and the library's documented circumcenter
// formula (mirrored, not imported), so A2 never trusts the implementation
// under test to grade itself.
// -----------------------------------------------------------------------------
const TEST_EPSILON = Math.pow(2, -52);

function buildMeshInfo(pxs, pys, n) {
    const coords = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) { coords[2 * i] = pxs[i]; coords[2 * i + 1] = pys[i]; }
    const tri = new DelaunayTriangulator(n);
    const triCount = tri.triangulate(coords, n);
    const triangles = tri.triangles, halfedges = tri.halfedges;

    const degreeSets = new Array(n);
    for (let i = 0; i < n; i++) degreeSets[i] = new Set();
    for (let t = 0; t < triCount; t++) {
        const a = triangles[3 * t], b = triangles[3 * t + 1], c = triangles[3 * t + 2];
        degreeSets[a].add(b); degreeSets[a].add(c);
        degreeSets[b].add(a); degreeSets[b].add(c);
        degreeSets[c].add(a); degreeSets[c].add(b);
    }
    const degree = new Int32Array(n);
    for (let i = 0; i < n; i++) degree[i] = degreeSets[i].size;

    const isHull = new Uint8Array(n);
    const trianglesLen = triCount * 3;
    for (let e = 0; e < trianglesLen; e++) {
        if (halfedges[e] === -1) {
            const ne = (e % 3 === 2) ? e - 2 : e + 1;
            isHull[triangles[ne]] = 1;
        }
    }

    const ccx = new Float64Array(triCount), ccy = new Float64Array(triCount);
    for (let t = 0; t < triCount; t++) {
        const ia = triangles[3 * t], ib = triangles[3 * t + 1], ic = triangles[3 * t + 2];
        const ax = coords[2 * ia], ay = coords[2 * ia + 1];
        const bx = coords[2 * ib], by = coords[2 * ib + 1];
        const cx = coords[2 * ic], cy = coords[2 * ic + 1];
        const dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
        const bl = dx * dx + dy * dy, cl = ex * ex + ey * ey;
        const D = dx * ey - dy * ex;
        const scale = bl + cl;
        if (scale === 0 || !(Math.abs(D) > TEST_EPSILON * scale)) {
            ccx[t] = (ax + bx + cx) / 3;
            ccy[t] = (ay + by + cy) / 3;
        } else {
            const d = 0.5 / D;
            ccx[t] = ax + (ey * bl - dy * cl) * d;
            ccy[t] = ay + (dx * cl - ex * bl) * d;
        }
    }

    return { triCount, triangles, degree, isHull, ccx, ccy };
}

// -----------------------------------------------------------------------------
// shared cloud sweep for A1 / A2 / A6 -- 4 shapes, all finite, no NaN, so
// original index == compacted index throughout.
// -----------------------------------------------------------------------------
function buildSweepClouds() {
    const clouds = [];
    {
        const rng = mulberry32(0x1001);
        const { pxs, pys } = genUniform(rng, 150, -500, 500);
        clouds.push({ name: "uniform", pxs, pys, n: 150 });
    }
    {
        const rng = mulberry32(0x1002);
        const { pxs, pys } = genClustered(rng, 120);
        clouds.push({ name: "clustered", pxs, pys, n: 120 });
    }
    {
        const rng = mulberry32(0x1003);
        const { pxs, pys } = genGridJittered(rng, 7, 10);
        clouds.push({ name: "grid", pxs, pys, n: 49 });
    }
    {
        const rng = mulberry32(0x1004);
        const { pxs, pys } = genNeedle(rng, 60);
        clouds.push({ name: "needle", pxs, pys, n: 60 });
    }
    return clouds;
}

// =============================================================================
// A1 TILING
// =============================================================================
test("A1 TILING: shoelace area of ALL clipped cells == bbox area across cloud x bbox sweep", () => {
    const clouds = buildSweepClouds();
    let siteChecks = 0, polysSummed = 0;
    for (const cloud of clouds) {
        const factory = createCellIndex(cloud.n);
        const handle = factory(cloud.pxs, cloud.pys, cloud.n);
        const buf = mkBuf(cloud.n);
        for (const bbox of sweepBBoxesFor(cloud.pxs, cloud.pys, cloud.n)) {
            let total = 0;
            for (let i = 0; i < cloud.n; i++) {
                const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
                siteChecks++;
                assert.ok(c === 0 || c >= 3, `${cloud.name}/${bbox.label} site ${i}: vertexCount ${c} not 0 or >=3`);
                if (c > 0) {
                    total += polyArea(buf, c);
                    polysSummed++;
                }
            }
            const bboxArea = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
            const relErr = Math.abs(total - bboxArea) / bboxArea;
            assert.ok(relErr < 1e-6,
                `${cloud.name}/${bbox.label}: tiling area mismatch total=${total} bboxArea=${bboxArea} relErr=${relErr}`);
        }
        handle.dispose();
    }
    assert.ok(siteChecks >= 1000, `A1 TILING sample too small: ${siteChecks}`);
    console.log("A1 TILING siteChecks=" + siteChecks + " polysSummed=" + polysSummed);
});

test("A1 TILING: exact duplicates -- exactly one member of each coincident group survives, tiling still holds", () => {
    const rng = mulberry32(0x1D0D);
    const { pxs, pys, n } = genCloudWithDuplicates(rng, 80, 10);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const bbox = sweepBBoxesFor(pxs, pys, n)[1]; // containing
    const buf = mkBuf(n);

    const groups = new Map();
    for (let i = 0; i < n; i++) {
        const key = pxs[i] + "," + pys[i];
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(i);
    }

    let total = 0;
    for (let i = 0; i < n; i++) {
        const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
        assert.ok(c === 0 || c >= 3, `dup-cloud site ${i}: vertexCount ${c} not 0 or >=3`);
        if (c > 0) total += polyArea(buf, c);
    }

    let dupGroupsChecked = 0;
    for (const [key, members] of groups) {
        if (members.length < 2) continue;
        let nonzero = 0;
        for (const idx of members) {
            const c = handle.cell(idx, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
            if (c > 0) nonzero++;
        }
        assert.equal(nonzero, 1, `coincident group ${key} (members ${members}): expected exactly 1 survivor, got ${nonzero}`);
        dupGroupsChecked++;
    }

    const bboxArea = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
    const relErr = Math.abs(total - bboxArea) / bboxArea;
    assert.ok(relErr < 1e-6, `dup-cloud tiling mismatch: total=${total} bboxArea=${bboxArea} relErr=${relErr}`);
    assert.ok(dupGroupsChecked >= 1, "no duplicate groups formed -- regenerate test data");
    handle.dispose();
    console.log("A1 TILING duplicate groups=" + dupGroupsChecked + " relErr=" + relErr);
});

// =============================================================================
// A1b CONTAINMENT
// =============================================================================
test("A1b CONTAINMENT: nearest-site cell contains the query point; other sites' cells do not", () => {
    const N = 300;
    const rng = mulberry32(0xC0C0);
    const { pxs, pys } = genUniform(rng, N, -200, 200);
    const factory = createCellIndex(N);
    const handle = factory(pxs, pys, N);
    const buf = mkBuf(N);

    const { minX, minY, maxX, maxY } = bboxOfFiniteLocal(pxs, pys, N);
    const spanX = maxX - minX, spanY = maxY - minY;
    const bx0 = minX + spanX * 0.25, bx1 = maxX - spanX * 0.25;
    const by0 = minY + spanY * 0.25, by1 = maxY - spanY * 0.25;

    const qrng = mulberry32(0xFACE);
    const SAMPLES = 5000;
    let sampled = 0, skippedTie = 0, checkedNearest = 0, checkedOthers = 0;

    for (let s = 0; s < SAMPLES; s++) {
        const qx = bx0 + qrng() * (bx1 - bx0);
        const qy = by0 + qrng() * (by1 - by0);
        sampled++;

        let best = -1, bestD = Infinity, secondD = Infinity;
        for (let i = 0; i < N; i++) {
            const dx = qx - pxs[i], dy = qy - pys[i];
            const d = dx * dx + dy * dy;
            if (d < bestD) { secondD = bestD; bestD = d; best = i; }
            else if (d < secondD) { secondD = d; }
        }
        const distBest = Math.sqrt(bestD), distSecond = Math.sqrt(secondD);
        if (distSecond - distBest < 1e-6) { skippedTie++; continue; }

        const c = handle.cell(best, bx0, by0, bx1, by1, buf);
        assert.ok(c >= 3, `nearest site ${best} returned a degenerate cell (c=${c}) for an interior query`);
        assert.ok(pointInPolygon(qx, qy, buf, c, 1e-7),
            `query (${qx},${qy}) not inside its nearest site ${best}'s cell`);
        checkedNearest++;

        for (let t = 0; t < 3; t++) {
            let j = (qrng() * N) | 0;
            if (j === best) j = (j + 1) % N;
            const cj = handle.cell(j, bx0, by0, bx1, by1, buf);
            if (cj === 0) continue;
            assert.ok(!strictlyInside(qx, qy, buf, cj, 1e-9),
                `other site ${j} strictly contains the query point belonging to nearest site ${best}`);
            checkedOthers++;
        }
    }

    assert.ok(checkedNearest >= SAMPLES * 0.9,
        `too many ties or degenerate cells: checked=${checkedNearest} skipped=${skippedTie}`);
    handle.dispose();
    console.log("A1b CONTAINMENT sampled=" + sampled + " checkedNearest=" + checkedNearest +
        " checkedOthers=" + checkedOthers + " skippedTie=" + skippedTie);
});

// =============================================================================
// A2 SIZING
// =============================================================================
test("A2 SIZING: vertexCount <= degree + 4 (interior) / degree + 5 (hull), across the cloud sweep", () => {
    const clouds = buildSweepClouds();
    let total = 0;
    for (const cloud of clouds) {
        const mesh = buildMeshInfo(cloud.pxs, cloud.pys, cloud.n);
        const factory = createCellIndex(cloud.n);
        const handle = factory(cloud.pxs, cloud.pys, cloud.n);
        const bbox = sweepBBoxesFor(cloud.pxs, cloud.pys, cloud.n)[1]; // containing
        const buf = mkBuf(cloud.n);
        for (let i = 0; i < cloud.n; i++) {
            const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
            if (c === 0) continue;
            const deg = mesh.degree[i];
            const bound = deg + (mesh.isHull[i] ? 5 : 4);
            assert.ok(c <= bound,
                `${cloud.name} site ${i}: vertexCount ${c} exceeds bound ${bound} (degree=${deg} hull=${mesh.isHull[i]})`);
            total++;
        }
        handle.dispose();
    }
    assert.ok(total >= 200, `A2 SIZING sample too small: ${total}`);
    console.log("A2 SIZING cases=" + total);
});

test("A2 SIZING: huge bbox -- interior site's cell equals its raw circumcenter fan exactly", () => {
    const rng = mulberry32(0x2A2A);
    const { pxs, pys } = genGridJittered(rng, 6, 10);
    const n = 36;
    const mesh = buildMeshInfo(pxs, pys, n);

    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const { minX, minY, maxX, maxY } = bboxOfFiniteLocal(pxs, pys, n);
    const HUGE = 1e7;
    const bx0 = minX - HUGE, by0 = minY - HUGE, bx1 = maxX + HUGE, by1 = maxY + HUGE;
    const buf = mkBuf(n);

    let checked = 0;
    for (let i = 0; i < n; i++) {
        if (mesh.isHull[i]) continue;
        const c = handle.cell(i, bx0, by0, bx1, by1, buf);
        if (c === 0) continue;
        assert.equal(c, mesh.degree[i], `interior site ${i}: vertexCount != degree under a huge bbox`);

        const incident = [];
        for (let t = 0; t < mesh.triCount; t++) {
            const a = mesh.triangles[3 * t], b = mesh.triangles[3 * t + 1], cc = mesh.triangles[3 * t + 2];
            if (a === i || b === i || cc === i) incident.push([mesh.ccx[t], mesh.ccy[t]]);
        }
        const used = new Array(incident.length).fill(false);
        for (let k = 0; k < c; k++) {
            const x = buf[2 * k], y = buf[2 * k + 1];
            let matched = -1;
            for (let m = 0; m < incident.length; m++) {
                if (used[m]) continue;
                if (Math.hypot(x - incident[m][0], y - incident[m][1]) < 1e-9) { matched = m; break; }
            }
            assert.ok(matched >= 0, `interior site ${i} vertex ${k} (${x},${y}) matches no incident circumcenter`);
            used[matched] = true;
        }
        checked++;
    }
    handle.dispose();
    assert.ok(checked >= 5, `too few interior sites checked: ${checked}`);
    console.log("A2 SIZING exactness cases=" + checked);
});

// =============================================================================
// A3 DEGENERATE MATRIX
// =============================================================================
test("A3 DEGENERATE MATRIX: n=0 -- any cell(i) throws (range [0,0) is empty)", () => {
    const factory = createCellIndex(5);
    const handle = factory(new Float32Array(0), new Float32Array(0), 0);
    assert.throws(() => handle.cell(0, -1, -1, 1, 1, mkBuf(5)), /lite-delaunay: /);
    handle.dispose();
});

test("A3 DEGENERATE MATRIX: n=1, n=2 -- valid cell() calls return 0, never throw", () => {
    for (const n of [1, 2]) {
        const rng = mulberry32(200 + n);
        const { pxs, pys } = genUniform(rng, n, -10, 10);
        const factory = createCellIndex(Math.max(n, 3));
        const handle = factory(pxs, pys, n);
        const buf = mkBuf(n);
        for (let i = 0; i < n; i++) {
            let c;
            assert.doesNotThrow(() => { c = handle.cell(i, -100, -100, 100, 100, buf); });
            assert.equal(c, 0, `n=${n} site ${i} must return 0 (degenerate build)`);
        }
        handle.dispose();
    }
});

test("A3 DEGENERATE MATRIX: all-coincident (x20) -- every cell() returns 0, no throw", () => {
    const n = 20;
    const pxs = new Float32Array(n).fill(3.5);
    const pys = new Float32Array(n).fill(-2.25);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const buf = mkBuf(n);
    for (let i = 0; i < n; i++) {
        let c;
        assert.doesNotThrow(() => { c = handle.cell(i, -10, -10, 10, 10, buf); });
        assert.equal(c, 0, `coincident site ${i} must return 0`);
    }
    handle.dispose();
});

test("A3 DEGENERATE MATRIX: all-collinear (30 points on a line) -- every cell() returns 0, no throw", () => {
    const n = 30;
    const rng = mulberry32(0x0C01);
    const pxs = new Float32Array(n), pys = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const t = rng() * 1000 - 500;
        pxs[i] = t; pys[i] = 2 * t + 3;
    }
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const buf = mkBuf(n);
    for (let i = 0; i < n; i++) {
        let c;
        assert.doesNotThrow(() => { c = handle.cell(i, -600, -1200, 600, 1200, buf); });
        assert.equal(c, 0, `collinear site ${i} must return 0`);
    }
    handle.dispose();
});

test("A3 DEGENERATE MATRIX: all-NaN input builds successfully; every cell() returns 0", () => {
    const n = 50;
    const pxs = new Float32Array(n).fill(NaN);
    const pys = new Float32Array(n).fill(NaN);
    const factory = createCellIndex(n);
    let handle;
    assert.doesNotThrow(() => { handle = factory(pxs, pys, n); });
    const buf = mkBuf(n);
    for (let i = 0; i < n; i++) {
        const c = handle.cell(i, -10, -10, 10, 10, buf);
        assert.equal(c, 0, `all-NaN site ${i} must return 0`);
    }
    handle.dispose();
});

test("A3 DEGENERATE MATRIX: mixed ~20% NaN -- NaN sites return 0, finite sites tile the bbox", () => {
    const n = 150;
    const rng = mulberry32(0x0C02);
    const { pxs, pys } = genUniform(rng, n, -300, 300);
    const nanFlag = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
        if (rng() < 0.2) { pxs[i] = NaN; nanFlag[i] = 1; }
    }
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const bbox = sweepBBoxesFor(pxs, pys, n)[1]; // containing bbox over the FINITE subset
    const buf = mkBuf(n);
    let total = 0, nanChecked = 0, finiteChecked = 0;
    for (let i = 0; i < n; i++) {
        const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
        if (nanFlag[i]) {
            assert.equal(c, 0, `NaN site ${i} must return 0`);
            nanChecked++;
        } else {
            finiteChecked++;
            if (c > 0) total += polyArea(buf, c);
        }
    }
    const bboxArea = (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0);
    const relErr = Math.abs(total - bboxArea) / bboxArea;
    assert.ok(relErr < 1e-6, `mixed-NaN tiling mismatch relErr=${relErr}`);
    assert.ok(nanChecked > 0 && finiteChecked > 0, "test setup: need both NaN and finite sites");
    handle.dispose();
    console.log("A3 DEGENERATE MATRIX mixed-NaN nan=" + nanChecked + " finite=" + finiteChecked + " relErr=" + relErr);
});

// =============================================================================
// A4 FAIL-CLOSED MATRIX
// =============================================================================
test("A4 FAIL-CLOSED: createCellIndex(maxPoints) validation matrix", () => {
    for (const bad of [-1, 1.5, NaN, undefined, null, "5", Infinity, -0.5]) {
        assert.throws(() => createCellIndex(bad), /lite-delaunay: /, `maxPoints=${String(bad)} must throw`);
    }
    assert.doesNotThrow(() => createCellIndex(0), "maxPoints=0 must NOT throw");
    assert.doesNotThrow(() => createCellIndex(-0), "maxPoints=-0 must NOT throw (integer zero)");
});

test("A4 FAIL-CLOSED: factory(pxs, pys, n) build validation matrix", () => {
    const MAXP = 5;
    const factory = createCellIndex(MAXP);
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

test("A4 FAIL-CLOSED: cell() index i validation matrix (0, 1, N-1, N, N+1, fractional, negative, NaN, -0)", () => {
    const n = 6;
    const rng = mulberry32(0xB0B0);
    const { pxs, pys } = genUniform(rng, n, -20, 20);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const buf = mkBuf(n);
    const bbox = sweepBBoxesFor(pxs, pys, n)[1];

    for (const i of [0, 1, n - 1]) {
        assert.doesNotThrow(() => handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf), `i=${i} must NOT throw`);
    }
    assert.doesNotThrow(() => handle.cell(-0, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf), "i=-0 must NOT throw");

    for (const i of [n, n + 1, -1, 0.5, NaN, -1.5]) {
        assert.throws(() => handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf), /lite-delaunay: /, `i=${i} must throw`);
    }
    handle.dispose();
});

test("A4 FAIL-CLOSED: cell() bbox validation matrix (non-finite, equal, inverted)", () => {
    const n = 6;
    const rng = mulberry32(0xB0B1);
    const { pxs, pys } = genUniform(rng, n, -20, 20);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const buf = mkBuf(n);

    const badBoxes = [
        [NaN, -10, 10, 10], [-10, NaN, 10, 10], [-10, -10, NaN, 10], [-10, -10, 10, NaN],
        [Infinity, -10, 10, 10], [-Infinity, -10, 10, 10],
        [-10, Infinity, 10, 10], [-10, -Infinity, 10, 10],
        [-10, -10, Infinity, 10], [-10, -10, -Infinity, 10],
        [-10, -10, 10, Infinity], [-10, -10, 10, -Infinity],
        [5, -10, 5, 10],    // bx0 === bx1
        [10, -10, 5, 10],   // bx0 > bx1
        [-10, 5, 10, 5],    // by0 === by1
        [-10, 10, 10, -10], // by0 > by1
    ];
    for (const [bx0, by0, bx1, by1] of badBoxes) {
        assert.throws(() => handle.cell(0, bx0, by0, bx1, by1, buf), /lite-delaunay: /,
            `bbox (${bx0},${by0},${bx1},${by1}) must throw`);
    }
    assert.doesNotThrow(() => handle.cell(0, -10, -10, 10, 10, buf));
    handle.dispose();
});

test("A4 FAIL-CLOSED: outXY too small throws (never truncates a real polygon)", () => {
    const rng = mulberry32(0xB0B2);
    const n = 40;
    const { pxs, pys } = genUniform(rng, n, -50, 50);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const bbox = sweepBBoxesFor(pxs, pys, n)[1];

    const scratchBuf = mkBuf(n);
    let target = -1, targetCount = 0;
    for (let i = 0; i < n; i++) {
        const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, scratchBuf);
        if (c >= 3) { target = i; targetCount = c; break; }
    }
    assert.ok(target >= 0, "test setup: no nonzero cell found");

    const fitBuf = new Float64Array(2 * targetCount);
    assert.doesNotThrow(() => handle.cell(target, bbox.x0, bbox.y0, bbox.x1, bbox.y1, fitBuf));

    const shortBuf = new Float64Array(2 * targetCount - 1);
    assert.throws(() => handle.cell(target, bbox.x0, bbox.y0, bbox.x1, bbox.y1, shortBuf), /lite-delaunay: /);

    const tinyBuf = new Float64Array(4); // room for 2 vertices -- never enough for a real cell
    assert.throws(() => handle.cell(target, bbox.x0, bbox.y0, bbox.x1, bbox.y1, tinyBuf), /lite-delaunay: /);

    handle.dispose();
});

test("A4 FAIL-CLOSED: cell() leaves outXY beyond 2*count as whatever was already there (sentinel probe)", () => {
    const rng = mulberry32(0xB0B3);
    const n = 40;
    const { pxs, pys } = genUniform(rng, n, -50, 50);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const bbox = sweepBBoxesFor(pxs, pys, n)[1];
    const buf = new Float64Array(2 * (n + 8));
    const SENTINEL = -123456.789;

    let checked = 0;
    for (let i = 0; i < n; i++) {
        buf.fill(SENTINEL);
        const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
        for (let k = 2 * c; k < buf.length; k++) {
            assert.equal(buf[k], SENTINEL, `site ${i}: outXY[${k}] modified past 2*count=${2 * c}`);
        }
        checked++;
    }
    handle.dispose();
    assert.equal(checked, n);
    console.log("A4 FAIL-CLOSED sentinel-probe sites=" + checked);
});

test("A4 FAIL-CLOSED: disposed handle throws on cell() and on a second dispose()", () => {
    const rng = mulberry32(0xB0B4);
    const n = 6;
    const { pxs, pys } = genUniform(rng, n, -10, 10);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    handle.dispose();
    assert.throws(() => handle.cell(0, -10, -10, 10, 10, mkBuf(n)), /lite-delaunay: /);
    assert.throws(() => handle.dispose(), /lite-delaunay: /);
});

// =============================================================================
// A5 LIFECYCLE
// =============================================================================
test("A5 LIFECYCLE: dispose-then-cell throws, double-dispose throws", () => {
    const rng = mulberry32(0x5001);
    const n = 8;
    const { pxs, pys } = genUniform(rng, n, -30, 30);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const bbox = sweepBBoxesFor(pxs, pys, n)[1];
    assert.doesNotThrow(() => handle.cell(0, bbox.x0, bbox.y0, bbox.x1, bbox.y1, mkBuf(n)));
    handle.dispose();
    assert.throws(() => handle.cell(0, bbox.x0, bbox.y0, bbox.x1, bbox.y1, mkBuf(n)));
    assert.throws(() => handle.dispose());
});

test("A5 LIFECYCLE: stale handle after slot reuse throws while the NEW handle answers correctly", () => {
    const factory = createCellIndex(20);
    const rngA = mulberry32(0x5A5A);
    const dataA = genUniform(rngA, 20, -50, -10);
    const rngB = mulberry32(0x5B5B);
    const dataB = genUniform(rngB, 20, 100, 200);

    const handleA = factory(dataA.pxs, dataA.pys, 20);
    const slotA = handleA._slot; // captured BEFORE dispose -- dispose() nulls handleA._slot
    handleA.dispose();
    const handleB = factory(dataB.pxs, dataB.pys, 20); // very likely reuses A's slot
    assert.equal(handleB._slot, slotA, "expected pooled slot reuse (LIFO free stack)");

    const bboxB = sweepBBoxesFor(dataB.pxs, dataB.pys, 20)[1];
    const buf = mkBuf(20);
    assert.doesNotThrow(() => handleB.cell(0, bboxB.x0, bboxB.y0, bboxB.x1, bboxB.y1, buf), "B before stale ops");

    assert.throws(() => handleA.cell(0, bboxB.x0, bboxB.y0, bboxB.x1, bboxB.y1, buf), /lite-delaunay: /);
    assert.throws(() => handleA.dispose(), /lite-delaunay: /);

    let c;
    assert.doesNotThrow(() => { c = handleB.cell(0, bboxB.x0, bboxB.y0, bboxB.x1, bboxB.y1, buf); });
    assert.ok(c === 0 || c >= 3);
    handleB.dispose();
});

test("A5 LIFECYCLE: 3 concurrent handles from one factory answer independently, disposed in shuffled order", () => {
    const factory = createCellIndex(60);
    const clouds = [];
    for (let i = 0; i < 3; i++) {
        const rng = mulberry32(0x6000 + i);
        const { pxs, pys } = genUniform(rng, 40, i * 500, i * 500 + 200);
        clouds.push({ pxs, pys, n: 40 });
    }
    const handles = clouds.map((c) => factory(c.pxs, c.pys, c.n));
    const bufs = clouds.map((c) => mkBuf(c.n));
    const bboxes = clouds.map((c) => sweepBBoxesFor(c.pxs, c.pys, c.n)[1]);

    function tilingSum(idx) {
        let total = 0;
        const bb = bboxes[idx];
        for (let i = 0; i < clouds[idx].n; i++) {
            const c = handles[idx].cell(i, bb.x0, bb.y0, bb.x1, bb.y1, bufs[idx]);
            if (c > 0) total += polyArea(bufs[idx], c);
        }
        return total;
    }

    const live = [true, true, true];
    function checkAllLive() {
        for (let idx = 0; idx < 3; idx++) {
            if (!live[idx]) continue;
            const bb = bboxes[idx];
            const total = tilingSum(idx);
            const area = (bb.x1 - bb.x0) * (bb.y1 - bb.y0);
            assert.ok(Math.abs(total - area) / area < 1e-6, `concurrent handle ${idx} tiling drift after sibling ops`);
        }
    }
    checkAllLive();

    const disposeOrder = [1, 0, 2];
    for (const idx of disposeOrder) {
        handles[idx].dispose();
        live[idx] = false;
        checkAllLive();
        assert.throws(() => handles[idx].cell(0, bboxes[idx].x0, bboxes[idx].y0, bboxes[idx].x1, bboxes[idx].y1, bufs[idx]),
            `handle ${idx} must throw after dispose`);
    }
});

test("A5 LIFECYCLE: build-after-dispose reuses the pooled slot (handle._slot identity)", () => {
    const factory = createCellIndex(10);
    const rng = mulberry32(0x7777);
    const { pxs, pys } = genUniform(rng, 10, -10, 10);
    const h1 = factory(pxs, pys, 10);
    const slot1 = h1._slot;
    h1.dispose();
    const h2 = factory(pxs, pys, 10);
    assert.equal(h2._slot, slot1, "expected the freed slot to be reused, not a fresh allocation");
    h2.dispose();
});

test("A5 LIFECYCLE / BOUNDARY: dispose-during-iteration does not corrupt prior results or a sibling handle", () => {
    const factory = createCellIndex(16);
    const rng = mulberry32(0x8888);
    const { pxs, pys } = genUniform(rng, 16, -50, 50);
    const handle = factory(pxs, pys, 16);
    const sibling = factory(pxs, pys, 16);
    const bbox = sweepBBoxesFor(pxs, pys, 16)[1];
    const buf = mkBuf(16);

    const collected = [];
    for (let i = 0; i < 16; i++) {
        if (i === 8) { handle.dispose(); continue; }
        if (i < 8) {
            const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
            collected.push({ i, c, poly: buf.slice(0, 2 * c) });
        } else {
            assert.throws(() => handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf));
        }
    }

    const cs = sibling.cell(0, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
    assert.ok(cs === 0 || cs >= 3);

    // pre-dispose results must still match a fresh rebuild over the same data
    const verifyFactory = createCellIndex(16);
    const verifyHandle = verifyFactory(pxs, pys, 16);
    const vbuf = mkBuf(16);
    for (const r of collected) {
        const vc = verifyHandle.cell(r.i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, vbuf);
        assert.equal(vc, r.c, `site ${r.i}: pre-dispose count doesn't match a fresh rebuild`);
        for (let k = 0; k < 2 * vc; k++) assert.equal(vbuf[k], r.poly[k], `site ${r.i}: pre-dispose vertex ${k} mismatch`);
    }
    verifyHandle.dispose();
    sibling.dispose();
});

// =============================================================================
// A6 GEOMETRY SANITY
// =============================================================================
test("A6 GEOMETRY SANITY: every emitted polygon is convex, finite, and inside-or-on the bbox", () => {
    const clouds = buildSweepClouds();
    let polysChecked = 0, vertsChecked = 0;
    for (const cloud of clouds) {
        const factory = createCellIndex(cloud.n);
        const handle = factory(cloud.pxs, cloud.pys, cloud.n);
        const buf = mkBuf(cloud.n);
        for (const bbox of sweepBBoxesFor(cloud.pxs, cloud.pys, cloud.n)) {
            for (let i = 0; i < cloud.n; i++) {
                const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
                if (c === 0) continue;
                assert.ok(isConvex(buf, c, 1e-9),
                    `${cloud.name}/${bbox.label} site ${i}: cell not convex`);
                for (let k = 0; k < c; k++) {
                    const x = buf[2 * k], y = buf[2 * k + 1];
                    assert.ok(Number.isFinite(x) && Number.isFinite(y),
                        `${cloud.name}/${bbox.label} site ${i} vertex ${k}: non-finite coordinate`);
                    assert.ok(insideOrOnBBox(x, y, bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                        `${cloud.name}/${bbox.label} site ${i} vertex ${k}: (${x},${y}) outside bbox`);
                    vertsChecked++;
                }
                polysChecked++;
            }
        }
        handle.dispose();
    }
    assert.ok(polysChecked >= 100, `A6 sample too small: ${polysChecked}`);
    console.log("A6 GEOMETRY SANITY polys=" + polysChecked + " verts=" + vertsChecked);
});

// =============================================================================
// A7 COPY SEMANTICS
// =============================================================================
test("A7 COPY SEMANTICS: mutating source pxs/pys after build does not change cell() answers", () => {
    const n = 80;
    const rng = mulberry32(0xA7A7);
    const { pxs, pys } = genUniform(rng, n, -50, 50);
    const snapPxs = pxs.slice(), snapPys = pys.slice();
    const bbox = sweepBBoxesFor(snapPxs, snapPys, n)[1];

    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);

    const buf1 = mkBuf(n);
    const before = [];
    for (let i = 0; i < n; i++) {
        const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf1);
        before.push(buf1.slice(0, 2 * c));
    }

    // mutate the SOURCE arrays in place after build -- collapse everything.
    for (let i = 0; i < n; i++) { pxs[i] = 9999; pys[i] = 9999; }

    const buf2 = mkBuf(n);
    let anyNonzero = false;
    for (let i = 0; i < n; i++) {
        const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf2);
        const b = before[i];
        assert.equal(2 * c, b.length, `site ${i}: vertex count changed after source mutation`);
        for (let k = 0; k < b.length; k++) {
            assert.equal(buf2[k], b[k], `site ${i}: vertex ${k} changed after source mutation`);
        }
        if (c > 0) anyNonzero = true;
    }
    assert.ok(anyNonzero, "expected at least one nonzero cell (copy-semantics sanity)");
    handle.dispose();
    console.log("A7 COPY SEMANTICS sites=" + n);
});

// =============================================================================
// BOUNDARY MATRIX extras (empty outXY, Float32Array outXY, re-entrant reuse)
// =============================================================================
test("BOUNDARY: cell() with an empty (zero-length) outXY succeeds when count is 0", () => {
    const n = 20;
    const pxs = new Float32Array(n).fill(1);
    const pys = new Float32Array(n).fill(1); // all-coincident -> degenerate -> every cell is 0
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    let c;
    assert.doesNotThrow(() => { c = handle.cell(0, -1, -1, 1, 1, new Float64Array(0)); });
    assert.equal(c, 0);
    handle.dispose();
});

test("BOUNDARY: outXY as Float32Array (reduced precision) still yields valid convex cells", () => {
    const rng = mulberry32(0xF32F);
    const n = 25;
    const { pxs, pys } = genUniform(rng, n, -30, 30);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const bbox = sweepBBoxesFor(pxs, pys, n)[1];
    const buf32 = new Float32Array(2 * (n + 8));
    let checked = 0;
    for (let i = 0; i < n; i++) {
        const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf32);
        assert.ok(c === 0 || c >= 3, `site ${i}: vertexCount ${c} not 0 or >=3 (f32 out)`);
        if (c > 0) {
            for (let k = 0; k < 2 * c; k++) assert.ok(Number.isFinite(buf32[k]), `site ${i}: non-finite f32 vertex`);
            checked++;
        }
    }
    handle.dispose();
    assert.ok(checked > 0, "expected at least one nonzero cell with a Float32Array output buffer");
});

test("BOUNDARY: re-entrant cell() calls on the SAME buffer across different sites leave no stale tail leakage", () => {
    const rng = mulberry32(0x9999);
    const n = 30;
    const { pxs, pys } = genUniform(rng, n, -40, 40);
    const factory = createCellIndex(n);
    const handle = factory(pxs, pys, n);
    const bbox = sweepBBoxesFor(pxs, pys, n)[1];
    const buf = mkBuf(n);

    let siteA = -1, siteB = -1, cA = 0, cB = 0;
    for (let i = 0; i < n && (siteA < 0 || siteB < 0); i++) {
        const c = handle.cell(i, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
        if (c === 0) continue;
        if (siteA < 0) { siteA = i; cA = c; }
        else if (c !== cA) { siteB = i; cB = c; }
    }
    assert.ok(siteA >= 0 && siteB >= 0, "test setup: need two differently-sized cells");

    const c1 = handle.cell(siteA, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
    const poly1 = buf.slice(0, 2 * c1);
    const c2 = handle.cell(siteB, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf); // SAME buffer, second call
    assert.equal(c1, cA);
    assert.equal(c2, cB);
    assert.notEqual(c1, c2, "need genuinely differing counts to prove no leakage");
    assert.ok(isConvex(buf, c2), "second call result corrupted");

    const c1b = handle.cell(siteA, bbox.x0, bbox.y0, bbox.x1, bbox.y1, buf);
    assert.equal(c1b, c1);
    for (let k = 0; k < 2 * c1; k++) assert.equal(buf[k], poly1[k]);
    handle.dispose();
});

// =============================================================================
// ADVERSARIAL
// =============================================================================
test("ADVERSARIAL: interleaved cell() calls from two concurrent handles into the SAME outXY buffer do not cross-contaminate", () => {
    const factory = createCellIndex(50);
    const rngA = mulberry32(0xAD01);
    const rngB = mulberry32(0xAD02);
    const cloudA = genUniform(rngA, 35, -100, 100);
    const cloudB = genGridJittered(rngB, 6, 12); // n=36, different shape entirely

    const handleA = factory(cloudA.pxs, cloudA.pys, 35);
    const handleB = factory(cloudB.pxs, cloudB.pys, 36);
    assert.notEqual(handleA._slot, handleB._slot, "test setup: expected two DIFFERENT concurrent pool slots");

    const sharedBuf = new Float64Array(2 * 64);
    const bboxA = sweepBBoxesFor(cloudA.pxs, cloudA.pys, 35)[1];
    const bboxB = sweepBBoxesFor(cloudB.pxs, cloudB.pys, 36)[1];

    const refBufA = new Float64Array(2 * 64);
    const cARef = handleA.cell(0, bboxA.x0, bboxA.y0, bboxA.x1, bboxA.y1, refBufA);
    assert.ok(cARef >= 3, "test setup: site 0 of cloud A must have a real cell");

    let mismatches = 0;
    for (let round = 0; round < 20; round++) {
        handleA.cell(0, bboxA.x0, bboxA.y0, bboxA.x1, bboxA.y1, sharedBuf);
        handleB.cell(round % 36, bboxB.x0, bboxB.y0, bboxB.x1, bboxB.y1, sharedBuf); // SAME buffer, sibling slot
        const cA2 = handleA.cell(0, bboxA.x0, bboxA.y0, bboxA.x1, bboxA.y1, sharedBuf);
        if (cA2 !== cARef) { mismatches++; continue; }
        for (let k = 0; k < 2 * cARef; k++) {
            if (sharedBuf[k] !== refBufA[k]) { mismatches++; break; }
        }
    }
    assert.equal(mismatches, 0,
        `handle A's answers drifted after interleaved handle-B writes into the shared buffer: ${mismatches}/20`);
    handleA.dispose();
    handleB.dispose();
});
