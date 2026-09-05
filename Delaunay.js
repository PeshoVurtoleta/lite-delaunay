/**
 * @zakkster/lite-delaunay
 * -----------------------------------------------------------------------------
 * Zero-GC 2D Delaunay triangulation via the Mapbox-style sweepline algorithm.
 *
 * Memory model: one allocation at construction time, sized for `maxPoints`.
 * Every subsequent `triangulate()` call writes into the same arena -- no
 * `new` in the hot path, no garbage produced per frame.
 *
 * Algorithm: ported from Mapbox's Delaunator (ISC-licensed) onto a strict
 * Int32Array arena. Same O(n log n) average-case complexity; same Delaunay
 * property guarantees; same halfedge mesh representation.
 *
 * Output topology (after `triangulate(coords, n)` returns `triCount`):
 *   - `triangles[3*k .. 3*k+2]` holds the three vertex indices of triangle `k`
 *     in counter-clockwise winding order (math convention; in screen-space
 *     where Y points down this looks clockwise).
 *   - `halfedges[3*k+i]` is the index of the OPPOSITE half-edge (across the
 *     shared edge in the neighbouring triangle), or -1 if the edge is on the
 *     convex hull. This is the standard half-edge mesh representation --
 *     suitable for Voronoi dual extraction, edge walks, neighbour queries.
 *   - Both arrays are valid only up to index `trianglesLen` (== `triCount * 3`).
 *     Indices past that are leftover bytes from a previous call.
 *
 * Degenerate inputs (returns 0, does not throw):
 *   - fewer than 3 input points
 *   - all points coincident
 *   - all points collinear (no 2D triangulation exists)
 *   Callers can detect these by checking the return value or `trianglesLen`.
 *
 * @module @zakkster/lite-delaunay
 */

/**
 * Package version. Kept in lockstep with package.json and llms.txt -- the three
 * are bumped in the same commit or not at all (packaging law).
 * @type {string}
 */
export const VERSION = "1.4.0";

// IEEE 754 double machine-epsilon. Used for the near-duplicate-point skip
// in the advancing-front loop (matches Mapbox Delaunator's tolerance).
const EPSILON = Math.pow(2, -52);

/**
 * Pre-allocated, zero-allocation Delaunay triangulator.
 *
 * Lifetime: construct once with the largest point count you expect to handle,
 * then call `triangulate()` as many times as you like -- including with smaller
 * point counts. The arena is never resized or re-allocated.
 *
 * @example
 * const tri = new DelaunayTriangulator(10_000);
 * for (const frame of frames) {
 *   const count = tri.triangulate(frame.coords, frame.n);
 *   for (let k = 0; k < count; k++) {
 *     const a = tri.triangles[k * 3];
 *     const b = tri.triangles[k * 3 + 1];
 *     const c = tri.triangles[k * 3 + 2];
 *     // ... emit triangle (a, b, c) ...
 *   }
 * }
 */
export class DelaunayTriangulator {
    /** Capacity (maximum points the arena was sized for). Set once, never changes. */
    maxPoints;
    /** Capacity-derived: 2N - 5 (the Euler upper bound on triangles for N points). */
    maxTriangles;
    /** Capacity-derived: ceil(sqrt(N)) + 1 (angular hash bucket count for hull lookup). */
    hashSize;

    /** Flat vertex-index array: triangle k's verts live at [3k, 3k+1, 3k+2]. Valid up to `trianglesLen`. */
    triangles;
    /** Half-edge twin array, parallel to `triangles`. -1 means the edge is on the convex hull. */
    halfedges;

    // -- Internal hull-tracking state (advancing-front bookkeeping) --
    hullPrev;
    hullNext;
    hullTri;
    hullHash;

    // -- Internal sort scratch --
    ids;
    dists;

    // -- Internal fixed-size stacks (legalize flip stack, quicksort range stack) --
    edgeStack;
    sortStack;

    /** Number of *valid* indices in `triangles` (== triangleCount × 3). Reset on every triangulate() call. */
    trianglesLen = 0;
    hullStart = 0;
    hullLen = 0;

    cx = 0;
    cy = 0;

    /**
     * Allocate the arena. This is the only `new` the library will ever do
     * (apart from constructor error messages on misuse).
     *
     * Memory footprint, in bytes, is approximately:
     *   triangles+halfedges:  (2N - 5) × 3 × 4 × 2  ~ 48N
     *   hull state:           N × 4 × 3              = 12N
     *   sort scratch:         N × 8                  =  8N
     *   fixed stacks:         ~16 KB                 (constant)
     * Total: ~68 bytes per max-point + 16 KB. For N=10_000 that's ~700 KB.
     *
     * @param {number} maxPoints The hard upper bound on input size. Pick the
     *   largest count you expect to ever pass to `triangulate()`. Must be a
     *   non-negative integer. Passing a tighter bound saves memory.
     * @throws {Error} If `maxPoints` is not a non-negative integer.
     */
    constructor(maxPoints) {
        if (!Number.isInteger(maxPoints) || maxPoints < 0) {
            throw new Error(`lite-delaunay: maxPoints must be a non-negative integer, got ${maxPoints}`);
        }

        this.maxPoints = maxPoints;
        this.maxTriangles = Math.max(2 * maxPoints - 5, 0);
        this.hashSize = Math.ceil(Math.sqrt(maxPoints)) + 1;

        this.triangles = new Int32Array(this.maxTriangles * 3);
        this.halfedges = new Int32Array(this.maxTriangles * 3);

        this.hullPrev = new Int32Array(maxPoints);
        this.hullNext = new Int32Array(maxPoints);
        this.hullTri = new Int32Array(maxPoints);
        this.hullHash = new Int32Array(this.hashSize);

        this.ids = new Int32Array(maxPoints);
        this.dists = new Float32Array(maxPoints);

        this.edgeStack = new Int32Array(4096); // Bumped for extreme pathological safety
        this.sortStack = new Int32Array(64);
    }

    /**
     * Triangulate `pointCount` points and populate `triangles` + `halfedges`.
     *
     * The input `coords` array MUST be flat & interleaved: `[x0, y0, x1, y1, ...]`.
     * `Float32Array` is recommended for cache-friendliness and to avoid silent
     * doubles-to-floats truncation later; `Float64Array` and plain `number[]`
     * also work (typed access is faster).
     *
     * **Zero allocations**: this method does not allocate. The only `new` it
     * can produce is an `Error` on a precondition failure (which is, by
     * definition, off the hot path).
     *
     * **Idempotent**: calling with the same input twice produces the same
     * topology. State from a previous call is fully reset.
     *
     * **Degenerate inputs return 0** (do not throw): empty input, single point,
     * two points, all-coincident points, or fully-collinear points. Detect
     * these by checking the return value.
     *
     * After this returns `k`, you may read:
     *   - `triangles[0 .. k*3 - 1]` -- vertex indices, CCW
     *   - `halfedges[0 .. k*3 - 1]` -- twin half-edge indices, or -1 on the hull
     *   - `trianglesLen` -- equals `k * 3`
     *
     * @param {Float32Array | Float64Array | number[]} coords Flat interleaved
     *   `[x, y, x, y, ...]`. Must contain at least `pointCount * 2` elements.
     * @param {number} pointCount Number of logical (x, y) points to process.
     *   Must be <= `maxPoints`. If less than 3, returns 0 with no work done.
     * @returns {number} The number of triangles generated (0 on degenerate input).
     * @throws {Error} If `pointCount` exceeds the arena's `maxPoints`.
     */
    triangulate(coords, pointCount) {
        // Reset state UNCONDITIONALLY -- must come before any early return so
        // a previous run's triangles don't leak into a degenerate call.
        this.trianglesLen = 0;
        this.hullLen = 0;

        if (pointCount < 3) return 0;
        if (pointCount > this.maxPoints) {
            throw new Error(`lite-delaunay: pointCount (${pointCount}) exceeds arena max (${this.maxPoints})`);
        }

        this.hullHash.fill(-1);

        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        // 1. Calculate Bounding Box
        for (let i = 0; i < pointCount; i++) {
            const idx = i << 1;
            const x = coords[idx];
            const y = coords[idx + 1];
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
            this.ids[i] = i;
        }

        const cx = (minX + maxX) / 2;
        const cy = (minY + maxY) / 2;
        this.cx = cx;
        this.cy = cy;

        // (ids[] was already initialized in the bbox pass above.)

        // 2. Find the Seed Triangle
        // ------------------------------------------------------------------
        // No sort is needed yet -- the seed-finding steps below are O(N)
        // argmin scans. Sorting before this point (as a previous version did)
        // (a) wastes O(N log N) work, and (b) produces nearly-sorted input
        // for the LATER circumcenter sort, which degrades Hoare quicksort
        // with a middle-element pivot to O(N^2).
        //
        // i1/i2 are seeded to -1 so we can DETECT degeneracy: if the search
        // never finds a distinct-from-i0 point (all points coincident) or
        // never finds a non-collinear third point (all points on one line),
        // we bail out cleanly. Without this, the advancing-front loop walks
        // a hull built from coincident vertices and never terminates.
        let i0 = 0, i1 = -1, i2 = -1;
        let minDist = Infinity;

        // Find the point closest to the bbox center
        for (let i = 0; i < pointCount; i++) {
            const dx = coords[i << 1] - cx;
            const dy = coords[(i << 1) + 1] - cy;
            const d = dx * dx + dy * dy;
            if (d < minDist) {
                i0 = i;
                minDist = d;
            }
        }
        const i0x = coords[i0 << 1];
        const i0y = coords[(i0 << 1) + 1];

        // Find the point closest to i0 (must be distinct, i.e. distSq > 0)
        minDist = Infinity;
        for (let i = 0; i < pointCount; i++) {
            const id = this.ids[i];
            if (id === i0) continue;
            const dx = coords[id << 1] - i0x;
            const dy = coords[(id << 1) + 1] - i0y;
            const distSq = dx * dx + dy * dy;
            if (distSq < minDist && distSq > 0) {
                i1 = id;
                minDist = distSq;
            }
        }
        // All points are coincident with i0 -- no triangulation possible.
        if (i1 === -1) return 0;
        let i1x = coords[i1 << 1];
        let i1y = coords[(i1 << 1) + 1];

        // Find the third point forming the smallest circumcircle.
        // For fully-collinear inputs every candidate has infinite circumradius,
        // so we'd silently keep the -1 sentinel -- that's the bail-out path.
        let minRadius = Infinity;
        for (let i = 0; i < pointCount; i++) {
            const id = this.ids[i];
            if (id === i0 || id === i1) continue;
            const r = this._circumradius(i0x, i0y, i1x, i1y, coords[id << 1], coords[(id << 1) + 1]);
            if (r < minRadius) {
                i2 = id;
                minRadius = r;
            }
        }
        // All points are collinear with i0 and i1 -- no 2D triangulation exists.
        if (i2 === -1) return 0;
        let i2x = coords[i2 << 1];
        let i2y = coords[(i2 << 1) + 1];

        // Swap to ensure counter-clockwise orientation (matches Delaunator).
        // _orient is mathematically identical to robust-predicates' orient2d:
        // returns negative for screen-CCW (math-CCW in y-down screen coords).
        if (this._orient(i0x, i0y, i1x, i1y, i2x, i2y) < 0) {
            const t = i1;
            i1 = i2;
            i2 = t;
            const tx = i1x;
            i1x = i2x;
            i2x = tx;
            const ty = i1y;
            i1y = i2y;
            i2y = ty;
        }

        // Calculate seed circumcenter to use as the origin for the hull hash
        this._circumcenter(i0x, i0y, i1x, i1y, i2x, i2y);

        // 3. Spatial Sort by Circumcenter Distance
        // ------------------------------------------------------------------
        // The advancing-front algorithm REQUIRES points to arrive in roughly
        // outward-spiral order from the seed circumcenter, so that each new
        // point is outside the current hull. Without this sort the hull
        // invariant breaks, halfedge pairings get scrambled, and the final
        // mesh contains overlapping triangles and Delaunay violations.
        // (Mapbox Delaunator does the same sort at the same point.)
        const ccx = this.cx, ccy = this.cy;
        for (let i = 0; i < pointCount; i++) {
            const idx = i << 1;
            const dx = coords[idx] - ccx;
            const dy = coords[idx + 1] - ccy;
            this.dists[i] = dx * dx + dy * dy;
        }
        this._quicksort(0, pointCount - 1);

        // 4. Initialize the Hull
        this.hullStart = i0;
        this.hullLen = 3;

        this.hullNext[i0] = this.hullPrev[i2] = i1;
        this.hullNext[i1] = this.hullPrev[i0] = i2;
        this.hullNext[i2] = this.hullPrev[i1] = i0;

        this.hullTri[i0] = 0;
        this.hullTri[i1] = 1;
        this.hullTri[i2] = 2;

        this.hullHash[this._hashKey(i0x, i0y)] = i0;
        this.hullHash[this._hashKey(i1x, i1y)] = i1;
        this.hullHash[this._hashKey(i2x, i2y)] = i2;

        this._addTriangle(i0, i1, i2, -1, -1, -1);

        let xp = 0, yp = 0;

        // 6. The Advancing Front
        for (let k = 0; k < pointCount; k++) {
            const i = this.ids[k];
            const x = coords[i << 1];
            const y = coords[(i << 1) + 1];

            // Duplicate-point check MUST happen BEFORE the seed-skip
            if (k > 0 && Math.abs(x - xp) <= EPSILON && Math.abs(y - yp) <= EPSILON) continue;
            xp = x;
            yp = y;

            if (i === i0 || i === i1 || i === i2) continue;

            // Find the visible edge on the hull
            let startKey = this._hashKey(x, y);
            let start = this.hullHash[startKey];

            // Hash bound guard prevents infinite loops on entirely stale buckets
            for (let j = 0; j < this.hashSize; j++) {
                if (start !== -1 && start !== this.hullNext[start]) break;
                startKey = (startKey + 1) % this.hashSize;
                start = this.hullHash[startKey];
            }

            start = this.hullPrev[start];
            let e = start;
            let q;
            while (true) {
                q = this.hullNext[e];
                // Walk while edge e->q is not yet visible from (x,y).
                // orient2d(x,y,e,q) >= 0 means (x,y) is left-of or on the edge --
                // edge is hull-interior side; keep walking. Break when < 0
                // (point is outside, edge is visible).
                if (this._orient(x, y, coords[e << 1], coords[(e << 1) + 1], coords[q << 1], coords[(q << 1) + 1]) < 0) {
                    break;
                }
                e = q;
                if (e === start) {
                    e = -1; // All points are collinear
                    break;
                }
            }
            if (e === -1) continue; // Collinear pathological case

            // Add the first triangle from the point
            let t = this._addTriangle(e, i, this.hullNext[e], -1, -1, this.hullTri[e]);

            // Recursively flip triangles from the point until they satisfy Delaunay condition
            this.hullTri[i] = this._legalize(t + 2, coords);
            this.hullTri[e] = t; // keep track of boundary triangles on the hull
            this.hullLen++;

            // Walk forward through the hull, adding more triangles and flipping recursively
            let next = this.hullNext[e];
            while (true) {
                q = this.hullNext[next];
                // Continue while edge next->q remains visible from (x,y), i.e. orient < 0.
                // Break when it stops being visible.
                if (this._orient(x, y, coords[next << 1], coords[(next << 1) + 1], coords[q << 1], coords[(q << 1) + 1]) >= 0) {
                    break;
                }
                t = this._addTriangle(next, i, q, this.hullTri[i], -1, this.hullTri[next]);
                this.hullTri[i] = this._legalize(t + 2, coords);
                this.hullNext[next] = next; // mark as removed
                this.hullLen--;
                next = q;
            }

            // Walk backward from the other side, adding more triangles and flipping.
            // The b/c halfedge parameters MUST be hullTri[e] and hullTri[q] (the
            // saved hull-triangle pointers for those two hull verts), NOT the
            // previous `t`. Getting these wrong produces a topology with the
            // right triangle count but scrambled halfedge pairing -- visible
            // only on inputs where the backward walk fires (roughly N >= ~20).
            if (e === start) {
                while (true) {
                    q = this.hullPrev[e];
                    if (this._orient(x, y, coords[q << 1], coords[(q << 1) + 1], coords[e << 1], coords[(e << 1) + 1]) >= 0) {
                        break;
                    }
                    t = this._addTriangle(q, i, e, -1, this.hullTri[e], this.hullTri[q]);
                    this._legalize(t + 2, coords);
                    this.hullTri[q] = t;
                    this.hullNext[e] = e; // mark as removed
                    this.hullLen--;
                    e = q;
                }
            }

            // Update the hull indices
            this.hullStart = this.hullPrev[i] = e;
            this.hullNext[e] = this.hullPrev[next] = i;
            this.hullNext[i] = next;

            // Save the two new edges in the hash table
            this.hullHash[this._hashKey(x, y)] = i;
            this.hullHash[this._hashKey(coords[e << 1], coords[(e << 1) + 1])] = e;
        }

        return this.trianglesLen / 3;
    }

    // -----------------------------------------------------------------
    // Internal Math & Helpers
    // -----------------------------------------------------------------

    _hashKey(x, y) {
        const dx = x - this.cx;
        const dy = y - this.cy;
        const p = dx / (Math.abs(dx) + Math.abs(dy));
        const a = (dy > 0 ? 3 - p : 1 + p) * 0.25; // mapped to [0, 1)
        return Math.floor(a * this.hashSize) % this.hashSize;
    }

    _legalize(a, coords) {
        let i = 0;
        let ar = 0;

        while (true) {
            const b = this.halfedges[a];
            const a0 = a - a % 3;
            ar = a0 + (a + 2) % 3;

            // If it's a hull edge, or we're done, pop from stack
            if (b === -1) {
                if (i === 0) break;
                a = this.edgeStack[--i];
                continue;
            }

            const b0 = b - b % 3;
            const al = a0 + (a + 1) % 3;
            const bl = b0 + (b + 2) % 3;

            const p0 = this.triangles[ar];
            const pr = this.triangles[a];
            const pl = this.triangles[al];
            const p1 = this.triangles[bl];

            const illegal = this._inCircle(
                coords[p0 << 1], coords[(p0 << 1) + 1],
                coords[pr << 1], coords[(pr << 1) + 1],
                coords[pl << 1], coords[(pl << 1) + 1],
                coords[p1 << 1], coords[(p1 << 1) + 1]
            );

            if (illegal) {
                // Flip the edge
                this.triangles[a] = p1;
                this.triangles[b] = p0;

                const hbl = this.halfedges[bl];
                if (hbl === -1) {
                    let e = this.hullStart;
                    do {
                        if (this.hullTri[e] === bl) {
                            this.hullTri[e] = a;
                            break;
                        }
                        e = this.hullPrev[e];
                    } while (e !== this.hullStart);
                }
                this._link(a, hbl);
                this._link(b, this.halfedges[ar]);
                this._link(ar, bl);

                const br = b0 + (b + 1) % 3;

                // Defer 'br' to stack, but KEEP 'a' in the active loop register
                if (i < this.edgeStack.length) {
                    this.edgeStack[i++] = br;
                }
            } else {
                if (i === 0) break;
                a = this.edgeStack[--i];
            }
        }
        return ar;
    }

    _link(a, b) {
        this.halfedges[a] = b;
        if (b !== -1) this.halfedges[b] = a;
    }

    _orient(px, py, qx, qy, rx, ry) {
        return (qy - py) * (rx - qx) - (qx - px) * (ry - qy);
    }

    _circumradius(ax, ay, bx, by, cx, cy) {
        const dx = bx - ax, dy = by - ay;
        const ex = cx - ax, ey = cy - ay;
        const bl = dx * dx + dy * dy;
        const cl = ex * ex + ey * ey;
        const d = 0.5 / (dx * ey - dy * ex);
        const x = (ey * bl - dy * cl) * d;
        const y = (dx * cl - ex * bl) * d;
        return x * x + y * y;
    }

    _circumcenter(ax, ay, bx, by, cx, cy) {
        const dx = bx - ax, dy = by - ay;
        const ex = cx - ax, ey = cy - ay;
        const bl = dx * dx + dy * dy;
        const cl = ex * ex + ey * ey;
        const d = 0.5 / (dx * ey - dy * ex);

        // Write directly to the pre-allocated instance properties
        this.cx = ax + (ey * bl - dy * cl) * d;
        this.cy = ay + (dx * cl - ex * bl) * d;
    }

    _inCircle(ax, ay, bx, by, cx, cy, px, py) {
        const dx = ax - px, dy = ay - py;
        const ex = bx - px, ey = by - py;
        const fx = cx - px, fy = cy - py;

        const ap = dx * dx + dy * dy;
        const bp = ex * ex + ey * ey;
        const cp = fx * fx + fy * fy;

        return dx * (ey * cp - bp * fy) -
            dy * (ex * cp - bp * fx) +
            ap * (ex * fy - ey * fx) < 0;
    }

    _addTriangle(i0, i1, i2, a, b, c) {
        const t0 = this.trianglesLen;
        const t1 = t0 + 1;
        const t2 = t0 + 2;

        this.triangles[t0] = i0;
        this.triangles[t1] = i1;
        this.triangles[t2] = i2;

        this._link(t0, a);
        this._link(t1, b);
        this._link(t2, c);

        this.trianglesLen += 3;
        return t0;
    }

    _quicksort(left, right) {
        const stack = this.sortStack;
        const dists = this.dists;
        const ids = this.ids;
        let stackPtr = 0;

        stack[stackPtr++] = left;
        stack[stackPtr++] = right;

        while (stackPtr > 0) {
            const r = stack[--stackPtr];
            const l = stack[--stackPtr];

            if (l >= r) continue;

            let i = l;
            let j = r;
            const pivotVal = dists[ids[(l + r) >> 1]];

            while (i <= j) {
                while (dists[ids[i]] < pivotVal) i++;
                while (dists[ids[j]] > pivotVal) j--;
                if (i <= j) {
                    const temp = ids[i];
                    ids[i] = ids[j];
                    ids[j] = temp;
                    i++;
                    j--;
                }
            }

            if (i - l > r - i) {
                if (l < j) {
                    stack[stackPtr++] = l;
                    stack[stackPtr++] = j;
                }
                if (i < r) {
                    stack[stackPtr++] = i;
                    stack[stackPtr++] = r;
                }
            } else {
                if (i < r) {
                    stack[stackPtr++] = i;
                    stack[stackPtr++] = r;
                }
                if (l < j) {
                    stack[stackPtr++] = l;
                    stack[stackPtr++] = j;
                }
            }
        }
    }
}

// ===========================================================================
// Spatial index (v1.1.0) -- charts-facing k-nearest-neighbour query surface.
// ===========================================================================
//
// `createSpatialIndex(maxPoints)` returns a BUILD function matching the
// SpatialIndexFactory contract that @zakkster/lite-charts injects for
// hit-testing dense bubble/scatter point clouds:
//
//   type SpatialIndexFactory = (pxs, pys, n) -> SpatialIndex
//   interface SpatialIndex {
//     findNearest(qx, qy, k, maxDistSq, outIndices, outDistSq) -> count
//     dispose() -> void
//   }
//
// Implementation is a UNIFORM GRID, not the Delaunay mesh walk. For pure
// radius-bounded k-NN a grid is simpler and, on the clustered pixel inputs a
// chart actually issues, at least as fast; the mesh walk (ROADMAP v1.2.0+)
// earns its keep only once the mesh is the substrate for Voronoi/interpolation
// too. The exported contract is identical either way, so this stays swappable.
// Crucially: triangulate() is NEVER called here.
//
// Allocation: findNearest allocates nothing (insertion sort into the
// caller-owned out arrays plus a slot-owned f64 scratch). A rebuild allocates
// exactly ONE small handle facade (~48 B, young-gen, minor-GC-collectible);
// all arenas, grids and scratch are pooled -- 0 B beyond the facade. NaN/
// Infinity points are compacted out at build time and can never be returned;
// a non-finite query fails closed to 0.
//
// Why a facade instead of returning the pooled slot directly: with an
// object-reused handle a stale reference is byte-indistinguishable from a
// fresh one, so "stale use throws" and "zero-object rebuild" are mutually
// exclusive. Fail-closed is law, so each build returns a fresh minimal wrapper
// { _slot, _gen } over a frozen shared prototype (methods are NOT re-created
// per build). Both methods compare `this._gen` against `slot.gen`; a stale or
// double-disposed facade throws, and dispose bumps slot.gen so EVERY
// outstanding facade of that build dies at once -- no aliasing corruption.

// Max k the query surface honours. Matches lite-charts'
// SPATIAL_INDEX_HIT_BUFFER_SIZE; the caller's out arrays are sized for this.
const SPATIAL_MAX_K = 8;

/**
 * Query up to `k` nearest points to (qx, qy), bounded by `maxDistSq`.
 *
 * Invoked as `handle.findNearest(...)` on a per-build facade. Writes ORIGINAL
 * point indices into `outIndices` and their SQUARED pixel distances into
 * `outDistSq`, both sorted nearest-first, and returns the count written (0..k).
 * Containment is INCLUSIVE (`d <= maxDistSq`) to match the charts linear-scan
 * `<=` semantics. Zero allocation per query.
 *
 * @this {object} per-build spatial-index facade ({ _slot, _gen })
 * @param {number} qx query x (pixels)
 * @param {number} qy query y (pixels)
 * @param {number} k desired neighbour count (clamped to SPATIAL_MAX_K)
 * @param {number} maxDistSq inclusive squared-distance cutoff
 * @param {Int32Array} outIndices caller-owned, length >= k
 * @param {Float32Array} outDistSq caller-owned, length >= k
 * @returns {number} neighbours written (0..k)
 * @throws {Error} if the handle has been disposed, or the out arrays are
 *   shorter than the (clamped) k
 */
function _spatialFindNearest(qx, qy, k, maxDistSq, outIndices, outDistSq) {
    // Generation stamp: stale (reused-slot) or disposed facade fails closed.
    const s = this._slot;
    if (s === null || this._gen !== s.gen) {
        throw new Error("lite-delaunay: findNearest called on a disposed spatial index");
    }
    // Fail closed on a non-finite query -- NaN comes from log-scale projections
    // and missing data; Infinity from an unclamped cursor. Never guess.
    if (qx !== qx || qy !== qy ||
        qx === Infinity || qx === -Infinity ||
        qy === Infinity || qy === -Infinity) {
        return 0;
    }
    if (k > SPATIAL_MAX_K) k = SPATIAL_MAX_K;
    if (k <= 0) return 0;
    // Fail closed on an under-sized output buffer -- silent under-reporting is
    // an unverified caller state (two compares; no allocation).
    if (outIndices.length < k || outDistSq.length < k) {
        throw new Error("lite-delaunay: findNearest output arrays shorter than k");
    }

    const m = s.m;
    if (m === 0) return 0;

    const cols = s.cols, rows = s.rows;
    const minX = s.minX, minY = s.minY;
    const invCellW = s.invCellW, invCellH = s.invCellH;
    const step = s.step;
    const cxs = s.cxs, cys = s.cys, remap = s.remap;
    const cellStart = s.cellStart, cellItems = s.cellItems;
    // Slot-owned f64 kth-best distances: ALL internal compares (insertion +
    // ring stop bound) use full f64 so an f32-rounding ULP at large coordinates
    // can never skip a marginally-closer next-ring point. The caller still
    // receives f32 in outDistSq.
    const kBest = s.kBest;

    // Locate the query's grid cell (clamp; invCell* is 0 on a degenerate axis).
    let qcx;
    const rx = (qx - minX) * invCellW;
    if (rx <= 0) qcx = 0; else if (rx >= cols) qcx = cols - 1; else qcx = rx | 0;
    let qcy;
    const ry = (qy - minY) * invCellH;
    if (ry <= 0) qcy = 0; else if (ry >= rows) qcy = rows - 1; else qcy = ry | 0;

    const maxRing = cols > rows ? cols : rows;
    let count = 0;

    // Expanding ring scan outward from the query cell.
    for (let r = 0; r <= maxRing; r++) {
        const cyLo = qcy - r, cyHi = qcy + r;
        const cxLo = qcx - r, cxHi = qcx + r;
        for (let gy = cyLo; gy <= cyHi; gy++) {
            if (gy < 0 || gy >= rows) continue;
            const yBorder = (gy === cyLo || gy === cyHi);
            const rowBase = gy * cols;
            for (let gx = cxLo; gx <= cxHi; gx++) {
                if (gx < 0 || gx >= cols) continue;
                // Ring r visits only the square BORDER at Chebyshev distance r;
                // interior cells were covered by a smaller ring already.
                if (!yBorder && gx !== cxLo && gx !== cxHi) continue;
                const cell = rowBase + gx;
                const e = cellStart[cell + 1];
                for (let t = cellStart[cell]; t < e; t++) {
                    const j = cellItems[t];
                    const dx = qx - cxs[j];
                    const dy = qy - cys[j];
                    const d = dx * dx + dy * dy;
                    if (d <= maxDistSq) {
                        // Insertion sort keyed on the f64 kBest scratch; the
                        // caller's f32 outDistSq is written in lockstep.
                        if (count < k) {
                            let p = count;
                            while (p > 0 && kBest[p - 1] > d) {
                                kBest[p] = kBest[p - 1];
                                outDistSq[p] = outDistSq[p - 1];
                                outIndices[p] = outIndices[p - 1];
                                p--;
                            }
                            kBest[p] = d;
                            outDistSq[p] = d;
                            outIndices[p] = remap[j];
                            count++;
                        } else if (d < kBest[count - 1]) {
                            let p = count - 1;
                            while (p > 0 && kBest[p - 1] > d) {
                                kBest[p] = kBest[p - 1];
                                outDistSq[p] = outDistSq[p - 1];
                                outIndices[p] = outIndices[p - 1];
                                p--;
                            }
                            kBest[p] = d;
                            outDistSq[p] = d;
                            outIndices[p] = remap[j];
                        }
                    }
                }
            }
        }
        // The next unscanned band (ring r+1) has a minimum possible distance of
        // r*step from the query point (step = smallest non-degenerate cell
        // side). Stop when nothing farther can enter the answer or beat the
        // current kth-best. step==0 means a fully-degenerate 1-cell axis --
        // maxRing already bounds the loop, so correctness holds without a bound.
        if (step > 0) {
            const bound = r * step;
            const boundSq = bound * bound;
            if (boundSq > maxDistSq) break;
            if (count >= k && boundSq >= kBest[count - 1]) break;
        }
    }
    return count;
}

/**
 * Release a per-build spatial-index facade and return its pooled slot to the
 * factory. Bumps the slot's generation stamp so ANY other outstanding facade of
 * the same build also dies (guarding against aliasing corruption), then detaches
 * this facade (`_slot = null`) so a second dispose on it fails closed too. Using
 * or double-disposing a disposed handle throws.
 *
 * @this {object} per-build spatial-index facade ({ _slot, _gen })
 * @throws {Error} if already disposed / stale
 */
function _spatialDispose() {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) {
        throw new Error("lite-delaunay: dispose called on an already-disposed spatial index");
    }
    this._slot = null;
    // Invalidate every facade that referenced this build, then free the slot.
    s.gen++;
    const pool = s._pool;
    pool.freeStack[pool.freeCount++] = s.poolIndex;
}

/**
 * Frozen shared prototype for the per-build facade. Both methods live here ONCE,
 * so a build never allocates fresh closures -- only the small facade object.
 */
const SPATIAL_FACADE_PROTO = Object.freeze({
    findNearest: _spatialFindNearest,
    dispose: _spatialDispose,
});

/**
 * Allocate one pooled slot: all per-slot arenas + scratch. Only ever called at
 * a new concurrent high-water mark. The slot carries no methods -- callers only
 * ever touch it through a facade.
 *
 * @param {number} maxPoints capacity
 * @param {object} pool the factory's pool bookkeeping
 * @param {number} poolIndex this slot's index in pool.slots
 * @returns {object} the pooled slot
 */
function _createSpatialSlot(maxPoints, pool, poolIndex) {
    // Grid never has more cells than points, so maxPoints+1 cells is a safe cap.
    const maxCells = maxPoints + 1;
    return {
        _pool: pool,
        poolIndex,
        // Generation stamp: bumped on every build and every dispose. A facade
        // is valid iff facade._gen === slot.gen.
        gen: 0,
        // Per-build grid parameters (set by _buildSpatialSlot).
        m: 0, cols: 0, rows: 0,
        minX: 0, minY: 0,
        invCellW: 0, invCellH: 0,
        step: 0, cells: 0,
        // Compacted finite points + compacted->original remap.
        cxs: new Float32Array(maxPoints),
        cys: new Float32Array(maxPoints),
        remap: new Int32Array(maxPoints),
        // Counting-sort grid: cellStart is CSR offsets; cellItems holds
        // compacted point indices grouped by cell; pointCell/cellCursor scratch.
        cellItems: new Int32Array(maxPoints),
        pointCell: new Int32Array(maxPoints),
        cellStart: new Int32Array(maxCells + 1),
        cellCursor: new Int32Array(maxCells + 1),
        // f64 kth-best distance scratch for exact internal compares.
        kBest: new Float64Array(SPATIAL_MAX_K),
    };
}

/**
 * (Re)build the grid for a slot from SoA pixel coordinates. Compacts only
 * finite points; the grid handles all-coincident / all-collinear input
 * naturally (it degrades to a 1-cell linear scan -- never wrong, maybe slow).
 * Zero allocation.
 *
 * @param {object} slot the pooled slot
 * @param {ArrayLike<number>} pxs x coordinates
 * @param {ArrayLike<number>} pys y coordinates
 * @param {number} n logical point count
 */
function _buildSpatialSlot(slot, pxs, pys, n) {
    const cxs = slot.cxs, cys = slot.cys, remap = slot.remap;
    let m = 0;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    // 1. Compact ONLY finite points (both coords finite). A NaN/Infinity point
    //    is dropped here so findNearest can never return it.
    for (let i = 0; i < n; i++) {
        const px = pxs[i], py = pys[i];
        if (px !== px || py !== py ||
            px === Infinity || px === -Infinity ||
            py === Infinity || py === -Infinity) {
            continue;
        }
        cxs[m] = px; cys[m] = py; remap[m] = i;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
        m++;
    }

    slot.m = m;
    if (m === 0) {
        // Nothing to index -- findNearest short-circuits to 0.
        slot.cells = 0; slot.cols = 0; slot.rows = 0;
        return;
    }

    // 2. Grid dimensions: ~1 point per cell. floor(sqrt(m)) keeps cols*rows <= m.
    const gridN = Math.max(1, Math.floor(Math.sqrt(m)));
    const rangeX = maxX - minX, rangeY = maxY - minY;
    const cols = rangeX > 0 ? gridN : 1;
    const rows = rangeY > 0 ? gridN : 1;
    const cellW = rangeX > 0 ? rangeX / cols : 0;
    const cellH = rangeY > 0 ? rangeY / rows : 0;
    const invCellW = cellW > 0 ? 1 / cellW : 0;
    const invCellH = cellH > 0 ? 1 / cellH : 0;
    let step = Infinity;
    if (cellW > 0) step = cellW;
    if (cellH > 0 && cellH < step) step = cellH;
    if (step === Infinity) step = 0;
    const cells = cols * rows;

    slot.minX = minX; slot.minY = minY;
    slot.cols = cols; slot.rows = rows;
    slot.invCellW = invCellW; slot.invCellH = invCellH;
    slot.step = step; slot.cells = cells;

    const cellStart = slot.cellStart, cellCursor = slot.cellCursor;
    const cellItems = slot.cellItems, pointCell = slot.pointCell;

    // 3. Counting sort of compacted points into CSR cell buckets.
    cellStart.fill(0, 0, cells + 1);
    for (let j = 0; j < m; j++) {
        let gx;
        const rxj = (cxs[j] - minX) * invCellW;
        if (rxj <= 0) gx = 0; else if (rxj >= cols) gx = cols - 1; else gx = rxj | 0;
        let gy;
        const ryj = (cys[j] - minY) * invCellH;
        if (ryj <= 0) gy = 0; else if (ryj >= rows) gy = rows - 1; else gy = ryj | 0;
        const c = gy * cols + gx;
        pointCell[j] = c;
        cellStart[c + 1]++;
    }
    for (let c = 1; c <= cells; c++) cellStart[c] += cellStart[c - 1];
    for (let c = 0; c < cells; c++) cellCursor[c] = cellStart[c];
    for (let j = 0; j < m; j++) {
        const c = pointCell[j];
        cellItems[cellCursor[c]++] = j;
    }
}

/**
 * Create a pooled spatial-index FACTORY sized for up to `maxPoints` points.
 *
 * The returned function is the SpatialIndexFactory lite-charts injects:
 * `(pxs, pys, n) -> SpatialIndex`. One factory serves MANY concurrent live
 * handles (a multi-series chart builds one index per series and disposes them
 * independently); each build acquires a free pooled slot and each dispose
 * returns it. A brand new slot is allocated only when concurrency reaches a new
 * high-water mark. A build allocates exactly one small handle facade (~48 B,
 * young-gen, minor-GC-collectible); all arenas, grids and scratch are pooled --
 * 0 B beyond the facade -- and findNearest is 0 B/query.
 *
 * Per-slot memory is approximately 24*maxPoints bytes plus the grid-cell
 * arrays (~= maxPoints cells).
 *
 * @example
 * // In a lite-charts config:
 * const chart = createBubbleChart({
 *   spatialIndex: createSpatialIndex(20_000),
 *   // ...
 * });
 *
 * @param {number} maxPoints hard upper bound on `n` per build. Must be a
 *   non-negative integer.
 * @returns {(pxs: ArrayLike<number>, pys: ArrayLike<number>, n: number) => object}
 *   the SpatialIndexFactory.
 * @throws {Error} if `maxPoints` is not a non-negative integer.
 */
export const createSpatialIndex = (maxPoints) => {
    if (!Number.isInteger(maxPoints) || maxPoints < 0) {
        throw new Error(`lite-delaunay: maxPoints must be a non-negative integer, got ${maxPoints}`);
    }

    // Pool bookkeeping. slots grows to the concurrent high-water mark; freeStack
    // + freeCount is a manual stack of reusable slot indices (never truncated,
    // so no backing-store churn after warm-up).
    const pool = { slots: [], freeStack: [], freeCount: 0 };

    return function buildSpatialIndex(pxs, pys, n) {
        // Fail closed at build time -- all guards live here, off the query path.
        if (!Number.isInteger(n) || n < 0) {
            throw new Error(`lite-delaunay: n must be a non-negative integer, got ${n}`);
        }
        if (n > maxPoints) {
            throw new Error(`lite-delaunay: n (${n}) exceeds spatial index max (${maxPoints})`);
        }
        if (pxs == null || pys == null) {
            throw new Error("lite-delaunay: pxs and pys are required");
        }
        if (pxs.length < n || pys.length < n) {
            throw new Error(`lite-delaunay: pxs/pys shorter than n (${n})`);
        }

        // Acquire a free slot, or grow the pool at a new high-water mark.
        let idx;
        if (pool.freeCount > 0) {
            idx = pool.freeStack[--pool.freeCount];
        } else {
            idx = pool.slots.length;
            pool.slots[idx] = _createSpatialSlot(maxPoints, pool, idx);
        }
        const slot = pool.slots[idx];

        _buildSpatialSlot(slot, pxs, pys, n);
        // New generation for this build; any facade from a prior build of this
        // slot is now stale and will throw.
        slot.gen++;
        // Fresh minimal facade (~48 B). Methods come from the frozen shared
        // proto, so nothing beyond this object is allocated.
        const facade = Object.create(SPATIAL_FACADE_PROTO);
        facade._slot = slot;
        facade._gen = slot.gen;
        return facade;
    };
};

// ===========================================================================
// Cell index (v1.2.0) -- charts-facing bbox-clipped Voronoi cell surface.
// ===========================================================================
//
// `createCellIndex(maxPoints)` returns a BUILD function matching the
// CellIndexFactory contract @zakkster/lite-charts injects to draw a Voronoi
// tessellation over a projected scatter (see LiteCharts/briefs/voronoi-cells.md,
// "The consumer contract"):
//
//   type CellIndexFactory = (pxs, pys, n) -> CellIndex
//   interface CellIndex {
//     cell(i, bx0, by0, bx1, by1, outXY) -> vertexCount   // ORIGINAL index i
//     dispose() -> void
//   }
//
// Unlike createSpatialIndex (a uniform grid), this DOES triangulate: it builds
// the Delaunay mesh once at build time, computes every triangle circumcenter
// once, and cell(i) walks the half-edges around site i to assemble that site's
// Voronoi polygon -- then CLIPS it to the caller's axis-aligned bbox with a
// zero-allocation Sutherland-Hodgman pass. Every returned polygon is finite,
// convex, closed (last vertex implicitly connects to first) and lies fully
// inside-or-on the bbox; a hull cell (unbounded in the true diagram) is closed
// by three synthetic far points that provably fall outside the bbox, so the
// clip result equals cell-intersect-bbox exactly -- no open-cell flag ever
// escapes. cell() writes into caller-owned interleaved outXY [x0,y0,x1,y1,...]
// and allocates nothing.
//
// Pooling / facade / generation semantics are IDENTICAL to createSpatialIndex:
// one factory, many concurrent handles; a build acquires a pooled slot and
// bumps its generation stamp; dispose bumps it again and returns the slot; a
// stale or disposed facade THROWS. A build allocates exactly one small facade
// (~48 B, young-gen). cell() is 0 B/query.

/**
 * Clip a convex polygon against a single axis-aligned half-plane
 * (Sutherland-Hodgman, one plane). Reads `inCount` interleaved vertices from
 * `inP`, writes the clipped interleaved vertices to `outP`, returns the new
 * vertex count. Zero allocation: a module-level function (never a per-build
 * closure), all state in f64 locals, intersection parameter in full f64.
 *
 * @param {Float64Array} inP  interleaved input polygon [x0,y0,x1,y1,...]
 * @param {number} inCount    input vertex count
 * @param {Float64Array} outP interleaved output buffer (caller guarantees room)
 * @param {number} axis       0 = clip on x, 1 = clip on y
 * @param {number} lim        the plane coordinate (bx0/bx1/by0/by1)
 * @param {boolean} keepGreater true keeps coord >= lim, false keeps coord <= lim
 * @returns {number} output vertex count
 */
function _clipHalfPlane(inP, inCount, outP, axis, lim, keepGreater) {
    let outCount = 0;
    let px = inP[2 * (inCount - 1)], py = inP[2 * (inCount - 1) + 1];
    let pCoord = axis === 0 ? px : py;
    let pInside = keepGreater ? (pCoord >= lim) : (pCoord <= lim);
    for (let k = 0; k < inCount; k++) {
        const cx = inP[2 * k], cy = inP[2 * k + 1];
        const cCoord = axis === 0 ? cx : cy;
        const cInside = keepGreater ? (cCoord >= lim) : (cCoord <= lim);
        if (cInside) {
            if (!pInside) {
                // Crossing into the half-plane: emit the boundary intersection.
                const t = (lim - pCoord) / (cCoord - pCoord);
                outP[2 * outCount] = px + t * (cx - px);
                outP[2 * outCount + 1] = py + t * (cy - py);
                outCount++;
            }
            outP[2 * outCount] = cx;
            outP[2 * outCount + 1] = cy;
            outCount++;
        } else if (pInside) {
            // Crossing out of the half-plane: emit the boundary intersection.
            const t = (lim - pCoord) / (cCoord - pCoord);
            outP[2 * outCount] = px + t * (cx - px);
            outP[2 * outCount + 1] = py + t * (cy - py);
            outCount++;
        }
        px = cx; py = cy; pCoord = cCoord; pInside = cInside;
    }
    return outCount;
}

/**
 * Write site `i`'s bbox-clipped Voronoi polygon into `outXY` and return its
 * vertex count. `i` is the ORIGINAL point index (as passed to the factory).
 * Zero allocation.
 *
 * Returns 0 (never a garbage polygon) when: `i` was a non-finite input point;
 * the build was degenerate (fewer than 3 finite points, or triangulate()
 * produced 0 triangles); `i` lost the triangulator's EPSILON dedup (an exact/
 * near-duplicate that owns no mesh vertex); or the cell does not intersect the
 * bbox (including a clip result that degenerates below 3 vertices).
 *
 * @this {object} per-build cell-index facade ({ _slot, _gen })
 * @param {number} i original point index
 * @param {number} bx0 bbox min x
 * @param {number} by0 bbox min y
 * @param {number} bx1 bbox max x (must be > bx0)
 * @param {number} by1 bbox max y (must be > by0)
 * @param {Float64Array|Float32Array} outXY caller-owned interleaved output
 * @returns {number} vertex count written (0, or 3..outXY.length/2)
 * @throws {Error} if the handle is disposed, `i` is not an integer in [0, n),
 *   the bbox is non-finite or not strictly ordered, or outXY is too small for
 *   the clipped cell (the loud escape -- never truncates)
 */
function _cellQuery(i, bx0, by0, bx1, by1, outXY) {
    // Generation stamp: stale (reused-slot) or disposed facade fails closed.
    const s = this._slot;
    if (s === null || this._gen !== s.gen) {
        throw new Error("lite-delaunay: cell called on a disposed cell index");
    }
    // `i` must be a valid original index -- fail LOUD (an out-of-range i is a
    // caller bug, not a missing cell).
    const n = s.n;
    if (!Number.isInteger(i) || i < 0 || i >= n) {
        throw new Error(`lite-delaunay: cell index i (${i}) out of range [0, ${n})`);
    }
    // bbox must be finite and strictly ordered -- a zero/negative-area or
    // non-finite bbox is an unverified caller state, not a degenerate cell.
    if (bx0 !== bx0 || by0 !== by0 || bx1 !== bx1 || by1 !== by1 ||
        bx0 === Infinity || bx0 === -Infinity || by0 === Infinity || by0 === -Infinity ||
        bx1 === Infinity || bx1 === -Infinity || by1 === Infinity || by1 === -Infinity ||
        !(bx0 < bx1) || !(by0 < by1)) {
        throw new Error("lite-delaunay: cell bbox must be finite with bx0 < bx1 and by0 < by1");
    }

    // Degenerate build -> no cells for anyone.
    if (s.degenerate) return 0;

    // Map original index -> compacted vertex; -1 means the point was non-finite
    // (never compacted). inEdge -1 means the vertex lost the triangulator's
    // EPSILON dedup and owns no mesh cell.
    const v = s.orig2v[i];
    if (v === -1) return 0;
    const inEdge = s.inEdge;
    const e0 = inEdge[v];
    if (e0 === -1) return 0;

    const tri = s.tri;
    const triangles = tri.triangles;
    const halfedges = tri.halfedges;
    const ccx = s.ccx, ccy = s.ccy;
    const S = s.scratch;
    const polyA = s.polyA, polyB = s.polyB;
    const maxPoints = s.maxPoints;

    const sx = S[2 * v], sy = S[2 * v + 1];

    // --- Fan walk (d3-delaunay cell walk) ---------------------------------
    // Collect the circumcenter of each triangle incident to v, in order around
    // v. inEdge's hull-priority guarantees an OPEN fan starts at one hull edge
    // and ends at the other, covering every incident triangle exactly once.
    let e = e0;
    let count = 0;
    let open = false;
    let lastNe = -1; // outgoing hull half-edge, set only on an open fan
    do {
        // Safety: a valid fan has at most (m - 1) circumcenters; anything more
        // is a malformed-mesh cycle -- fail closed rather than overrun polyA.
        if (count >= maxPoints) return 0;
        const t = (e / 3) | 0;
        polyA[2 * count] = ccx[t];
        polyA[2 * count + 1] = ccy[t];
        count++;
        // nextHalfedge(e): e in {3t,3t+1,3t+2} -> the next edge of triangle t.
        const ne = (e % 3 === 2) ? e - 2 : e + 1;
        // Defensive: on a well-formed mesh every walked edge is incoming to v,
        // so triangles[nextHalfedge(e)] === v always. If that invariant is ever
        // broken, fail closed -- never build a far fan from a mislabeled edge.
        if (triangles[ne] !== v) return 0;
        e = halfedges[ne];
        if (e === -1) { lastNe = ne; open = true; break; }             // hull end
    } while (e !== e0);

    // --- Hull far-fan -----------------------------------------------------
    // For an open cell, close the fan with three synthetic far points so the
    // subsequent bbox clip yields cell-intersect-bbox exactly. The two boundary
    // Voronoi rays are dual to the two hull edges at the site: each emanates
    // from ITS circumcenter (c0 for the start edge, c_last for the end edge) in
    // the outward perpendicular-bisector direction of that hull edge. The far
    // ray tips MUST be anchored at those circumcenters (not the site), otherwise
    // the chord back to the circumcenter is tilted off the true ray and the
    // clipped cells no longer tile the bbox.
    if (open) {
        // Start hull edge (p -> v): the incoming hull edge e0 (halfedges[e0]==-1).
        // Its triangle's third vertex = sum of the triangle's three vertex ids
        // minus p and v (the ids are distinct, so the sum trick is exact).
        const t0 = (e0 / 3) | 0;
        const p = triangles[e0];
        const third0 = triangles[3 * t0] + triangles[3 * t0 + 1] + triangles[3 * t0 + 2] - p - v;
        const px = S[2 * p], py = S[2 * p + 1];
        // Edge direction p->v and the two candidate normals (dy,-dx)/(-dy,dx).
        // Pick the one pointing AWAY from the third vertex: the mesh sits on the
        // third-vertex side, so the outward Voronoi ray points the other way.
        // (Sign is derived per-triangle so it holds under either hull winding
        // the triangulator emits -- see the CCW note at the top of this file.)
        let ex0 = sx - px, ey0 = sy - py;
        const mx0 = (px + sx) * 0.5, my0 = (py + sy) * 0.5;
        const tx0 = S[2 * third0], ty0 = S[2 * third0 + 1];
        let ux = ey0, uy = -ex0;
        if (ux * (tx0 - mx0) + uy * (ty0 - my0) > 0) { ux = -ey0; uy = ex0; }
        const ul = Math.sqrt(ux * ux + uy * uy);
        if (!(ul > 0)) return 0; // degenerate hull edge -- fail closed
        ux /= ul; uy /= ul;

        // End hull edge (v -> q): the outgoing hull edge lastNe.
        const tL = (lastNe / 3) | 0;
        const qne = (lastNe % 3 === 2) ? lastNe - 2 : lastNe + 1;
        const q = triangles[qne];
        const third1 = triangles[3 * tL] + triangles[3 * tL + 1] + triangles[3 * tL + 2] - v - q;
        const qx = S[2 * q], qy = S[2 * q + 1];
        let ex1 = qx - sx, ey1 = qy - sy;
        const mx1 = (sx + qx) * 0.5, my1 = (sy + qy) * 0.5;
        const tx1 = S[2 * third1], ty1 = S[2 * third1 + 1];
        let wx = ey1, wy = -ex1;
        if (wx * (tx1 - mx1) + wy * (ty1 - my1) > 0) { wx = -ey1; wy = ex1; }
        const wl = Math.sqrt(wx * wx + wy * wy);
        if (!(wl > 0)) return 0;
        wx /= wl; wy /= wl;

        // Bisector b = normalize(u + w). u+w vanishes only at a ~180deg aperture
        // (float pathology: a real hull-vertex aperture is < 180deg); fail
        // closed to the perpendicular of u turned toward w's side.
        let bx = ux + wx, by = uy + wy;
        const blb = Math.sqrt(bx * bx + by * by);
        if (blb > 1e-9) {
            bx /= blb; by /= blb;
        } else {
            bx = -uy; by = ux;
            if (bx * wx + by * wy < 0) { bx = uy; by = -ux; }
        }

        // The ray tips anchor at their circumcenters; the bisector tip anchors
        // at the site. FAR must dominate BOTH the bbox extent C (farthest bbox
        // corner from the site) AND D (site-to-circumcenter distance), so all
        // three far points and both bridge chords (far_w -> far_b -> far_u,
        // split by the bisector so each subtends < 90deg) clear the bbox: with
        // FAR = 3*(C + D) the bridge clearance is >= FAR*cos45 - D > C. The two
        // ray edges (far_u -> c0, c_last -> far_w) lie exactly on the true rays,
        // so the clip yields cell-intersect-bbox exactly.
        const c0x = ccx[t0], c0y = ccy[t0];
        const cLx = ccx[tL], cLy = ccy[tL];
        const dxm = Math.max(Math.abs(sx - bx0), Math.abs(sx - bx1));
        const dym = Math.max(Math.abs(sy - by0), Math.abs(sy - by1));
        const C = Math.sqrt(dxm * dxm + dym * dym);
        const d0 = Math.sqrt((c0x - sx) * (c0x - sx) + (c0y - sy) * (c0y - sy));
        const dL = Math.sqrt((cLx - sx) * (cLx - sx) + (cLy - sy) * (cLy - sy));
        const D = d0 > dL ? d0 : dL;
        const FAR = 3 * (C + D);

        // Append after the last circumcenter, in fan order: far(w), far(b), far(u).
        polyA[2 * count] = cLx + FAR * wx; polyA[2 * count + 1] = cLy + FAR * wy; count++;
        polyA[2 * count] = sx + FAR * bx; polyA[2 * count + 1] = sy + FAR * by; count++;
        polyA[2 * count] = c0x + FAR * ux; polyA[2 * count + 1] = c0y + FAR * uy; count++;
    }

    if (count < 3) return 0;

    // --- Clip against the four bbox half-planes (ping-pong polyA<->polyB) ---
    count = _clipHalfPlane(polyA, count, polyB, 0, bx0, true);  // x >= bx0
    if (count < 3) return 0;
    count = _clipHalfPlane(polyB, count, polyA, 0, bx1, false); // x <= bx1
    if (count < 3) return 0;
    count = _clipHalfPlane(polyA, count, polyB, 1, by0, true);  // y >= by0
    if (count < 3) return 0;
    count = _clipHalfPlane(polyB, count, polyA, 1, by1, false); // y <= by1
    if (count < 3) return 0;

    // Fail LOUD if the caller's buffer cannot hold the clipped cell -- never
    // truncate a polygon into a garbage shape.
    if (2 * count > outXY.length) {
        throw new Error(`lite-delaunay: outXY too small for ${count}-vertex cell`);
    }
    for (let k = 0; k < count; k++) {
        outXY[2 * k] = polyA[2 * k];
        outXY[2 * k + 1] = polyA[2 * k + 1];
    }
    return count;
}

/**
 * Release a per-build cell-index facade and return its pooled slot to the
 * factory. Identical semantics to `_spatialDispose`: bump the slot generation
 * (invalidating every outstanding facade of this build), detach this facade so
 * a second dispose fails closed, and push the slot back on the free stack.
 *
 * @this {object} per-build cell-index facade ({ _slot, _gen })
 * @throws {Error} if already disposed / stale
 */
function _cellDispose() {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) {
        throw new Error("lite-delaunay: dispose called on an already-disposed cell index");
    }
    this._slot = null;
    s.gen++;
    const pool = s._pool;
    pool.freeStack[pool.freeCount++] = s.poolIndex;
}

/**
 * Frozen shared prototype for the per-build cell-index facade. Both methods
 * live here ONCE, so a build never allocates fresh closures.
 */
const CELL_FACADE_PROTO = Object.freeze({
    cell: _cellQuery,
    dispose: _cellDispose,
});

/**
 * Allocate one pooled cell-index slot: the triangulator arena, the interleave
 * scratch, per-triangle circumcenter arrays, the inEdge / orig2v maps, and the
 * Sutherland-Hodgman ping-pong buffers. Only ever called at a new concurrent
 * high-water mark. The slot carries no methods -- callers only touch it through
 * a facade.
 *
 * @param {number} maxPoints capacity
 * @param {object} pool the factory's pool bookkeeping
 * @param {number} poolIndex this slot's index in pool.slots
 * @returns {object} the pooled slot
 */
function _createCellSlot(maxPoints, pool, poolIndex) {
    const maxTriangles = Math.max(2 * maxPoints - 5, 0);
    return {
        _pool: pool,
        poolIndex,
        // Generation stamp: bumped on every build and every dispose.
        gen: 0,
        maxPoints,
        // Per-build state (set by _buildCellSlot).
        n: 0, m: 0, triCount: 0, degenerate: true,
        // The mesh engine and its f64 interleave scratch [x0,y0,x1,y1,...].
        tri: new DelaunayTriangulator(maxPoints),
        scratch: new Float64Array(2 * maxPoints),
        // Compacted finite coords (kept for parity with the spatial slot).
        cxs: new Float32Array(maxPoints),
        cys: new Float32Array(maxPoints),
        // Per-triangle circumcenters, computed once per build.
        ccx: new Float64Array(maxTriangles),
        ccy: new Float64Array(maxTriangles),
        // inEdge[v] = an incoming half-edge for compacted vertex v (-1 = none,
        // i.e. the vertex lost the EPSILON dedup); orig2v[i] = compacted vertex
        // for original index i (-1 = non-finite / no cell).
        inEdge: new Int32Array(maxPoints),
        orig2v: new Int32Array(maxPoints),
        // Sutherland-Hodgman ping-pong scratch. A hull fan is at most (m-1)
        // circumcenters + 3 far points, and each of the 4 clip planes adds at
        // most 1 vertex, so maxPoints + 8 vertices is a safe cap.
        polyA: new Float64Array(2 * (maxPoints + 8)),
        polyB: new Float64Array(2 * (maxPoints + 8)),
    };
}

/**
 * (Re)build the mesh + circumcenters for a slot from SoA pixel coordinates.
 * Compacts only finite points, triangulates, computes every circumcenter once,
 * and builds the inEdge map. Zero allocation.
 *
 * @param {object} slot the pooled slot
 * @param {ArrayLike<number>} pxs x coordinates
 * @param {ArrayLike<number>} pys y coordinates
 * @param {number} n logical point count
 */
function _buildCellSlot(slot, pxs, pys, n) {
    const cxs = slot.cxs, cys = slot.cys, orig2v = slot.orig2v, S = slot.scratch;
    slot.n = n;
    slot.degenerate = true;
    slot.triCount = 0;

    // 1. Compact ONLY finite points; record original index -> compacted vertex
    //    and interleave the compacted coords into the f64 scratch in one pass.
    //    (x !== x style, no Number.isFinite in the loop.)
    orig2v.fill(-1, 0, n);
    let m = 0;
    for (let i = 0; i < n; i++) {
        const px = pxs[i], py = pys[i];
        if (px !== px || py !== py ||
            px === Infinity || px === -Infinity ||
            py === Infinity || py === -Infinity) {
            continue;
        }
        cxs[m] = px; cys[m] = py; orig2v[i] = m;
        S[2 * m] = px; S[2 * m + 1] = py;
        m++;
    }
    slot.m = m;

    // Fewer than 3 finite points: no triangulation exists -- every cell() is 0.
    if (m < 3) return;

    const triCount = slot.tri.triangulate(S, m);
    if (triCount === 0) return; // collinear / coincident -- degenerate stays true
    slot.triCount = triCount;
    slot.degenerate = false;

    const tri = slot.tri;
    const triangles = tri.triangles;
    const halfedges = tri.halfedges;
    const ccx = slot.ccx, ccy = slot.ccy;

    // 2. Compute ALL circumcenters once.
    for (let t = 0; t < triCount; t++) {
        const ia = triangles[3 * t], ib = triangles[3 * t + 1], ic = triangles[3 * t + 2];
        const ax = S[2 * ia], ay = S[2 * ia + 1];
        const bx = S[2 * ib], by = S[2 * ib + 1];
        const cx = S[2 * ic], cy = S[2 * ic + 1];
        const dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
        const bl = dx * dx + dy * dy, cl = ex * ex + ey * ey;
        const D = dx * ey - dy * ex;
        const scale = bl + cl;
        // Relative degeneracy guard: |D| is twice the triangle area; when it
        // drops to EPSILON*scale the 0.5/D division would overflow toward
        // Infinity, so fail closed to the centroid (finite, inside the
        // triangle). This fires only on the overflow boundary -- ordinary thin
        // triangles keep their finite, far true circumcenters (the clip copes).
        // The `!(|D| > ...)` form also catches a NaN denominator.
        if (scale === 0 || !(Math.abs(D) > EPSILON * scale)) {
            ccx[t] = (ax + bx + cx) / 3;
            ccy[t] = (ay + by + cy) / 3;
        } else {
            const d = 0.5 / D;
            ccx[t] = ax + (ey * bl - dy * cl) * d;
            ccy[t] = ay + (dx * cl - ex * bl) * d;
        }
    }

    // 3. Build inEdge with the d3-delaunay "inedges" pattern so a hull fan
    //    starts at the hull: a hull-incoming edge (halfedges[e] == -1) always
    //    wins, otherwise the first edge seen wins. Vertices skipped by the
    //    triangulator's EPSILON dedup keep inEdge -1.
    const inEdge = slot.inEdge;
    inEdge.fill(-1, 0, m);
    const trianglesLen = triCount * 3;
    for (let e = 0; e < trianglesLen; e++) {
        const ne = (e % 3 === 2) ? e - 2 : e + 1; // nextHalfedge(e)
        const p = triangles[ne];
        if (halfedges[e] === -1 || inEdge[p] === -1) inEdge[p] = e;
    }
}

/**
 * Create a pooled cell-index FACTORY sized for up to `maxPoints` points.
 *
 * The returned function is the CellIndexFactory lite-charts injects:
 * `(pxs, pys, n) -> CellIndex`. It mirrors {@link createSpatialIndex} exactly --
 * pooled factory-factory, SoA input, NaN compaction, ORIGINAL indices,
 * generation-stamped facades -- but instead of a k-NN grid it builds the
 * Delaunay mesh, precomputes every circumcenter, and answers `cell(i, ...)` by
 * walking the half-edges around site i and CLIPPING the resulting Voronoi
 * polygon to the caller's axis-aligned bbox.
 *
 * Memory model: a build allocates exactly one small facade (~48 B, young-gen,
 * minor-GC-collectible); the mesh arena, scratch, circumcenter arrays and clip
 * buffers are all pooled -- 0 B beyond the facade -- and `cell()` is 0 B/query.
 * One factory serves many concurrent live handles; a new slot is allocated only
 * at a new concurrent high-water mark.
 *
 * SIZING RULE (documented so callers can size outXY): a bbox-clipped Voronoi
 * cell of an INTERIOR site has at most (degree + 4) vertices, and of a HULL site
 * at most (degree + 5). A caller-owned buffer of 2 * 64 floats (64 vertices)
 * covers every non-adversarial cloud; if a clipped cell needs more, `cell()`
 * THROWS rather than truncating -- the loud escape.
 *
 * @example
 * // In a lite-charts scatter config (cells is all-or-nothing opt-in):
 * const chart = createScatterChart({
 *   data,
 *   cells: { index: createCellIndex(20_000), colorKey: 'zone' },
 * });
 * // The chart calls factory(pxs, pys, n) at extract time, disposes the old
 * // index first, and per frame walks each visible cell(i, ...) at 0 B.
 *
 * @param {number} maxPoints hard upper bound on `n` per build. Must be a
 *   non-negative integer.
 * @returns {(pxs: ArrayLike<number>, pys: ArrayLike<number>, n: number) => object}
 *   the CellIndexFactory.
 * @throws {Error} if `maxPoints` is not a non-negative integer.
 */
export const createCellIndex = (maxPoints) => {
    if (!Number.isInteger(maxPoints) || maxPoints < 0) {
        throw new Error(`lite-delaunay: maxPoints must be a non-negative integer, got ${maxPoints}`);
    }

    // Pool bookkeeping, identical in shape to createSpatialIndex.
    const pool = { slots: [], freeStack: [], freeCount: 0 };

    return function buildCellIndex(pxs, pys, n) {
        // Fail closed at build time -- all guards live here, off the query path.
        if (!Number.isInteger(n) || n < 0) {
            throw new Error(`lite-delaunay: n must be a non-negative integer, got ${n}`);
        }
        if (n > maxPoints) {
            throw new Error(`lite-delaunay: n (${n}) exceeds cell index max (${maxPoints})`);
        }
        if (pxs == null || pys == null) {
            throw new Error("lite-delaunay: pxs and pys are required");
        }
        if (pxs.length < n || pys.length < n) {
            throw new Error(`lite-delaunay: pxs/pys shorter than n (${n})`);
        }

        // Acquire a free slot, or grow the pool at a new high-water mark.
        let idx;
        if (pool.freeCount > 0) {
            idx = pool.freeStack[--pool.freeCount];
        } else {
            idx = pool.slots.length;
            pool.slots[idx] = _createCellSlot(maxPoints, pool, idx);
        }
        const slot = pool.slots[idx];

        _buildCellSlot(slot, pxs, pys, n);
        // New generation for this build; any facade from a prior build of this
        // slot is now stale and will throw.
        slot.gen++;
        const facade = Object.create(CELL_FACADE_PROTO);
        facade._slot = slot;
        facade._gen = slot.gen;
        return facade;
    };
};

// ===========================================================================
// Field index (v1.3.0) -- scattered-data to regular-grid interpolation.
// ===========================================================================
//
// `createFieldIndex(maxPoints)` returns a BUILD function
// `(pxs, pys, n) -> fieldIndex` that triangulates the Delaunay mesh once and
// then answers point-location and barycentric-interpolation queries against it.
// One mesh serves MANY scalar fields: `pxs`/`pys` fix the geometry at build
// time, and every query takes the `zs` values per-call (indexed by ORIGINAL
// point index), so a caller can interpolate temperature, then pressure, then
// elevation over the same cloud without rebuilding.
//
// The core is a remembering visibility (straight) walk over the half-edge mesh:
// `locate(qx, qy)` starts from the last hit triangle (`slot.cursor`), so a
// coherent stream of nearby queries -- a raster scan, a dragged cursor -- costs
// O(1) amortised steps; a cold jump costs O(sqrt(T)). The walk provably
// terminates on a Delaunay mesh; a step budget (`stepCap`) covers f64-degenerate
// cycling by falling back to an O(T) linear scan -- never wrong, maybe slow.
//
// Pooling / facade / generation semantics are IDENTICAL to createCellIndex and
// createSpatialIndex: one factory, many concurrent handles; a build acquires a
// pooled slot and bumps its generation stamp; dispose bumps it again and returns
// the slot; a stale or disposed facade THROWS. A build allocates exactly one
// small facade (~48 B, young-gen). Every query is 0 B. There are no
// circumcenters -- this is a lighter slot than the cell index.

/**
 * Cold error-message factory for the field index. All throw strings are built
 * (and thrown) HERE, off every hot method body -- the guard CONDITIONS live in
 * the methods, but not the byte cost of their messages.
 *
 * @param {string} code message selector
 * @param {*} [a] first interpolated value
 * @param {*} [b] second interpolated value
 * @returns {never} always throws
 */
function _fieldThrow(code, a, b) {
    switch (code) {
        case 'maxp':
            throw new Error(`lite-delaunay: maxPoints must be a non-negative integer, got ${a}`);
        case 'n':
            throw new Error(`lite-delaunay: n must be a non-negative integer, got ${a}`);
        case 'nmax':
            throw new Error(`lite-delaunay: n (${a}) exceeds field index max (${b})`);
        case 'arr':
            throw new Error("lite-delaunay: pxs and pys are required");
        case 'arrlen':
            throw new Error(`lite-delaunay: pxs/pys shorter than n (${a})`);
        case 'disposed':
            throw new Error(`lite-delaunay: ${a} called on a disposed field index`);
        case 'ddispose':
            throw new Error("lite-delaunay: dispose called on an already-disposed field index");
        case 'trange':
            throw new Error(`lite-delaunay: triangle t (${a}) out of range [0, ${b})`);
        case 'outshort':
            throw new Error(`lite-delaunay: ${a} too short (need length >= ${b})`);
        case 'zs':
            throw new Error(`lite-delaunay: zs required with length >= n (${a})`);
        case 'grid':
            throw new Error("lite-delaunay: gridW and gridH must be positive integers");
        case 'outgrid':
            throw new Error("lite-delaunay: outGrid must be a Float32Array/Float64Array of length >= gridW*gridH");
        case 'bbox':
            throw new Error("lite-delaunay: bbox must be finite with bx0 < bx1 and by0 < by1");
    }
}

/**
 * Cold O(T) fallback locator: scan every triangle for the one that contains
 * (qx, qy) using the same three-directed-edge cross test as the walk. Returns
 * the first containing triangle, or -1 if the point is outside the hull. No
 * allocation, no strings; only reached when the walk exhausts its step budget.
 *
 * @param {object} slot the pooled slot
 * @param {number} qx query x
 * @param {number} qy query y
 * @returns {number} containing compacted-mesh triangle id, or -1
 */
function _fieldLinearScan(slot, qx, qy) {
    const triangles = slot.tri.triangles;
    const S = slot.scratch;
    const orientSign = slot.orientSign;
    const T = slot.triCount;
    for (let t = 0; t < T; t++) {
        const e0 = 3 * t;
        let inside = true;
        for (let j = 0; j < 3; j++) {
            const e = e0 + j;
            const va = triangles[e];
            const eb = (j === 2) ? e0 : e + 1; // nextHalfedge(e)
            const vb = triangles[eb];
            const x1 = S[2 * va], y1 = S[2 * va + 1];
            const x2 = S[2 * vb], y2 = S[2 * vb + 1];
            const cross = orientSign * ((x2 - x1) * (qy - y1) - (y2 - y1) * (qx - x1));
            if (cross < 0) { inside = false; break; }
        }
        if (inside) return t;
    }
    return -1;
}

/**
 * Remembering visibility (straight) walk from a given start triangle. Steps
 * across the first directed edge whose outward cross is negative; a hull edge
 * (halfedge -1) on that side means the query is outside the convex hull, so
 * return -1. If no edge is crossed the triangle contains q (on-edge counts as
 * inside), so return it. On step-budget exhaustion fall back to the linear
 * scan. Module-level, zero allocation, never throws, never builds a string.
 *
 * @param {object} slot the pooled slot
 * @param {number} qx query x
 * @param {number} qy query y
 * @param {number} t start triangle id (clamped into [0, T))
 * @returns {number} containing compacted-mesh triangle id, or -1
 */
function _fieldWalkFrom(slot, qx, qy, t) {
    const triangles = slot.tri.triangles;
    const halfedges = slot.tri.halfedges;
    const S = slot.scratch;
    const orientSign = slot.orientSign;
    const T = slot.triCount;
    if (t < 0) t = 0; else if (t >= T) t = T - 1;
    let budget = slot.stepCap;
    while (budget > 0) {
        budget--;
        const e0 = 3 * t;
        let moved = 0;
        for (let j = 0; j < 3; j++) {
            const e = e0 + j;
            const va = triangles[e];
            const eb = (j === 2) ? e0 : e + 1; // nextHalfedge(e)
            const vb = triangles[eb];
            const x1 = S[2 * va], y1 = S[2 * va + 1];
            const x2 = S[2 * vb], y2 = S[2 * vb + 1];
            const cross = orientSign * ((x2 - x1) * (qy - y1) - (y2 - y1) * (qx - x1));
            if (cross < 0) {
                const tw = halfedges[e];
                if (tw === -1) return -1; // strictly outside a hull-edge half-plane
                t = (tw / 3) | 0;
                moved = 1;
                break;
            }
        }
        if (moved === 0) return t; // contained
    }
    return _fieldLinearScan(slot, qx, qy);
}

/**
 * Compute the barycentric weights (w0, w1, w2) of (qx, qy) in triangle `t` and
 * write them into `out`. A near-zero-area triangle (relative guard mirroring the
 * cell circumcenter guard) writes NaN,NaN,NaN and returns false. Otherwise
 * returns true iff q is inside-or-on `t` (every weight >= -1e-9). Zero
 * allocation. w0/w1/w2 correspond to the triangle's three mesh vertices in order.
 *
 * @param {object} slot the pooled slot
 * @param {number} t compacted-mesh triangle id (caller guarantees in range)
 * @param {number} qx query x
 * @param {number} qy query y
 * @param {Float32Array|Float64Array|number[]} out length >= 3
 * @returns {boolean} true iff inside-or-on
 */
function _fieldBary(slot, t, qx, qy, out) {
    const triangles = slot.tri.triangles;
    const S = slot.scratch;
    const e0 = 3 * t;
    const ia = triangles[e0], ib = triangles[e0 + 1], ic = triangles[e0 + 2];
    const ax = S[2 * ia], ay = S[2 * ia + 1];
    const bx = S[2 * ib], by = S[2 * ib + 1];
    const cx = S[2 * ic], cy = S[2 * ic + 1];
    const v0x = bx - ax, v0y = by - ay;
    const v1x = cx - ax, v1y = cy - ay;
    const d = v0x * v1y - v0y * v1x; // signed doubled area
    const scale = (v0x * v0x + v0y * v0y) + (v1x * v1x + v1y * v1y);
    // Relative degeneracy guard: mirror the cell circumcenter guard's
    // `!(|d| > EPSILON*scale)` form, which also traps a NaN denominator.
    if (scale === 0 || !(Math.abs(d) > EPSILON * scale)) {
        out[0] = NaN; out[1] = NaN; out[2] = NaN;
        return false;
    }
    const qx0 = qx - ax, qy0 = qy - ay;
    const w1 = (qx0 * v1y - qy0 * v1x) / d;
    const w2 = (v0x * qy0 - v0y * qx0) / d;
    const w0 = 1 - w1 - w2;
    out[0] = w0; out[1] = w1; out[2] = w2;
    return (w0 >= -1e-9 && w1 >= -1e-9 && w2 >= -1e-9);
}

/**
 * Locate the compacted-mesh triangle containing (qx, qy). Returns a triangle id
 * in [0, triangleCount), or -1 for a non-finite query, a degenerate build, or a
 * point outside the convex hull. Updates the walk cursor on a hit for coherence.
 * z-agnostic.
 *
 * @this {object} per-build field-index facade ({ _slot, _gen })
 * @param {number} qx query x
 * @param {number} qy query y
 * @returns {number} triangle id, or -1
 * @throws {Error} if the handle is disposed or stale
 */
function _fieldLocate(qx, qy) {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _fieldThrow('disposed', 'locate');
    if (!Number.isFinite(qx) || !Number.isFinite(qy)) return -1;
    if (s.degenerate) return -1;
    const t = _fieldWalkFrom(s, qx, qy, s.cursor);
    if (t >= 0) s.cursor = t;
    return t;
}

/**
 * Write the barycentric weights of (qx, qy) in triangle `t` into `outW3`
 * (ALWAYS all three), returning true iff q is inside-or-on `t`. A near-zero-area
 * triangle writes NaN,NaN,NaN and returns false.
 *
 * @this {object} per-build field-index facade ({ _slot, _gen })
 * @param {number} t compacted-mesh triangle id in [0, triangleCount)
 * @param {number} qx query x
 * @param {number} qy query y
 * @param {Float32Array|Float64Array|number[]} outW3 length >= 3
 * @returns {boolean} true iff inside-or-on
 * @throws {Error} if disposed/stale, `t` out of range, or `outW3` too short
 */
function _fieldBarycentric(t, qx, qy, outW3) {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _fieldThrow('disposed', 'barycentric');
    const T = s.triCount;
    if (!Number.isInteger(t) || t < 0 || t >= T) _fieldThrow('trange', t, T);
    if (outW3 == null || outW3.length < 3) _fieldThrow('outshort', 'outW3', 3);
    return _fieldBary(s, t, qx, qy, outW3);
}

/**
 * Write triangle `t`'s three ORIGINAL point indices (via vert2orig) into
 * `outI3`. Reserves per-triangle site access for future contour / TIN work.
 *
 * @this {object} per-build field-index facade ({ _slot, _gen })
 * @param {number} t compacted-mesh triangle id in [0, triangleCount)
 * @param {Int32Array|number[]} outI3 length >= 3
 * @returns {void}
 * @throws {Error} if disposed/stale, `t` out of range, or `outI3` too short
 */
function _fieldTriangleVertices(t, outI3) {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _fieldThrow('disposed', 'triangleVertices');
    const T = s.triCount;
    if (!Number.isInteger(t) || t < 0 || t >= T) _fieldThrow('trange', t, T);
    if (outI3 == null || outI3.length < 3) _fieldThrow('outshort', 'outI3', 3);
    const triangles = s.tri.triangles;
    const v2o = s.vert2orig;
    const e0 = 3 * t;
    outI3[0] = v2o[triangles[e0]];
    outI3[1] = v2o[triangles[e0 + 1]];
    outI3[2] = v2o[triangles[e0 + 2]];
}

/**
 * Number of triangles in the built mesh. 0 on a degenerate build.
 *
 * @this {object} per-build field-index facade ({ _slot, _gen })
 * @returns {number}
 * @throws {Error} if the handle is disposed or stale
 */
function _fieldTriangleCount() {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _fieldThrow('disposed', 'triangleCount');
    return s.degenerate ? 0 : s.triCount;
}

/**
 * Interpolate the scalar field `zs` at (qx, qy): locate the containing triangle,
 * compute barycentric weights, return the weighted sum of `zs` at the triangle's
 * three ORIGINAL vertex indices. NaN when the query is non-finite, the build is
 * degenerate, the point is outside the hull, or the area guard fails. A NaN-z
 * corner propagates arithmetically (no branch). `zs` is indexed by ORIGINAL
 * point index. Zero allocation.
 *
 * @this {object} per-build field-index facade ({ _slot, _gen })
 * @param {Float32Array|Float64Array|number[]} zs scalar values, length >= n
 * @param {number} qx query x
 * @param {number} qy query y
 * @returns {number} interpolated value, or NaN
 * @throws {Error} if disposed/stale or `zs` missing/short
 */
function _fieldInterpolate(zs, qx, qy) {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _fieldThrow('disposed', 'interpolate');
    const n = s.n;
    if (zs == null || zs.length < n) _fieldThrow('zs', n);
    if (!Number.isFinite(qx) || !Number.isFinite(qy)) return NaN;
    if (s.degenerate) return NaN;
    const t = _fieldWalkFrom(s, qx, qy, s.cursor);
    if (t < 0) return NaN;
    s.cursor = t;
    const triangles = s.tri.triangles;
    const S = s.scratch;
    const v2o = s.vert2orig;
    const e0 = 3 * t;
    const ia = triangles[e0], ib = triangles[e0 + 1], ic = triangles[e0 + 2];
    const ax = S[2 * ia], ay = S[2 * ia + 1];
    const bx = S[2 * ib], by = S[2 * ib + 1];
    const cx = S[2 * ic], cy = S[2 * ic + 1];
    const v0x = bx - ax, v0y = by - ay;
    const v1x = cx - ax, v1y = cy - ay;
    const d = v0x * v1y - v0y * v1x;
    const scale = (v0x * v0x + v0y * v0y) + (v1x * v1x + v1y * v1y);
    if (scale === 0 || !(Math.abs(d) > EPSILON * scale)) return NaN;
    const qx0 = qx - ax, qy0 = qy - ay;
    const w1 = (qx0 * v1y - qy0 * v1x) / d;
    const w2 = (v0x * qy0 - v0y * qx0) / d;
    const w0 = 1 - w1 - w2;
    return w0 * zs[v2o[ia]] + w1 * zs[v2o[ib]] + w2 * zs[v2o[ic]];
}

/**
 * Rasterize the scalar field `zs` onto a regular grid, writing into `outGrid`
 * and returning the count of FINITE cells written.
 *
 * GRID CONTRACT (the consumer contract -- document exactly this): row-major,
 * index = row*gridW + col; col 0 at bx0 (xMin), row 0 at by0 (yMin) --
 * MATHEMATICAL +y-up orientation, deliberately NOT screen convention (a
 * pixel-space consumer flips rows itself, on its own cold path). Cell-CENTER
 * sampling: x_j = bx0 + (j + 0.5) * (bx1 - bx0) / gridW,
 * y_i = by0 + (i + 0.5) * (by1 - by0) / gridH. Cells outside the hull get NaN.
 * A degenerate build fills the whole gridW*gridH prefix with NaN and returns 0.
 *
 * The mesh values are gathered once into the pooled `zv` (zv[v] = zs[orig(v)])
 * so the inner loop reads `zv[vertex]` directly, and the scan is serpentine
 * (boustrophedon) so the walk cursor stays coherent at row ends -- but WRITES
 * stay row-major. A NaN-z corner propagates arithmetically, confined to its
 * incident triangles.
 *
 * @this {object} per-build field-index facade ({ _slot, _gen })
 * @param {Float32Array|Float64Array|number[]} zs scalar values, length >= n
 * @param {number} gridW grid columns (positive integer)
 * @param {number} gridH grid rows (positive integer)
 * @param {number} bx0 bbox min x
 * @param {number} by0 bbox min y
 * @param {number} bx1 bbox max x (must be > bx0)
 * @param {number} by1 bbox max y (must be > by0)
 * @param {Float32Array|Float64Array} outGrid length >= gridW*gridH
 * @returns {number} count of finite cells written
 * @throws {Error} if disposed/stale; `zs` missing/short; gridW/gridH not
 *   positive integers; outGrid wrong type or too short; or the bbox is
 *   non-finite or not strictly ordered
 */
function _fieldSampleField(zs, gridW, gridH, bx0, by0, bx1, by1, outGrid) {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _fieldThrow('disposed', 'sampleField');
    const n = s.n;
    if (zs == null || zs.length < n) _fieldThrow('zs', n);
    if (!Number.isInteger(gridW) || gridW <= 0 || !Number.isInteger(gridH) || gridH <= 0) {
        _fieldThrow('grid');
    }
    if (outGrid == null ||
        !(outGrid instanceof Float32Array || outGrid instanceof Float64Array) ||
        outGrid.length < gridW * gridH) {
        _fieldThrow('outgrid');
    }
    if (bx0 !== bx0 || by0 !== by0 || bx1 !== bx1 || by1 !== by1 ||
        bx0 === Infinity || bx0 === -Infinity || by0 === Infinity || by0 === -Infinity ||
        bx1 === Infinity || bx1 === -Infinity || by1 === Infinity || by1 === -Infinity ||
        !(bx0 < bx1) || !(by0 < by1)) {
        _fieldThrow('bbox');
    }

    const total = gridW * gridH;
    // Degenerate build: NaN the whole prefix, return 0 (tested ONCE, not per cell).
    if (s.degenerate) {
        for (let k = 0; k < total; k++) outGrid[k] = NaN;
        return 0;
    }

    // GATHER once: zv[v] = zs[orig(v)] so the grid loop reads zv[vertex] with no
    // double indirection.
    const m = s.m;
    const v2o = s.vert2orig;
    const zv = s.zv;
    for (let v = 0; v < m; v++) zv[v] = zs[v2o[v]];

    const triangles = s.tri.triangles;
    const S = s.scratch;
    const dx = (bx1 - bx0) / gridW;
    const dy = (by1 - by0) / gridH;
    const T = s.triCount;
    let cursor = s.cursor;
    if (cursor < 0) cursor = 0; else if (cursor >= T) cursor = T - 1;
    let count = 0;

    for (let row = 0; row < gridH; row++) {
        const qy = by0 + (row + 0.5) * dy;
        const base = row * gridW;
        const ltr = (row & 1) === 0; // serpentine: alternate scan direction
        for (let cc = 0; cc < gridW; cc++) {
            const col = ltr ? cc : (gridW - 1 - cc);
            const qx = bx0 + (col + 0.5) * dx;
            const t = _fieldWalkFrom(s, qx, qy, cursor);
            let val = NaN;
            if (t >= 0) {
                cursor = t;
                const e0 = 3 * t;
                const ia = triangles[e0], ib = triangles[e0 + 1], ic = triangles[e0 + 2];
                const ax = S[2 * ia], ay = S[2 * ia + 1];
                const bx = S[2 * ib], by = S[2 * ib + 1];
                const cx = S[2 * ic], cy = S[2 * ic + 1];
                const v0x = bx - ax, v0y = by - ay;
                const v1x = cx - ax, v1y = cy - ay;
                const d = v0x * v1y - v0y * v1x;
                const scale = (v0x * v0x + v0y * v0y) + (v1x * v1x + v1y * v1y);
                if (!(scale === 0) && Math.abs(d) > EPSILON * scale) {
                    const qx0 = qx - ax, qy0 = qy - ay;
                    const w1 = (qx0 * v1y - qy0 * v1x) / d;
                    const w2 = (v0x * qy0 - v0y * qx0) / d;
                    const w0 = 1 - w1 - w2;
                    // zv[vertex]: NaN-z propagates arithmetically, no branch.
                    val = w0 * zv[ia] + w1 * zv[ib] + w2 * zv[ic];
                }
            }
            outGrid[base + col] = val; // WRITES stay row-major
            if (Number.isFinite(val)) count++;
        }
    }
    s.cursor = cursor;
    return count;
}

/**
 * Release a per-build field-index facade and return its pooled slot to the
 * factory. Byte-identical semantics to `_cellDispose`: bump the slot generation
 * (invalidating every outstanding facade of this build), detach this facade so a
 * second dispose fails closed, and push the slot back on the free stack.
 *
 * @this {object} per-build field-index facade ({ _slot, _gen })
 * @throws {Error} if already disposed / stale
 */
function _fieldDispose() {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _fieldThrow('ddispose');
    this._slot = null;
    s.gen++;
    const pool = s._pool;
    pool.freeStack[pool.freeCount++] = s.poolIndex;
}

/**
 * Frozen shared prototype for the per-build field-index facade. Every method
 * lives here ONCE, so a build never allocates fresh closures.
 */
const FIELD_FACADE_PROTO = Object.freeze({
    locate: _fieldLocate,
    barycentric: _fieldBarycentric,
    triangleVertices: _fieldTriangleVertices,
    triangleCount: _fieldTriangleCount,
    interpolate: _fieldInterpolate,
    sampleField: _fieldSampleField,
    dispose: _fieldDispose,
});

/**
 * Allocate one pooled field-index slot: the triangulator arena, the interleave
 * scratch, the orig2v / vert2orig maps and the per-vertex z-gather buffer. Only
 * ever called at a new concurrent high-water mark. No circumcenters -- this is a
 * lighter slot than the cell index. The slot carries no methods.
 *
 * @param {number} maxPoints capacity
 * @param {object} pool the factory's pool bookkeeping
 * @param {number} poolIndex this slot's index in pool.slots
 * @returns {object} the pooled slot
 */
function _createFieldSlot(maxPoints, pool, poolIndex) {
    return {
        _pool: pool,
        poolIndex,
        // Generation stamp: bumped on every build and every dispose.
        gen: 0,
        maxPoints,
        // Per-build state (set by _buildFieldSlot).
        n: 0, m: 0, triCount: 0, degenerate: true,
        // Walk state: last-hit triangle, mesh winding sign, step budget.
        cursor: 0, orientSign: 1, stepCap: 0,
        // The mesh engine and its f64 interleave scratch [x0,y0,x1,y1,...].
        tri: new DelaunayTriangulator(maxPoints),
        scratch: new Float64Array(2 * maxPoints),
        // orig2v[i] = compacted vertex for original index i (-1 = non-finite);
        // vert2orig[v] = original index for compacted vertex v.
        orig2v: new Int32Array(maxPoints),
        vert2orig: new Int32Array(maxPoints),
        // Per-vertex z-gather buffer (zv[v] = zs[vert2orig[v]]) reused per
        // sampleField call so the grid loop reads zv[vertex] with zero alloc.
        zv: new Float64Array(maxPoints),
    };
}

/**
 * (Re)build the mesh for a slot from SoA coordinates. Compacts only finite
 * points, triangulates, derives the global winding sign and the walk step cap.
 * Zero allocation. No circumcenters, no inEdge map.
 *
 * @param {object} slot the pooled slot
 * @param {ArrayLike<number>} pxs x coordinates
 * @param {ArrayLike<number>} pys y coordinates
 * @param {number} n logical point count
 */
function _buildFieldSlot(slot, pxs, pys, n) {
    const orig2v = slot.orig2v, vert2orig = slot.vert2orig, S = slot.scratch;
    slot.n = n;
    slot.degenerate = true;
    slot.triCount = 0;
    slot.cursor = 0;
    slot.orientSign = 1;
    slot.stepCap = 0;

    // 1. Compact ONLY finite points; record original<->compacted index maps and
    //    interleave the compacted coords into the f64 scratch in one pass.
    orig2v.fill(-1, 0, n);
    let m = 0;
    for (let i = 0; i < n; i++) {
        const px = pxs[i], py = pys[i];
        if (px !== px || py !== py ||
            px === Infinity || px === -Infinity ||
            py === Infinity || py === -Infinity) {
            continue;
        }
        orig2v[i] = m; vert2orig[m] = i;
        S[2 * m] = px; S[2 * m + 1] = py;
        m++;
    }
    slot.m = m;

    // Fewer than 3 finite points: no triangulation exists -- degenerate stays.
    if (m < 3) return;

    const triCount = slot.tri.triangulate(S, m);
    if (triCount === 0) return; // collinear / coincident -- degenerate stays true
    slot.triCount = triCount;
    slot.degenerate = false;

    // Global winding sign from triangle 0's signed doubled area (the mesh has
    // uniform winding, so one sign serves every containment test).
    const triangles = slot.tri.triangles;
    const ia = triangles[0], ib = triangles[1], ic = triangles[2];
    const ax = S[2 * ia], ay = S[2 * ia + 1];
    const bx = S[2 * ib], by = S[2 * ib + 1];
    const cx = S[2 * ic], cy = S[2 * ic + 1];
    const d0 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    slot.orientSign = d0 >= 0 ? 1 : -1;
    // Step budget: O(sqrt(T)) expected walk length, generously padded; on
    // exhaustion the walk falls back to a linear scan -- never wrong, maybe slow.
    slot.stepCap = 4 * Math.ceil(Math.sqrt(triCount)) + 64;
}

/**
 * Create a pooled field-index FACTORY sized for up to `maxPoints` points.
 *
 * The returned function is `(pxs, pys, n) -> fieldIndex`. It mirrors
 * {@link createCellIndex} exactly -- pooled factory-factory, SoA input, NaN
 * compaction, ORIGINAL indices, generation-stamped facades -- but instead of a
 * Voronoi-cell surface it triangulates the mesh ONCE and answers point-location
 * and barycentric-interpolation queries with a remembering visibility walk. One
 * mesh serves MANY scalar fields: geometry is fixed at build, and each query
 * takes its `zs` per-call (indexed by ORIGINAL point index).
 *
 * GRID CONTRACT (`sampleField`): row-major, index = row*gridW + col; col 0 at
 * bx0 (xMin), row 0 at by0 (yMin) -- MATHEMATICAL +y-up orientation, deliberately
 * NOT screen convention. A pixel-space consumer flips rows itself and derives
 * its own present-mask, on its own cold path. Cell-CENTER sampling; cells
 * outside the hull get NaN; a degenerate build NaN-fills and returns 0.
 *
 * `zs` semantics: per-call, indexed by ORIGINAL point index; Float32Array,
 * Float64Array, or number[] accepted; throws if missing or shorter than the `n`
 * of THIS build. A NaN z-value propagates arithmetically into its incident
 * triangles' cells -- no NaN branch on the hot path.
 *
 * Memory model: a build allocates exactly one small facade (~48 B, young-gen);
 * the mesh arena, scratch, index maps and z-gather buffer are all pooled -- 0 B
 * beyond the facade -- and every query is 0 B. Per-slot memory is
 * ~100*maxPoints bytes + 16 KB (no circumcenters -- lighter than the cell slot).
 * A new slot is allocated only at a new concurrent high-water mark.
 *
 * @example
 * // CONSUMED lite-charts wiring: charts 1.16.0 rasterizes this into its
 * // scatter heatmap (sampleField, batched cold per refresh) and 1.17.0 draws
 * // contour isolines over it (triangleCount + triangleVertices). locate and
 * // barycentric are not yet consumed by charts but are frozen surface (semver).
 * const chart = createScatterChart({
 *   data,
 *   field: { index: createFieldIndex(20_000), value: "temperature" },
 * });
 *
 * @param {number} maxPoints hard upper bound on `n` per build. Must be a
 *   non-negative integer.
 * @returns {(pxs: ArrayLike<number>, pys: ArrayLike<number>, n: number) => object}
 *   the field-index factory.
 * @throws {Error} if `maxPoints` is not a non-negative integer.
 */
export const createFieldIndex = (maxPoints) => {
    if (!Number.isInteger(maxPoints) || maxPoints < 0) {
        _fieldThrow('maxp', maxPoints);
    }

    // Pool bookkeeping, identical in shape to createCellIndex.
    const pool = { slots: [], freeStack: [], freeCount: 0 };

    return function buildFieldIndex(pxs, pys, n) {
        // Fail closed at build time -- all guards live here, off the query path.
        if (!Number.isInteger(n) || n < 0) _fieldThrow('n', n);
        if (n > maxPoints) _fieldThrow('nmax', n, maxPoints);
        if (pxs == null || pys == null) _fieldThrow('arr');
        if (pxs.length < n || pys.length < n) _fieldThrow('arrlen', n);

        // Acquire a free slot, or grow the pool at a new high-water mark.
        let idx;
        if (pool.freeCount > 0) {
            idx = pool.freeStack[--pool.freeCount];
        } else {
            idx = pool.slots.length;
            pool.slots[idx] = _createFieldSlot(maxPoints, pool, idx);
        }
        const slot = pool.slots[idx];

        _buildFieldSlot(slot, pxs, pys, n);
        // New generation for this build; any facade from a prior build of this
        // slot is now stale and will throw.
        slot.gen++;
        const facade = Object.create(FIELD_FACADE_PROTO);
        facade._slot = slot;
        facade._gen = slot.gen;
        return facade;
    };
};

// ===========================================================================
// Cluster index (v1.4.0) -- convex hull + alpha-shape cluster outlines.
// ===========================================================================
//
// `createClusterIndex(maxPoints)` returns a BUILD function
// `(pxs, pys, n) -> clusterIndex` that triangulates the Delaunay mesh once and
// then answers two boundary-extraction queries against it:
//
//   - `convexHull(outIndices)` walks the triangulator's own hull ring and emits
//     the ordered hull as ORIGINAL site indices. This is the alpha -> +Infinity
//     degenerate of the alpha shape, computed directly from the ring in O(h).
//   - `alphaShape(alpha, outIndices, outLoopEnds)` keeps every triangle whose
//     circumradius is <= alpha, then traces the boundary (edges owned by exactly
//     one kept triangle) into one or more CONCATENATED loops. `alpha` is a RADIUS
//     in input (pixel) units. The two-buffer output expresses the multiple
//     disjoint loops (and interior holes) a single flat count cannot.
//
// Pooling / facade / generation semantics are IDENTICAL to createFieldIndex and
// createCellIndex: one factory, many concurrent handles; a build acquires a
// pooled slot and bumps its generation stamp; dispose bumps it again and returns
// the slot; a stale or disposed facade THROWS. A build allocates exactly one
// small facade (~48 B, young-gen). Every query is 0 B. The slot is the field
// slot MINUS the z-gather buffer PLUS three cluster arrays: per-triangle
// circumradius^2 (`crSq`), the per-triangle keep flag (`kept`), and the
// boundary-walk visited marks (`visited`).
//
// WINDING (the documented ABSOLUTE convention): the mesh is clockwise in math
// coordinates (+y up), which reads counter-clockwise in screen coordinates
// (+y down). Outer loops and the convex hull are emitted in that SAME sense
// (screen-CCW / math-CW), interior hole loops in the OPPOSITE sense; outer and
// hole loops therefore always wind oppositely. The emit rule is chosen ONCE per
// call from `slot.orientSign` (measured winding), so the convention survives a
// theoretically mirror-wound mesh without any per-edge branch.

/**
 * Cold error-message factory for the cluster index. All throw strings are built
 * (and thrown) HERE, off every hot method body -- the guard CONDITIONS live in
 * the methods, but not the byte cost of their messages.
 *
 * @param {string} code message selector
 * @param {*} [a] first interpolated value
 * @param {*} [b] second interpolated value
 * @returns {never} always throws
 */
function _clusterThrow(code, a, b) {
    switch (code) {
        case 'maxp':
            throw new Error(`lite-delaunay: maxPoints must be a non-negative integer, got ${a}`);
        case 'n':
            throw new Error(`lite-delaunay: n must be a non-negative integer, got ${a}`);
        case 'nmax':
            throw new Error(`lite-delaunay: n (${a}) exceeds cluster index max (${b})`);
        case 'arr':
            throw new Error("lite-delaunay: pxs and pys are required");
        case 'arrlen':
            throw new Error(`lite-delaunay: pxs/pys shorter than n (${a})`);
        case 'disposed':
            throw new Error(`lite-delaunay: ${a} called on a disposed cluster index`);
        case 'ddispose':
            throw new Error("lite-delaunay: dispose called on an already-disposed cluster index");
        case 'alpha':
            throw new Error(`lite-delaunay: alpha must be a finite number > 0, got ${a}`);
        case 'outtype':
            throw new Error(`lite-delaunay: ${a} must be an Int32Array`);
        case 'outshort':
            throw new Error(`lite-delaunay: ${a} too short (need length >= ${b})`);
        case 'walk':
            throw new Error("lite-delaunay: alpha-shape boundary walk exceeded its step cap (degenerate mesh)");
        case 'ring':
            throw new Error("lite-delaunay: convex-hull ring walk exceeded its step cap (degenerate hull)");
        // Self-guarding: a mistyped code must fail CLOSED, never fall through
        // to an undefined return.
        default:
            throw new Error(`lite-delaunay: internal error (unknown cluster throw code "${code}")`);
    }
}

/**
 * Precompute every triangle's circumradius^2 into the pooled `crSq`. Cold, once
 * per build (only when the build is non-degenerate). Uses the SAME dx/dy/ex/ey/
 * bl/cl/D/scale circumcenter math as the cell circumcenter guard, but on the
 * degeneracy boundary (`scale === 0 || !(|D| > EPSILON*scale)`) writes NaN --
 * NOT the cell path's centroid fallback. A cell needs a finite interior point;
 * an alpha test must FAIL CLOSED. Because `NaN <= alphaSq` is always false, a
 * triangle degenerate at the f64 NOISE FLOOR can NEVER be kept at ANY alpha.
 * A merely THIN triangle is above that floor and keeps honest alpha-shape
 * semantics: its circumradius is huge but finite, so it is kept only by a
 * correspondingly huge alpha -- the guard traps arithmetic garbage, it does
 * not reclassify thin-but-real geometry.
 *
 * @param {object} slot the pooled slot (slot.triCount already set, non-zero)
 * @returns {void}
 */
function _buildClusterRadii(slot) {
    const triangles = slot.tri.triangles;
    const S = slot.scratch;
    const crSq = slot.crSq;
    const triCount = slot.triCount;
    for (let t = 0; t < triCount; t++) {
        const ia = triangles[3 * t], ib = triangles[3 * t + 1], ic = triangles[3 * t + 2];
        const ax = S[2 * ia], ay = S[2 * ia + 1];
        const bx = S[2 * ib], by = S[2 * ib + 1];
        const cx = S[2 * ic], cy = S[2 * ic + 1];
        const dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
        const bl = dx * dx + dy * dy, cl = ex * ex + ey * ey;
        const D = dx * ey - dy * ex;
        const scale = bl + cl;
        // Fail closed to NaN on the overflow boundary (also traps a NaN D), so
        // the triangle drops out of every alpha keep-set.
        if (scale === 0 || !(Math.abs(D) > EPSILON * scale)) {
            crSq[t] = NaN;
        } else {
            const d = 0.5 / D;
            const ux = (ey * bl - dy * cl) * d;
            const uy = (dx * cl - ex * bl) * d;
            crSq[t] = ux * ux + uy * uy;
        }
    }
}

/**
 * Allocate one pooled cluster-index slot: the field slot MINUS the z-gather
 * buffer, PLUS the three cluster arrays. The `cursor` / `orientSign` / `stepCap`
 * fields are declared (inert for cluster queries) so {@link _buildFieldSlot} can
 * be called VERBATIM to compact + triangulate + derive the winding sign. Only
 * ever called at a new concurrent high-water mark. The slot carries no methods.
 *
 * @param {number} maxPoints capacity
 * @param {object} pool the factory's pool bookkeeping
 * @param {number} poolIndex this slot's index in pool.slots
 * @returns {object} the pooled slot
 */
function _createClusterSlot(maxPoints, pool, poolIndex) {
    return {
        _pool: pool,
        poolIndex,
        // Generation stamp: bumped on every build and every dispose.
        gen: 0,
        maxPoints,
        // Per-build state (set by _buildFieldSlot).
        n: 0, m: 0, triCount: 0, degenerate: true,
        // Declared so _buildFieldSlot runs verbatim; cursor/stepCap are inert for
        // cluster queries, orientSign carries the measured winding sign.
        cursor: 0, orientSign: 1, stepCap: 0,
        // The mesh engine and its f64 interleave scratch [x0,y0,x1,y1,...].
        tri: new DelaunayTriangulator(maxPoints),
        scratch: new Float64Array(2 * maxPoints),
        // orig2v[i] = compacted vertex for original index i (-1 = non-finite);
        // vert2orig[v] = original index for compacted vertex v.
        orig2v: new Int32Array(maxPoints),
        vert2orig: new Int32Array(maxPoints),
        // Cluster arrays (max triangles = 2N - 5, max halfedges = 3*(2N-5)):
        // per-triangle circumradius^2, per-triangle keep flag, per-halfedge marks.
        crSq: new Float64Array(2 * maxPoints),
        kept: new Uint8Array(2 * maxPoints),
        visited: new Uint8Array(6 * maxPoints),
    };
}

/**
 * (Re)build a cluster slot: run {@link _buildFieldSlot} VERBATIM (compact finite
 * points, triangulate, derive the winding sign), then precompute circumradii for
 * a non-degenerate build. Zero allocation.
 *
 * @param {object} slot the pooled slot
 * @param {ArrayLike<number>} pxs x coordinates
 * @param {ArrayLike<number>} pys y coordinates
 * @param {number} n logical point count
 * @returns {void}
 */
function _buildClusterSlot(slot, pxs, pys, n) {
    _buildFieldSlot(slot, pxs, pys, n);
    if (!slot.degenerate) _buildClusterRadii(slot);
}

/**
 * Emit the ordered convex hull as ORIGINAL site indices into `outIndices`,
 * returning the vertex count `h` (<= n). A degenerate build (n < 3, collinear,
 * all-duplicate) returns 0 -- NOT a throw. Two-phase and O(h): phase 1 walks the
 * triangulator's own hull ring counting `h` (capped at slot.m against a
 * degenerate self-loop), validates `outIndices.length >= h`, then phase 2
 * re-walks emitting. Emission sense matches the alpha-shape outer-loop
 * convention (screen-CCW / math-CW); the ring is already that sense, and the
 * `orientSign` guard reverses the walk for a theoretically mirror-wound mesh.
 * Zero allocation.
 *
 * @this {object} per-build cluster-index facade ({ _slot, _gen })
 * @param {Int32Array} outIndices caller-owned, length >= h (n is the safe bound)
 * @returns {number} hull vertex count h
 * @throws {Error} if disposed/stale, `outIndices` not an Int32Array, too short,
 *   or the ring walk overruns its cap
 */
function _clusterConvexHull(outIndices) {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _clusterThrow('disposed', 'convexHull');
    if (outIndices == null || !(outIndices instanceof Int32Array)) _clusterThrow('outtype', 'outIndices');
    if (s.degenerate) return 0;
    const tri = s.tri;
    // Screen-CCW / math-CW ring: hullNext for the measured mesh (orientSign < 0),
    // hullPrev to reverse for a theoretically mirror-wound mesh.
    const ring = s.orientSign < 0 ? tri.hullNext : tri.hullPrev;
    const start = tri.hullStart;
    const m = s.m;
    // Phase 1: count (cap against a degenerate ring self-loop).
    let h = 0;
    let e = start;
    do {
        h++;
        if (h > m) _clusterThrow('ring');
        e = ring[e];
    } while (e !== start);
    if (outIndices.length < h) _clusterThrow('outshort', 'outIndices', h);
    // Phase 2: emit ORIGINAL indices.
    const v2o = s.vert2orig;
    let w = 0;
    e = start;
    do {
        outIndices[w++] = v2o[e];
        e = ring[e];
    } while (e !== start);
    return h;
}

/**
 * Extract the alpha-shape boundary into `outIndices` / `outLoopEnds`, returning
 * the loop count. Keep every triangle whose circumradius is <= `alpha`; the
 * boundary is the set of half-edges owned by exactly one kept triangle. Loops are
 * CONCATENATED into `outIndices` as ORIGINAL indices (implicit closure last ->
 * first); `outLoopEnds[i]` is the EXCLUSIVE end offset of loop i. 0 loops is
 * legal (alpha keeps nothing, or a degenerate build).
 *
 * `alpha` is a finite RADIUS > 0 in input units. Infinity is NOT a hull alias --
 * it throws; a large FINITE alpha lawfully degenerates to the convex hull (every
 * triangle kept -> the outer boundary is the hull). NaN / +/-0 / negative /
 * -Infinity / non-numbers all throw before any arithmetic touches the value.
 *
 * Two-phase, zero allocation. Pass 0 fills the keep flags. PHASE 1 counts loops
 * and edges (writing NOTHING to the caller's buffers) with a hard step cap.
 * VALIDATE-BEFORE-WRITE: `outIndices` / `outLoopEnds` length is checked against
 * the counted totals BEFORE phase 2 writes a single byte. PHASE 2 re-scans in the
 * identical order and emits. The emit rule (which vertex per edge, which rotation)
 * is chosen ONCE from `slot.orientSign`, so outer loops come out screen-CCW /
 * math-CW and holes opposite regardless of mesh winding, with no per-edge branch.
 *
 * @this {object} per-build cluster-index facade ({ _slot, _gen })
 * @param {number} alpha finite radius > 0
 * @param {Int32Array} outIndices caller-owned, length >= total boundary edges
 *   (safe bound 3n)
 * @param {Int32Array} outLoopEnds caller-owned, length >= loop count (safe bound n)
 * @returns {number} loop count
 * @throws {Error} if disposed/stale; `alpha` not a finite number > 0; either out
 *   buffer not an Int32Array or too short for the counted totals; or the boundary
 *   walk overruns its step cap
 */
function _clusterAlphaShape(alpha, outIndices, outLoopEnds) {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _clusterThrow('disposed', 'alphaShape');
    // Alpha door: refuses null/undefined/NaN/+0/-0/negative/-Infinity/Infinity/
    // strings without ever applying + or * to a non-number.
    if (typeof alpha !== "number" || alpha !== alpha || alpha <= 0 || alpha === Infinity) {
        _clusterThrow('alpha', alpha);
    }
    if (outIndices == null || !(outIndices instanceof Int32Array)) _clusterThrow('outtype', 'outIndices');
    if (outLoopEnds == null || !(outLoopEnds instanceof Int32Array)) _clusterThrow('outtype', 'outLoopEnds');
    const triCount = s.triCount;
    if (s.degenerate || triCount === 0) return 0;

    const alphaSq = alpha * alpha;
    const triangles = s.tri.triangles;
    const halfedges = s.tri.halfedges;
    const crSq = s.crSq;
    const kept = s.kept;
    const visited = s.visited;
    const trianglesLen = 3 * triCount;

    // Pass 0: keep flags. NaN crSq (degenerate triangle) fails `<=` -> never kept.
    for (let t = 0; t < triCount; t++) kept[t] = crSq[t] <= alphaSq ? 1 : 0;
    visited.fill(0, 0, trianglesLen);

    // PHASE 1: count loops + edges. No writes to caller buffers. The next-rotation
    // successor traces the same loop PARTITION for either mesh winding (a mirror
    // mesh only reverses each loop), so the counts are orientation-independent.
    let totalEdges = 0, loopCount = 0, steps = 0;
    for (let e = 0; e < trianglesLen; e++) {
        if (!kept[(e / 3) | 0] || visited[e]) continue;
        const he = halfedges[e];
        if (!(he === -1 || !kept[(he / 3) | 0])) continue; // not a boundary edge
        let f = e;
        do {
            visited[f] = 1;
            totalEdges++;
            if (++steps > trianglesLen) _clusterThrow('walk');
            // Successor: rotate around the head vertex through kept triangles.
            let g = (f % 3 === 2) ? f - 2 : f + 1;
            let tw = halfedges[g];
            while (tw !== -1 && kept[(tw / 3) | 0]) {
                g = (tw % 3 === 2) ? tw - 2 : tw + 1;
                tw = halfedges[g];
            }
            f = g;
        } while (f !== e);
        loopCount++;
    }

    // Absolute validate-before-write door: the documented 3n / n bounds guarantee
    // a contract-compliant caller never trips this.
    if (outIndices.length < totalEdges) _clusterThrow('outshort', 'outIndices', totalEdges);
    if (outLoopEnds.length < loopCount) _clusterThrow('outshort', 'outLoopEnds', loopCount);

    // PHASE 2: identical ascending scan, now emitting. Rule picked ONCE by winding.
    visited.fill(0, 0, trianglesLen);
    const v2o = s.vert2orig;
    let w = 0, k = 0;
    if (s.orientSign < 0) {
        // Measured mesh (math-CW): emit the edge TAIL, next-rotation successor.
        for (let e = 0; e < trianglesLen; e++) {
            if (!kept[(e / 3) | 0] || visited[e]) continue;
            const he = halfedges[e];
            if (!(he === -1 || !kept[(he / 3) | 0])) continue;
            let f = e;
            do {
                visited[f] = 1;
                outIndices[w++] = v2o[triangles[f]];
                let g = (f % 3 === 2) ? f - 2 : f + 1;
                let tw = halfedges[g];
                while (tw !== -1 && kept[(tw / 3) | 0]) {
                    g = (tw % 3 === 2) ? tw - 2 : tw + 1;
                    tw = halfedges[g];
                }
                f = g;
            } while (f !== e);
            outLoopEnds[k++] = w;
        }
    } else {
        // Mirror-wound mesh (theoretical): emit the edge HEAD, prev-rotation
        // successor, so the documented screen-CCW outer convention still holds.
        for (let e = 0; e < trianglesLen; e++) {
            if (!kept[(e / 3) | 0] || visited[e]) continue;
            const he = halfedges[e];
            if (!(he === -1 || !kept[(he / 3) | 0])) continue;
            let f = e;
            do {
                visited[f] = 1;
                const nf = (f % 3 === 2) ? f - 2 : f + 1;
                outIndices[w++] = v2o[triangles[nf]];
                let g = (f % 3 === 0) ? f + 2 : f - 1;
                let tw = halfedges[g];
                while (tw !== -1 && kept[(tw / 3) | 0]) {
                    g = (tw % 3 === 0) ? tw + 2 : tw - 1;
                    tw = halfedges[g];
                }
                f = g;
            } while (f !== e);
            outLoopEnds[k++] = w;
        }
    }
    return loopCount;
}

/**
 * Release a per-build cluster-index facade and return its pooled slot to the
 * factory. Byte-identical semantics to `_fieldDispose`: bump the slot generation
 * (invalidating every outstanding facade of this build), detach this facade so a
 * second dispose fails closed, and push the slot back on the free stack.
 *
 * @this {object} per-build cluster-index facade ({ _slot, _gen })
 * @throws {Error} if already disposed / stale
 */
function _clusterDispose() {
    const s = this._slot;
    if (s === null || this._gen !== s.gen) _clusterThrow('ddispose');
    this._slot = null;
    s.gen++;
    const pool = s._pool;
    pool.freeStack[pool.freeCount++] = s.poolIndex;
}

/**
 * Frozen shared prototype for the per-build cluster-index facade. Every method
 * lives here ONCE, so a build never allocates fresh closures.
 */
const CLUSTER_FACADE_PROTO = Object.freeze({
    convexHull: _clusterConvexHull,
    alphaShape: _clusterAlphaShape,
    dispose: _clusterDispose,
});

/**
 * Create a pooled cluster-index FACTORY sized for up to `maxPoints` points.
 *
 * The returned function is `(pxs, pys, n) -> clusterIndex`. It mirrors
 * {@link createFieldIndex} exactly -- pooled factory-factory, SoA input, NaN/
 * Infinity compaction, ORIGINAL indices, generation-stamped facades -- but its
 * surface is boundary extraction: `convexHull` and `alphaShape`. Built fresh per
 * refresh per point group by lite-charts' `outlines` layer.
 *
 * THE FOUR CONTRACT ANSWERS (quotable, for the consumer brief cluster-outlines.md):
 *
 * (a) SIZING BOUNDS. `alphaShape` needs `outIndices.length >= total boundary
 *     edges`. A planar triangulation has at most `3n - 6` edges, and a pinch-
 *     point vertex is counted PER EDGE (already covered by the edge bound), so
 *     the tight bound is `3n - 6`; the safe recommendation is `3n`. `outLoopEnds`
 *     needs `>= loopCount`; every loop has `>= 3` edges so the tight bound is
 *     `n - 2`, safe `n`. `convexHull` needs `outIndices.length >= h`, bounded by
 *     `n`. Because the method validates the counted totals BEFORE writing, an
 *     exactly-3n / n caller can never trip the short-buffer door.
 *
 * (b) HOLE LOOPS ARE EMITTED. A kept annulus yields its inner rim as its own
 *     loop. Orientation (the ABSOLUTE convention): outer loops are emitted CCW in
 *     screen coordinates (+y down) = CW in math coordinates (+y up); hole loops
 *     wind OPPOSITE; outer and hole always wind oppositely; `convexHull` is the
 *     SAME sense as an outer loop. Stated in BOTH senses so it survives a screen-
 *     space consumer.
 *
 * (c) COINCIDENT DUPLICATES. When several coincident points share a hull/boundary
 *     vertex, the emitted ORIGINAL index is the sweep-order-FIRST duplicate (the
 *     triangulator sorts by distance from the seed circumcenter and skips
 *     near-duplicates within its EPSILON tolerance). This is DETERMINISTIC for
 *     identical input arrays; it is NOT guaranteed to be the lowest index and MAY
 *     change if the input order changes.
 *
 * (d) PERF. See bench/bench.js: warm-handle convexHull / alphaShape throughput at
 *     n = 1k/10k/100k, and the small-n full-cycle (factory -> convexHull ->
 *     alphaShape -> dispose) table charts' per-refresh unit is measured against.
 *
 * FAIL-CLOSED: a triangle degenerate at the f64 noise floor (the relative-area
 * guard `|D| <= EPSILON*scale`, shared with the cell index) carries NaN `crSq`
 * and can never be kept at any alpha -- while a merely thin triangle keeps its
 * honest huge-but-finite circumradius and is kept only by a correspondingly
 * huge alpha; a degenerate build returns 0 from both methods; the alpha door
 * refuses every non-finite / non-positive value before arithmetic; the boundary
 * and ring walks are step-capped; and both buffers are validated before any write.
 *
 * Memory model: a build allocates exactly one small facade (~48 B, young-gen);
 * the mesh arena, scratch, index maps and cluster arrays are all pooled -- 0 B
 * beyond the facade -- and every query is 0 B. Per-slot memory is ~116*maxPoints
 * bytes + 16 KB (the field slot minus the z-gather buffer, plus crSq/kept/
 * visited). A new slot is allocated only at a new concurrent high-water mark.
 *
 * @example
 * // lite-charts v1.18.0 consumer (brief cluster-outlines.md; charts consumes
 * // this AFTER lite-delaunay 1.4.0 publishes to npm):
 * const chart = createScatterChart({
 *   data,
 *   outlines: { index: createClusterIndex(20_000), groupKey: 'zone', alpha: 25 },
 * });
 * // Per refresh, charts packs each group's pxs/pys and calls the factory once
 * // per group, then convexHull (alpha absent) or alphaShape (alpha set), 0 B/frame.
 *
 * @param {number} maxPoints hard upper bound on `n` per build. Must be a
 *   non-negative integer.
 * @returns {(pxs: ArrayLike<number>, pys: ArrayLike<number>, n: number) => object}
 *   the cluster-index factory.
 * @throws {Error} if `maxPoints` is not a non-negative integer.
 */
export const createClusterIndex = (maxPoints) => {
    if (!Number.isInteger(maxPoints) || maxPoints < 0) {
        _clusterThrow('maxp', maxPoints);
    }

    // Pool bookkeeping, identical in shape to createFieldIndex.
    const pool = { slots: [], freeStack: [], freeCount: 0 };

    return function buildClusterIndex(pxs, pys, n) {
        // Fail closed at build time -- all guards live here, off the query path.
        if (!Number.isInteger(n) || n < 0) _clusterThrow('n', n);
        if (n > maxPoints) _clusterThrow('nmax', n, maxPoints);
        if (pxs == null || pys == null) _clusterThrow('arr');
        if (pxs.length < n || pys.length < n) _clusterThrow('arrlen', n);

        // Acquire a free slot, or grow the pool at a new high-water mark.
        let idx;
        if (pool.freeCount > 0) {
            idx = pool.freeStack[--pool.freeCount];
        } else {
            idx = pool.slots.length;
            pool.slots[idx] = _createClusterSlot(maxPoints, pool, idx);
        }
        const slot = pool.slots[idx];

        _buildClusterSlot(slot, pxs, pys, n);
        // New generation for this build; any facade from a prior build of this
        // slot is now stale and will throw.
        slot.gen++;
        const facade = Object.create(CLUSTER_FACADE_PROTO);
        facade._slot = slot;
        facade._gen = slot.gen;
        return facade;
    };
};
