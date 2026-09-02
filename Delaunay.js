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
export const VERSION = "1.1.0";

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
 *   spatialIndexFactory: createSpatialIndex(20_000),
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