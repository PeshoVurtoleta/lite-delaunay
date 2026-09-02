/**
 * @zakkster/lite-delaunay
 * -----------------------------------------------------------------------------
 * Zero-GC 2D Delaunay triangulation via the Mapbox-style sweepline algorithm.
 *
 * Memory model: one allocation at construction time, sized for `maxPoints`.
 * Every subsequent `triangulate()` call writes into the same arena -- no
 * `new` in the hot path, no garbage produced per frame.
 */

/**
 * Pre-allocated, zero-allocation Delaunay triangulator.
 *
 * Construct **once** with the largest point count you expect to ever pass to
 * `triangulate()`. The arena is sized at that point and never re-allocated.
 * Subsequent calls to `triangulate()` are all in-place -- zero allocations
 * in the hot path.
 *
 * @example
 * ```ts
 * import { DelaunayTriangulator } from "@zakkster/lite-delaunay";
 *
 * const tri = new DelaunayTriangulator(10_000);
 * const coords = new Float32Array([0, 0, 100, 0, 100, 100, 0, 100]);
 *
 * const count = tri.triangulate(coords, 4); // -> 2
 * for (let k = 0; k < count; k++) {
 *   const a = tri.triangles[k * 3];
 *   const b = tri.triangles[k * 3 + 1];
 *   const c = tri.triangles[k * 3 + 2];
 *   // ...
 * }
 * ```
 */
export class DelaunayTriangulator {
    /**
     * The hard upper bound on input size, set in the constructor. The arena's
     * typed arrays are sized for this many points; `triangulate(coords, n)`
     * with `n > maxPoints` throws.
     */
    readonly maxPoints: number;

    /**
     * The maximum number of triangles the output arena can hold.
     * Equals `Math.max(2 * maxPoints - 5, 0)` -- Euler's upper bound for a
     * planar triangulation of N points.
     */
    readonly maxTriangles: number;

    /**
     * The size of the angular hash table used internally to accelerate hull
     * lookup. Equals `Math.ceil(Math.sqrt(maxPoints)) + 1`. Exposed for
     * diagnostics; you should not need it.
     */
    readonly hashSize: number;

    /**
     * Flat triangle vertex-index array. Triangle `k`'s three vertices are at
     * indices `[3*k, 3*k+1, 3*k+2]`, in counter-clockwise winding order
     * (math convention -- in screen-space where Y points down this looks CW).
     *
     * **Only valid up to index `trianglesLen - 1`.** Bytes past that are
     * leftover from previous calls and must not be read.
     *
     * The array is `Int32Array(maxTriangles * 3)`, allocated once.
     */
    readonly triangles: Int32Array;

    /**
     * Flat half-edge twin array, parallel to `triangles`. `halfedges[3*k + i]`
     * is the index (in this same flat indexing) of the half-edge across the
     * shared edge in the neighbouring triangle, or `-1` if that edge lies on
     * the convex hull.
     *
     * This is the standard half-edge mesh representation. It lets you walk
     * the dual graph (for Voronoi), find triangle neighbours, traverse the
     * hull, etc., all in O(1) per step.
     *
     * **Only valid up to index `trianglesLen - 1`.**
     *
     * The array is `Int32Array(maxTriangles * 3)`, allocated once.
     */
    readonly halfedges: Int32Array;

    /**
     * Number of valid entries in `triangles` / `halfedges`. Equals
     * `triangleCount * 3`. Updated by every `triangulate()` call (including
     * reset to 0 on degenerate input).
     */
    readonly trianglesLen: number;

    /** Index of the first point on the convex hull. */
    readonly hullStart: number;

    /** Number of vertices currently on the convex hull. */
    readonly hullLen: number;

    /** Flat array for traversing the hull forward. `hullNext[i]` gives the next vertex index. */
    readonly hullNext: Int32Array;

    /** Flat array for traversing the hull backward. `hullPrev[i]` gives the previous vertex index. */
    readonly hullPrev: Int32Array;

    /**
     * Allocate the arena for up to `maxPoints` input vertices.
     *
     * @param maxPoints Hard upper bound on input size. Must be a non-negative
     *   integer. Memory footprint is roughly **68 bytes per max-point + ~16 KB**
     *   of fixed overhead. For `maxPoints = 10_000` that's ~700 KB.
     * @throws If `maxPoints` is not a non-negative integer.
     */
    constructor(maxPoints: number);

    /**
     * Triangulate `pointCount` points from a flat interleaved coordinate array.
     *
     * Writes results into `triangles` and `halfedges`, updates `trianglesLen`,
     * and returns the triangle count.
     *
     * **Performance guarantees:**
     *   - Zero allocations on the success path.
     *   - O(n log n) average-case; O(n^2) worst-case (pathological cocircular sets).
     *   - Same algorithm and numerical behaviour as Mapbox Delaunator.
     *
     * **Degenerate inputs** return `0` and do not throw:
     *   - `pointCount < 3`
     *   - all input points coincident
     *   - all input points collinear (no 2D triangulation exists)
     *
     * In every degenerate case, `trianglesLen` is reset to 0 so a previous
     * call's output cannot leak through.
     *
     * @param coords Flat interleaved `[x0, y0, x1, y1, ...]`. Must contain at
     *   least `pointCount * 2` elements. `Float32Array` is recommended for
     *   memory bandwidth; `Float64Array` and plain `number[]` are accepted.
     * @param pointCount Number of logical (x, y) points to process. Must be
     *   `<= maxPoints`.
     * @returns The number of triangles generated. Read
     *   `triangles[0 .. returnValue*3 - 1]` for vertex indices and
     *   `halfedges[0 .. returnValue*3 - 1]` for the half-edge twin table.
     * @throws If `pointCount > maxPoints`.
     */
    triangulate(
        coords: Float32Array | Float64Array | ArrayLike<number>,
        pointCount: number
    ): number;
}

/**
 * Package version. Kept in lockstep with `package.json` and `llms.txt`.
 */
export const VERSION: string;

/**
 * A pre-built, allocation-free k-nearest-neighbour query over a fixed set of
 * SoA pixel coordinates. Produced by a {@link SpatialIndexFactory}. This is the
 * interface `@zakkster/lite-charts` injects to hit-test dense bubble/scatter
 * point clouds; the implementation is a uniform grid.
 */
export interface SpatialIndex {
    /**
     * Write up to `k` nearest points to `(qx, qy)`, by squared pixel distance,
     * into the caller-owned `outIndices` / `outDistSq` (both length `>= k`),
     * sorted nearest-first, filtered to points within `maxDistSq`.
     *
     * `outIndices` receives ORIGINAL point indices (as passed to the factory);
     * `outDistSq` receives SQUARED pixel distances. Containment is INCLUSIVE
     * (`d <= maxDistSq`). Points whose coordinates were NaN/Infinity at build
     * time are never returned. A non-finite `(qx, qy)` returns `0` (fail closed).
     *
     * Zero allocation. `k` is clamped to 8 (the charts hit-buffer size).
     *
     * @returns the number of neighbours written (`0 .. k`).
     * @throws if the handle has been disposed.
     */
    findNearest(
        qx: number,
        qy: number,
        k: number,
        maxDistSq: number,
        outIndices: Int32Array,
        outDistSq: Float32Array
    ): number;

    /**
     * Release this handle back to its factory pool. The backing arena lives as
     * long as the factory. Using or double-disposing a disposed handle throws.
     */
    dispose(): void;
}

/**
 * Builds a {@link SpatialIndex} over the SoA pixel coordinates `pxs` / `pys`
 * (the first `n` entries of each). `n` must be `<= maxPoints` (throws otherwise).
 * A build allocates one small handle facade (~48 B, minor-GC-collectible); all
 * arenas, grids and scratch are pooled -- 0 B beyond the facade.
 */
export type SpatialIndexFactory = (
    pxs: ArrayLike<number>,
    pys: ArrayLike<number>,
    n: number
) => SpatialIndex;

/**
 * Create a pooled spatial-index factory sized for up to `maxPoints` points.
 *
 * The returned function matches the `SpatialIndexFactory` contract that
 * `@zakkster/lite-charts` injects via config. One factory serves many
 * concurrent live handles (one per series in a multi-series chart), disposed
 * independently; each build reuses a pooled slot and allocates only one small
 * handle facade (~48 B) -- 0 B beyond it. A new slot is allocated only at a new
 * concurrent high-water mark. Per-slot memory is ~24*maxPoints bytes plus grid
 * cells. `findNearest` is 0 B/query.
 *
 * @param maxPoints hard upper bound on `n` per build. Must be a non-negative
 *   integer.
 * @throws if `maxPoints` is not a non-negative integer.
 */
export function createSpatialIndex(maxPoints: number): SpatialIndexFactory;
