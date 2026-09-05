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

/**
 * A pre-built, allocation-free bbox-clipped Voronoi-cell query over a fixed set
 * of SoA pixel coordinates. Produced by a {@link CellIndexFactory}. This is the
 * interface `@zakkster/lite-charts` injects to draw a Voronoi tessellation over
 * a projected scatter (see the v1.2.0 cells layer). Unlike {@link SpatialIndex}
 * (a uniform grid), this builds the Delaunay mesh and precomputes every triangle
 * circumcenter at build time.
 */
export interface CellIndex {
    /**
     * Write site `i`'s Voronoi polygon, CLIPPED to the axis-aligned bbox, into
     * the caller-owned interleaved `outXY` (`[x0, y0, x1, y1, ...]`), and return
     * the vertex count written. `i` is the ORIGINAL point index (as passed to
     * the factory). Every returned polygon is finite, convex, and closed (the
     * last vertex implicitly connects to the first) and lies fully inside-or-on
     * the bbox -- hull cells are clipped, never flagged.
     *
     * Returns `0` (never a garbage polygon) when: `i` was a non-finite input
     * point; the build was degenerate (fewer than 3 finite points, or the
     * triangulation collapsed -- all-collinear / all-coincident); `i` lost the
     * triangulator's EPSILON dedup (an exact/near-duplicate that owns no mesh
     * vertex); or the cell does not intersect the bbox.
     *
     * Zero allocation per call.
     *
     * @param i original point index; must be an integer in `[0, n)`.
     * @param bx0 bbox min x
     * @param by0 bbox min y
     * @param bx1 bbox max x (must be `> bx0`)
     * @param by1 bbox max y (must be `> by0`)
     * @param outXY caller-owned interleaved output buffer.
     * @returns vertex count written (`0`, or `3 .. outXY.length / 2`).
     * @throws if the handle has been disposed; if `i` is not an integer in
     *   `[0, n)`; if the bbox is non-finite or not strictly ordered; or if the
     *   clipped cell needs more vertices than `outXY` can hold (the loud escape
     *   -- it never truncates).
     */
    cell(
        i: number,
        bx0: number,
        by0: number,
        bx1: number,
        by1: number,
        outXY: Float64Array | Float32Array
    ): number;

    /**
     * Release this handle back to its factory pool. The backing mesh arena lives
     * as long as the factory. Using or double-disposing a disposed handle throws.
     */
    dispose(): void;
}

/**
 * Builds a {@link CellIndex} over the SoA pixel coordinates `pxs` / `pys` (the
 * first `n` entries of each; `NaN`/`Infinity` legal, compacted out at build).
 * `n` must be `<= maxPoints` (throws otherwise). A build allocates one small
 * handle facade (~48 B, minor-GC-collectible); the mesh arena, circumcenter
 * arrays and clip buffers are pooled -- 0 B beyond the facade.
 */
export type CellIndexFactory = (
    pxs: ArrayLike<number>,
    pys: ArrayLike<number>,
    n: number
) => CellIndex;

/**
 * Create a pooled cell-index factory sized for up to `maxPoints` points.
 *
 * Mirrors {@link createSpatialIndex} exactly (pooled factory-factory, SoA input,
 * NaN compaction, ORIGINAL indices, generation-stamped facades) but, instead of
 * a k-NN grid, builds the Delaunay mesh, precomputes every circumcenter, and
 * answers `cell(i, ...)` by walking the half-edges around site `i` and CLIPPING
 * the resulting Voronoi polygon to the caller's axis-aligned bbox with a
 * zero-allocation Sutherland-Hodgman pass. A build allocates one small handle
 * facade (~48 B); everything else is pooled and `cell()` is 0 B/query. A new
 * slot is allocated only at a new concurrent high-water mark.
 *
 * SIZING RULE (so callers can size `outXY`): a bbox-clipped Voronoi cell of an
 * INTERIOR site has at most `degree + 4` vertices, and of a HULL site at most
 * `degree + 5`. A caller-owned buffer of `2 * 64` floats (64 vertices) covers
 * every non-adversarial cloud; a clipped cell that needs more makes `cell()`
 * THROW rather than truncate.
 *
 * @param maxPoints hard upper bound on `n` per build. Must be a non-negative
 *   integer.
 * @throws if `maxPoints` is not a non-negative integer.
 */
export function createCellIndex(maxPoints: number): CellIndexFactory;

/**
 * A pre-built, allocation-free point-location and scalar-field interpolation
 * query over a fixed set of SoA coordinates. Produced by a
 * {@link FieldIndexFactory}. Unlike {@link CellIndex} (a Voronoi-cell surface),
 * this triangulates the Delaunay mesh ONCE and walks it to answer queries. ONE
 * mesh serves MANY scalar fields: geometry is fixed at build, and each query
 * takes its `zs` per-call (indexed by ORIGINAL point index).
 */
export interface FieldIndex {
    /**
     * Locate the COMPACTED-MESH triangle containing `(qx, qy)`. Returns a
     * triangle id in `[0, triangleCount())`, or `-1` for a non-finite query, a
     * degenerate build, or a point outside the convex hull. z-agnostic. Updates
     * the walk cursor on a hit for coherence. Zero allocation.
     *
     * @throws if the handle is disposed or stale.
     */
    locate(qx: number, qy: number): number;

    /**
     * Write the three barycentric weights of `(qx, qy)` in triangle `t` into
     * `outW3` (ALWAYS all three), and return `true` iff `q` is inside-or-on `t`
     * (every weight `>= -1e-9`). A near-zero-area triangle writes `NaN,NaN,NaN`
     * and returns `false`. Zero allocation.
     *
     * @param t compacted-mesh triangle id in `[0, triangleCount())`.
     * @param outW3 caller-owned output, length `>= 3`.
     * @throws if the handle is disposed/stale, `t` is not an integer in range,
     *   or `outW3` is too short.
     */
    barycentric(
        t: number,
        qx: number,
        qy: number,
        outW3: Float32Array | Float64Array | number[]
    ): boolean;

    /**
     * Write triangle `t`'s three ORIGINAL point indices (as passed to the
     * factory) into `outI3`. Reserves per-triangle site access for future
     * contour / TIN work. Zero allocation.
     *
     * @param t compacted-mesh triangle id in `[0, triangleCount())`.
     * @param outI3 caller-owned output, length `>= 3`.
     * @throws if the handle is disposed/stale, `t` is not an integer in range,
     *   or `outI3` is too short.
     */
    triangleVertices(t: number, outI3: Int32Array | number[]): void;

    /**
     * Number of triangles in the built mesh. `0` on a degenerate build.
     *
     * @throws if the handle is disposed or stale.
     */
    triangleCount(): number;

    /**
     * Interpolate the scalar field `zs` at `(qx, qy)`: locate the containing
     * triangle, compute barycentric weights, return the weighted sum of `zs` at
     * the triangle's three ORIGINAL vertex indices. Returns `NaN` when the query
     * is non-finite, the build is degenerate, the point is outside the hull, or
     * the area guard fails; a `NaN`-z corner propagates arithmetically. `zs` is
     * indexed by ORIGINAL point index. Zero allocation.
     *
     * @param zs scalar values, length `>= n`.
     * @throws if the handle is disposed/stale, or `zs` is missing or shorter
     *   than `n`.
     */
    interpolate(
        zs: Float32Array | Float64Array | number[],
        qx: number,
        qy: number
    ): number;

    /**
     * Rasterize the scalar field `zs` onto a `gridW x gridH` regular grid over
     * the bbox, writing into `outGrid`, and return the count of FINITE cells
     * written.
     *
     * GRID CONTRACT: row-major, `index = row*gridW + col`; col 0 at `bx0`
     * (xMin), row 0 at `by0` (yMin) -- MATHEMATICAL +y-up orientation,
     * deliberately NOT screen convention (a pixel-space consumer flips rows
     * itself). Cell-CENTER sampling. Cells outside the hull get `NaN`; a
     * degenerate build fills the `gridW*gridH` prefix with `NaN` and returns
     * `0`. Zero allocation.
     *
     * @param zs scalar values, length `>= n`.
     * @param outGrid caller-owned output, length `>= gridW*gridH`.
     * @throws if the handle is disposed/stale; `zs` is missing/short;
     *   `gridW`/`gridH` are not positive integers; `outGrid` is the wrong type
     *   or too short; or the bbox is non-finite or not strictly ordered
     *   (`bx0 < bx1`, `by0 < by1`).
     */
    sampleField(
        zs: Float32Array | Float64Array | number[],
        gridW: number,
        gridH: number,
        bx0: number,
        by0: number,
        bx1: number,
        by1: number,
        outGrid: Float32Array | Float64Array
    ): number;

    /**
     * Release this handle back to its factory pool. The backing mesh arena lives
     * as long as the factory. Using or double-disposing a disposed handle throws.
     */
    dispose(): void;
}

/**
 * Builds a {@link FieldIndex} over the SoA coordinates `pxs` / `pys` (the first
 * `n` entries of each; `NaN`/`Infinity` legal, compacted out at build). `n` must
 * be `<= maxPoints` (throws otherwise). A build allocates one small handle facade
 * (~48 B, minor-GC-collectible); the mesh arena, scratch, index maps and z-gather
 * buffer are pooled -- 0 B beyond the facade.
 */
export type FieldIndexFactory = (
    pxs: ArrayLike<number>,
    pys: ArrayLike<number>,
    n: number
) => FieldIndex;

/**
 * Create a pooled field-index factory sized for up to `maxPoints` points.
 *
 * Mirrors {@link createCellIndex} exactly (pooled factory-factory, SoA input,
 * NaN compaction, ORIGINAL indices, generation-stamped facades) but, instead of
 * a Voronoi-cell surface, triangulates the mesh ONCE and answers point-location
 * and barycentric-interpolation queries with a remembering visibility walk. ONE
 * mesh serves MANY scalar fields: geometry is fixed at build, and each query
 * takes its `zs` per-call (indexed by ORIGINAL point index). A build allocates
 * one small handle facade (~48 B); everything else is pooled and every query is
 * 0 B. A new slot is allocated only at a new concurrent high-water mark. Per-slot
 * memory is `~100*maxPoints` bytes + 16 KB (no circumcenters).
 *
 * @param maxPoints hard upper bound on `n` per build. Must be a non-negative
 *   integer.
 * @throws if `maxPoints` is not a non-negative integer.
 */
export function createFieldIndex(maxPoints: number): FieldIndexFactory;

/**
 * A per-build cluster-index handle: boundary extraction (convex hull +
 * alpha-shape outlines) over one point group, built by a
 * {@link ClusterIndexFactory}. Built fresh per group per refresh by the
 * lite-charts `outlines` layer. All emitted indices are ORIGINAL site indices.
 *
 * Winding (the absolute convention, both senses): outer loops and the convex
 * hull are emitted counter-clockwise in SCREEN coordinates (+y down) =
 * clockwise in math coordinates (+y up); interior hole loops wind OPPOSITE;
 * outer and hole loops always wind oppositely.
 */
export interface ClusterIndex {
    /**
     * Write the ordered convex hull into `outIndices` and return the vertex
     * count `h` (`<= n`). A degenerate build (n < 3, collinear, all-duplicate)
     * returns `0` -- a value, not a throw. Validates `outIndices.length >= h`
     * BEFORE writing anything. Zero allocation.
     *
     * @param outIndices caller-owned `Int32Array`, safe bound `n`
     * @throws if the handle is disposed/stale, `outIndices` is not an
     *   `Int32Array`, or it is shorter than the counted hull
     */
    convexHull(outIndices: Int32Array): number;

    /**
     * Extract the alpha-shape boundary and return the loop count. Keeps every
     * triangle whose circumradius is `<= alpha` (`alpha` is a RADIUS in input
     * units); the boundary is the set of edges owned by exactly one kept
     * triangle, traced into closed loops. Loops are CONCATENATED into
     * `outIndices` (implicit closure last -> first); `outLoopEnds[i]` is the
     * EXCLUSIVE end offset of loop i. `0` loops is legal (alpha keeps nothing,
     * or a degenerate build). Interior hole loops are emitted (see the
     * interface doc for orientation). A triangle degenerate at the f64 noise
     * floor carries `NaN` circumradius and can never be kept (a merely thin
     * triangle keeps its honest huge-but-finite circumradius). Counts first and
     * validates BOTH buffers against the exact totals BEFORE writing (a
     * `3n` / `n`-sized caller can never trip the short-buffer throw). A large
     * FINITE alpha lawfully degenerates to the convex hull; `Infinity` itself
     * throws. Zero allocation.
     *
     * @param alpha finite radius > 0 in input units
     * @param outIndices caller-owned `Int32Array`, tight bound `3n - 6`, safe `3n`
     * @param outLoopEnds caller-owned `Int32Array`, tight bound `n - 2`, safe `n`
     * @throws if the handle is disposed/stale; `alpha` is not a finite number
     *   > 0 (`NaN`, `+/-0`, negatives, `Infinity` all refused); either buffer
     *   is not an `Int32Array` or is shorter than the counted need; or a
     *   boundary walk overruns its step cap
     */
    alphaShape(alpha: number, outIndices: Int32Array, outLoopEnds: Int32Array): number;

    /**
     * Release the handle and return its pooled slot to the factory. Any later
     * use of this handle (or any other facade of the same build) THROWS;
     * double-dispose THROWS.
     */
    dispose(): void;
}

/**
 * Builds a {@link ClusterIndex} over the SoA coordinates `pxs` / `pys` (the
 * first `n` entries of each; `NaN`/`Infinity` legal, compacted out at build).
 * `n` must be `<= maxPoints` (throws otherwise). A build allocates one small
 * handle facade (~48 B, minor-GC-collectible); the mesh arena, scratch, index
 * maps and cluster arrays are pooled -- 0 B beyond the facade.
 */
export type ClusterIndexFactory = (
    pxs: ArrayLike<number>,
    pys: ArrayLike<number>,
    n: number
) => ClusterIndex;

/**
 * Create a pooled cluster-index factory sized for up to `maxPoints` points.
 *
 * Mirrors {@link createFieldIndex} exactly (pooled factory-factory, SoA input,
 * NaN compaction, ORIGINAL indices, generation-stamped facades) but its surface
 * is boundary extraction: `convexHull` (the triangulator's own ordered hull
 * ring, O(h)) and `alphaShape` (keep circumradius <= alpha, trace the boundary
 * into concatenated loops). Sizing bounds for caller-owned buffers: `outIndices`
 * tight `3n - 6` / safe `3n`; `outLoopEnds` tight `n - 2` / safe `n`;
 * `convexHull` bound `n`. A build allocates one small handle facade (~48 B);
 * everything else is pooled and every query is 0 B. A new slot is allocated only
 * at a new concurrent high-water mark. Per-slot memory is `~116*maxPoints`
 * bytes + 16 KB (circumradii only, no circumcenters).
 *
 * @param maxPoints hard upper bound on `n` per build. Must be a non-negative
 *   integer.
 * @throws if `maxPoints` is not a non-negative integer.
 */
export function createClusterIndex(maxPoints: number): ClusterIndexFactory;
