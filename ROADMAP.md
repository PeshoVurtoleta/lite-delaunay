# lite-delaunay -- roadmap

Written 2026-09-02, from the lite-charts side of the fence. Each release below
is sequenced by which CHART capability it unlocks -- the point of this package
inside the suite is to be the geometry engine that lite-charts (and later
lite-charts-gl) injects but never imports. Every item is one
`cd LiteDelaunay && claude` session under the standard pipeline
(planner -> coder -> reviewer -> qa) and the standard law: zero runtime deps,
zero alloc on any hot path, fail closed, ASCII-only, single Delaunay.js.

## Where 1.0.0 stands

`DelaunayTriangulator` -- one class, one method: `triangulate(coords, n)` over
a pre-allocated Int32Array arena, half-edge mesh output (`triangles`,
`halfedges`, `trianglesLen`). Degenerate inputs return 0 triangles (no throw,
no hang). That is a triangulation KERNEL, not yet a chart-facing capability:
lite-charts README names lite-delaunay as the intended default spatial index,
but 1.0.0 cannot satisfy the contract. Nothing in 1.0.0 changes below --
every release is additive on top of the mesh.

## The consumer contract that drives v1.1.0

lite-charts (Charts.js, "spatial index" notes around line 3860) hit-tests
dense bubble/scatter through an injected factory:

```
type SpatialIndexFactory = (pxs, pys, n) -> SpatialIndex

interface SpatialIndex {
  // Write up to k nearest indices (by pixel distance from qx,qy) into
  // outIndices/outDistSq, filtered to points within maxDistSq.
  // Returns the count written (0..k). Output arrays are caller-owned,
  // stable refs -- zero alloc per query.
  findNearest(qx, qy, k, maxDistSq, outIndices, outDistSq) -> number
  dispose() -> void
}
```

Gaps vs 1.0.0: (a) no query at all -- triangulate-only; (b) charts hand SoA
`pxs`/`pys`, triangulate wants interleaved `coords`; (c) charts rebuild the
index on every data change and dispose the old one; (d) k up to 8
(SPATIAL_INDEX_HIT_BUFFER_SIZE), maxDistSq-filtered, count-returning.

## v1.1.0 -- `createSpatialIndex` (unlocks: charts hover on dense bubble/scatter)

The release that makes `spatialIndex: createSpatialIndex(maxPoints)` a
one-line wire-up in a chart config.

- `export const createSpatialIndex = (maxPoints) => (pxs, pys, n) => index` --
  a factory-factory: the OUTER call allocates everything once (triangulator
  arena, an interleave scratch Float64Array of 2*maxPoints, neighbor-walk
  scratch, a seen-stamp Int32Array + generation counter); the INNER call
  (charts' rebuild-on-data-change) interleaves pxs/pys into the scratch,
  triangulates, and returns a handle with ZERO fresh allocation. Throws
  (fail closed) if n > maxPoints.
- `findNearest` = point location by remembering-stones mesh walk (start at the
  last-hit triangle, walk toward the query point -- near-O(1) for the coherent
  mouse-move queries charts actually issue), then k-NN by breadth-first
  expansion over half-edge vertex rings with the maxDistSq cutoff, insertion
  into the caller's outIndices/outDistSq kept sorted (k <= 8, insertion sort
  is right). 0 B/query.
- **Degenerate fallback is mandatory, not optional:** all-collinear or
  all-coincident input triangulates to 0 triangles. The contract still
  requires correct answers, so the handle must detect trianglesLen === 0 at
  build time and flip to a linear-scan findNearest over the same scratch.
  Fail closed means "never wrong", not "never slow".
- `dispose()` releases the handle back to the factory (clear the last-hit
  cursor, bump the generation stamp); the arena itself lives as long as the
  outer factory. Using a disposed handle throws.
- Honesty check for the planner: for pure radius-bounded k-NN a ~60-line
  uniform grid is simpler and usually faster. The delaunay walk earns its keep
  on wildly non-uniform point densities (where a grid degenerates) and because
  the mesh is ALREADY the substrate for everything below -- one structure,
  many capabilities. If the planner disagrees after measuring, ship the grid
  inside the same `createSpatialIndex` and keep the mesh walk for v1.2.0+;
  the exported contract must not change either way.
- Gate: contract-shape test mirroring the charts interface; a
  brute-force-vs-index equivalence sweep (random + clustered + collinear +
  coincident inputs); 0 B/query and pooled rebuilds (~48 B facade per build --
  what shipped; "stale use throws" rules out 0 B/rebuild) under the torture
  harness; coherent-query walk length bounded.

## v1.2.0 -- Voronoi cells (SHIPPED: `createCellIndex`)

Shaped by a real consumer: LiteCharts/briefs/voronoi-cells.md ("The consumer
contract") REPLACED the earlier sketch below. Rather than expose raw
circumcenters + an open/closed cell flag, v1.2.0 ships a single charts-facing
factory that does the whole job and hands back a polygon that is always safe to
draw:

- `export const createCellIndex = (maxPoints) => (pxs, pys, n) => cellIndex` --
  a pooled factory-factory mirroring `createSpatialIndex` exactly (SoA input,
  NaN compaction, ORIGINAL indices, generation-stamped ~48 B facade, one factory
  / many concurrent handles). The build triangulates ONCE and precomputes every
  circumcenter.
- `cell(i, bx0, by0, bx1, by1, outXY) -> vertexCount` -- walk the half-edges
  around site `i` and write its Voronoi polygon, CLIPPED to the axis-aligned
  bbox (zero-allocation Sutherland-Hodgman), into the caller-owned interleaved
  buffer. 0 B/query.
- Bbox clipping lives INSIDE the library (d3-delaunay's `voronoi(bounds)`
  precedent): every returned cell is finite, convex, closed and inside-or-on the
  bbox, or absent (returns 0). There is NO open-cell flag -- hull cells are
  closed by a provably-outside far-fan and clipped, so the caller never sees an
  unbounded ray. This is the correction to the old sketch's "flagged open cell".
- Near-degenerate circumcenters (d ~ 0) fail closed to the triangle centroid --
  no Infinity vertex, as the sketch already required.
- SIZING correction: a bbox-clipped cell of an interior site has at most
  degree + 4 vertices; a HULL site at most degree + 5 (the far-fan adds the
  extra corner). The brief's "degree + 4" bound is the interior case; hull cells
  need the +5. A 64-vertex caller buffer covers every non-adversarial cloud;
  overflow THROWS rather than truncates.
- What it enables in lite-charts: the scatter `cells` tessellation layer
  (cell-shaded scatter -- every reading owns a colored region, the classic
  station-map/coverage view) plus a free hover-cell highlight. "Fat" hover
  itself turned out to need NOTHING from delaunay (it is a charts-side
  tolerance policy over the existing k=1 spatial query -- see the brief's D1).

## v1.3.0 -- mesh interpolation (unlocks: contour + field charts from scattered data)

- `locate(qx, qy) -> triangleIndex | -1` (exposes the v1.1.0 walk as public
  API) and `barycentric(t, qx, qy, outW3)` -- together: zero-alloc scattered
  data interpolation `z(q) = w0*z[a] + w1*z[b] + w2*z[c]`.
- `sampleField(zValues, gridW, gridH, bbox, outGrid)` -- rasterize the
  interpolated surface into a caller-owned Float32Array grid in one call
  (walk-based location makes row-major sampling near-linear).
- What it enables: lite-charts contour/isoline charts and
  heatmap-from-scatter -- today's heatmap kernel needs a regular grid;
  sampleField turns irregular sensor/telemetry point clouds into exactly
  that grid, so the EXISTING grid kernel renders fields with no new chart
  kernel. Natural-neighbor interpolation is explicitly OUT (needs per-query
  virtual insertion; revisit only with a measured need).

## v1.4.0 -- hulls and outlines (unlocks: cluster overlays on scatter)

- `convexHull(outIndices) -> count` -- read off the halfedges === -1 boundary,
  ordered; free given the mesh.
- `alphaShape(alpha, outIndices) -> count` -- concave outline by dropping
  triangles with circumradius > 1/alpha, then boundary extraction. Enables
  cluster-outline annotations in lite-charts (feed the polygon straight into
  the v1.7.0 annotation layer as a range/point overlay set).

## Explicitly NOT planned

- Constrained Delaunay (holes/enforced edges) -- a different algorithm class;
  no chart in the pipeline needs it.
- Robust exact-arithmetic predicates -- 1.0.0's fast f64 predicates are the
  documented trade-off; charts feed pixel coordinates, well inside f64 comfort.
- Auto-growing arenas -- fixed capacity is the point; charts know n.
- 3D / d-dimensional -- out of suite scope.

## Sequencing note

v1.1.0 is the only release lite-charts is BLOCKED on (its README already
points at lite-delaunay as the default index; today that wiring is
impossible). v1.2.0-v1.4.0 each unlock a NEW chart capability and can be
scheduled on demand when the corresponding lite-charts brief is picked up --
write the matching lite-charts brief first, then cut the delaunay release it
needs, so the API is shaped by a real consumer, never speculatively.
