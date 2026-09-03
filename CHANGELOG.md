# Changelog

All notable changes to `@zakkster/lite-delaunay` are documented here. The format
is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [1.3.0] - 2026-09-03

### Added

- `createFieldIndex(maxPoints)` -- a pooled field-index factory (0 B/query; ~48 B
  facade per build) for scattered-data point location and scalar-field
  interpolation. The returned function `(pxs, pys, n) -> FieldIndex` triangulates
  the Delaunay mesh ONCE; the handle exposes `locate`, `barycentric`,
  `triangleVertices`, `triangleCount`, `interpolate`, `sampleField` and
  `dispose`. One mesh serves many scalar fields: geometry is fixed at build, and
  each query takes its `zs` per-call, indexed by ORIGINAL point index.
  - Point location is a remembering visibility (straight) walk over the half-edge
    mesh, starting from the last hit triangle (coherent O(1) amortised, cold
    O(sqrt(T))). The walk provably terminates on a Delaunay mesh; a step budget
    covers f64-degenerate cycling by falling back to an O(T) linear scan -- never
    wrong, maybe slow.
  - `sampleField(zs, gridW, gridH, bx0, by0, bx1, by1, outGrid) -> finiteCount`
    rasterizes the field onto a regular grid. GRID CONTRACT: row-major,
    `index = row*gridW + col`, col 0 at `bx0` (xMin), row 0 at `by0` (yMin) --
    MATHEMATICAL +y-up orientation, deliberately NOT screen convention (a
    pixel-space consumer flips rows and derives its own present-mask, on its own
    cold path). Cell-CENTER sampling; cells outside the hull get `NaN`; a
    degenerate build NaN-fills the `gridW*gridH` prefix and returns `0`.
  - Fail closed everywhere (NaN is the absent value, never 0 -- 0 is a legal
    field value): a non-finite or outside-the-hull query yields `-1` / `NaN`; a
    degenerate build yields `-1` / `NaN` / `triangleCount() === 0`; a near-zero
    area triangle makes `barycentric` write `NaN,NaN,NaN` and return `false`; a
    `NaN` z-value propagates arithmetically, confined to its incident triangles.
    Out-of-range / NaN queries are VALUES, never throws. Disposed / stale handle
    use, `t` out of `[0, triangleCount())`, a too-short out-array, a missing /
    short `zs`, non-positive `gridW`/`gridH`, a wrong-type / too-short `outGrid`,
    and a non-finite or non-strictly-ordered bbox all THROW.
  - Pooling, facade and generation semantics are IDENTICAL to `createCellIndex`:
    one factory serves many concurrent live handles, builds acquire a pooled slot
    and disposes return it, a new slot is allocated only at a new concurrent
    high-water mark, every query is 0 B, and a stale or double-disposed handle
    throws. Per-slot memory is `~100*maxPoints` bytes + 16 KB (no circumcenters).
  - `triangleVertices` and `triangleCount` reserve per-triangle site access for
    future contour / TIN work.
  - Torture harness extended with a field-index rebuild storm (build/dispose
    interleaved with `locate`/`interpolate`/`sampleField`, lite-leak tracked) and
    a field steady-state phase (one build, ~200k walk/interpolate queries plus
    256 `sampleField` rasterizations, gated to `major=0`).

## [1.2.0] - 2026-09-03

### Added

- `createCellIndex(maxPoints)` -- a pooled cell-index factory (0 B/query; ~48 B
  facade per build) that satisfies the `CellIndexFactory` contract
  `@zakkster/lite-charts` injects to draw a Voronoi tessellation over a projected
  scatter (see LiteCharts/briefs/voronoi-cells.md). The returned function
  `(pxs, pys, n) -> CellIndex` builds the Delaunay mesh and precomputes every
  triangle circumcenter once at build time; the handle exposes
  `cell(i, bx0, by0, bx1, by1, outXY) -> vertexCount` and `dispose()`.
  - `cell(i)` writes site `i`'s Voronoi polygon, CLIPPED to the caller's
    axis-aligned bbox (zero-allocation Sutherland-Hodgman on fixed scratch),
    into the caller-owned interleaved `outXY` and returns the vertex count. `i`
    is the ORIGINAL point index. Every returned polygon is finite, convex,
    closed (last vertex implicitly connects to the first) and fully inside-or-on
    the bbox. Zero allocation per call.
  - Hull cells (unbounded in the true diagram) are CLOSED by three synthetic far
    points, anchored at the boundary circumcenters and split by the angle
    bisector, that provably fall outside the bbox -- so the clip yields
    cell-intersect-bbox exactly and adjacent cells tile the bbox with no seams
    (far-fan clip exactness). Hull cells are clipped, never flagged.
  - Fail closed everywhere: a non-finite input point, a degenerate build (fewer
    than 3 finite points, or all-collinear / all-coincident), an EPSILON-dedup
    duplicate that owns no mesh vertex, or a cell that misses the bbox all return
    `0` (never a garbage polygon). Near-degenerate circumcenters fall back to the
    triangle centroid -- no `Infinity`/`NaN` vertex is ever stored. Out-of-range
    `i`, a non-finite / non-strictly-ordered bbox, an `outXY` too small for the
    clipped cell, and disposed / stale handle use all THROW (never truncate).
  - Pooling, facade and generation semantics are identical to
    `createSpatialIndex`: one factory serves many concurrent live handles,
    builds acquire a pooled slot and disposes return it, a new slot is allocated
    only at a new concurrent high-water mark, `cell()` is 0 B/query, and a stale
    or double-disposed handle throws.
  - SIZING RULE (documented): a bbox-clipped Voronoi cell of an interior site has
    at most `degree + 4` vertices, of a hull site at most `degree + 5`; a
    caller-owned buffer of `2 * 64` floats covers every non-adversarial cloud,
    and overflow throws rather than truncates.
- `test/torture.mjs` gains a cell-index rebuild storm (build/dispose cycles
  interleaved with `cell()` queries; `tracker.size()` back to 0) and a
  steady-state phase (~200k `cell()` queries, 0 major GC), folded into the single
  GATE line. The existing spatial-index budgets are unchanged.

## [1.1.1] - 2026-09-02

### Fixed

- Docs: the lite-charts wiring example used the config key
  `spatialIndexFactory:`; the real key is `spatialIndex:`. Fixed in README.md,
  llms.txt and the `createSpatialIndex` JSDoc example. Found by the lite-charts
  integration session while wiring the live demo. No code change.

### Changed

- The root-level correctness probe `Delaunay.test.js` (halfedge twin pairing +
  the Delaunay in-circle property + exact triangle counts on known inputs) is
  ported to `node:test` as `test/DelaunayProperty.test.js`, ASCII-normalized,
  and wired into `npm test` (7 tests). The root file is removed. Repo-only:
  the probe was never part of the published tarball.

## [1.1.0] - 2026-09-02

### Added

- `createSpatialIndex(maxPoints)` -- a pooled spatial-index factory
  (0 B/query; see facade note below) that satisfies the `SpatialIndexFactory` contract
  `@zakkster/lite-charts` injects for hit-testing dense bubble/scatter point
  clouds. The returned function `(pxs, pys, n) -> SpatialIndex` builds a uniform
  grid over the SoA pixel coordinates; the handle exposes
  `findNearest(qx, qy, k, maxDistSq, outIndices, outDistSq) -> count` and
  `dispose()`.
  - `findNearest` writes ORIGINAL point indices and SQUARED pixel distances into
    the caller-owned output arrays, sorted nearest-first, with INCLUSIVE
    (`d <= maxDistSq`) containment to match the charts linear-scan semantics.
    Zero allocation per query.
  - NaN / Infinity points are compacted out at build time and can never be
    returned; a non-finite query returns `0` (fail closed).
  - Degenerate input (all-coincident or all-collinear) is handled by the grid
    itself, which degrades to a small-cell / linear scan -- never wrong, maybe
    slow. `triangulate()` is not involved.
  - One factory serves many concurrent live handles (one per series in a
    multi-series chart), disposed independently. Builds acquire a pooled slot
    and disposes return it; a new slot is allocated only at a new concurrent
    high-water mark. A build allocates exactly one small handle facade (~48 B,
    young-gen, minor-GC-collectible); all arenas, grids and scratch are pooled
    -- 0 B beyond the facade -- and `findNearest` is 0 B/query. The facade
    carries a generation stamp: using or double-disposing a disposed handle
    throws, and a stale handle onto a rebuilt slot throws rather than silently
    returning the new build's points.
- `VERSION` export (`"1.1.0"`), kept in lockstep with `package.json` and
  `llms.txt`.
- `test/torture.mjs` -- the mandatory zero-GC gate (retention via
  `@zakkster/lite-leak`, allocation + rebuild-churn via
  `@zakkster/lite-gc-profiler`). Wired to `npm run torture`; `npm run verify`
  now runs test + torture + bench.

### Changed

- Normalized pre-existing non-ASCII glyphs in source comments to ASCII
  equivalents (`--`, `->`, `-`, `ceil(sqrt(N))`, `~`, `<=`, `>=`, `^2`).
  Comment-only; zero behavior change. (`U+00D7` and `U+00B5` are retained per
  the ASCII-source law.)

### Fixed

- Test/bench imports now reference `Delaunay.js` (was `Delaunay.d.ts`).

All 1.0.0 exports and behavior are unchanged; this release is purely additive.

## [1.0.0]

### Added

- Initial release. `DelaunayTriangulator` -- a pre-allocated, zero-GC 2D
  Delaunay triangulator (Mapbox-style sweepline algorithm) over a static
  `Int32Array` arena. One allocation at construction; `triangulate(coords, n)`
  writes into the same arena on every call with no per-frame garbage. Half-edge
  mesh output (`triangles`, `halfedges`, `trianglesLen`). Degenerate inputs
  (fewer than 3 points, all-coincident, all-collinear) return 0 without throwing
  or hanging.
