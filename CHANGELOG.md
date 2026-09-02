# Changelog

All notable changes to `@zakkster/lite-delaunay` are documented here. The format
is based on Keep a Changelog, and this project adheres to Semantic Versioning.

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
