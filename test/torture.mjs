// test/torture.mjs -- node --expose-gc test/torture.mjs
// ---------------------------------------------------------------------------
// The mandatory zero-GC gate for @zakkster/lite-delaunay's spatial index.
//
// Phase 1 (retention): 4096 factory build/dispose cycles, each tracked by
//   @zakkster/lite-leak inside a lite-signal owner. Clean owner disposal
//   auto-untracks; after gc + settle, tracker.size() must return to 0 with no
//   audit findings. The owner-cascade and collection-growth kernels watch for
//   orphaned owners and an unbounded pool.
// Phase 2 (allocation): one steady-state handle, ~200k coherent + jump queries
//   over a skewed clustered input with NaN holes, gated to major=0 / pause<=4ms.
// Phase 2b (rebuild churn): after warm-up, 2000 build/dispose cycles must not
//   grow heapUsed beyond noise (< 64 KB) -- proves the pool reuses slots.
//
// No gate output is a FAIL. process.exitCode = 1 on any FAIL.

import { GcProfiler, checkNoGc } from '@zakkster/lite-gc-profiler';
import {
  createLeakTracker,
  createOwnerCascadeOrphanKernel,
  createCollectionGrowthKernel,
} from '@zakkster/lite-leak';
import { effect } from '@zakkster/lite-signal';

import { createSpatialIndex } from '../Delaunay.js';

const CYCLES = 4096;
const HOT = 200000;
const N = 5000;
const MAXP = 5000;
const K = 8;

// --- skewed clustered input with ~5% NaN (log-scale / missing-data holes) ---
// A 10^4:1 density skew: a few tight clusters near the origin plus rare far
// outliers, so a naive uniform grid would degenerate -- exactly the input the
// index must survive.
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(0xC0FFEE);
const pxs = new Float32Array(N);
const pys = new Float32Array(N);
for (let i = 0; i < N; i++) {
  const c = i % 8;
  let x = (c * 1.3) + rng() * 2.0;     // dense clusters, ~sub-pixel spread
  let y = (c * 1.1) + rng() * 2.0;
  if (rng() < 0.01) { x = rng() * 2e4; y = rng() * 2e4; }  // 10^4:1 far outliers
  if (rng() < 0.05) { x = NaN; }        // ~5% NaN holes
  pxs[i] = x; pys[i] = y;
}
const maxDistSq = 64 * 64;  // charts' prMaxSq: a plausible max bubble radius^2

// ---------------------------------------------------------------------------
// leak tracker + kernels
// ---------------------------------------------------------------------------
const leaks = [];
const warns = [];
const tracker = createLeakTracker({
  name: 'delaunay-spatial-torture',
  onLeak: (r) => leaks.push(r.kind + ':' + String(r.tag)),
  onWarning: (w) => warns.push(w.kind + ':' + w.reason),
});
tracker.registerKernel(createOwnerCascadeOrphanKernel());

// slotWatch reflects the pool's high-water mark: a handle is recorded once, by
// identity, and never removed. Correct pooling keeps this at the concurrency
// HWM (== 1 for serial churn); a pool that leaked a slot per build would grow
// it without bound. Seed it with one real cycle so the growth kernel (which
// rejects empty collections) has something to watch.
const factory = createSpatialIndex(MAXP);
const slotWatch = [];
{
  const h = factory(pxs, pys, N);
  slotWatch.push(h._slot);   // watch the pooled SLOT, not the per-build facade
  h.dispose();
}
tracker.registerKernel(createCollectionGrowthKernel({
  collections: [slotWatch],
  window: 8,
  minSamples: 4,
}));

// ---------------------------------------------------------------------------
// phase 1: retention torture
// ---------------------------------------------------------------------------
for (let i = 0; i < CYCLES; i++) {
  // effect(fn) returns a disposer; track() inside its body auto-registers
  // onCleanup(untrack), so disposing the effect drives size() back down.
  const dispose = effect(() => {
    const handle = factory(pxs, pys, N);
    // Record pool HWM by SLOT identity (reused slot -> already present -> no
    // push); the facade is fresh each build and would grow this unboundedly.
    const slot = handle._slot;
    let known = false;
    for (let s = 0; s < slotWatch.length; s++) {
      if (slotWatch[s] === slot) { known = true; break; }
    }
    if (!known) slotWatch.push(slot);
    // Held-value contract: capture a DETACHED primitive (the generation
    // stamp), never the handle itself.
    const gen = handle._gen | 0;
    tracker.track(handle, () => { void gen; }, 'spatial', { audit: true });
    handle.dispose();
  });
  dispose();  // owner disposal -> onCleanup(untrack) -> size decrements
}

globalThis.gc?.();
await new Promise((r) => setTimeout(r, 50));

const live = tracker.size();
const findings = tracker.audit();

// ---------------------------------------------------------------------------
// phase 2: allocation + GC torture (steady-state handle, stepped queries)
// ---------------------------------------------------------------------------
const outI = new Int32Array(K);
const outD = new Float32Array(K);
const gc = new GcProfiler().start();

const index = factory(pxs, pys, N);  // built ONCE, outside the loop
let mx = 5, my = 5;
for (let i = 0; i < HOT; i++) {
  // coherent mouse-like drift...
  mx += (rng() - 0.5) * 3;
  my += (rng() - 0.5) * 3;
  // ...with periodic random jumps (defeats the walk's coherence assumption).
  if ((i & 1023) === 0) { mx = rng() * 40; my = rng() * 40; }
  index.findNearest(mx, my, K, maxDistSq, outI, outD);
  if ((i & 8191) === 0) {
    gc.sampleHeap(performance.now(), process.memoryUsage().heapUsed);
  }
}
index.dispose();

await new Promise((r) => setTimeout(r, 50));
const s = gc.summary();
const report = checkNoGc(s, { maxMajor: 0, maxPauseMs: 4 });
gc.stop();

// ---------------------------------------------------------------------------
// phase 2b: rebuild churn -- each build allocates one ~48 B facade (young-gen
// garbage), but the pooled arenas must not grow. After churn + gc + settle the
// facades are collected, so a heap delta vs the pre-churn baseline beyond noise
// (< 64 KB) means a real retention leak.
// ---------------------------------------------------------------------------
for (let i = 0; i < 8; i++) { const h = factory(pxs, pys, N); h.dispose(); }  // warm-up
globalThis.gc?.();
await new Promise((r) => setTimeout(r, 20));
const heapBefore = process.memoryUsage().heapUsed;
for (let i = 0; i < 2000; i++) {
  const h = factory(pxs, pys, N);
  h.findNearest(mx, my, K, maxDistSq, outI, outD);
  h.dispose();
}
globalThis.gc?.();
await new Promise((r) => setTimeout(r, 50));
globalThis.gc?.();
const heapAfter = process.memoryUsage().heapUsed;
const rebuildBytes = heapAfter - heapBefore;
const REBUILD_LIMIT = 64 * 1024;

// ---------------------------------------------------------------------------
// gate
// ---------------------------------------------------------------------------
const ok = report.ok &&
  live === 0 &&
  leaks.length === 0 &&
  findings.length === 0 &&
  rebuildBytes < REBUILD_LIMIT;

console.log(
  'GATE leak=size ' + live + '/0 findings=' + findings.length +
  ' warnings=' + warns.length +
  ' | gc major=' + s.gc.major + ' minor=' + s.gc.minor +
  ' maxMs=' + s.gc.maxMs.toFixed(2) +
  ' | rebuild=' + rebuildBytes + ' bytes | ' + (ok ? 'ok' : 'FAIL')
);

if (!ok) {
  for (const v of report.violations) {
    console.error('  violation ' + v.metric + ' limit=' + v.limit + ' actual=' + v.actual);
  }
  for (const f of findings) console.error('  finding ' + f.kind + ':' + (f.reason || ''));
  for (const l of leaks) console.error('  leak ' + l);
  if (rebuildBytes >= REBUILD_LIMIT) {
    console.error('  rebuild-growth ' + rebuildBytes + ' >= ' + REBUILD_LIMIT + ' bytes');
  }
  process.exitCode = 1;
}
