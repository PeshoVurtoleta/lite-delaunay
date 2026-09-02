// Side-by-side: lite-delaunay (zero-GC) vs Mapbox Delaunator (per-call allocation)
import { DelaunayTriangulator } from "../Delaunay.js";
import Delaunator from "delaunator";

function makePoints(n, seed = 1) {
    let s = seed >>> 0;
    const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    const c = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { c[i*2] = rng() * 1000; c[i*2+1] = rng() * 1000; }
    return c;
}

function median(arr) { const s = [...arr].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; }

function bench(n) {
    const coords = makePoints(n, n);
    const batch = n < 500 ? 200 : n < 5000 ? 30 : n < 20000 ? 5 : 1;
    const runs = 7;

    // ── lite-delaunay ──
    const tri = new DelaunayTriangulator(n);
    for (let i = 0; i < 5; i++) tri.triangulate(coords, n);  // warmup

    if (global.gc) global.gc();
    let heapBefore = process.memoryUsage().heapUsed;
    const liteTimes = [];
    for (let r = 0; r < runs; r++) {
        const t0 = performance.now();
        for (let b = 0; b < batch; b++) tri.triangulate(coords, n);
        liteTimes.push((performance.now() - t0) / batch);
    }
    if (global.gc) global.gc();
    const liteHeap = (process.memoryUsage().heapUsed - heapBefore) / 1024;

    // ── Mapbox Delaunator ──
    // Note: Delaunator.from() takes [[x,y],...] but `new Delaunator(flat)` is fast path
    const dCoords = new Float64Array(coords);  // Delaunator wants Float64
    for (let i = 0; i < 5; i++) new Delaunator(dCoords);  // warmup

    if (global.gc) global.gc();
    heapBefore = process.memoryUsage().heapUsed;
    const mapTimes = [];
    for (let r = 0; r < runs; r++) {
        const t0 = performance.now();
        for (let b = 0; b < batch; b++) new Delaunator(dCoords);
        mapTimes.push((performance.now() - t0) / batch);
    }
    if (global.gc) global.gc();
    const mapHeap = (process.memoryUsage().heapUsed - heapBefore) / 1024;

    return {
        n,
        liteMs: median(liteTimes).toFixed(3),
        mapMs: median(mapTimes).toFixed(3),
        liteHeap: liteHeap.toFixed(1),
        mapHeap: mapHeap.toFixed(1),
        ratio: (median(mapTimes) / median(liteTimes)).toFixed(2),
    };
}

console.log("n\t| lite-delaunay\t\t\t| Mapbox Delaunator\t\t| lite is");
console.log("\t| ms (median)\theap Δ (KB)\t| ms (median)\theap Δ (KB)\t|");
console.log("─".repeat(95));
for (const n of [100, 1000, 5000, 10000, 50000, 100000]) {
    const r = bench(n);
    console.log(`${r.n}\t| ${r.liteMs}\t\t${r.liteHeap}\t\t| ${r.mapMs}\t\t${r.mapHeap}\t\t| ${r.ratio}× speed`);
}
