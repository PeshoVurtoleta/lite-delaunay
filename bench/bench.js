// Honest numbers for the README. Runs each size 5x and reports median.
import { DelaunayTriangulator } from "../Delaunay.js";

function makePoints(n, seed = 1) {
    let s = seed >>> 0;
    const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    const c = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { c[i*2] = rng() * 1000; c[i*2+1] = rng() * 1000; }
    return c;
}

function median(arr) {
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)];
}

function bench(n, runs = 7) {
    const coords = makePoints(n);
    const tri = new DelaunayTriangulator(n);

    // How many calls per timed batch so we get a reliable ms reading
    const batch = n < 500 ? 200 : n < 5000 ? 30 : n < 20000 ? 5 : 1;

    // Warmup
    for (let i = 0; i < 5; i++) tri.triangulate(coords, n);

    if (global.gc) global.gc();
    const heapBefore = process.memoryUsage().heapUsed;

    const times = [];
    for (let r = 0; r < runs; r++) {
        const t0 = performance.now();
        for (let b = 0; b < batch; b++) tri.triangulate(coords, n);
        times.push((performance.now() - t0) / batch);
    }

    if (global.gc) global.gc();
    const heapAfter = process.memoryUsage().heapUsed;

    const ms = median(times);
    const heapDelta = (heapAfter - heapBefore) / 1024;
    const opsPerSec = (1000 / ms).toFixed(0);
    const ptsPerSec = ((n / ms) * 1000 / 1e6).toFixed(1);

    return { n, ms: ms.toFixed(3), opsPerSec, ptsPerSec, heapDelta: heapDelta.toFixed(1) };
}

console.log("n\t\tms (median)\tops/sec\tMpts/sec\theap Δ (KB, 5 runs)");
console.log("─".repeat(80));
for (const n of [100, 1000, 5000, 10000, 50000, 100000]) {
    const r = bench(n);
    console.log(`${r.n}\t\t${r.ms}\t\t${r.opsPerSec}\t${r.ptsPerSec}\t\t${r.heapDelta}`);
}
