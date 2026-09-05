// Honest numbers for the README. Runs each size 5x and reports median.
import {
    DelaunayTriangulator,
    createSpatialIndex,
    createCellIndex,
    createFieldIndex,
    createClusterIndex,
} from "../Delaunay.js";

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

// ---------------------------------------------------------------------------
// Pooled index surfaces: createSpatialIndex (v1.1.0), createCellIndex
// (v1.2.0), createFieldIndex (v1.3.0). SoA input, same LCG family as the core
// bench. Build numbers are WARM rebuilds (the pool is already at its
// high-water mark; the first build pays the arena, exactly as documented).
// Query loops run steady-state on one live handle with caller-owned out
// buffers, so heap deltas show the per-query allocation story.
// ---------------------------------------------------------------------------

function makeSoA(n, seed = 1) {
    let s = seed >>> 0;
    const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    const pxs = new Float32Array(n);
    const pys = new Float32Array(n);
    for (let i = 0; i < n; i++) { pxs[i] = rng() * 1000; pys[i] = rng() * 1000; }
    return { pxs, pys };
}

function makeQueries(count, seed) {
    let s = seed >>> 0;
    const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
    const qxs = new Float64Array(count);
    const qys = new Float64Array(count);
    for (let i = 0; i < count; i++) { qxs[i] = rng() * 1000; qys[i] = rng() * 1000; }
    return { qxs, qys };
}

let sink = 0; // query results accumulate here; printed at the end to defeat DCE

function benchWarmBuild(factory, pxs, pys, n, reps = 7) {
    factory(pxs, pys, n).dispose(); // reach the pool high-water mark first
    const times = [];
    for (let r = 0; r < reps; r++) {
        const t0 = performance.now();
        const h = factory(pxs, pys, n);
        times.push(performance.now() - t0);
        h.dispose();
    }
    return median(times);
}

// Median ops/sec over `runs` timed batches of `ops` calls each.
function timeOps(fn, ops, runs = 5) {
    fn(Math.min(ops, 10000)); // warmup
    const times = [];
    for (let r = 0; r < runs; r++) {
        const t0 = performance.now();
        fn(ops);
        times.push(performance.now() - t0);
    }
    return ops / (median(times) / 1000);
}

const QN = 65536;
const QMASK = QN - 1;

function benchSpatialIdx(n) {
    const { pxs, pys } = makeSoA(n, n ^ 0x2545f491);
    const factory = createSpatialIndex(n);
    const buildMs = benchWarmBuild(factory, pxs, pys, n);
    const h = factory(pxs, pys, n);

    const outI = new Int32Array(8);
    const outD = new Float32Array(8);
    const { qxs, qys } = makeQueries(QN, 0xdecafbad);
    // maxDistSq 4e6 covers the whole 1000x1000 domain -> true k-NN, no cutoff.
    const runK = (k) => (ops) => {
        let acc = 0;
        for (let q = 0; q < ops; q++) {
            acc += h.findNearest(qxs[q & QMASK], qys[q & QMASK], k, 4e6, outI, outD);
        }
        sink += acc;
    };

    if (global.gc) global.gc();
    const heapBefore = process.memoryUsage().heapUsed;
    const k1 = timeOps(runK(1), 200000);
    const k8 = timeOps(runK(8), 200000);
    if (global.gc) global.gc();
    const heapDelta = ((process.memoryUsage().heapUsed - heapBefore) / 1024).toFixed(1);

    h.dispose();
    return {
        n,
        buildMs: buildMs.toFixed(3),
        k1: (k1 / 1e6).toFixed(2),
        k8: (k8 / 1e6).toFixed(2),
        heapDelta,
    };
}

function benchCellIdx(n) {
    const { pxs, pys } = makeSoA(n, n ^ 0x5bd1e995);
    const factory = createCellIndex(n);
    const buildMs = benchWarmBuild(factory, pxs, pys, n);
    const h = factory(pxs, pys, n);

    // 64-vertex capacity; the degree+5 hull rule never comes near this on a
    // random cloud. Full-domain bbox so hull cells exercise the clip path.
    const outXY = new Float64Array(128);

    // One untimed full pass: warms the walk and measures the mean cell size.
    let verts = 0;
    for (let i = 0; i < n; i++) verts += h.cell(i, 0, 0, 1000, 1000, outXY);
    const avgVerts = (verts / n).toFixed(1);

    let cursor = 0;
    const run = (ops) => {
        let acc = 0;
        let i = cursor;
        for (let q = 0; q < ops; q++) {
            acc += h.cell(i, 0, 0, 1000, 1000, outXY);
            i++;
            if (i === n) i = 0;
        }
        cursor = i;
        sink += acc;
    };

    if (global.gc) global.gc();
    const heapBefore = process.memoryUsage().heapUsed;
    const cellsPerSec = timeOps(run, 200000);
    if (global.gc) global.gc();
    const heapDelta = ((process.memoryUsage().heapUsed - heapBefore) / 1024).toFixed(1);

    h.dispose();
    return {
        n,
        buildMs: buildMs.toFixed(3),
        cells: (cellsPerSec / 1e6).toFixed(2),
        avgVerts,
        heapDelta,
    };
}

function benchFieldIdx(n) {
    const { pxs, pys } = makeSoA(n, n ^ 0x27d4eb2f);
    const factory = createFieldIndex(n);
    const buildMs = benchWarmBuild(factory, pxs, pys, n);
    const h = factory(pxs, pys, n);

    const zs = new Float64Array(n);
    for (let i = 0; i < n; i++) zs[i] = Math.sin(pxs[i] * 0.01) + Math.cos(pys[i] * 0.01);

    // Coherent drift: the per-frame probe / isoline case the remembering walk
    // is built for. Queries near the domain edge fall outside the hull and
    // return NaN -- that is the honest consumer mix, counted at full cost.
    let qx = 500, qy = 500, dx = 2.31, dy = 1.77;
    const runCoherent = (ops) => {
        let acc = 0;
        for (let q = 0; q < ops; q++) {
            qx += dx; if (qx < 1 || qx > 999) { dx = -dx; qx += dx + dx; }
            qy += dy; if (qy < 1 || qy > 999) { dy = -dy; qy += dy + dy; }
            const z = h.interpolate(zs, qx, qy);
            if (z === z) acc += z;
        }
        sink += acc;
    };

    // Random jumps: worst case for a remembering walk (O(sqrt T) crossings).
    const { qxs, qys } = makeQueries(QN, 0xfeedc0de);
    const runRandom = (ops) => {
        let acc = 0;
        for (let q = 0; q < ops; q++) {
            const z = h.interpolate(zs, qxs[q & QMASK], qys[q & QMASK]);
            if (z === z) acc += z;
        }
        sink += acc;
    };

    // sampleField: 64x64 grid, bbox inset so most cells land inside the hull.
    const grid = new Float32Array(64 * 64);
    const runGrids = (ops) => {
        let acc = 0;
        for (let g = 0; g < ops; g++) {
            acc += h.sampleField(zs, 64, 64, 100, 100, 900, 900, grid);
        }
        sink += acc;
    };

    if (global.gc) global.gc();
    const heapBefore = process.memoryUsage().heapUsed;
    const coherent = timeOps(runCoherent, 200000);
    const random = timeOps(runRandom, 50000);
    const gridsPerSec = timeOps(runGrids, 64);
    if (global.gc) global.gc();
    const heapDelta = ((process.memoryUsage().heapUsed - heapBefore) / 1024).toFixed(1);

    h.dispose();
    return {
        n,
        buildMs: buildMs.toFixed(3),
        coherent: (coherent / 1e6).toFixed(2),
        random: (random / 1e6).toFixed(2),
        gridCells: (gridsPerSec * 4096 / 1e6).toFixed(1),
        heapDelta,
    };
}

const IDX_SIZES = [1000, 10000, 100000];

console.log("\ncreateSpatialIndex -- warm rebuild + true k-NN queries");
console.log("-".repeat(80));
console.log("n\t\trebuild ms\tk=1 Mq/s\tk=8 Mq/s\theap d (KB)");
for (const n of IDX_SIZES) {
    const r = benchSpatialIdx(n);
    console.log(`${r.n}\t\t${r.buildMs}\t\t${r.k1}\t\t${r.k8}\t\t${r.heapDelta}`);
}

console.log("\ncreateCellIndex -- warm rebuild + bbox-clipped Voronoi cell extraction");
console.log("-".repeat(80));
console.log("n\t\trebuild ms\tMcells/s\tavg verts\theap d (KB)");
for (const n of IDX_SIZES) {
    const r = benchCellIdx(n);
    console.log(`${r.n}\t\t${r.buildMs}\t\t${r.cells}\t\t${r.avgVerts}\t\t${r.heapDelta}`);
}

console.log("\ncreateFieldIndex -- warm rebuild + interpolate + 64x64 sampleField");
console.log("-".repeat(80));
console.log("n\t\trebuild ms\tdrift Mq/s\tjump Mq/s\tgrid Mcells/s\theap d (KB)");
for (const n of IDX_SIZES) {
    const r = benchFieldIdx(n);
    console.log(`${r.n}\t\t${r.buildMs}\t\t${r.coherent}\t\t${r.random}\t\t${r.gridCells}\t\t${r.heapDelta}`);
}

function benchClusterIdx(n) {
    const { pxs, pys } = makeSoA(n, n ^ 0x1b873593);
    const factory = createClusterIndex(n);
    const buildMs = benchWarmBuild(factory, pxs, pys, n);
    const h = factory(pxs, pys, n);

    // Caller-owned buffers at the documented safe bounds (3n / n).
    const outI = new Int32Array(3 * n);
    const outE = new Int32Array(n);
    // Alpha ~ 3x the mean point spacing of the 1000x1000 cloud: keeps the dense
    // interior, drops sliver fringe -- the realistic outline setting.
    const alpha = 3 * (1000 / Math.sqrt(n));

    const runHull = (ops) => {
        let acc = 0;
        for (let q = 0; q < ops; q++) acc += h.convexHull(outI);
        sink += acc;
    };
    const runAlpha = (ops) => {
        let acc = 0;
        for (let q = 0; q < ops; q++) acc += h.alphaShape(alpha, outI, outE);
        sink += acc;
    };

    // convexHull is O(h); alphaShape is O(T) per call -- scale the batch so each
    // timed run stays in a reliable-milliseconds regime.
    const alphaOps = n <= 1000 ? 20000 : n <= 10000 ? 2000 : 200;

    if (global.gc) global.gc();
    const heapBefore = process.memoryUsage().heapUsed;
    const hullPerSec = timeOps(runHull, 200000);
    const alphaPerSec = timeOps(runAlpha, alphaOps);
    if (global.gc) global.gc();
    const heapDelta = ((process.memoryUsage().heapUsed - heapBefore) / 1024).toFixed(1);

    h.dispose();
    return {
        n,
        buildMs: buildMs.toFixed(3),
        hull: (hullPerSec / 1e6).toFixed(2),
        alphaShape: alphaPerSec >= 1e6 ? (alphaPerSec / 1e6).toFixed(2) : (alphaPerSec / 1e3).toFixed(1) + "k",
        heapDelta,
    };
}

// The charts per-refresh unit: ONE factory sized for the largest group, then
// per group build -> convexHull -> alphaShape -> dispose, fresh every refresh.
function benchClusterCycle(n, factory, poolPx, poolPy) {
    const outI = new Int32Array(3 * n);
    const outE = new Int32Array(n);
    const alpha = 2.5 * (1000 / Math.sqrt(n));

    const run = (ops) => {
        let acc = 0;
        for (let q = 0; q < ops; q++) {
            const handle = factory(poolPx, poolPy, n);
            acc += handle.convexHull(outI);
            acc += handle.alphaShape(alpha, outI, outE);
            handle.dispose();
        }
        sink += acc;
    };

    const ops = n <= 64 ? 20000 : 10000;
    if (global.gc) global.gc();
    const heapBefore = process.memoryUsage().heapUsed;
    const cyclesPerSec = timeOps(run, ops);
    if (global.gc) global.gc();
    const heapDelta = ((process.memoryUsage().heapUsed - heapBefore) / 1024).toFixed(1);

    return {
        n,
        usPerCycle: (1e6 / cyclesPerSec).toFixed(2),
        cyclesPerSec: (cyclesPerSec / 1e3).toFixed(1) + "k",
        heapDelta,
    };
}

console.log("\ncreateClusterIndex -- warm rebuild + convexHull / alphaShape queries");
console.log("-".repeat(80));
console.log("n\t\trebuild ms\thull Mq/s\talphaShape q/s\theap d (KB)");
for (const n of IDX_SIZES) {
    const r = benchClusterIdx(n);
    console.log(`${r.n}\t\t${r.buildMs}\t\t${r.hull}\t\t${r.alphaShape}\t\t${r.heapDelta}`);
}

console.log("\ncreateClusterIndex -- small-n full cycle (build -> hull -> alpha -> dispose)");
console.log("-".repeat(80));
console.log("n\t\tus/cycle\tcycles/s\theap d (KB)\t<- charts' per-group refresh unit");
{
    const CYCLE_MAX = 256;
    const { pxs, pys } = makeSoA(CYCLE_MAX, 0x85ebca6b);
    const cycleFactory = createClusterIndex(CYCLE_MAX);
    cycleFactory(pxs, pys, CYCLE_MAX).dispose(); // reach the pool high-water mark
    for (const n of [8, 16, 32, 64, 128, 256]) {
        const r = benchClusterCycle(n, cycleFactory, pxs, pys);
        console.log(`${r.n}\t\t${r.usPerCycle}\t\t${r.cyclesPerSec}\t\t${r.heapDelta}`);
    }
}

// A read of sink keeps every query loop observable -- V8 cannot elide them.
console.log(`\n(checksum ${sink.toFixed(2)} -- keeps the loops live, value meaningless)`);
