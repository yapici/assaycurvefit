// ── Shared test fixtures ──────────────────────────────────────────
// Deterministic synthetic datasets. No RNG: the "noise" is a fixed
// repeating pattern so every run and every machine sees identical data
// and golden regression values stay stable.

/** Fixed pseudo-noise in [-1, 1); index-based, deterministic. */
export function noise(i) {
  return Math.sin(i * 12.9898) * 43758.5453 % 1;
}

/** Evaluate a 4PL at [A, B, C, D] = [Bottom, Hill, EC50, Top]. */
function eval4PL(x, [A, B, C, D]) {
  if (x <= 0) return A;
  return D + (A - D) / (1 + Math.pow(x / C, B));
}

/**
 * Build a dose-response dataset spanning `decades` log10 units centred on
 * the true EC50, with `nConc` concentrations and `nReps` replicates each.
 *
 * @param {object} opts
 * @param {number[]} opts.params  true 4PL [A, B, C, D]
 * @param {number} opts.nConc     concentrations (default 8)
 * @param {number} opts.nReps     replicates per concentration (default 3)
 * @param {number} opts.decades   log10 span around EC50 (default 4)
 * @param {number} opts.noiseAmp  additive noise amplitude in Y units (default 0)
 */
export function make4PLData({ params, nConc = 8, nReps = 3, decades = 4, noiseAmp = 0 }) {
  const [, , C] = params;
  const xData = [], yData = [];
  let k = 0;
  for (let i = 0; i < nConc; i++) {
    const logX = Math.log10(C) - decades / 2 + (decades * i) / (nConc - 1);
    const x = Math.pow(10, logX);
    for (let r = 0; r < nReps; r++) {
      xData.push(x);
      yData.push(eval4PL(x, params) + noiseAmp * noise(k++));
    }
  }
  return { xData, yData };
}

// Same curve shape expressed at three concentration scales. The fitted
// EC50/Hill/plateaus must be scale-equivariant: only the EC50 changes.
export const SCALES = {
  unit: 1,        // e.g. nM entered as plain numbers — well conditioned
  micromolar: 1e-6,
  nanomolar: 1e-9, // molar notation, the scale the absolute Jacobian step broke
};

/** Analytic gradient of the 4PL w.r.t. [A, B, C, D]. */
export function grad4PL(x, [A, B, C, D]) {
  if (x <= 0) return [1, 0, 0, 0];
  const u = Math.pow(x / C, B);
  const den = (1 + u) * (1 + u);
  return [
    1 / (1 + u),                             // dA
    -(A - D) * u * Math.log(x / C) / den,    // dB
    (A - D) * B * u / (C * den),             // dC
    u / (1 + u),                             // dD
  ];
}

// ── Seeded Gaussian noise ─────────────────────────────────────────
// The sin-based `noise` above is deterministic but not normally distributed,
// which is fine for "does it converge" tests and useless for checking that a
// 95% interval actually covers 95% of the time. These give reproducible
// Gaussian draws instead.

/** mulberry32: small, fast, well-distributed seeded PRNG. */
export function seededRandom(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draws via the Box-Muller transform. */
export function seededNormal(seed) {
  const rand = seededRandom(seed);
  let spare = null;
  return function next() {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = rand(); const v = rand();
    if (u < 1e-12) u = 1e-12; // log(0) guard
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

/** As make4PLData, but with genuine Gaussian noise from `seed`. */
export function make4PLDataGaussian({ params, nConc = 8, nReps = 3, decades = 4, sd = 0, seed = 1 }) {
  const normal = seededNormal(seed);
  const [, , C] = params;
  const xData = [], yData = [];
  for (let i = 0; i < nConc; i++) {
    const logX = Math.log10(C) - decades / 2 + (decades * i) / (nConc - 1);
    const x = Math.pow(10, logX);
    for (let r = 0; r < nReps; r++) {
      xData.push(x);
      yData.push(eval4PLExported(x, params) + sd * normal());
    }
  }
  return { xData, yData };
}

/** Exported evaluator so the Gaussian builder can share the 4PL definition. */
export function eval4PLExported(x, [A, B, C, D]) {
  if (x <= 0) return B > 0 ? A : D;
  return D + (A - D) / (1 + Math.pow(x / C, B));
}
