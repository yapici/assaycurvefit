// ── Beyond the Wald interval ──────────────────────────────────────
// The standard errors in inference.js come from linearising the model at the
// solution. That linearisation is exact for a model linear in its parameters
// and an approximation for everything else -- and the 4PL is emphatically
// everything else.
//
// Where the approximation fails, it fails in a specific and unhelpful
// direction: it produces intervals that are SYMMETRIC and too NARROW. A
// plateau the doses never reach has a likelihood surface that is a steep wall
// on one side and a long flat valley on the other; a parabola fitted at the
// bottom of that has no way to express it, so it reports a tidy +/- that
// understates the upside badly.
//
// Two ways out, with different costs and different failure modes:
//
//   Profile likelihood -- walk each parameter away from its estimate,
//     re-optimising everything else at each step, and find where the sum of
//     squares rises past a threshold. Follows the actual shape of the
//     likelihood, so it gives asymmetric intervals and can honestly report
//     that one side is unbounded. Deterministic. Costs tens of refits per
//     parameter.
//
//   Bootstrap -- resample the residuals, refit, and read the interval off the
//     percentiles of the resulting distribution. Makes no assumption about the
//     shape of anything, and additionally exposes bias. Costs hundreds of
//     refits, and is random, so it is seeded here: the same data must give the
//     same interval twice.
//
// Both are opt-in. The Wald interval is right often enough, and cheap enough,
// to stay the default.

import {
  levenbergMarquardt, sumSquaredResiduals, fitModel, withLogParams, makeFitConstraint,
} from "./lm.js";
import { tInv } from "./distributions.js";
import { groupByConcentration } from "./stats.js";

/** mulberry32: small seeded PRNG, so a bootstrap is reproducible. */
function seededRandom(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Wrap a model so parameter `index` is held at `value` and hidden from the optimiser. */
function withFixedParam(modelFn, index, value) {
  return (x, free) => {
    const full = [];
    let f = 0;
    for (let i = 0; i <= free.length; i++) full.push(i === index ? value : free[f++]);
    return modelFn(x, full);
  };
}

const dropIndex = (arr, index) => arr.filter((_, i) => i !== index);
const insertAt = (arr, index, value) => {
  const out = [...arr];
  out.splice(index, 0, value);
  return out;
};

/**
 * Minimum sum of squares achievable with parameter `index` pinned at `value`.
 *
 * Warm-started from the previous point on the profile. That continuation is
 * not an optimisation detail: restarting cold at each step lets the optimiser
 * jump to a different branch of the likelihood surface, which puts a
 * discontinuity in the profile and produces an interval endpoint that is not
 * where the threshold is actually crossed.
 */
function profileAt({ xData, yData, modelFn, index, value, startFree, weights, constrain }) {
  if (startFree.length === 0) {
    const full = insertAt([], index, value);
    return { ssr: sumSquaredResiduals(xData, yData, modelFn, full, weights), free: [] };
  }
  const fixedFn = withFixedParam(modelFn, index, value);
  const innerConstrain = constrain
    ? (proposed, prev) => {
        const full = constrain(insertAt(proposed, index, value), insertAt(prev, index, value));
        return dropIndex(full, index);
      }
    : null;
  const r = levenbergMarquardt(xData, yData, fixedFn, startFree, {
    weights, constrain: innerConstrain,
  });
  return { ssr: r.ssr, free: r.params };
}

/**
 * Profile-likelihood confidence interval for one parameter.
 *
 * The interval is every value of the parameter whose best achievable sum of
 * squares stays under
 *
 *   SSR_min * (1 + t^2(alpha/2, n-p) / (n-p))
 *
 * which is the standard profile-t region. Where the model IS linear this
 * reduces exactly to the Wald interval, which is the check the tests pin it
 * against.
 *
 * @returns {{lo, hi, loBounded, hiBounded, threshold, evaluations}}
 *   A null endpoint with `bounded` false means the sum of squares never rose
 *   past the threshold in that direction: the data do not bound the parameter
 *   on that side, which is a finding rather than a failure.
 */
export function profileInterval({
  xData, yData, modelFn, params, index, ssr, alpha = 0.05,
  weights = null, constrain = null, scale = null,
  maxSteps = 40, maxBisect = 40, tolerance = 1e-4,
}) {
  const n = xData.length;
  const p = params.length;
  const dof = n - p;
  const blank = {
    lo: null, hi: null, loBounded: false, hiBounded: false,
    threshold: null, evaluations: 0,
  };
  if (dof <= 0 || !isFinite(ssr) || ssr < 0) return blank;

  const t = tInv(alpha / 2, dof);
  const threshold = ssr * (1 + (t * t) / dof);
  const theta0 = params[index];
  const startFree = dropIndex(params, index);

  // Step scale: the Wald standard error when the caller has one, since the
  // profile interval is usually the same order of magnitude. Falls back to the
  // parameter's own magnitude.
  const step0 = scale && isFinite(scale) && scale > 0
    ? scale
    : Math.max(Math.abs(theta0) * 0.1, 1e-6);

  let evaluations = 0;
  const evaluate = (value, warm) => {
    evaluations++;
    return profileAt({ xData, yData, modelFn, index, value, startFree: warm, weights, constrain });
  };

  const search = (sign) => {
    let step = 0.5 * step0;
    let insideTheta = theta0;
    let warm = startFree;

    // ── Bracket: walk out geometrically until the threshold is crossed ──
    let outsideTheta = null;
    for (let i = 0; i < maxSteps; i++) {
      const value = insideTheta + sign * step;
      if (!isFinite(value)) break;
      const r = evaluate(value, warm);
      if (!isFinite(r.ssr) || r.ssr > threshold) { outsideTheta = value; break; }
      insideTheta = value;
      warm = r.free;
      step *= 1.7; // geometric, so a flat valley is crossed in a few dozen steps
    }
    if (outsideTheta === null) return { value: null, bounded: false };

    // ── Bisect between the last inside point and the first outside one ──
    let lo = insideTheta, hi = outsideTheta;
    let insideWarm = warm;
    for (let i = 0; i < maxBisect; i++) {
      if (Math.abs(hi - lo) <= tolerance * step0) break;
      const mid = (lo + hi) / 2;
      const r = evaluate(mid, insideWarm);
      if (isFinite(r.ssr) && r.ssr <= threshold) { lo = mid; insideWarm = r.free; }
      else hi = mid;
    }
    return { value: (lo + hi) / 2, bounded: true };
  };

  const down = search(-1);
  const up = search(+1);

  return {
    lo: down.value, hi: up.value,
    loBounded: down.bounded, hiBounded: up.bounded,
    threshold, evaluations,
  };
}

/**
 * Profile-likelihood intervals for every parameter.
 *
 * @param {(number|null)[]} [se] Wald standard errors, used only to scale the
 *   search. Passing them makes the search converge in far fewer refits.
 */
export function profileIntervals({ params, se = null, ...rest }) {
  return params.map((_, i) =>
    profileInterval({ ...rest, params, index: i, scale: se ? se[i] : null }),
  );
}

/**
 * Bootstrap confidence intervals by resampling residuals.
 *
 * The residuals are resampled, added back to the FITTED values, and the model
 * refitted; the spread of the refits is the sampling distribution. Because the
 * residuals are attached to the fitted curve rather than to the observations,
 * this holds the design fixed, which is what you want for a dose-response
 * assay where the concentrations are chosen rather than sampled.
 *
 * Weighted fits resample the WEIGHTED residuals sqrt(w) * r, which are the
 * ones the model treats as identically distributed, and rescale on the way
 * back. Resampling raw residuals under a weighted fit would redistribute the
 * heteroscedasticity at random across the dose range.
 *
 * @param {object} options
 * @param {"residual"|"stratified"} [options.method] "stratified" resamples
 *   within each concentration group, preserving a variance structure that the
 *   weights do not describe. Needs replicates; falls back with a warning.
 * @param {number} [options.nBoot] Resamples. 1000+ for a publishable interval;
 *   the default trades a little precision in the percentile for interactivity.
 * @param {number} [options.seed] Fixed by default: the same data must give the
 *   same interval every time it is fitted.
 */
export function bootstrapIntervals({
  xData, yData, modelFn, params, alpha = 0.05,
  weights = null, constrain = null,
  nBoot = 400, seed = 20240101, method = "residual",
}) {
  const n = xData.length;
  const p = params.length;
  const blank = (warning) => ({
    ci: params.map(() => null), se: params.map(() => null),
    bias: params.map(() => null), draws: [], nSuccess: 0, nBoot, method, warning,
  });
  if (n <= p) return blank("Bootstrap needs more observations than parameters.");

  const yPred = xData.map(x => modelFn(x, params));
  const rootW = weights ? weights.map(Math.sqrt) : null;
  // Scaled residuals: the quantity the fit treats as identically distributed.
  const scaled = yData.map((y, i) => (y - yPred[i]) * (rootW ? rootW[i] : 1));

  // Residuals from a least-squares fit are shrunk relative to the true errors
  // by roughly sqrt(1 - p/n), because p degrees of freedom went into the fit.
  // Inflating restores the scale; without it the bootstrap interval is
  // systematically narrow, which is the failure it exists to avoid.
  const inflation = Math.sqrt(n / (n - p));
  const pool = scaled.map(r => r * inflation);

  // Which residuals each observation may draw from.
  let donors;
  let warning = null;
  if (method === "stratified") {
    const groups = groupByConcentration(xData, yData);
    if (groups.every(g => g.n >= 2)) {
      donors = new Array(n);
      for (const g of groups) for (const i of g.indices) donors[i] = g.indices;
    } else {
      warning = "Stratified bootstrap needs replicates at every concentration; " +
        "resampled from the pooled residuals instead.";
      donors = null;
    }
  }

  const rand = seededRandom(seed);
  const draws = [];
  let failures = 0;

  for (let b = 0; b < nBoot; b++) {
    const ySim = new Array(n);
    for (let i = 0; i < n; i++) {
      const from = donors ? donors[i] : null;
      const j = from
        ? from[Math.floor(rand() * from.length)]
        : Math.floor(rand() * n);
      ySim[i] = yPred[i] + pool[j] / (rootW ? rootW[i] : 1);
    }
    const r = levenbergMarquardt(xData, ySim, modelFn, params, { weights, constrain });
    if (r.converged && r.params.every(v => isFinite(v))) draws.push(r.params);
    else failures++;
  }

  if (draws.length < 20) {
    return blank(
      `Bootstrap did not converge often enough to form an interval ` +
      `(${draws.length} of ${nBoot} resamples succeeded).`,
    );
  }

  const quantile = (sorted, q) => {
    // Type-7 (linear interpolation), the convention R and NumPy use.
    const h = (sorted.length - 1) * q;
    const lo = Math.floor(h), hi = Math.ceil(h);
    return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
  };

  const ci = [], seOut = [], bias = [];
  for (let k = 0; k < p; k++) {
    const col = draws.map(d => d[k]).sort((a, b) => a - b);
    const mean = col.reduce((s, v) => s + v, 0) / col.length;
    const variance = col.reduce((s, v) => s + (v - mean) ** 2, 0) / (col.length - 1);
    ci.push({ lo: quantile(col, alpha / 2), hi: quantile(col, 1 - alpha / 2) });
    seOut.push(Math.sqrt(variance));
    // Bootstrap mean minus the original estimate. A bias comparable to the
    // standard error means the estimator itself is skewed here, and the
    // symmetric Wald interval is centred in the wrong place.
    bias.push(mean - params[k]);
  }

  return {
    ci, se: seOut, bias, draws,
    nSuccess: draws.length, nBoot, method: donors ? "stratified" : "residual",
    warning: warning ?? (failures > 0
      ? `${failures} of ${nBoot} resamples failed to converge and were discarded.`
      : null),
  };
}

// ── Entry point ───────────────────────────────────────────────────
// Kept here rather than folded into fitModel for two reasons. The honest one
// is cost: these intervals are hundreds of refits, and that should be a
// decision the caller makes rather than a surprise inside the default path.
// The structural one is that resampling has to re-run the optimiser, so
// putting it inside lm.js would make the two modules import each other.

const EC50_INDEX = 2;

/**
 * Fit, then compute intervals that do not assume the likelihood is a parabola.
 *
 * Takes the same arguments as `fitModel` and returns the same object with
 * extra fields, so it can be swapped in wherever a more careful interval is
 * wanted.
 *
 * All the resampling happens in FITTING space -- with log10(EC50) in place of
 * EC50 -- for the same reason the original fit does: it is the space where the
 * problem is well conditioned. Profile and percentile intervals are both
 * equivariant under a monotone transformation, so raising the endpoints to the
 * power of ten gives exactly the interval that would have been obtained by
 * profiling the EC50 itself, and it cannot come back negative.
 *
 * Weights are held at the values the IRLS loop converged on. The intervals are
 * therefore conditional on the weighting, which is the usual convention: the
 * alternative is to re-derive weights inside every resample, which propagates
 * the weights' own noise into the interval and is rarely what is meant.
 *
 * @param {object} [options]
 * @param {"profile"|"bootstrap"|"both"|"none"} [options.intervals]
 * @param {object} [options.bootstrapOptions] Passed to bootstrapIntervals.
 * @returns The fit result plus `profile` and/or `bootstrap`, each carrying a
 *   `ci` array aligned with the caller-visible `params`.
 */
export function fitModelWithIntervals(xData, yData, modelFn, is5PL, options = {}) {
  const {
    intervals = "profile", alpha = 0.05, bootstrapOptions = {}, ...fitOptions
  } = options;

  const fit = fitModel(xData, yData, modelFn, is5PL, fitOptions);
  if (!fit || intervals === "none") return fit;

  // Reconstruct the space the optimiser searched. toLog is the exact inverse
  // of the toLinear that produced `fit.params`, so this recovers the solution
  // rather than approximating it.
  const log = withLogParams(modelFn, [EC50_INDEX]);
  const fitSpaceParams = log.toLog(fit.params);
  const constrain = makeFitConstraint(is5PL);
  const weights = fit.weighting?.weights ?? null;
  // Whatever was actually minimised: the weighted sum when weights are in play.
  const ssr = weights && fit.wssr != null ? fit.wssr : fit.ssr;

  const shared = {
    xData, yData, modelFn: log.modelFn, params: fitSpaceParams,
    alpha, weights, constrain,
  };

  // Back to the caller's linear scale. Only the EC50 slot is transformed, and
  // an unbounded endpoint stays unbounded.
  const toLinearCi = (ci, slot) => {
    if (!ci) return null;
    const map = (v) => (v == null ? null : slot === EC50_INDEX ? Math.pow(10, v) : v);
    return { lo: map(ci.lo), hi: map(ci.hi) };
  };

  const out = { ...fit };

  if (intervals === "profile" || intervals === "both") {
    // Wald standard errors scale the search. In fitting space that means the
    // log-space error for the EC50, not the delta-method one on the linear scale.
    const fitSpaceSe = fit.se
      ? fit.se.map((v, i) => (i === EC50_INDEX ? fit.logEC50?.se ?? null : v))
      : null;
    const raw = profileIntervals({ ...shared, se: fitSpaceSe, ssr });
    out.profile = {
      ci: raw.map((r, i) => toLinearCi(r, i)),
      bounded: raw.map(r => ({ lo: r.loBounded, hi: r.hiBounded })),
      threshold: raw[0]?.threshold ?? null,
      evaluations: raw.reduce((s, r) => s + r.evaluations, 0),
      alpha,
    };
  }

  if (intervals === "bootstrap" || intervals === "both") {
    const boot = bootstrapIntervals({ ...shared, ...bootstrapOptions });
    out.bootstrap = {
      ...boot,
      ci: boot.ci.map((c, i) => toLinearCi(c, i)),
      // The log-space draws are the meaningful ones for the EC50; the linear
      // standard error would describe a distribution that is not symmetric.
      se: boot.se,
      bias: boot.bias,
      draws: undefined, // the raw draws are large; recompute if they are wanted
    };
  }

  return out;
}
