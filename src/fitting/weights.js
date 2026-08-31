// ── Regression weights ────────────────────────────────────────────
// Assay response is typically heteroscedastic: replicate spread grows with
// signal. Unweighted least squares therefore lets the high end of the curve
// dominate the fit, and fits the low end -- where the interesting potency
// differences usually live -- comparatively poorly.
//
// Weighting corrects that by dividing each squared residual by an estimate of
// that point's variance. Which estimate is right is an empirical question
// about the assay, not a universal constant:
//
//   none    variance constant            (homoscedastic)
//   1/Y     variance proportional to Y   (counting/Poisson-like noise)
//   1/Y^2   variance proportional to Y^2 (constant CV) -- the usual default
//           for ligand-binding and immunoassay work
//   1/SD^2  variance measured directly from the replicates at each dose
//
// 1/Y^2 is the common recommendation, but it is a hypothesis: it is correct
// when the coefficient of variation is constant. With enough replicates,
// 1/SD^2 tests that assumption instead of assuming it.

import { groupByConcentration } from "./stats.js";

export const WEIGHTING_TYPES = ["none", "1/Y", "1/Y^2", "1/SD^2"];

// A fitted baseline within this many replicate standard deviations of zero is
// not resolved from it, making relative weighting inadvisable. See
// baselineCaution.
const BASELINE_SD_MULTIPLE = 3;

// Fallback threshold, as a fraction of the peak response, used only when there
// are no replicates from which to measure noise at the low end.
const BASELINE_CAUTION_FRACTION = 0.05;

/**
 * Build the per-observation weight vector for a fit.
 *
 * Relative weights are computed from the PREDICTED values, never the observed
 * ones. Weighting by observed y biases the fit downward: a point that happened
 * to read low gets a larger weight purely for having been noisy in one
 * direction. Using the curve makes the weights a property of the model, which
 * is why this has to be iterated (see fitModel's IRLS loop).
 *
 * @param {string} type      One of WEIGHTING_TYPES.
 * @param {object} ctx
 * @param {number[]} ctx.yPred  Current fitted values.
 * @param {number[]} ctx.xData  Needed only for 1/SD^2 grouping.
 * @param {number[]} ctx.yData  Needed only for 1/SD^2 grouping.
 * @returns {{weights: number[]|null, warning: string|null}}
 *   `weights` is null when weighting is not applicable; `warning` then says
 *   why. Callers should fall back to an unweighted fit and surface the reason
 *   rather than silently fitting something the user did not ask for.
 */
export function buildWeights(type, { yPred, xData, yData }) {
  if (!type || type === "none") return { weights: null, warning: null };

  if (!WEIGHTING_TYPES.includes(type)) {
    return { weights: null, warning: `Unknown weighting type "${type}".` };
  }

  if (type === "1/SD^2") return sdWeights(xData, yData);

  // Two distinct failure modes, which deserve different treatment.
  //
  // Hard refusal: a fitted value at or below zero makes 1/Y and 1/Y^2
  // arithmetically undefined or sign-flipped. There is no fit to be had.
  //
  // Caution: a strictly positive but near-zero baseline is computable, yet
  // leans hard on the constant-CV assumption exactly where it is least
  // plausible -- real assays have a noise floor, so absolute error does not
  // shrink to nothing as the signal does. The fit proceeds and says so.
  const scale = Math.max(...yPred.map(Math.abs));
  if (!isFinite(scale) || scale === 0) {
    return { weights: null, warning: "Fitted curve is degenerate; cannot weight." };
  }

  const lowest = Math.min(...yPred);
  if (lowest <= 0) {
    return {
      weights: null,
      warning:
        `${type} weighting needs a strictly positive fitted curve, but it ` +
        `reaches ${lowest.toPrecision(3)}. This usually means the response was ` +
        `normalised or background-subtracted to a zero baseline; fit the ` +
        `raw response instead, or use unweighted or 1/SD^2 weighting.`,
    };
  }

  const power = type === "1/Y^2" ? 2 : 1;
  const weights = yPred.map(v => 1 / Math.pow(v, power));

  return {
    weights,
    warning: baselineCaution({ type, lowest, scale, power, xData, yData }),
  };
}

/**
 * Is the fitted baseline actually resolved from zero?
 *
 * Relative weighting says a point's variance shrinks in proportion to its
 * expected value, so points near a zero baseline are treated as almost
 * noiseless and dominate the fit. Real assays have a noise floor, so that is
 * false, and the consequence is measurable: on curves whose true baseline is
 * zero, 1/Y^2 pulls the fitted EC50 to roughly 0.75x its true value, while
 * unweighted fits of the same data recover it.
 *
 * The right comparison is against the noise where it matters, not against a
 * fixed fraction of the peak. A baseline of 10 is well resolved in an assay
 * whose replicates scatter by 1.5, and not resolved at all in one where they
 * scatter by 20 -- and a peak-relative rule cannot tell those apart, because
 * on heteroscedastic data the residual spread at the top of the curve says
 * nothing about the spread at the bottom.
 *
 * So the noise estimate is taken from the replicate group nearest the
 * baseline. Without replicates there is nothing to measure, and the rule falls
 * back to a peak-relative heuristic.
 *
 * This warns rather than refuses. Whether the constant-CV assumption holds is
 * a fact about the assay, established during development, not something to be
 * settled from a single plate -- see estimateVariancePower.
 */
function baselineCaution({ type, lowest, scale, power, xData, yData }) {
  const ratio = Math.round(Math.pow(scale / lowest, power));
  const tail =
    `so the lowest points carry ${ratio}x the weight of the highest. ` +
    `Relative weighting assumes a constant CV, which breaks down at a zero ` +
    `baseline; consider 1/SD^2 or an unweighted fit.`;

  let noise = null;
  if (xData && yData) {
    const replicated = groupByConcentration(xData, yData).filter(g => g.n >= 2 && g.sd > 0);
    if (replicated.length) {
      const nearest = replicated.reduce((best, g) =>
        Math.abs(g.mean - lowest) < Math.abs(best.mean - lowest) ? g : best);
      noise = nearest.sd;
    }
  }

  if (noise !== null) {
    if (lowest < BASELINE_SD_MULTIPLE * noise) {
      return `${type} weighting applied, but the fitted baseline ` +
        `(${lowest.toPrecision(3)}) is within ${BASELINE_SD_MULTIPLE} replicate ` +
        `SDs of zero (SD ${noise.toPrecision(3)} nearby), so it is not resolved ` +
        `from zero -- ${tail}`;
    }
    return null;
  }

  if (lowest < scale * BASELINE_CAUTION_FRACTION) {
    return `${type} weighting applied, but the fitted baseline ` +
      `(${lowest.toPrecision(3)}) is under ${BASELINE_CAUTION_FRACTION * 100}% of ` +
      `the peak response (${scale.toPrecision(3)}) and there are no replicates ` +
      `from which to judge the noise there -- ${tail}`;
  }
  return null;
}

/**
 * Weights from the observed replicate variance at each concentration.
 *
 * This is the only scheme that measures the variance structure rather than
 * assuming it, but it needs enough replicates to estimate an SD at all. With
 * few replicates the SD is itself so noisy that the weights add variance
 * rather than removing it, so require at least three per dose.
 */
function sdWeights(xData, yData) {
  if (!xData || !yData) {
    return { weights: null, warning: "1/SD^2 weighting requires the raw data." };
  }
  const groups = groupByConcentration(xData, yData);

  const thin = groups.filter(g => g.n < 3);
  if (thin.length) {
    return {
      weights: null,
      warning:
        `1/SD^2 weighting needs at least 3 replicates per concentration; ` +
        `${thin.length} of ${groups.length} have fewer. Use 1/Y^2 instead, ` +
        `which assumes a constant CV rather than measuring it.`,
    };
  }

  const flat = groups.filter(g => g.sd <= 0);
  if (flat.length) {
    return {
      weights: null,
      warning:
        `1/SD^2 weighting is undefined where replicates are identical ` +
        `(zero SD at ${flat.length} concentration(s)).`,
    };
  }

  const weights = new Array(xData.length).fill(1);
  for (const g of groups) {
    const w = 1 / (g.sd * g.sd);
    for (const i of g.indices) weights[i] = w;
  }
  return { weights, warning: null };
}

/** Weighted sum of squared residuals, the quantity a weighted fit minimises. */
export function weightedSSR(residuals, weights) {
  if (!weights) return residuals.reduce((s, r) => s + r * r, 0);
  return residuals.reduce((s, r, i) => s + weights[i] * r * r, 0);
}

/**
 * Have the weights stopped moving?
 *
 * Compared on a relative scale, because absolute weight magnitudes depend
 * entirely on the units of the response.
 */
export function weightsConverged(a, b, tol = 1e-6) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.abs(a[i]) + Math.abs(b[i]);
    if (denom > 0 && (2 * Math.abs(a[i] - b[i])) / denom > tol) return false;
  }
  return true;
}

/**
 * Estimate the variance structure of an assay from its replicates.
 *
 * Choosing a weighting scheme is choosing a hypothesis about how variance
 * scales with signal. With replicates, that hypothesis is measurable rather
 * than assumed: fit a power-of-the-mean model
 *
 *   SD  =  a * mean^theta        =>       log SD  =  log a + theta * log mean
 *
 * by regressing log SD on log mean across concentration groups. The exponent
 * names the appropriate scheme:
 *
 *   theta ~ 0     SD constant             -> unweighted
 *   theta ~ 0.5   variance ~ mean         -> 1/Y
 *   theta ~ 1     variance ~ mean^2       -> 1/Y^2   (constant CV)
 *
 * This is the approach recommended for ligand-binding calibration curves
 * (determine the weighting function during assay development, then confirm it
 * at validation) rather than reaching for 1/Y^2 by default.
 *
 * Treat the result as a guide, not a verdict: with a handful of concentrations
 * and a few replicates each, theta is imprecisely determined. `se` and `groups`
 * say how much weight the estimate itself deserves.
 *
 * @returns {{theta, se, recommended, groups, warning}} `theta` is null when it
 *   cannot be estimated, with `warning` explaining why.
 */
export function estimateVariancePower(xData, yData) {
  const nothing = (warning) => ({
    theta: null, se: null, recommended: null, groups: 0, warning,
  });

  if (!xData || !yData) return nothing("Variance structure requires the raw data.");

  // Only groups with enough replicates for a meaningful SD, and with both
  // mean and SD strictly positive so the logs exist.
  const usable = groupByConcentration(xData, yData)
    .filter(g => g.n >= 3 && g.sd > 0 && g.mean > 0);

  if (usable.length < 3) {
    return nothing(
      `Estimating the variance structure needs at least 3 concentrations with ` +
      `3+ replicates each and a positive mean; found ${usable.length}.`,
    );
  }

  const lx = usable.map(g => Math.log(g.mean));
  const ly = usable.map(g => Math.log(g.sd));
  const n = usable.length;
  const mx = lx.reduce((s, v) => s + v, 0) / n;
  const my = ly.reduce((s, v) => s + v, 0) / n;
  const sxx = lx.reduce((s, v) => s + (v - mx) ** 2, 0);
  if (sxx <= 0) return nothing("All concentration groups share the same mean response.");

  const sxy = lx.reduce((s, v, i) => s + (v - mx) * (ly[i] - my), 0);
  const theta = sxy / sxx;
  const intercept = my - theta * mx;

  // Standard error of the slope, so callers can see how well determined it is.
  const resid = ly.map((v, i) => v - (intercept + theta * lx[i]));
  const se = n > 2
    ? Math.sqrt(resid.reduce((s, r) => s + r * r, 0) / (n - 2) / sxx)
    : null;

  // Snap to the nearest scheme the fitter actually offers.
  const candidates = [
    { theta: 0, type: "none" },
    { theta: 0.5, type: "1/Y" },
    { theta: 1, type: "1/Y^2" },
  ];
  const recommended = candidates
    .reduce((best, c) => (Math.abs(c.theta - theta) < Math.abs(best.theta - theta) ? c : best))
    .type;

  return { theta, se, recommended, groups: n, warning: null };
}
