// ── Replicate-based lack-of-fit test ──────────────────────────────
// Asks the question R² cannot: is the MODEL adequate, or is it missing
// structure the data actually contain?
//
// R² and RMSE conflate two different things -- noise in the measurement, and
// systematic deviation of the curve from the data. A noisy assay caps R² no
// matter how right the model is; a precise assay can show a poor R² for a
// perfect model, and an excellent one for a model that misses a real shoulder.
//
// Replicates separate the two. Replicates at the SAME concentration differ only
// by measurement error, so their scatter estimates pure error with no reference
// to the model at all. Whatever residual variation is left over is the model's
// fault:
//
//   SSR  =  SS(pure error)  +  SS(lack of fit)
//
// Each carries its own degrees of freedom, so the ratio of mean squares is an
// F-statistic. Under the null hypothesis "the model is correct", lack-of-fit
// variance is just more measurement error and F ~ 1. A significantly large F
// says the curve is systematically wrong somewhere, and the per-group
// breakdown says where.
//
// The test needs replicates (to estimate pure error at all) and more distinct
// concentrations than parameters (or the model can thread every group mean
// exactly, leaving nothing to test).

import { groupByConcentration } from "./stats.js";
import { fPValue } from "./distributions.js";

/**
 * Partition the residual sum of squares into pure error and lack of fit.
 *
 * @param {number[]} xData   Concentrations, one per observation.
 * @param {number[]} yData   Observed responses.
 * @param {number[]} yPred   Fitted values from the converged model.
 * @param {number} p         Number of FREE parameters the fit estimated.
 * @param {object} [options]
 * @param {number[]} [options.weights] Fitting weights. When the fit was
 *   weighted the decomposition must use the same metric, or the two sums of
 *   squares are in different units and their ratio is meaningless.
 * @param {number} [options.alpha] Significance level for the reported verdict.
 * @returns {object} `applicable` false with a `reason` when the design cannot
 *   support the test; otherwise the full decomposition.
 */
export function lackOfFitTest(xData, yData, yPred, p, options = {}) {
  const { weights = null, alpha = 0.05 } = options;

  const notApplicable = (reason) => ({
    applicable: false, reason,
    F: null, pValue: null, ssPureError: null, ssLackOfFit: null,
    dfPureError: null, dfLackOfFit: null, msPureError: null, msLackOfFit: null,
    sdPureError: null, significant: null, alpha, groups: [],
  });

  if (!xData || !yData || !yPred) return notApplicable("Lack-of-fit requires the raw data and fitted values.");
  const n = xData.length;
  if (n === 0) return notApplicable("No data.");

  const groups = groupByConcentration(xData, yData);
  const m = groups.length;

  const dfPureError = n - m;
  const dfLackOfFit = m - p;

  if (dfPureError <= 0) {
    return notApplicable(
      `Lack-of-fit needs replicate measurements: all ${n} points are at ` +
      `distinct concentrations, so there is no model-free estimate of pure error.`,
    );
  }
  if (dfLackOfFit <= 0) {
    return notApplicable(
      `Lack-of-fit needs more concentrations than parameters; the model has ` +
      `${p} free parameters and the design has only ${m} distinct ` +
      `concentrations, which it can reproduce exactly.`,
    );
  }

  const w = (i) => (weights ? weights[i] : 1);

  // Pure error: scatter of replicates about their own group mean. The mean is
  // weighted whenever the fit was, so that both sums of squares are formed
  // under the same metric.
  let ssPureError = 0;
  const groupDetail = [];
  for (const g of groups) {
    let sw = 0, swy = 0;
    for (const i of g.indices) { sw += w(i); swy += w(i) * yData[i]; }
    const gMean = sw > 0 ? swy / sw : g.mean;

    let sPure = 0;
    for (const i of g.indices) sPure += w(i) * (yData[i] - gMean) ** 2;
    ssPureError += sPure;

    // The group's contribution to lack of fit: how far the CURVE sits from the
    // group mean, which is the part of the residual the replicates cannot
    // explain. This is what localises a misfit to a dose.
    let sLof = 0;
    for (const i of g.indices) sLof += w(i) * (gMean - yPred[i]) ** 2;

    groupDetail.push({
      x: g.x, n: g.n, mean: g.mean, weightedMean: gMean,
      predicted: yPred[g.indices[0]],
      deviation: gMean - yPred[g.indices[0]],
      ssPureError: sPure, ssLackOfFit: sLof,
    });
  }

  const ssr = yData.reduce((s, y, i) => s + w(i) * (y - yPred[i]) ** 2, 0);
  // Algebraically SSR - SSPE, but summing the group terms is the numerically
  // better-behaved route: the subtraction cancels catastrophically on a good
  // fit, and can land a whisker below zero.
  const ssLackOfFit = Math.max(0, groupDetail.reduce((s, g) => s + g.ssLackOfFit, 0));

  if (ssPureError <= 0) {
    return notApplicable(
      "Replicates at every concentration are identical, giving a pure-error " +
      "estimate of zero. This usually means the values are means rather than " +
      "individual measurements; the test needs the raw replicates.",
    );
  }

  const msPureError = ssPureError / dfPureError;
  const msLackOfFit = ssLackOfFit / dfLackOfFit;
  const F = msLackOfFit / msPureError;
  const pValue = fPValue(F, dfLackOfFit, dfPureError);

  return {
    applicable: true, reason: null,
    F, pValue, alpha,
    significant: pValue < alpha,
    ssr, ssPureError, ssLackOfFit,
    dfPureError, dfLackOfFit,
    msPureError, msLackOfFit,
    // The replicate-only estimate of measurement noise, in response units.
    // Unlike Sy.x it does not depend on the model being right, so it is the
    // honest floor: no model can fit better than this.
    sdPureError: Math.sqrt(msPureError),
    groups: groupDetail.sort((a, b) => b.ssLackOfFit - a.ssLackOfFit),
  };
}

/**
 * One-line interpretation of a lack-of-fit result, for display.
 *
 * Phrased around the decision the user has to make. A significant result is
 * not "a bad fit" -- it means the deviation from the curve is larger than the
 * assay's own noise, which in a precise assay can happen while R² is still
 * 0.999 and the curve looks perfect by eye.
 */
export function describeLackOfFit(result) {
  if (!result || !result.applicable) return result?.reason ?? null;
  if (result.significant) {
    return `The curve deviates from the replicate means by more than measurement ` +
      `error (F = ${result.F.toPrecision(3)}, p = ${result.pValue.toPrecision(2)}). ` +
      `The model is missing real structure; check the largest-deviation ` +
      `concentrations before trusting the parameters.`;
  }
  return `No detectable lack of fit (F = ${result.F.toPrecision(3)}, ` +
    `p = ${result.pValue.toPrecision(2)}): the scatter about the curve is ` +
    `consistent with the assay's own replicate noise.`;
}
