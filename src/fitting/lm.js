// ── Levenberg-Marquardt curve fitting ─────────────────────────────
// Damped least-squares optimiser plus the multi-start wrappers that drive it.

import { model4PL, makeConstrainedModel, computeBiologicalEC50 } from "./models.js";
import { matMul, matTranspose, solveLU } from "./linalg.js";
import { rSquared, computeAIC, computeAICc, computeBIC } from "./stats.js";
import {
  parameterCovariance, parameterIntervals, correlationMatrix, backTransformLog10,
} from "./inference.js";
import { buildWeights, weightsConverged, estimateVariancePower } from "./weights.js";
import { lackOfFitTest } from "./lackoffit.js";
import { identifiabilityWarnings } from "./identifiability.js";

export function residuals(xData, yData, modelFn, params) {
  return xData.map((x, i) => yData[i] - modelFn(x, params));
}

/**
 * Sum of squared residuals, optionally weighted.
 *
 * With weights this is the quantity a weighted fit minimises. It is NOT
 * comparable across different weighting schemes -- the units change with the
 * weights -- so never compare information criteria between a weighted and an
 * unweighted fit.
 */
export function sumSquaredResiduals(xData, yData, modelFn, params, weights = null) {
  const r = residuals(xData, yData, modelFn, params);
  if (!weights) return r.reduce((s, v) => s + v * v, 0);
  return r.reduce((s, v, i) => s + weights[i] * v * v, 0);
}

// Optimal relative step for a central difference is ~cbrt(machine epsilon).
const CBRT_EPS = Math.cbrt(Number.EPSILON); // ~6.06e-6

/**
 * Numerical Jacobian by central differences.
 *
 * The step is RELATIVE to each parameter's own magnitude. A fixed absolute
 * step cannot work here because the parameters are not commensurate: a molar
 * EC50 is ~1e-9 while a luminescence plateau is ~5e4. An absolute 1e-8 step
 * perturbs the former by 10x (destroying the derivative, sometimes reversing
 * its sign) and the latter by 2e-13 (below the resolution of the difference,
 * so the result is rounding noise).
 *
 * @param {number} [eps] Optional fixed absolute step, for testing. Omit to use
 *   the relative step, which is what callers should do.
 */
export function jacobian(xData, modelFn, params, eps = null) {
  const n = xData.length;
  const p = params.length;
  const J = Array.from({ length: n }, () => new Array(p).fill(0));
  for (let j = 0; j < p; j++) {
    // Pure relative step, falling back to the absolute value only when the
    // parameter is exactly zero (where a relative step is undefined). A
    // parameter sitting at zero enters the 4PL linearly, so a unit-scaled
    // step is accurate there regardless.
    const scale = Math.abs(params[j]);
    const h = eps !== null ? eps : CBRT_EPS * (scale > 0 ? scale : 1);
    const pPlus = [...params];
    const pMinus = [...params];
    pPlus[j] += h;
    pMinus[j] -= h;
    for (let i = 0; i < n; i++) {
      J[i][j] = (modelFn(xData[i], pPlus) - modelFn(xData[i], pMinus)) / (2 * h);
    }
  }
  return J;
}

/**
 * Wrap a model so the optimiser sees log10 of the parameters at `logIndices`.
 *
 * Fitting log10(EC50) rather than EC50 is standard practice (it is what Prism
 * does) and buys three things:
 *   - scale invariance: the fit no longer depends on whether concentrations
 *     were entered in M, nM or arbitrary units;
 *   - positivity for free: 10^t > 0 for every real t, so the EC50 cannot be
 *     driven negative and needs no clamping;
 *   - a well-conditioned Jacobian column, since log10(EC50) is O(1) even when
 *     EC50 is 1e-9.
 *
 * It is also the correct space for a confidence interval on a potency, which
 * is why the transform lives here rather than in the caller.
 */
export function withLogParams(modelFn, logIndices) {
  const set = new Set(logIndices);
  const toLinear = (p) => p.map((v, i) => (set.has(i) ? Math.pow(10, v) : v));
  const toLog = (p) => p.map((v, i) => (set.has(i) ? Math.log10(v) : v));
  return {
    toLinear,
    toLog,
    modelFn: (x, p) => modelFn(x, toLinear(p)),
  };
}

/**
 * Levenberg-Marquardt least-squares optimiser.
 *
 * @param {object} [options]
 * @param {function} [options.constrain] Called as (proposed, previous) after
 *   each step; returns the parameter vector to actually try. Use it to keep
 *   parameters in a valid region. Defaults to reverting non-finite entries.
 *   Callers that fit log10(EC50) do not need a positivity constraint on it.
 * @param {number[]} [options.weights] Per-observation weights. Minimises
 *   sum(w_i * r_i^2) instead of sum(r_i^2). The returned `ssr` is then the
 *   WEIGHTED sum, since that is what was minimised.
 */
export function levenbergMarquardt(xData, yData, modelFn, initialParams, options = {}) {
  const {
    maxIter = 500, tol = 1e-10, lambdaInit = 0.01, lambdaUp = 10, lambdaDown = 0.1,
    lambdaMax = 1e12, constrain = null, weights = null,
  } = options;
  let params = [...initialParams];
  let lambda = lambdaInit;
  let ssr = sumSquaredResiduals(xData, yData, modelFn, params, weights);
  let converged = false;
  let improved = false;

  // Weighted least squares is ordinary least squares on residuals and Jacobian
  // rows scaled by sqrt(w): the normal equations (J^T W J) d = J^T W r become
  // (J'^T J') d = J'^T r' with J' = sqrt(W) J and r' = sqrt(W) r.
  const rootW = weights ? weights.map(Math.sqrt) : null;

  for (let iter = 0; iter < maxIter; iter++) {
    let r = residuals(xData, yData, modelFn, params);
    let J = jacobian(xData, modelFn, params);
    if (rootW) {
      r = r.map((v, i) => rootW[i] * v);
      J = J.map((row, i) => row.map(v => rootW[i] * v));
    }
    const Jt = matTranspose(J);
    const JtJ = matMul(Jt, J);
    const Jtr = Jt.map(row => row.reduce((s, v, i) => s + v * r[i], 0));

    // Damping
    const A = JtJ.map((row, i) => row.map((v, j) => i === j ? v + lambda * (v + 1e-6) : v));
    const delta = solveLU(A, Jtr);
    if (!delta) { lambda *= lambdaUp; continue; }

    const proposed = params.map((p, i) => p + delta[i]);
    // Previously this dispatched on `modelFn === model4PL`, which silently did
    // nothing once the model was wrapped in a closure. Constraints are now
    // supplied explicitly by the caller, which knows its own parameter layout.
    const newParams = constrain
      ? constrain(proposed, params)
      : proposed.map((v, i) => (isFinite(v) ? v : params[i]));

    const newSSR = sumSquaredResiduals(xData, yData, modelFn, newParams, weights);
    if (newSSR < ssr) {
      improved = true;
      if (Math.abs(ssr - newSSR) / (ssr + 1e-15) < tol) { converged = true; params = newParams; ssr = newSSR; break; }
      params = newParams;
      ssr = newSSR;
      lambda *= lambdaDown;
    } else {
      lambda *= lambdaUp;
      // No improving step exists even under heavy damping, so we are at a
      // minimum to within numerical precision. Without this the loop spins to
      // maxIter and reports failure for a fit that is actually exact: the SSR
      // criterion above only fires on an *improving* step, which a fit sitting
      // on the machine-precision floor can never produce.
      // Only counts as convergence if we made progress at all -- a run that
      // never found a single downhill step is a genuine failure.
      if (lambda > lambdaMax) { converged = improved; break; }
    }
  }
  return { params, ssr, converged };
}

// Initial parameter estimation
export function estimateInitialParams(xData, yData, is5PL = false) {
  const sortedIndices = xData.map((_, i) => i).sort((a, b) => xData[a] - xData[b]);
  const sortedY = sortedIndices.map(i => yData[i]);
  const sortedX = sortedIndices.map(i => xData[i]);

  const A = sortedY.slice(0, Math.max(1, Math.floor(sortedY.length * 0.2))).reduce((a, b) => a + b, 0) /
    Math.max(1, Math.floor(sortedY.length * 0.2));
  const D = sortedY.slice(-Math.max(1, Math.floor(sortedY.length * 0.2))).reduce((a, b) => a + b, 0) /
    Math.max(1, Math.floor(sortedY.length * 0.2));

  const midY = (A + D) / 2;
  let C = sortedX[Math.floor(sortedX.length / 2)];
  for (let i = 0; i < sortedY.length - 1; i++) {
    if ((sortedY[i] - midY) * (sortedY[i + 1] - midY) <= 0) {
      C = (sortedX[i] + sortedX[i + 1]) / 2;
      break;
    }
  }
  if (C <= 0) C = sortedX[Math.floor(sortedX.length / 2)] || 1;

  const B = A > D ? 1 : -1;
  // 5PL uses (EC50/x) instead of (x/C), so Hill slope sign is flipped vs 4PL
  if (is5PL) return [A, A > D ? -1 : 1, C, D, 1];
  return [A, B, C, D];
}

// EC50 sits at index 2 in both the 4PL [A, B, C, D] and 5PL
// [Bottom, Hill, EC50, Top, S] parameter vectors.
const EC50_INDEX = 2;

/**
 * Put a 4PL solution into canonical form: Hill slope positive.
 *
 * The 4PL has an exact two-fold symmetry. Because
 *
 *   D + (A - D)/(1 + (x/C)^B)  ==  A + (D - A)/(1 + (x/C)^-B)
 *
 * the vectors [A, B, C, D] and [D, -B, C, A] describe the SAME curve, with
 * identical SSR and identical EC50. Which one the optimiser lands on depends
 * on nothing more than the starting point, so an unconstrained fit returns
 * either at random -- in practice the flipped one most of the time.
 *
 * That makes A and D meaningless as reported quantities: sometimes A is the
 * zero-dose asymptote and sometimes it is the infinite-dose one. It also
 * inverts the UI's "min"/"max" labels, and makes a confidence interval on A
 * uninterpretable, since it may describe either end of the curve.
 *
 * Fixing B > 0 resolves it: A is then always the response as x -> 0 and D
 * always the response as x -> infinity, which is what the labels claim.
 *
 * Applies to the unconstrained 4PL only. A constrained fit cannot be flipped
 * (the swap would move a value into a slot the caller pinned), and the 5PL has
 * no equivalent symmetry because the asymmetry exponent S does not commute
 * with the slope inversion.
 */
function canonicalise4PL(params) {
  const [A, B, C, D] = params;
  return B < 0 ? [D, -B, C, A] : [A, B, C, D];
}

/**
 * Standard errors and confidence intervals for a converged fit.
 *
 * Everything happens in FITTING space -- the space the optimiser actually
 * searched, with log10(EC50) in place of EC50 -- and is then mapped back onto
 * the caller-visible parameter layout. Two things make that mapping non-trivial:
 *
 *  - a constrained fit (1PL/2PL/3PL) searches only the FREE parameters, so the
 *    covariance matrix is smaller than the 4-slot vector the caller sees. Fixed
 *    slots get a null SE, which is correct: they were not estimated, so they
 *    carry no uncertainty from this fit.
 *  - the EC50 interval must be back-transformed from log space rather than
 *    formed on the linear scale, which is what makes it asymmetric.
 *
 * @param {number[]|null} freeIndices Caller-visible index of each free
 *   parameter, or null when every slot is free.
 * @returns Fields to merge into the fit result. `se`/`ci` are aligned with the
 *   caller-visible `params`; `logEC50` carries the log-space estimate that the
 *   EC50 interval derives from.
 */
function buildInference({
  xData, yData, fitSpaceParams, fitModelFn, ssr,
  logIndices, slotCount, freeIndices, weights = null, alpha = 0.05,
}) {
  const empty = {
    dof: null, syx: null, se: null, ci: null, logEC50: null,
    cov: null, correlation: null,
  };

  const J = jacobian(xData, fitModelFn, fitSpaceParams);
  const covResult = parameterCovariance(J, ssr, weights);
  if (!covResult) return empty;

  const { cov, dof, syx } = covResult;
  const { se, ci, tCrit } = parameterIntervals(fitSpaceParams, cov, dof, alpha);

  // Expand from the free-parameter vector onto the caller-visible slots.
  const slots = (fill) => {
    const out = new Array(slotCount).fill(null);
    fitSpaceParams.forEach((_, freeIdx) => {
      const slot = freeIndices ? freeIndices[freeIdx] : freeIdx;
      out[slot] = fill(freeIdx);
    });
    return out;
  };

  const seOut = slots(i => se[i]);
  const ciOut = slots(i => ci[i]);

  // Map the log10 slots back to the linear scale the caller reports.
  let logEC50 = null;
  for (const logIdx of logIndices) {
    const slot = freeIndices ? freeIndices[logIdx] : logIdx;
    const back = backTransformLog10(fitSpaceParams[logIdx], se[logIdx], tCrit);
    seOut[slot] = back.se;
    ciOut[slot] = back.ci;
    if (slot === EC50_INDEX) {
      logEC50 = { value: fitSpaceParams[logIdx], se: se[logIdx], ci: ci[logIdx] };
    }
  }

  return {
    dof, syx, se: seOut, ci: ciOut, logEC50,
    cov, correlation: correlationMatrix(cov),
  };
}

/**
 * Fit a single model with multi-start, return full stats.
 *
 * The optimiser runs in log10(EC50) space; `params` in the returned object is
 * back-transformed to a linear EC50 so callers see the historical layout.
 */
export function fitModel(xData, yData, modelFn, is5PL, options = {}) {
  const { weighting = "none", maxWeightIterations = 10 } = options;
  const init = estimateInitialParams(xData, yData, is5PL);
  const perturbations = [
    init,
    init.map((p, i) => i === 1 ? -p : p),
    init.map((p, i) => i === EC50_INDEX ? p * 2 : p),
    init.map((p, i) => i === EC50_INDEX ? p * 0.5 : p),
  ];

  const log = withLogParams(modelFn, [EC50_INDEX]);
  // log10(EC50) needs no constraint; only the 5PL asymmetry S must stay positive.
  const constrain = is5PL
    ? (proposed, prev) => proposed.map((v, i) => {
        if (!isFinite(v)) return prev[i];
        if (i === 4 && v <= 0) return Math.abs(v) || 0.1;
        return v;
      })
    : null;

  let best = null;
  for (const startParams of perturbations) {
    if (!(startParams[EC50_INDEX] > 0)) continue; // log10 undefined
    const result = levenbergMarquardt(
      xData, yData, log.modelFn, log.toLog(startParams), { constrain },
    );
    if (!best || result.ssr < best.ssr) best = result;
  }
  if (!best) return null;

  // ── Iteratively reweighted least squares ────────────────────────
  // Relative weights depend on the fitted curve, and the fitted curve depends
  // on the weights, so the two are solved together: fit, recompute weights
  // from the new predictions, refit, until the weights stop moving. Seeding
  // from the converged unweighted fit means this usually takes 2-3 passes.
  //
  // If weighting turns out not to be applicable -- most often because the
  // response was normalised or background-subtracted to a zero baseline, where
  // relative weights are undefined -- the unweighted fit is kept and the
  // reason is reported rather than silently fitting something else.
  let weights = null;
  let weightWarning = null;
  let weightIterations = 0;

  if (weighting && weighting !== "none") {
    let current = best;
    for (let iter = 0; iter < maxWeightIterations; iter++) {
      const predicted = xData.map(x => log.modelFn(x, current.params));
      const built = buildWeights(weighting, { yPred: predicted, xData, yData });
      // A warning can accompany usable weights (a near-zero baseline is
      // computable but inadvisable), so record it either way; only a null
      // weight vector means weighting could not be applied at all.
      weightWarning = built.warning;
      if (!built.weights) { weights = null; break; }
      if (weightsConverged(weights, built.weights)) break;

      weights = built.weights;
      weightIterations = iter + 1;
      const refit = levenbergMarquardt(
        xData, yData, log.modelFn, current.params, { constrain, weights },
      );
      current = refit;
    }
    if (weights) best = current;
  }

  // Resolve the mirror ambiguity before anything is derived from the params,
  // so the reported A/D and their intervals always mean the same thing. The
  // curve, and therefore the SSR and every goodness-of-fit statistic, is
  // unchanged; only the labelling of the two asymptotes is.
  if (!is5PL) best = { ...best, params: canonicalise4PL(best.params) };

  // Inference is computed in FITTING space (log10 EC50), where the Wald
  // interval is well behaved, then back-transformed. See buildInference.
  const inference = buildInference({
    xData, yData, fitSpaceParams: best.params, fitModelFn: log.modelFn,
    ssr: best.ssr, logIndices: [EC50_INDEX], slotCount: is5PL ? 5 : 4,
    freeIndices: null, weights,
  });

  best = { ...best, params: log.toLinear(best.params) };

  const n = xData.length;
  const k = is5PL ? 5 : 4;
  const yPred = xData.map(x => modelFn(x, best.params));
  const r2 = rSquared(yData, yPred);

  // `best.ssr` is whatever was minimised -- weighted when weights are in play.
  // The plain unweighted SSR is reported alongside it so the familiar
  // goodness-of-fit numbers stay on a comparable scale across weighting
  // choices; the information criteria necessarily use the minimised objective.
  const wssr = weights ? best.ssr : null;
  const ssrUnweighted = weights
    ? yData.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0)
    : best.ssr;

  const rmse = Math.sqrt(ssrUnweighted / n);
  const aicc = computeAICc(n, k, best.ssr);
  const bic = computeBIC(n, k, best.ssr);
  const aic = computeAIC(n, k, best.ssr);

  // Biological EC50 for 5PL (differs from parametric EC50 when S ≠ 1)
  const bioEC50 = is5PL ? computeBiologicalEC50(modelFn, best.params) : null;

  // Whether the curve is ADEQUATE, as opposed to how tightly it fits. Needs
  // replicates, so it self-reports as inapplicable rather than failing when
  // the design cannot support it. Uses the same weights the fit minimised.
  const lackOfFit = lackOfFitTest(xData, yData, yPred, k, { weights });

  // Whether the parameters are IDENTIFIABLE, as opposed to how precise they
  // look. Deliberately geometric, so it still reports when the covariance
  // matrix came back singular -- which is itself the strongest such finding.
  const identifiability = identifiabilityWarnings(
    { params: best.params, ...inference }, xData, modelFn, { is5PL },
  );

  return {
    ...best, r2, rmse, yPred, aicc, bic, aic, k, n, bioEC50, ...inference,
    lackOfFit, identifiability,
    ssr: ssrUnweighted, wssr,
    weighting: {
      requested: weighting || "none",
      applied: weights ? weighting : "none",
      warning: weightWarning,
      iterations: weightIterations,
      weights,
      // Measured from the replicates, independent of what was requested, so a
      // caller can see whether the chosen scheme matches the data. The engine
      // reports this rather than overriding the user: with a handful of doses
      // the exponent is imprecise, and picking a weighting is an
      // assay-development decision, not something to infer from one plate.
      variance: estimateVariancePower(xData, yData),
    },
  };
}

// Fit constrained model, returns result with full 4PL params for display
export function fitConstrainedModel(xData, yData, fixedMap) {
  const freeIndices = [0, 1, 2, 3].filter(i => !(i in fixedMap));
  const k = freeIndices.length;
  const constrainedFn = makeConstrainedModel(fixedMap);

  // Initial estimates from full 4PL estimates
  const fullInit = estimateInitialParams(xData, yData, false);
  // Override fixed params in init
  for (const [idx, val] of Object.entries(fixedMap)) {
    fullInit[parseInt(idx)] = val;
  }
  const freeInit = freeIndices.map(i => fullInit[i]);

  // Multi-start
  const perturbations = [freeInit];
  // Perturb EC50 (which is index 2 in full params)
  const ec50FreeIdx = freeIndices.indexOf(2);
  if (ec50FreeIdx >= 0) {
    perturbations.push(freeInit.map((p, i) => i === ec50FreeIdx ? p * 2 : p));
    perturbations.push(freeInit.map((p, i) => i === ec50FreeIdx ? p * 0.5 : p));
  }
  // Perturb slope if free
  const slopeFreeIdx = freeIndices.indexOf(1);
  if (slopeFreeIdx >= 0) {
    perturbations.push(freeInit.map((p, i) => i === slopeFreeIdx ? -p : p));
  }

  // Fit log10(EC50) when the EC50 is free, for the same reasons as fitModel.
  // Its position within the FREE vector depends on which params are fixed.
  const log = withLogParams(constrainedFn, ec50FreeIdx >= 0 ? [ec50FreeIdx] : []);

  let best = null;
  for (const startParams of perturbations) {
    if (ec50FreeIdx >= 0 && !(startParams[ec50FreeIdx] > 0)) continue;
    const result = levenbergMarquardt(
      xData, yData, log.modelFn, log.toLog(startParams),
    );
    if (!best || result.ssr < best.ssr) best = result;
  }
  if (!best) return null;

  // As in fitModel, inference is computed in fitting space. `freeIndices` maps
  // the free-parameter vector back onto the 4 caller-visible slots; the fixed
  // slots come back with a null SE, since this fit did not estimate them.
  const inference = buildInference({
    xData, yData, fitSpaceParams: best.params, fitModelFn: log.modelFn,
    ssr: best.ssr, logIndices: ec50FreeIdx >= 0 ? [ec50FreeIdx] : [],
    slotCount: 4, freeIndices,
  });

  best = { ...best, params: log.toLinear(best.params) };

  // Expand to full 4PL params for display
  const fullParams = [0, 0, 0, 0];
  let freeIdx = 0;
  for (let i = 0; i < 4; i++) {
    if (i in fixedMap) fullParams[i] = fixedMap[i];
    else fullParams[i] = best.params[freeIdx++];
  }

  const n = xData.length;
  const yPred = xData.map(x => model4PL(x, fullParams));
  const r2 = rSquared(yData, yPred);
  const rmse = Math.sqrt(best.ssr / n);
  const aicc = computeAICc(n, k, best.ssr);
  const bic = computeBIC(n, k, best.ssr);
  const aic = computeAIC(n, k, best.ssr);

  return {
    params: fullParams, ssr: best.ssr, converged: best.converged,
    r2, rmse, yPred, aicc, bic, aic, k, n, bioEC50: null, ...inference,
    // p is the number of FREE parameters, not the 4 display slots: a
    // constrained fit spends fewer degrees of freedom, which leaves more for
    // lack of fit and makes the test more sensitive, not less.
    lackOfFit: lackOfFitTest(xData, yData, yPred, k),
    identifiability: identifiabilityWarnings(
      { params: fullParams, ...inference }, xData, model4PL, { is5PL: false },
    ),
  };
}
