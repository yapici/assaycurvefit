// ── Levenberg-Marquardt curve fitting ─────────────────────────────
// Damped least-squares optimiser plus the multi-start wrappers that drive it.

import { model4PL, makeConstrainedModel, computeBiologicalEC50 } from "./models.js";
import { matMul, matTranspose, solveLU } from "./linalg.js";
import { rSquared, computeAIC, computeAICc, computeBIC } from "./stats.js";

export function residuals(xData, yData, modelFn, params) {
  return xData.map((x, i) => yData[i] - modelFn(x, params));
}

export function sumSquaredResiduals(xData, yData, modelFn, params) {
  return residuals(xData, yData, modelFn, params).reduce((s, r) => s + r * r, 0);
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
 */
export function levenbergMarquardt(xData, yData, modelFn, initialParams, options = {}) {
  const {
    maxIter = 500, tol = 1e-10, lambdaInit = 0.01, lambdaUp = 10, lambdaDown = 0.1,
    lambdaMax = 1e12, constrain = null,
  } = options;
  let params = [...initialParams];
  let lambda = lambdaInit;
  let ssr = sumSquaredResiduals(xData, yData, modelFn, params);
  let converged = false;
  let improved = false;

  for (let iter = 0; iter < maxIter; iter++) {
    const r = residuals(xData, yData, modelFn, params);
    const J = jacobian(xData, modelFn, params);
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

    const newSSR = sumSquaredResiduals(xData, yData, modelFn, newParams);
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
 * Fit a single model with multi-start, return full stats.
 *
 * The optimiser runs in log10(EC50) space; `params` in the returned object is
 * back-transformed to a linear EC50 so callers see the historical layout.
 */
export function fitModel(xData, yData, modelFn, is5PL) {
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
  best = { ...best, params: log.toLinear(best.params) };

  const n = xData.length;
  const k = is5PL ? 5 : 4;
  const yPred = xData.map(x => modelFn(x, best.params));
  const r2 = rSquared(yData, yPred);
  const rmse = Math.sqrt(best.ssr / n);
  const aicc = computeAICc(n, k, best.ssr);
  const bic = computeBIC(n, k, best.ssr);
  const aic = computeAIC(n, k, best.ssr);

  // Biological EC50 for 5PL (differs from parametric EC50 when S ≠ 1)
  const bioEC50 = is5PL ? computeBiologicalEC50(modelFn, best.params) : null;

  return { ...best, r2, rmse, yPred, aicc, bic, aic, k, n, bioEC50 };
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

  return { params: fullParams, ssr: best.ssr, converged: best.converged, r2, rmse, yPred, aicc, bic, aic, k, n, bioEC50: null };
}
