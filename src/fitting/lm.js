// ── Levenberg-Marquardt curve fitting ─────────────────────────────
// Damped least-squares optimiser plus the multi-start wrappers that drive it.

import { model4PL, model5PL, makeConstrainedModel, computeBiologicalEC50 } from "./models.js";
import { matMul, matTranspose, solveLU } from "./linalg.js";
import { rSquared, computeAIC, computeAICc, computeBIC } from "./stats.js";

export function residuals(xData, yData, modelFn, params) {
  return xData.map((x, i) => yData[i] - modelFn(x, params));
}

export function sumSquaredResiduals(xData, yData, modelFn, params) {
  return residuals(xData, yData, modelFn, params).reduce((s, r) => s + r * r, 0);
}

// Numerical Jacobian
export function jacobian(xData, modelFn, params, eps = 1e-8) {
  const n = xData.length;
  const p = params.length;
  const J = Array.from({ length: n }, () => new Array(p).fill(0));
  for (let j = 0; j < p; j++) {
    const pPlus = [...params];
    const pMinus = [...params];
    pPlus[j] += eps;
    pMinus[j] -= eps;
    for (let i = 0; i < n; i++) {
      J[i][j] = (modelFn(xData[i], pPlus) - modelFn(xData[i], pMinus)) / (2 * eps);
    }
  }
  return J;
}

// Levenberg-Marquardt optimizer
export function levenbergMarquardt(xData, yData, modelFn, initialParams, options = {}) {
  const { maxIter = 500, tol = 1e-10, lambdaInit = 0.01, lambdaUp = 10, lambdaDown = 0.1 } = options;
  let params = [...initialParams];
  let lambda = lambdaInit;
  let ssr = sumSquaredResiduals(xData, yData, modelFn, params);
  let converged = false;

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

    const newParams = params.map((p, i) => p + delta[i]);
    
    // Keep EC50/C positive for known models
    if (modelFn === model4PL && newParams[2] <= 0) newParams[2] = Math.abs(newParams[2]) || 1e-6;
    if (modelFn === model5PL) {
      if (newParams[2] <= 0) newParams[2] = Math.abs(newParams[2]) || 1e-6;
      if (newParams[4] <= 0) newParams[4] = Math.abs(newParams[4]) || 0.1;
    }
    // For constrained models (closures), keep all params finite and EC50-like params positive
    if (modelFn !== model4PL && modelFn !== model5PL) {
      for (let i = 0; i < newParams.length; i++) {
        if (!isFinite(newParams[i])) newParams[i] = params[i];
      }
    }

    const newSSR = sumSquaredResiduals(xData, yData, modelFn, newParams);
    if (newSSR < ssr) {
      if (Math.abs(ssr - newSSR) / (ssr + 1e-15) < tol) { converged = true; params = newParams; ssr = newSSR; break; }
      params = newParams;
      ssr = newSSR;
      lambda *= lambdaDown;
    } else {
      lambda *= lambdaUp;
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

// Fit a single model with multi-start, return full stats
export function fitModel(xData, yData, modelFn, is5PL) {
  const init = estimateInitialParams(xData, yData, is5PL);
  const perturbations = [
    init,
    init.map((p, i) => i === 1 ? -p : p),
    init.map((p, i) => i === 2 ? p * 2 : p),
    init.map((p, i) => i === 2 ? p * 0.5 : p),
  ];
  let best = null;
  for (const startParams of perturbations) {
    try {
      const result = levenbergMarquardt(xData, yData, modelFn, startParams);
      if (!best || result.ssr < best.ssr) best = result;
    } catch (e) { /* skip */ }
  }
  if (!best) return null;

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

  let best = null;
  for (const startParams of perturbations) {
    try {
      // Wrap to keep EC50-like params positive
      const result = levenbergMarquardt(xData, yData, constrainedFn, startParams);
      // Expand and check EC50 (index 2 in full params) is positive
      const fullCheck = [0, 0, 0, 0];
      let fi = 0;
      for (let i = 0; i < 4; i++) {
        if (i in fixedMap) fullCheck[i] = fixedMap[i];
        else fullCheck[i] = result.params[fi++];
      }
      if (fullCheck[2] <= 0) continue; // skip if EC50 is negative
      if (!best || result.ssr < best.ssr) best = result;
    } catch (e) { /* skip */ }
  }
  if (!best) return null;

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
