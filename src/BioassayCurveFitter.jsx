import { useState, useCallback, useMemo, useRef, useEffect } from "react";

// ── Multi-Compound CSV Parser ─────────────────────────────────────
// Handles GraphPad Prism-style format:
//   Row 0: compound names (duplicated headers = replicates)
//   Col 0: concentrations (Log10 M or raw)
//   Adjacent duplicate headers = replicate columns for same compound
function parsePrismCSV(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error("File has fewer than 2 rows.");
  const headers = lines[0].split(",").map(h => h.trim());
  // Group columns by compound name preserving order
  const compoundMap = new Map();
  const compoundOrder = [];
  for (let i = 1; i < headers.length; i++) {
    const name = headers[i] || `Compound ${i}`;
    if (!compoundMap.has(name)) { compoundMap.set(name, []); compoundOrder.push(name); }
    compoundMap.get(name).push(i);
  }
  if (compoundOrder.length === 0) throw new Error("No compound columns found.");
  const rows = lines.slice(1).map(l => l.split(",").map(v => v.trim()));
  const compounds = compoundOrder.map(name => {
    const colIndices = compoundMap.get(name);
    const points = rows.map(row => {
      const conc = parseFloat(row[0]);
      const reps = colIndices.map(ci => parseFloat(row[ci])).filter(v => !isNaN(v));
      const avg = reps.length ? reps.reduce((a, b) => a + b, 0) / reps.length : NaN;
      return { conc, avg, reps };
    }).filter(p => !isNaN(p.conc) && !isNaN(p.avg));
    return { name, nReplicates: colIndices.length, points };
  });
  return compounds;
}

// Convert a parsed compound back to CSV text for the existing textarea/parser
function compoundToCSV(compound) {
  const repCount = compound.nReplicates;
  const repHeaders = Array.from({ length: repCount }, (_, i) => `Rep${i + 1}`).join(",");
  const header = `Concentration,${repHeaders}`;
  const rows = compound.points.map(p => {
    const repVals = p.reps.map(v => v.toString()).join(",");
    return `${Math.pow(10, p.conc)},${repVals}`;
  });
  return [header, ...rows].join("\n");
}

// ── Numerical Engine ──────────────────────────────────────────────
// 4PL: y = D + (A - D) / (1 + (x/C)^B)
// 5PL: y = Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)^S
// 3PL: 4PL with B fixed to 1 (Hill slope = 1)
// 2PL: 4PL with A and D fixed (fit B and C only)
// 1PL: 4PL with A, B, D fixed (fit C / EC50 only)

function model4PL(x, params) {
  const [A, B, C, D] = params;
  if (x <= 0) return A;
  return D + (A - D) / (1 + Math.pow(x / C, B));
}

function model5PL(x, params) {
  const [Bottom, Hill, EC50, Top, S] = params;
  if (x <= 0) return Bottom;
  return Bottom + (Top - Bottom) / Math.pow(1 + Math.pow(EC50 / x, Hill), S);
}

// Create a constrained model function that wraps the full 4PL
// fixedMap: object mapping param index -> fixed value, e.g. { 1: 1.0 } fixes B=1
function makeConstrainedModel(fixedMap) {
  return function constrainedModel(x, freeParams) {
    // Expand free params into full 4PL params
    const fullParams = [0, 0, 0, 0];
    let freeIdx = 0;
    for (let i = 0; i < 4; i++) {
      if (i in fixedMap) {
        fullParams[i] = fixedMap[i];
      } else {
        fullParams[i] = freeParams[freeIdx++];
      }
    }
    return model4PL(x, fullParams);
  };
}

// Fit constrained model, returns result with full 4PL params for display
function fitConstrainedModel(xData, yData, fixedMap) {
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

// Get the drawing/evaluation model function for a given model type
// 1PL, 2PL, 3PL, 4PL all use model4PL (with full 4-param vector)
function getModelFn(mType) {
  return mType === "5PL" ? model5PL : model4PL;
}

function residuals(xData, yData, modelFn, params) {
  return xData.map((x, i) => yData[i] - modelFn(x, params));
}

function sumSquaredResiduals(xData, yData, modelFn, params) {
  return residuals(xData, yData, modelFn, params).reduce((s, r) => s + r * r, 0);
}

// Numerical Jacobian
function jacobian(xData, modelFn, params, eps = 1e-8) {
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

// Matrix operations
function matMul(A, B) {
  const m = A.length, n = B[0].length, k = B.length;
  const C = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let l = 0; l < k; l++)
        C[i][j] += A[i][l] * B[l][j];
  return C;
}

function matTranspose(A) {
  const m = A.length, n = A[0].length;
  const T = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      T[j][i] = A[i][j];
  return T;
}

// Solve Ax = b using LU decomposition with partial pivoting
function solveLU(A, b) {
  const n = A.length;
  const LU = A.map(row => [...row]);
  const P = Array.from({ length: n }, (_, i) => i);
  
  for (let k = 0; k < n; k++) {
    let maxVal = 0, maxIdx = k;
    for (let i = k; i < n; i++) {
      if (Math.abs(LU[i][k]) > maxVal) {
        maxVal = Math.abs(LU[i][k]);
        maxIdx = i;
      }
    }
    if (maxVal < 1e-15) return null;
    if (maxIdx !== k) {
      [LU[k], LU[maxIdx]] = [LU[maxIdx], LU[k]];
      [P[k], P[maxIdx]] = [P[maxIdx], P[k]];
    }
    for (let i = k + 1; i < n; i++) {
      LU[i][k] /= LU[k][k];
      for (let j = k + 1; j < n; j++) {
        LU[i][j] -= LU[i][k] * LU[k][j];
      }
    }
  }

  const pb = P.map(i => b[i]);
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    y[i] = pb[i];
    for (let j = 0; j < i; j++) y[i] -= LU[i][j] * y[j];
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = y[i];
    for (let j = i + 1; j < n; j++) x[i] -= LU[i][j] * x[j];
    x[i] /= LU[i][i];
  }
  return x;
}

// Levenberg-Marquardt optimizer
function levenbergMarquardt(xData, yData, modelFn, initialParams, options = {}) {
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
function estimateInitialParams(xData, yData, is5PL = false) {
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

// Compute R²
function rSquared(yData, yPred) {
  const yMean = yData.reduce((a, b) => a + b, 0) / yData.length;
  const ssTot = yData.reduce((s, y) => s + (y - yMean) ** 2, 0);
  const ssRes = yData.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0);
  return 1 - ssRes / ssTot;
}

// Information criteria for model comparison
function computeAIC(n, k, ssr) {
  // k = number of parameters, n = number of observations
  // AIC = n * ln(SSR/n) + 2k
  return n * Math.log(ssr / n) + 2 * k;
}

function computeAICc(n, k, ssr) {
  // Corrected AIC for small sample sizes
  const aic = computeAIC(n, k, ssr);
  if (n - k - 1 <= 0) return Infinity;
  return aic + (2 * k * (k + 1)) / (n - k - 1);
}

function computeBIC(n, k, ssr) {
  return n * Math.log(ssr / n) + k * Math.log(n);
}

// Compute biological EC50 for 5PL: concentration where response = (Top + Bottom) / 2
function computeBiologicalEC50(modelFn, params) {
  const [Bottom, Hill, EC50, Top, S] = params;
  const targetY = (Top + Bottom) / 2;
  // Bisection in log-space
  let lo = 1e-15, hi = 1e15;
  const yLo = modelFn(lo, params);
  const yHi = modelFn(hi, params);
  const increasing = yHi > yLo;
  for (let i = 0; i < 100; i++) {
    const mid = Math.sqrt(lo * hi);
    const yMid = modelFn(mid, params);
    if ((increasing && yMid > targetY) || (!increasing && yMid < targetY)) hi = mid;
    else lo = mid;
  }
  return Math.sqrt(lo * hi);
}

// Fit a single model with multi-start, return full stats
function fitModel(xData, yData, modelFn, is5PL) {
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

// Group replicate data by concentration, compute stats
function groupByConcentration(xData, yData) {
  const groups = {};
  xData.forEach((x, i) => {
    const key = x.toString();
    if (!groups[key]) groups[key] = { x, values: [], indices: [] };
    groups[key].values.push(yData[i]);
    groups[key].indices.push(i);
  });
  return Object.values(groups).map(g => {
    const vals = g.values;
    const n = vals.length;
    const mean = vals.reduce((a, b) => a + b, 0) / n;
    const variance = n > 1 ? vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
    const sd = Math.sqrt(variance);
    const sem = n > 1 ? sd / Math.sqrt(n) : 0;
    return { x: g.x, mean, sd, sem, n, values: vals, indices: g.indices };
  }).sort((a, b) => a.x - b.x);
}

// ── Grubbs' Test for Outliers ─────────────────────────────────────
// Accurate inverse t-distribution via regularized incomplete beta function

// Log-gamma function (Lanczos approximation)
function lnGamma(z) {
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  }
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

// Regularized incomplete beta function I_x(a, b) via continued fraction (Lentz)
function betaIncomplete(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  // Use symmetry if needed for convergence
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - betaIncomplete(1 - x, b, a);
  }
  const lnBeta = lnGamma(a) + lnGamma(b) - lnGamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta) / a;

  // Lentz's continued fraction
  let f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d; f = d;

  for (let m = 1; m <= 200; m++) {
    // Even step
    let num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + num * d; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d;
    c = 1 + num / c; if (Math.abs(c) < 1e-30) c = 1e-30;
    f *= d * c;

    // Odd step
    num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + num * d; if (Math.abs(d) < 1e-30) d = 1e-30; d = 1 / d;
    c = 1 + num / c; if (Math.abs(c) < 1e-30) c = 1e-30;
    f *= d * c;

    if (Math.abs(d * c - 1) < 1e-12) break;
  }
  return front * f;
}

// CDF of t-distribution: P(T <= t) for df degrees of freedom
function tCDF(t, df) {
  const x = df / (df + t * t);
  const ib = betaIncomplete(x, df / 2, 0.5);
  if (t >= 0) return 1 - 0.5 * ib;
  return 0.5 * ib;
}

// Inverse t-distribution via bisection + Newton refinement
// Returns t such that P(T > t) = p (upper tail)
function tInv(p, df) {
  if (df <= 0) return Infinity;
  if (p <= 0) return Infinity;
  if (p >= 0.5) return 0;

  // Target: find t such that tCDF(t, df) = 1 - p
  const target = 1 - p;

  // Bisection to get close
  let lo = 0, hi = 5;
  while (tCDF(hi, df) < target) hi *= 2;

  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    if (tCDF(mid, df) < target) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12) break;
  }
  return (lo + hi) / 2;
}

function grubbsCriticalG(n, alpha) {
  if (n < 3) return Infinity;
  const df = n - 2;
  const p = alpha / (2 * n); // two-sided Grubbs uses alpha/(2n) for t lookup
  const t = tInv(p, df);
  const tSq = t * t;
  return ((n - 1) / Math.sqrt(n)) * Math.sqrt(tSq / (n - 2 + tSq));
}

function grubbsTest(values, alpha = 0.05) {
  const n = values.length;
  if (n < 3) return { outliers: [], details: [] };
  
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1));
  
  if (sd < 1e-15) return { outliers: [], details: [] };
  
  const gCrit = grubbsCriticalG(n, alpha);
  const details = values.map((v, i) => {
    const g = Math.abs(v - mean) / sd;
    return { index: i, value: v, g, gCrit, isOutlier: g > gCrit, deviation: v - mean };
  });
  
  return {
    outliers: details.filter(d => d.isOutlier),
    details,
    mean,
    sd,
    gCrit,
  };
}

// Run Grubbs on all concentration groups, return per-group results + flat outlier index set
function runGrubbsAllGroups(xData, yData, alpha = 0.05) {
  const grouped = groupByConcentration(xData, yData);
  const outlierIndices = new Set();
  const groupResults = [];

  for (const g of grouped) {
    if (g.n < 3) {
      groupResults.push({ x: g.x, n: g.n, tested: false, result: null, indices: g.indices, outlierCount: 0 });
      continue;
    }
    const result = grubbsTest(g.values, alpha);
    for (const ol of result.outliers) {
      outlierIndices.add(g.indices[ol.index]);
    }
    groupResults.push({
      x: g.x,
      n: g.n,
      tested: true,
      result,
      indices: g.indices,
      outlierCount: result.outliers.length,
    });
  }

  return { outlierIndices, groupResults, totalOutliers: outlierIndices.size };
}

// ── Chart drawing ─────────────────────────────────────────────────
function drawChart(canvas, xData, yData, fitResult, modelType, options = {}, theme = {}) {
  const { pointView = "individual", errorBarType = "sd", outlierIndices = null, excludedIndices: exclSet = null } = options;
  const t = theme;
  const grouped = groupByConcentration(xData, yData);
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  const pad = { top: 30, right: 30, bottom: 55, left: 70 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Log scale for X
  const logX = xData.filter(x => x > 0).map(x => Math.log10(x));
  const allY = [...yData];
  
  let xMin, xMax;
  if (logX.length > 0) {
    xMin = Math.floor(Math.min(...logX)) - 0.5;
    xMax = Math.ceil(Math.max(...logX)) + 0.5;
  } else {
    xMin = -2; xMax = 4;
  }
  
  // Generate curve points for Y range
  const curveY = [];
  if (fitResult) {
    const modelFn = getModelFn(modelType);
    for (let i = 0; i <= 200; i++) {
      const lx = xMin + (xMax - xMin) * (i / 200);
      curveY.push(modelFn(Math.pow(10, lx), fitResult.params));
    }
  }
  const allYValues = [...allY, ...curveY];
  
  let yMin = Math.min(...allYValues);
  let yMax = Math.max(...allYValues);
  const yPad = (yMax - yMin) * 0.1 || 1;
  yMin -= yPad;
  yMax += yPad;

  const toCanvasX = (lx) => pad.left + ((lx - xMin) / (xMax - xMin)) * plotW;
  const toCanvasY = (y) => pad.top + ((yMax - y) / (yMax - yMin)) * plotH;

  // Store coordinate metadata on canvas for tooltip hit-testing
  canvas._chartMeta = { pad, plotW, plotH, xMin, xMax, yMin, yMax, W, H, toCanvasX, toCanvasY, errorBarGroups: [], theme: t };

  // Background
  ctx.fillStyle = t.canvas || "#0a0f1a";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = t.grid || "rgba(100,140,180,0.08)";
  ctx.lineWidth = 1;
  for (let lx = Math.ceil(xMin); lx <= Math.floor(xMax); lx++) {
    const cx = toCanvasX(lx);
    ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
  }
  const yTicks = 6;
  for (let i = 0; i <= yTicks; i++) {
    const y = yMin + (yMax - yMin) * (i / yTicks);
    const cy = toCanvasY(y);
    ctx.beginPath(); ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + plotW, cy); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = t.axis || "rgba(140,170,210,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = t.axisLabel || "rgba(160,190,230,0.6)";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  for (let lx = Math.ceil(xMin); lx <= Math.floor(xMax); lx++) {
    ctx.fillText(`10^${lx}`, toCanvasX(lx), pad.top + plotH + 18);
  }
  ctx.textAlign = "right";
  for (let i = 0; i <= yTicks; i++) {
    const y = yMin + (yMax - yMin) * (i / yTicks);
    ctx.fillText(y.toFixed(2), pad.left - 8, toCanvasY(y) + 4);
  }

  // Axis titles
  ctx.fillStyle = t.label || "rgba(180,210,240,0.7)";
  ctx.font = "12px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("Concentration (log scale)", pad.left + plotW / 2, H - 8);
  ctx.save();
  ctx.translate(16, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Response", 0, 0);
  ctx.restore();

  // Fitted curve
  if (fitResult) {
    const modelFn = getModelFn(modelType);
    const curveColor = modelType === "5PL" ? (t.purple || "#a855f7") : (t.blue || "#3b9eff");
    const curveShadow = modelType === "5PL" ? (t.purpleBg || "rgba(168,85,247,0.4)") : (t.blueBg || "rgba(59,158,255,0.4)");
    ctx.strokeStyle = curveColor;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = curveShadow;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const lx = xMin + (xMax - xMin) * (i / 200);
      const x = Math.pow(10, lx);
      const y = modelFn(x, fitResult.params);
      const cx = toCanvasX(lx), cy = toCanvasY(y);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // EC50 line
    const ec50 = fitResult.params[2];
    if (ec50 > 0) {
      const lec50 = Math.log10(ec50);
      if (lec50 >= xMin && lec50 <= xMax) {
        const midY = modelFn(ec50, fitResult.params);
        const cx = toCanvasX(lec50), cy = toCanvasY(midY);
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = t.orangeBorder || "rgba(255,180,50,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + plotW, cy); ctx.stroke();
        ctx.setLineDash([]);
        
        // EC50 marker
        ctx.fillStyle = t.orange || "rgba(255,180,50,0.9)";
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = t.orange || "rgba(255,180,50,0.7)";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(`EC50: ${ec50.toExponential(3)}`, cx + 10, cy - 8);
      }
    }

    // Biological EC50 for 5PL
    if (modelType === "5PL" && fitResult.bioEC50) {
      const bioEc50 = fitResult.bioEC50;
      if (bioEc50 > 0) {
        const lbec50 = Math.log10(bioEc50);
        if (lbec50 >= xMin && lbec50 <= xMax) {
          const midY = modelFn(bioEc50, fitResult.params);
          const bx = toCanvasX(lbec50), by = toCanvasY(midY);
          ctx.setLineDash([2, 3]);
          ctx.strokeStyle = (t.tealGlow || "rgba(0,230,180,") + "0.4)";
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(bx, pad.top); ctx.lineTo(bx, pad.top + plotH); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(pad.left, by); ctx.lineTo(pad.left + plotW, by); ctx.stroke();
          ctx.setLineDash([]);

          ctx.fillStyle = t.teal || "#00e6b4";
          ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI * 2); ctx.fill();
          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.textAlign = "left";
          ctx.fillText(`Bio EC50: ${bioEc50.toExponential(3)}`, bx + 10, by + 14);
        }
      }
    }
  }

  // Data points
  const errorBarGroups = []; // collected for tooltip hit-testing
  if (pointView === "errorbars") {
    // Error bar view: mean ± SD or SEM per concentration, excluding excluded points
    grouped.forEach((g) => {
      if (g.x <= 0) return;
      // Filter out excluded indices
      const activeVals = [];
      g.indices.forEach((idx, j) => {
        if (!exclSet || !exclSet.has(idx)) activeVals.push(g.values[j]);
      });
      if (activeVals.length === 0) return;
      const n = activeVals.length;
      const nTotal = g.n;
      const mean = activeVals.reduce((a, b) => a + b, 0) / n;
      const variance = n > 1 ? activeVals.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
      const sd = Math.sqrt(variance);
      const sem = n > 1 ? sd / Math.sqrt(n) : 0;
      const cv = mean !== 0 ? (sd / Math.abs(mean)) * 100 : 0;

      const lx = Math.log10(g.x);
      const cx = toCanvasX(lx);
      const cyMean = toCanvasY(mean);
      const errVal = errorBarType === "sem" ? sem : sd;
      const cyHi = toCanvasY(mean + errVal);
      const cyLo = toCanvasY(mean - errVal);

      // Store for tooltip
      errorBarGroups.push({ x: g.x, cx, cyMean, mean, sd, sem, cv, n, nTotal, errVal, errorBarType });

      // Error bar line
      if (n > 1) {
        ctx.strokeStyle = (t.tealGlow || "rgba(0,230,180,") + "0.55)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx, cyHi); ctx.lineTo(cx, cyLo); ctx.stroke();
        // Caps
        const capW = 4;
        ctx.beginPath(); ctx.moveTo(cx - capW, cyHi); ctx.lineTo(cx + capW, cyHi); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx - capW, cyLo); ctx.lineTo(cx + capW, cyLo); ctx.stroke();
      }

      // Mean point
      ctx.fillStyle = t.teal || "#00e6b4";
      ctx.beginPath(); ctx.arc(cx, cyMean, 5, 0, Math.PI * 2); ctx.fill();
    });
  } else {
    // Individual points view
    xData.forEach((x, i) => {
      if (x <= 0) return;
      const lx = Math.log10(x);
      const cx = toCanvasX(lx), cy = toCanvasY(yData[i]);
      const isExcluded = exclSet && exclSet.has(i);
      const isOutlier = !isExcluded && outlierIndices && outlierIndices.has(i);
      
      if (isExcluded) {
        // Dimmed with strikethrough
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = t.red || "#ff506a";
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
        // Strikethrough line
        ctx.strokeStyle = (t.redGlow || "rgba(255,80,106,") + "0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy); ctx.stroke();
        ctx.globalAlpha = 1;
      } else if (isOutlier) {
        // Outlier flagged (red + X)
        ctx.fillStyle = t.red || "#ff506a";
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
        const s = 6;
        ctx.strokeStyle = (t.redGlow || "rgba(255,80,106,") + "0.7)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx - s, cy - s); ctx.lineTo(cx + s, cy + s); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx + s, cy - s); ctx.lineTo(cx - s, cy + s); ctx.stroke();
      } else {
        // Normal point
        ctx.fillStyle = t.teal || "#00e6b4";
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
      }
    });
  }

  // Attach error bar groups to meta for tooltip
  canvas._chartMeta.errorBarGroups = errorBarGroups;
}

// ── Residuals chart ───────────────────────────────────────────────
function drawResiduals(canvas, xData, yData, fitResult, modelType, theme = {}) {
  const t = theme;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = { top: 20, right: 30, bottom: 40, left: 70 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const modelFn = getModelFn(modelType);
  const res = xData.map((x, i) => yData[i] - modelFn(x, fitResult.params));
  const logX = xData.filter(x => x > 0).map(x => Math.log10(x));

  let xMin = Math.floor(Math.min(...logX)) - 0.5;
  let xMax = Math.ceil(Math.max(...logX)) + 0.5;
  const rMax = Math.max(Math.abs(Math.min(...res)), Math.abs(Math.max(...res))) * 1.3 || 1;

  const toCanvasX = (lx) => pad.left + ((lx - xMin) / (xMax - xMin)) * plotW;
  const toCanvasY = (r) => pad.top + ((rMax - r) / (2 * rMax)) * plotH;

  ctx.fillStyle = t.canvas || "#0a0f1a";
  ctx.fillRect(0, 0, W, H);

  // Zero line
  ctx.strokeStyle = t.orangeBorder || "rgba(255,180,50,0.3)";
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  const zeroY = toCanvasY(0);
  ctx.beginPath(); ctx.moveTo(pad.left, zeroY); ctx.lineTo(pad.left + plotW, zeroY); ctx.stroke();
  ctx.setLineDash([]);

  // Points
  xData.forEach((x, i) => {
    if (x <= 0) return;
    const lx = Math.log10(x);
    const cx = toCanvasX(lx), cy = toCanvasY(res[i]);
    const color = res[i] >= 0 ? (t.teal || "#00e6b4") : (t.red || "#ff6b8a");
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    
    // Line to zero
    ctx.strokeStyle = color + "44";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, zeroY); ctx.stroke();
  });

  ctx.fillStyle = t.axisLabel || "rgba(160,190,230,0.6)";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("Residuals", pad.left + plotW / 2, H - 6);
}

// ── Sample Data ───────────────────────────────────────────────────
const EXAMPLE_DATASETS = {
  "Full sigmoid (5 reps, outliers)": `Concentration,Rep1,Rep2,Rep3,Rep4,Rep5
0.001,0.12,0.15,0.10,0.13,0.11
0.003,0.14,0.17,0.13,0.15,0.16
0.01,0.18,0.22,0.19,0.20,0.21
0.03,0.30,0.35,0.28,0.32,0.31
0.1,0.58,0.65,0.55,0.61,0.59
0.3,1.35,1.48,1.30,1.40,1.38
1,3.10,3.28,3.05,3.18,3.12
3,4.20,4.35,4.15,4.28,4.22
10,4.62,4.78,4.55,4.70,6.25
30,4.80,4.92,4.76,4.85,4.82
100,4.88,4.98,4.84,4.90,4.86
300,4.92,5.02,4.88,3.15,4.94`,

  "Incomplete top plateau": `Concentration,Rep1,Rep2,Rep3
0.001,0.08,0.11,0.09
0.003,0.10,0.13,0.11
0.01,0.15,0.19,0.16
0.03,0.28,0.34,0.30
0.1,0.62,0.71,0.65
0.3,1.45,1.60,1.50
1,2.80,3.05,2.88
3,3.65,3.88,3.72`,

  "Incomplete bottom plateau": `Concentration,Rep1,Rep2,Rep3
0.3,1.40,1.55,1.45
1,2.85,3.02,2.90
3,3.90,4.10,3.95
10,4.50,4.68,4.55
30,4.78,4.92,4.82
100,4.90,5.02,4.94
300,4.95,5.06,4.98
1000,4.98,5.08,5.00`,

  "No plateaus (mid-curve only)": `Concentration,Rep1,Rep2,Rep3
0.1,0.55,0.63,0.58
0.3,1.30,1.45,1.35
1,2.70,2.90,2.78
3,3.60,3.80,3.68
10,4.15,4.35,4.22
30,4.45,4.60,4.50`,

  "Steep Hill slope (cooperative)": `Concentration,Rep1,Rep2,Rep3
0.001,0.10,0.12,0.11
0.003,0.11,0.13,0.12
0.01,0.12,0.15,0.13
0.03,0.14,0.18,0.15
0.1,0.20,0.25,0.22
0.3,0.85,1.10,0.95
1,4.50,4.70,4.58
3,4.85,5.00,4.90
10,4.90,5.05,4.95
30,4.92,5.06,4.96
100,4.93,5.07,4.97`,

  "Shallow Hill slope": `Concentration,Rep1,Rep2,Rep3
0.001,0.50,0.58,0.53
0.003,0.65,0.74,0.68
0.01,0.90,1.02,0.95
0.03,1.25,1.38,1.30
0.1,1.75,1.90,1.82
0.3,2.30,2.48,2.38
1,2.90,3.10,2.98
3,3.35,3.55,3.42
10,3.70,3.88,3.78
30,3.95,4.12,4.02
100,4.15,4.30,4.20
300,4.30,4.45,4.35`,

  "Decreasing response": `Concentration,Rep1,Rep2,Rep3
0.001,4.90,5.05,4.95
0.003,4.88,5.02,4.92
0.01,4.82,4.98,4.88
0.03,4.65,4.80,4.72
0.1,4.10,4.30,4.18
0.3,3.05,3.25,3.12
1,1.55,1.72,1.62
3,0.65,0.78,0.70
10,0.25,0.35,0.28
30,0.14,0.20,0.16
100,0.10,0.15,0.12`,

  "Asymmetric (5PL)": `Concentration,Rep1,Rep2,Rep3
0.001,0.22,0.25,0.20
0.003,0.32,0.36,0.30
0.01,0.52,0.58,0.48
0.03,0.85,0.94,0.80
0.1,1.55,1.68,1.48
0.3,2.60,2.75,2.50
1,3.85,3.98,3.78
3,4.52,4.65,4.45
10,4.82,4.95,4.78
30,4.92,5.02,4.88
100,4.96,5.05,4.92
300,4.98,5.06,4.94`,

  "High variability": `Concentration,Rep1,Rep2,Rep3,Rep4,Rep5
0.001,0.05,0.22,0.12,0.18,0.08
0.01,0.10,0.30,0.18,0.25,0.14
0.1,0.40,0.85,0.60,0.72,0.48
1,2.10,3.40,2.70,3.15,2.35
10,4.00,5.20,4.55,4.90,4.15
100,4.50,5.40,4.90,5.15,4.65
1000,4.60,5.50,5.00,5.20,4.75`,

  "Simple (no replicates)": `Concentration,Response
0.001,0.10
0.01,0.15
0.1,0.55
1,2.80
10,4.55
100,4.90
1000,4.98`,
};

const SAMPLE_DATA = EXAMPLE_DATASETS["Full sigmoid (5 reps, outliers)"];

// ── Main Component ────────────────────────────────────────────────
export default function BioassayCurveFitter() {
  const [rawData, setRawData] = useState(SAMPLE_DATA);
  const [modelType, setModelType] = useState("Auto");
  const [normalize, setNormalize] = useState(false);
  const [fixedMin, setFixedMin] = useState("");
  const [fixedMax, setFixedMax] = useState("");
  const [fixedHill, setFixedHill] = useState("1");
  const [showPdfModal, setShowPdfModal] = useState(false);
  const [pdfSections, setPdfSections] = useState({
    modelInfo: true,
    modelParams: true,
    plot: true,
    rawData: true,
    fitParams: true,
    paramR2: true,
    paramRMSE: true,
    paramSSR: true,
    paramAIC: false,
    paramAICc: true,
    paramBIC: true,
    paramEC50: true,
    paramBioEC50: true,
    paramConverged: true,
    paramNK: true,
    modelComparison: true,
    outlierResults: true,
    backgroundInfo: true,
    normalizationInfo: true,
  });
  const [pdfGenerating, setPdfGenerating] = useState(false);

  // Pre-populate fixed param fields from data when switching to constrained models
  useEffect(() => {
    if (!["1PL", "2PL", "3PL"].includes(modelType)) return;
    // Try to estimate from parsedData, or fall back to raw data parsing
    let yVals = null;
    if (parsedData && parsedData.yData && parsedData.yData.length > 0) {
      yVals = parsedData.yData;
    } else {
      try {
        const { yData } = parseData(rawData);
        if (yData && yData.length > 0) yVals = yData;
      } catch (e) { /* ignore */ }
    }
    if (!yVals) return;

    if (modelType === "1PL" || modelType === "2PL") {
      // Only pre-populate if fields are empty
      if (fixedMin === "") {
        const sorted = [...yVals].sort((a, b) => a - b);
        const lowN = Math.max(1, Math.floor(sorted.length * 0.2));
        const estMin = sorted.slice(0, lowN).reduce((a, b) => a + b, 0) / lowN;
        setFixedMin(estMin.toPrecision(4));
      }
      if (fixedMax === "") {
        const sorted = [...yVals].sort((a, b) => a - b);
        const lowN = Math.max(1, Math.floor(sorted.length * 0.2));
        const estMax = sorted.slice(-lowN).reduce((a, b) => a + b, 0) / lowN;
        setFixedMax(estMax.toPrecision(4));
      }
    }
  }, [modelType]); // only on model switch, not on every data change
  const [fitResult, setFitResult] = useState(null);
  const [error, setError] = useState(null);
  const [parsedData, setParsedData] = useState(null);
  const [showResiduals, setShowResiduals] = useState(false);
  const [weightsType, setWeightsType] = useState("none");
  const [comparison, setComparison] = useState(null); // { fit4PL, fit5PL, selected, reason }
  const [activeModel, setActiveModel] = useState("4PL"); // which model is currently displayed
  const [pointView, setPointView] = useState("individual"); // "individual" or "errorbars"
  const [errorBarType, setErrorBarType] = useState("sd"); // "sd" or "sem"
  const [grubbsAlpha, setGrubbsAlpha] = useState(0.05);
  const [grubbsResults, setGrubbsResults] = useState(null);
  const [showOutliers, setShowOutliers] = useState(false);
  const [excludedIndices, setExcludedIndices] = useState(new Set()); // manually excluded data point indices
  const [selectedGrubbsGroup, setSelectedGrubbsGroup] = useState(null); // concentration key for expanded view
  const [bgRawData, setBgRawData] = useState("");
  const [bgEnabled, setBgEnabled] = useState(false);
  const [bgStats, setBgStats] = useState(null); // { mean, sd, n, values }
  const [theme, setTheme] = useState("dark"); // "dark" or "light"
  const [isMobile, setIsMobile] = useState(false);

  // Responsive breakpoint
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Theme color palettes
  const t = useMemo(() => {
    const dark = {
      bg: "linear-gradient(160deg, #060a12 0%, #0c1324 40%, #0a1020 100%)",
      text: "#c8daf0",
      textMuted: "rgba(160,190,230,0.5)",
      textDim: "rgba(140,170,210,0.4)",
      textFaint: "rgba(140,170,210,0.3)",
      label: "rgba(160,190,230,0.7)",
      labelDim: "rgba(160,190,230,0.6)",
      panel: "rgba(12,20,40,0.8)",
      panelBorder: "rgba(60,100,160,0.15)",
      panelBorderLight: "rgba(60,100,160,0.08)",
      input: "rgba(6,10,20,0.8)",
      inputBorder: "rgba(60,100,160,0.2)",
      canvas: "#0a0f1a",
      grid: "rgba(100,140,180,0.08)",
      axis: "rgba(140,170,210,0.3)",
      axisLabel: "rgba(160,190,230,0.6)",
      teal: "#00e6b4",
      tealBg: "rgba(0,230,180,0.12)",
      tealBorder: "rgba(0,230,180,0.3)",
      tealGlow: "rgba(0,230,180,",
      blue: "#3b9eff",
      blueBg: "rgba(59,158,255,0.15)",
      blueBorder: "rgba(59,158,255,0.3)",
      purple: "#a855f7",
      purpleBg: "rgba(168,85,247,0.15)",
      purpleBorder: "rgba(168,85,247,0.3)",
      orange: "#ffb432",
      orangeBg: "rgba(255,180,50,0.15)",
      orangeBorder: "rgba(255,180,50,0.3)",
      red: "#ff6b8a",
      redBg: "rgba(255,80,106,0.12)",
      redBorder: "rgba(255,80,106,0.3)",
      redGlow: "rgba(255,80,100,",
      tooltip: "rgba(10,16,30,0.92)",
      tooltipBorder: "rgba(80,120,180,0.25)",
      scrollTrack: "rgba(0,0,0,0.2)",
      scrollThumb: "rgba(100,140,200,0.3)",
      btnInactive: "rgba(6,10,20,0.5)",
      btnInactiveBorder: "rgba(60,100,160,0.1)",
      btnInactiveText: "rgba(160,190,230,0.4)",
    };
    const light = {
      bg: "linear-gradient(160deg, #f0f4f8 0%, #e8edf4 40%, #f2f5fa 100%)",
      text: "#1a2a40",
      textMuted: "rgba(60,80,110,0.6)",
      textDim: "rgba(60,80,110,0.5)",
      textFaint: "rgba(60,80,110,0.35)",
      label: "rgba(40,60,90,0.75)",
      labelDim: "rgba(40,60,90,0.6)",
      panel: "rgba(255,255,255,0.85)",
      panelBorder: "rgba(60,100,160,0.15)",
      panelBorderLight: "rgba(60,100,160,0.08)",
      input: "rgba(245,248,252,0.9)",
      inputBorder: "rgba(60,100,160,0.2)",
      canvas: "#f8fafd",
      grid: "rgba(60,100,160,0.08)",
      axis: "rgba(60,80,120,0.25)",
      axisLabel: "rgba(40,60,100,0.55)",
      teal: "#009e7e",
      tealBg: "rgba(0,158,126,0.1)",
      tealBorder: "rgba(0,158,126,0.3)",
      tealGlow: "rgba(0,158,126,",
      blue: "#2563eb",
      blueBg: "rgba(37,99,235,0.1)",
      blueBorder: "rgba(37,99,235,0.3)",
      purple: "#7c3aed",
      purpleBg: "rgba(124,58,237,0.1)",
      purpleBorder: "rgba(124,58,237,0.3)",
      orange: "#d97706",
      orangeBg: "rgba(217,119,6,0.1)",
      orangeBorder: "rgba(217,119,6,0.3)",
      red: "#e11d48",
      redBg: "rgba(225,29,72,0.08)",
      redBorder: "rgba(225,29,72,0.25)",
      redGlow: "rgba(225,29,72,",
      tooltip: "rgba(255,255,255,0.95)",
      tooltipBorder: "rgba(60,100,160,0.2)",
      scrollTrack: "rgba(0,0,0,0.05)",
      scrollThumb: "rgba(60,100,160,0.2)",
      btnInactive: "rgba(240,243,248,0.8)",
      btnInactiveBorder: "rgba(60,100,160,0.12)",
      btnInactiveText: "rgba(60,80,110,0.45)",
    };
    return theme === "dark" ? dark : light;
  }, [theme]);

  const mainCanvasRef = useRef(null);
  const residCanvasRef = useRef(null);
  const tooltipRef = useRef(null);
  const chartContainerRef = useRef(null);

  const parseData = useCallback((text) => {
    const lines = text.trim().split("\n").filter(l => l.trim());
    const xData = [], yData = [];
    let startIdx = 0;

    // Detect format: is it tab-delimited with comma-formatted numbers?
    // e.g. "1000.000000\t47,189.7\t44,534.5"
    const firstLine = lines[0] || "";
    const hasTabDelim = firstLine.includes("\t");

    // Check if header row (first token is non-numeric)
    const firstToken = hasTabDelim
      ? firstLine.split("\t")[0].trim()
      : firstLine.split(/[,\t]/)[0].trim();
    if (firstToken && isNaN(parseFloat(firstToken.replace(/,/g, "")))) startIdx = 1;

    // Helper: parse a number that may have thousands commas (e.g. "47,189.7" → 47189.7)
    const parseNum = (s) => {
      if (!s) return NaN;
      s = s.trim();
      // If the string has commas and a decimal point, treat commas as thousands separators
      // e.g. "47,189.7" → "47189.7"
      // But also handle plain comma-separated values like "0.01,0.5"
      if (s.includes(",") && s.includes(".")) {
        // Thousands-separator pattern: digits,digits with optional decimal
        s = s.replace(/,/g, "");
      }
      return Number(s);
    };

    for (let i = startIdx; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      if (hasTabDelim) {
        // Tab-delimited: first column = concentration, remaining = replicate responses
        // Each value may contain commas as thousands separators
        const cols = line.split("\t").map(s => s.trim()).filter(s => s);
        if (cols.length < 2) continue;
        const conc = parseNum(cols[0]);
        if (isNaN(conc)) continue;

        for (let j = 1; j < cols.length; j++) {
          const resp = parseNum(cols[j]);
          if (!isNaN(resp)) {
            xData.push(conc);
            yData.push(resp);
          }
        }
      } else {
        // CSV or space-delimited: try to detect multi-column replicates
        // First, try splitting on comma (standard CSV)
        let parts = line.split(",").map(s => s.trim());
        
        // Check if we have simple two-column CSV (no thousands commas)
        // vs. multi-column with plain numbers
        if (parts.length >= 2) {
          const allNumeric = parts.every(p => !isNaN(Number(p)));
          if (allNumeric) {
            // Simple CSV with multiple columns: col 0 = conc, rest = replicates
            const conc = Number(parts[0]);
            if (!isNaN(conc)) {
              for (let j = 1; j < parts.length; j++) {
                const resp = Number(parts[j]);
                if (!isNaN(resp)) {
                  xData.push(conc);
                  yData.push(resp);
                }
              }
            }
            continue;
          }
        }

        // Fallback: space-delimited
        parts = line.split(/\s+/).map(s => parseNum(s));
        if (parts.length >= 2 && !isNaN(parts[0])) {
          const conc = parts[0];
          for (let j = 1; j < parts.length; j++) {
            if (!isNaN(parts[j])) {
              xData.push(conc);
              yData.push(parts[j]);
            }
          }
        }
      }
    }
    return { xData, yData };
  }, []);

  // Parse background values: accepts a flat list of numbers (any delimiter)
  const parseBgValues = useCallback((text) => {
    if (!text.trim()) return null;
    const values = [];
    // Handle same formats as main parser: tab, comma, space, newline delimited
    // Also handle comma-formatted thousands separators
    const tokens = text.replace(/\n/g, "\t").split(/[\t]+/);
    for (const token of tokens) {
      // Each token might contain comma-separated values or comma-formatted numbers
      const parts = token.split(",").map(s => s.trim()).filter(s => s);
      // Check if it looks like thousands-separated: "47,189.7" → single number
      // vs comma-delimited: "100,200,300" → multiple numbers
      const rejoined = parts.join(",");
      if (parts.length >= 2 && rejoined.includes(".")) {
        // Could be thousands-formatted; try parsing as single number
        const asOne = Number(rejoined.replace(/,/g, ""));
        if (!isNaN(asOne)) { values.push(asOne); continue; }
      }
      // Otherwise treat each comma-part as separate
      for (const p of parts) {
        const v = Number(p.replace(/,/g, ""));
        if (!isNaN(v) && p.trim()) values.push(v);
      }
    }
    if (values.length === 0) return null;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sd = values.length > 1
      ? Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1))
      : 0;
    return { mean, sd, n: values.length, values };
  }, []);

  const runFit = useCallback(() => {
    try {
      setError(null);
      setComparison(null);
      const { xData, yData: yRaw } = parseData(rawData);
      if (xData.length < 4) { setError("Need at least 4 data points"); return; }

      // Background subtraction
      let bgSub = null;
      let yData = yRaw;
      if (bgEnabled && bgRawData.trim()) {
        bgSub = parseBgValues(bgRawData);
        if (!bgSub) { setError("Could not parse background values"); return; }
        setBgStats(bgSub);
        yData = yRaw.map(y => y - bgSub.mean);
      } else {
        setBgStats(null);
      }

      // Normalization: scale to 0-100% using raw min/max
      let normMin = 0, normMax = 1, normalized = false;
      if (normalize) {
        normMin = Math.min(...yData);
        normMax = Math.max(...yData);
        const range = normMax - normMin;
        if (range > 0) {
          yData = yData.map(y => ((y - normMin) / range) * 100);
          normalized = true;
        }
      }

      setParsedData({ xData, yData, yRaw, bgSubtracted: bgSub ? bgSub.mean : 0, normMin, normMax, normalized });

      if (modelType === "Auto") {
        // Fit 4PL and 5PL, compare
        const fit4 = fitModel(xData, yData, model4PL, false);
        const fit5 = xData.length >= 5 ? fitModel(xData, yData, model5PL, true) : null;

        if (!fit4 && !fit5) { setError("Fitting failed for both models. Check your data."); return; }
        if (!fit5) {
          setComparison({ fit4PL: fit4, fit5PL: null, selected: "4PL", reason: "Too few points for 5PL" });
          setActiveModel("4PL");
          setFitResult(fit4);
          return;
        }

        // Compare using AICc (preferred for small n) and BIC
        const deltaAICc = fit4.aicc - fit5.aicc; // positive => 5PL better
        const deltaBIC = fit4.bic - fit5.bic;
        const eParam = fit5.params[4];
        const eNearOne = Math.abs(eParam - 1) < 0.05;

        let selected, reason;
        if (eNearOne) {
          selected = "4PL";
          reason = `5PL asymmetry parameter S≈${eParam.toFixed(3)} (near 1.0); extra parameter not justified`;
        } else if (deltaAICc > 2 && deltaBIC > 0) {
          selected = "5PL";
          reason = `5PL preferred: ΔAICc=${deltaAICc.toFixed(1)} (>2 threshold), ΔBIC=${deltaBIC.toFixed(1)}`;
        } else if (deltaAICc < -2) {
          selected = "4PL";
          reason = `4PL preferred: ΔAICc=${deltaAICc.toFixed(1)} favors simpler model`;
        } else {
          selected = "4PL";
          reason = `Models within ΔAICc=${deltaAICc.toFixed(1)}; 4PL preferred by parsimony`;
        }

        setComparison({ fit4PL: fit4, fit5PL: fit5, selected, reason });
        setActiveModel(selected);
        setFitResult(selected === "4PL" ? fit4 : fit5);
      } else if (modelType === "1PL") {
        // Fix A, B=fixedHill, D; fit only C (EC50)
        const aVal = parseFloat(fixedMin), dVal = parseFloat(fixedMax), hVal = parseFloat(fixedHill);
        if (isNaN(aVal) || isNaN(dVal)) { setError("1PL requires min and max asymptote values"); return; }
        if (isNaN(hVal) || hVal === 0) { setError("1PL requires a non-zero Hill slope value"); return; }
        const result = fitConstrainedModel(xData, yData, { 0: aVal, 1: hVal, 3: dVal });
        if (!result) { setError("1PL fitting failed. Check your data."); return; }
        setActiveModel("1PL");
        setFitResult(result);
      } else if (modelType === "2PL") {
        // Fix A and D; fit B and C
        const aVal = parseFloat(fixedMin), dVal = parseFloat(fixedMax);
        if (isNaN(aVal) || isNaN(dVal)) { setError("2PL requires min and max asymptote values"); return; }
        const result = fitConstrainedModel(xData, yData, { 0: aVal, 3: dVal });
        if (!result) { setError("2PL fitting failed. Check your data."); return; }
        setActiveModel("2PL");
        setFitResult(result);
      } else if (modelType === "3PL") {
        // 4PL with B fixed to fixedHill; fit A, C, D
        const hVal = parseFloat(fixedHill);
        if (isNaN(hVal) || hVal === 0) { setError("3PL requires a non-zero Hill slope value"); return; }
        const result = fitConstrainedModel(xData, yData, { 1: hVal });
        if (!result) { setError("3PL fitting failed. Check your data."); return; }
        setActiveModel("3PL");
        setFitResult(result);
      } else {
        // Manual 4PL or 5PL
        if (modelType === "5PL" && xData.length < 5) { setError("Need at least 5 data points for 5PL"); return; }
        const modelFn = getModelFn(modelType);
        const result = fitModel(xData, yData, modelFn, modelType === "5PL");
        if (!result) { setError("Fitting failed to converge. Check your data."); return; }
        setActiveModel(modelType);
        setFitResult(result);
        setComparison(null);
      }
    } catch (e) {
      setError("Error: " + e.message);
    }
  }, [rawData, modelType, parseData, bgEnabled, bgRawData, parseBgValues, normalize, fixedMin, fixedMax, fixedHill]);

  // Merged set of outlier + excluded indices for chart display
  const chartOutlierIndices = useMemo(() => {
    const s = new Set(excludedIndices);
    if (showOutliers && grubbsResults) {
      for (const idx of grubbsResults.outlierIndices) s.add(idx);
    }
    return s.size > 0 ? s : null;
  }, [excludedIndices, showOutliers, grubbsResults]);

  // Draw charts whenever data changes
  useEffect(() => {
    if (mainCanvasRef.current && parsedData) {
      drawChart(mainCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, { pointView, errorBarType, outlierIndices: chartOutlierIndices, excludedIndices }, t);
    } else if (mainCanvasRef.current) {
      const canvas = mainCanvasRef.current;
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.getBoundingClientRect().width * dpr;
      canvas.height = canvas.getBoundingClientRect().height * dpr;
      ctx.scale(dpr, dpr);
      ctx.fillStyle = t.canvas || "#0a0f1a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, [parsedData, fitResult, activeModel, pointView, errorBarType, chartOutlierIndices, excludedIndices, t]);

  useEffect(() => {
    if (residCanvasRef.current && parsedData && fitResult && showResiduals) {
      drawResiduals(residCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, t);
    }
  }, [parsedData, fitResult, activeModel, showResiduals, t]);

  // Resize handler
  useEffect(() => {
    const handleResize = () => {
      if (mainCanvasRef.current && parsedData) {
        drawChart(mainCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, { pointView, errorBarType, outlierIndices: chartOutlierIndices, excludedIndices }, t);
      }
      if (residCanvasRef.current && parsedData && fitResult && showResiduals) {
        drawResiduals(residCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, t);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [parsedData, fitResult, activeModel, showResiduals, pointView, errorBarType, chartOutlierIndices, excludedIndices, t]);

  // Tooltip handler for main chart
  useEffect(() => {
    const canvas = mainCanvasRef.current;
    const tooltip = tooltipRef.current;
    if (!canvas || !tooltip) return;

    const handleMouseMove = (e) => {
      const meta = canvas._chartMeta;
      if (!meta || !parsedData) { tooltip.style.display = "none"; return; }

      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const { pad, plotW, plotH, xMin, xMax, yMin, yMax } = meta;
      const mt = meta.theme || {};

      // Check if mouse is within plot area
      if (mx < pad.left || mx > pad.left + plotW || my < pad.top || my > pad.top + plotH) {
        tooltip.style.display = "none";
        return;
      }

      // Convert mouse position to data coordinates
      const logXMouse = xMin + (mx - pad.left) / plotW * (xMax - xMin);
      const yMouse = yMax - (my - pad.top) / plotH * (yMax - yMin);

      let nearestDist = Infinity;
      let nearestInfo = null;
      const hitRadius = 12;

      // In error bar mode, check proximity to aggregated group points
      if (pointView === "errorbars" && meta.errorBarGroups && meta.errorBarGroups.length > 0) {
        for (const g of meta.errorBarGroups) {
          const dist = Math.sqrt((mx - g.cx) ** 2 + (my - g.cyMean) ** 2);
          if (dist < 16 && dist < nearestDist) {
            nearestDist = dist;
            nearestInfo = { type: "errorbar", group: g };
          }
        }
      } else {
        // Check proximity to individual data points (within 12px)
        parsedData.xData.forEach((x, i) => {
          if (x <= 0) return;
          const lx = Math.log10(x);
          const cx = meta.toCanvasX(lx);
          const cy = meta.toCanvasY(parsedData.yData[i]);
          const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
          if (dist < hitRadius && dist < nearestDist) {
            nearestDist = dist;
            nearestInfo = { type: "point", x, y: parsedData.yData[i], index: i };
          }
        });
      }

      // If no data point nearby, show curve value
      if (!nearestInfo && fitResult) {
        const modelFn = getModelFn(activeModel);
        const xVal = Math.pow(10, logXMouse);
        const yFit = modelFn(xVal, fitResult.params);
        const cyFit = meta.toCanvasY(yFit);
        if (Math.abs(my - cyFit) < 20) {
          nearestInfo = { type: "curve", x: xVal, y: yFit };
        }
      }

      if (nearestInfo) {
        const containerRect = chartContainerRef.current ? chartContainerRef.current.getBoundingClientRect() : rect;

        if (nearestInfo.type === "errorbar") {
          const g = nearestInfo.group;
          const fmtX = g.x < 0.01 || g.x >= 10000 ? g.x.toExponential(3) : g.x.toPrecision(4);
          const fmtMean = Math.abs(g.mean) < 0.01 || Math.abs(g.mean) >= 100000 ? g.mean.toExponential(4) : g.mean.toFixed(1);
          const fmtSD = g.sd < 0.01 ? g.sd.toExponential(2) : g.sd < 100 ? g.sd.toFixed(2) : g.sd.toFixed(0);
          const fmtSEM = g.sem < 0.01 ? g.sem.toExponential(2) : g.sem < 100 ? g.sem.toFixed(2) : g.sem.toFixed(0);
          const cvColor = g.cv > 20 ? (mt.red || "#ff6b8a") : g.cv > 10 ? (mt.orange || "#ffb432") : (mt.teal || "#00e6b4");
          const nLabel = g.n < g.nTotal ? `${g.n}/${g.nTotal}` : `${g.n}`;

          tooltip.innerHTML = [
            `<div style="color:${mt.teal || '#00e6b4'};font-weight:600;margin-bottom:3px">Conc: ${fmtX}</div>`,
            `<div style="display:grid;grid-template-columns:auto auto;gap:1px 10px;font-size:9px">`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">Mean</span><span>${fmtMean}</span>`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">SD</span><span>±${fmtSD}</span>`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">SEM</span><span>±${fmtSEM}</span>`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">%CV</span><span style="color:${cvColor}">${g.cv.toFixed(1)}%</span>`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">n</span><span>${nLabel}</span>`,
            `</div>`,
          ].join("");

          let tx = e.clientX - containerRect.left + 14;
          let ty = e.clientY - containerRect.top - 80;
          const tw = tooltip.offsetWidth || 160;
          if (tx + tw > containerRect.width - 8) tx = e.clientX - containerRect.left - tw - 10;
          if (ty < 4) ty = 4;
          tooltip.style.left = tx + "px";
          tooltip.style.top = ty + "px";
          tooltip.style.display = "block";
        } else {
          const fmtX = nearestInfo.x < 0.01 || nearestInfo.x >= 10000
            ? nearestInfo.x.toExponential(3) : nearestInfo.x.toPrecision(4);
          const fmtY = Math.abs(nearestInfo.y) < 0.01 || Math.abs(nearestInfo.y) >= 100000
            ? nearestInfo.y.toExponential(3) : nearestInfo.y.toFixed(1);

          let label = nearestInfo.type === "curve" ? "Fit" : "Data";
          tooltip.innerHTML = `<span style="color:${nearestInfo.type === "curve" ? (activeModel === "4PL" ? (mt.blue || "#3b9eff") : (mt.purple || "#a855f7")) : (mt.teal || "#00e6b4")}">${label}</span>&nbsp; x: ${fmtX}&nbsp; y: ${fmtY}`;
          
          let tx = e.clientX - containerRect.left + 14;
          let ty = e.clientY - containerRect.top - 28;
          const tw = tooltip.offsetWidth || 160;
          if (tx + tw > containerRect.width - 8) tx = e.clientX - containerRect.left - tw - 10;
          if (ty < 4) ty = 4;
          tooltip.style.left = tx + "px";
          tooltip.style.top = ty + "px";
          tooltip.style.display = "block";
        }
      } else {
        tooltip.style.display = "none";
      }
    };

    const handleMouseLeave = () => {
      tooltip.style.display = "none";
    };

    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseleave", handleMouseLeave);
    return () => {
      canvas.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("mouseleave", handleMouseLeave);
    };
  }, [parsedData, fitResult, activeModel, pointView]);

  const paramLabels = activeModel === "5PL"
    ? ["Bottom", "Hill", "EC50", "Top", "S (asymmetry)"]
    : ["A (min)", "B (slope)", "C (EC50)", "D (max)"];

  // Which params are fixed for constrained models (indices into 4PL param vector)
  const fixedParams = activeModel === "1PL" ? new Set([0, 1, 3])
    : activeModel === "2PL" ? new Set([0, 3])
    : activeModel === "3PL" ? new Set([1])
    : new Set();

  const hasReplicates = useMemo(() => {
    if (!parsedData) return false;
    const uniqueConc = new Set(parsedData.xData).size;
    return parsedData.xData.length > uniqueConc;
  }, [parsedData]);

  // Grouped data with Grubbs results per concentration
  const groupedData = useMemo(() => {
    if (!parsedData) return [];
    return groupByConcentration(parsedData.xData, parsedData.yData);
  }, [parsedData]);

  // Run Grubbs test on all groups
  const runGrubbs = useCallback(() => {
    if (!parsedData) return;
    const result = runGrubbsAllGroups(parsedData.xData, parsedData.yData, grubbsAlpha);
    setGrubbsResults(result);
    setShowOutliers(true);
    // Auto-select the first group that has outliers, or the first group
    const firstOutlierGroup = result.groupResults.find(g => g.outlierCount > 0);
    setSelectedGrubbsGroup(firstOutlierGroup ? firstOutlierGroup.x.toString() : (groupedData[0] ? groupedData[0].x.toString() : null));
  }, [parsedData, grubbsAlpha, groupedData]);

  // Toggle exclusion of a specific data point index
  const toggleExclusion = useCallback((idx) => {
    setExcludedIndices(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  // Exclude all detected outliers at once
  const excludeAllOutliers = useCallback(() => {
    if (!grubbsResults) return;
    setExcludedIndices(prev => {
      const next = new Set(prev);
      for (const idx of grubbsResults.outlierIndices) next.add(idx);
      return next;
    });
  }, [grubbsResults]);

  // Clear all exclusions
  const clearExclusions = useCallback(() => {
    setExcludedIndices(new Set());
  }, []);

  // Refit model using only non-excluded data
  const refitWithoutExcluded = useCallback(() => {
    if (!parsedData || excludedIndices.size === 0) return;
    const xFiltered = [], yFiltered = [];
    parsedData.xData.forEach((x, i) => {
      if (!excludedIndices.has(i)) { xFiltered.push(x); yFiltered.push(parsedData.yData[i]); }
    });
    if (xFiltered.length < 4) { setError("Too few points remaining after exclusion"); return; }
    setError(null);

    if (modelType === "Auto") {
      const fit4 = fitModel(xFiltered, yFiltered, model4PL, false);
      const fit5 = xFiltered.length >= 5 ? fitModel(xFiltered, yFiltered, model5PL, true) : null;
      if (!fit4 && !fit5) { setError("Fitting failed"); return; }
      if (!fit5) { setActiveModel("4PL"); setFitResult(fit4); return; }
      const deltaAICc = fit4.aicc - fit5.aicc;
      const eNearOne = Math.abs(fit5.params[4] - 1) < 0.05;
      const selected = (eNearOne || deltaAICc <= 2) ? "4PL" : "5PL";
      setActiveModel(selected);
      setFitResult(selected === "4PL" ? fit4 : fit5);
      setComparison({ fit4PL: fit4, fit5PL: fit5, selected, reason: "Refit after exclusion" });
    } else if (["1PL", "2PL", "3PL"].includes(modelType)) {
      const aVal = parseFloat(fixedMin), dVal = parseFloat(fixedMax), hVal = parseFloat(fixedHill);
      if (["1PL", "2PL"].includes(modelType) && (isNaN(aVal) || isNaN(dVal))) {
        setError(`${modelType} requires min and max asymptote values`); return;
      }
      if (["1PL", "3PL"].includes(modelType) && (isNaN(hVal) || hVal === 0)) {
        setError(`${modelType} requires a non-zero Hill slope value`); return;
      }
      const fixedMap = modelType === "1PL" ? { 0: aVal, 1: hVal, 3: dVal }
        : modelType === "2PL" ? { 0: aVal, 3: dVal }
        : { 1: hVal };
      const result = fitConstrainedModel(xFiltered, yFiltered, fixedMap);
      if (!result) { setError("Fitting failed"); return; }
      setActiveModel(modelType);
      setFitResult(result);
    } else {
      const modelFn = getModelFn(modelType);
      const result = fitModel(xFiltered, yFiltered, modelFn, modelType === "5PL");
      if (!result) { setError("Fitting failed"); return; }
      setActiveModel(modelType);
      setFitResult(result);
    }
  }, [parsedData, excludedIndices, modelType, fixedMin, fixedMax, fixedHill]);

  const exportCSV = useCallback(() => {
    if (!parsedData || !fitResult) return;
    const modelFn = getModelFn(activeModel);
    const hasBg = parsedData.bgSubtracted > 0;
    let csv = "";
    csv += `# Model: ${activeModel}\n`;
    if (hasBg) csv += `# Background subtracted: ${parsedData.bgSubtracted.toFixed(2)}\n`;
    if (parsedData.normalized) csv += `# Normalized: 0-100% (min=${parsedData.normMin.toFixed(4)}, max=${parsedData.normMax.toFixed(4)})\n`;
    csv += hasBg
      ? "Concentration,Raw,BgSubtracted,Fitted,Residual\n"
      : "Concentration,Observed,Fitted,Residual\n";
    parsedData.xData.forEach((x, i) => {
      const fitted = modelFn(x, fitResult.params);
      const resid = parsedData.yData[i] - fitted;
      if (hasBg && parsedData.yRaw) {
        csv += `${x},${parsedData.yRaw[i]},${parsedData.yData[i].toFixed(6)},${fitted.toFixed(6)},${resid.toFixed(6)}\n`;
      } else {
        csv += `${x},${parsedData.yData[i]},${fitted.toFixed(6)},${resid.toFixed(6)}\n`;
      }
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `bioassay_${activeModel}_fit.csv`; a.click();
    URL.revokeObjectURL(url);
  }, [parsedData, fitResult, activeModel]);

  const exportImage = useCallback((format) => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const mimeType = format === "jpeg" ? "image/jpeg" : "image/png";
    const quality = format === "jpeg" ? 0.95 : undefined;
    // For JPEG, fill with background color first (canvas transparency becomes black)
    let exportCanvas = canvas;
    if (format === "jpeg") {
      exportCanvas = document.createElement("canvas");
      exportCanvas.width = canvas.width;
      exportCanvas.height = canvas.height;
      const ctx = exportCanvas.getContext("2d");
      ctx.fillStyle = t.canvas || "#0a0f1a";
      ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
      ctx.drawImage(canvas, 0, 0);
    }
    const dataUrl = exportCanvas.toDataURL(mimeType, quality);
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `bioassay_${activeModel}_fit.${format}`;
    a.click();
  }, [activeModel, t]);

  // PDF Report Generation
  const generatePdfReport = useCallback(async () => {
    if (!parsedData || !fitResult) return;
    setPdfGenerating(true);
    try {
      const { jsPDF } = await import("jspdf");
      const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
      const pw = 210, margin = 15;
      const cw = pw - 2 * margin;
      let y = margin;
      const gray = [100, 100, 100];
      const dark = [30, 30, 30];
      const accent = [0, 150, 120];
      const lineH = 5;

      const addTitle = (text) => { doc.setFontSize(16); doc.setFont("helvetica","bold"); doc.setTextColor(...dark); doc.text(text, margin, y); y += 8; };
      const addSection = (text) => { if (y > 260) { doc.addPage(); y = margin; } doc.setFontSize(11); doc.setFont("helvetica","bold"); doc.setTextColor(...accent); doc.text(text, margin, y); y += 2; doc.setDrawColor(...accent); doc.setLineWidth(0.3); doc.line(margin, y, margin + cw, y); y += 5; doc.setFontSize(9); doc.setFont("helvetica","normal"); doc.setTextColor(...gray); };
      const addText = (text, opts = {}) => { const sz = opts.size || 9; const clr = opts.color || gray; doc.setFontSize(sz); doc.setFont("helvetica", opts.bold ? "bold" : "normal"); doc.setTextColor(...clr); const lines = doc.splitTextToSize(text, cw); if (y + lines.length * (sz * 0.4) > 280) { doc.addPage(); y = margin; } doc.text(lines, margin, y); y += lines.length * (sz * 0.4) + 2; doc.setFontSize(9); doc.setFont("helvetica","normal"); doc.setTextColor(...gray); };
      const addKeyVal = (key, val) => { doc.setFontSize(9); doc.setFont("helvetica","bold"); doc.setTextColor(...dark); doc.text(key + ":", margin, y); doc.setFont("helvetica","normal"); doc.setTextColor(...gray); doc.text(String(val), margin + 35, y); y += lineH; };
      // Sanitize text for jsPDF (replace unsupported Unicode with ASCII)
      const sanitize = (s) => String(s).replace(/\u0394/g,"Delta ").replace(/\u2248/g,"~").replace(/--/g,"--").replace(/\u2019/g,"'").replace(/\u03B1/g,"alpha").replace(/\u00B2/g,"2").replace(/\u2265/g,">=").replace(/\u2264/g,"<=");

      // Header
      addTitle("Bioassay Curve Fitting Report");
      doc.setFontSize(8); doc.setTextColor(...gray); doc.setFont("helvetica","normal");
      doc.text("Generated " + new Date().toLocaleString() + "  |  assaycurvefit.com  |  " + (typeof __APP_VERSION__ !== "undefined" ? __APP_VERSION__ : "dev"), margin, y); y += 8;

      // Model info
      if (pdfSections.modelInfo) {
        addSection("Model");
        addKeyVal("Model Type", activeModel);
        if (pdfSections.modelParams) {
          const pLabels = activeModel === "5PL" ? ["Bottom","Hill","EC50","Top","S"] : ["A (min)","B (slope)","C (EC50)","D (max)"];
          pLabels.forEach((label, i) => {
            const val = fitResult.params[i];
            const suffix = fixedParams.has(i) ? "  [fixed]" : "";
            addKeyVal(label, val.toExponential(6) + suffix);
          });
        }
        if (pdfSections.paramEC50) {
          addKeyVal("EC50", fitResult.params[activeModel === "5PL" ? 2 : 2].toExponential(6));
        }
        if (pdfSections.paramBioEC50 && activeModel === "5PL" && fitResult.bioEC50) {
          addKeyVal("Biological EC50", fitResult.bioEC50.toExponential(6));
        }
        y += 2;
      }

      // Fit statistics (granular)
      if (pdfSections.fitParams) {
        addSection("Goodness of Fit");
        if (pdfSections.paramR2) addKeyVal("R^2", fitResult.r2.toFixed(8));
        if (pdfSections.paramRMSE) addKeyVal("RMSE", fitResult.rmse.toFixed(6));
        if (pdfSections.paramSSR) addKeyVal("SSR", fitResult.ssr.toExponential(4));
        if (pdfSections.paramAIC) addKeyVal("AIC", fitResult.aic.toFixed(2));
        if (pdfSections.paramAICc) addKeyVal("AICc", fitResult.aicc.toFixed(2));
        if (pdfSections.paramBIC) addKeyVal("BIC", fitResult.bic.toFixed(2));
        if (pdfSections.paramConverged) addKeyVal("Converged", fitResult.converged ? "Yes" : "No");
        if (pdfSections.paramNK) { addKeyVal("N data points", fitResult.n); addKeyVal("K parameters", fitResult.k); }
        y += 2;
      }

      // Model comparison
      if (pdfSections.modelComparison && comparison && comparison.fit5PL) {
        addSection("Model Comparison (Auto Selection)");
        const headers = ["Metric", "4PL", "5PL"];
        const rows = [
          ["R^2", comparison.fit4PL.r2.toFixed(6), comparison.fit5PL.r2.toFixed(6)],
          ["AICc", comparison.fit4PL.aicc.toFixed(2), comparison.fit5PL.aicc.toFixed(2)],
          ["BIC", comparison.fit4PL.bic.toFixed(2), comparison.fit5PL.bic.toFixed(2)],
          ["SSR", comparison.fit4PL.ssr.toExponential(3), comparison.fit5PL.ssr.toExponential(3)],
          ["S param", "--", comparison.fit5PL.params[4].toFixed(4)],
        ];
        // Table header
        doc.setFontSize(8); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
        doc.text(headers[0], margin, y); doc.text(headers[1], margin + 50, y); doc.text(headers[2], margin + 90, y); y += 1;
        doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
        doc.setFont("helvetica","normal"); doc.setTextColor(...gray);
        for (const row of rows) {
          doc.text(row[0], margin, y); doc.text(row[1], margin + 50, y); doc.text(row[2], margin + 90, y); y += lineH;
        }
        y += 1;
        addText("Selected: " + comparison.selected + " -- " + sanitize(comparison.reason), { size: 8, color: accent, bold: true }); y += 2;
      }

      // Background info
      if (pdfSections.backgroundInfo && parsedData.bgSubtracted > 0) {
        addSection("Background Subtraction");
        addKeyVal("Background Mean", parsedData.bgSubtracted.toFixed(4));
        y += 2;
      }

      // Normalization info
      if (pdfSections.normalizationInfo && parsedData.normalized) {
        addSection("Normalization");
        addText("Data normalized to 0-100% using raw min=" + parsedData.normMin.toFixed(4) + ", max=" + parsedData.normMax.toFixed(4), { size: 9 });
        y += 2;
      }

      // Plot
      if (pdfSections.plot && mainCanvasRef.current) {
        if (y > 160) { doc.addPage(); y = margin; }
        addSection("Fitted Curve");
        const canvas = mainCanvasRef.current;
        const imgData = canvas.toDataURL("image/png");
        const imgW = cw;
        const imgH = (canvas.height / canvas.width) * imgW;
        if (y + imgH > 280) { doc.addPage(); y = margin; }
        doc.addImage(imgData, "PNG", margin, y, imgW, Math.min(imgH, 100));
        y += Math.min(imgH, 100) + 5;
      }

      // Raw data table
      if (pdfSections.rawData) {
        if (y > 200) { doc.addPage(); y = margin; }
        addSection("Raw Data");
        const groups = groupByConcentration(parsedData.xData, parsedData.yData);
        // Table header
        doc.setFontSize(7); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
        const colX = margin, colMean = margin + 22, colSD = margin + 48, colSEM = margin + 70, colCV = margin + 90, colN = margin + 112, colVals = margin + 122;
        doc.text("Conc", colX, y); doc.text("Mean", colMean, y); doc.text("SD", colSD, y); doc.text("SEM", colSEM, y); doc.text("%CV", colCV, y); doc.text("n", colN, y); doc.text("Values", colVals, y);
        y += 1;
        doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
        doc.setFont("helvetica","normal"); doc.setTextColor(...gray); doc.setFontSize(6.5);
        for (const g of groups) {
          if (y > 275) { doc.addPage(); y = margin; }
          const vals = g.values;
          const n = vals.length;
          const mean = vals.reduce((a, b) => a + b, 0) / n;
          const sd = n > 1 ? Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : 0;
          const sem = n > 1 ? sd / Math.sqrt(n) : 0;
          const cv = mean !== 0 ? (sd / Math.abs(mean)) * 100 : 0;
          doc.text(g.x.toString(), colX, y);
          doc.text(mean.toFixed(4), colMean, y);
          doc.text(n > 1 ? sd.toFixed(4) : "--", colSD, y);
          doc.text(n > 1 ? sem.toFixed(4) : "--", colSEM, y);
          doc.text(n > 1 ? cv.toFixed(1) + "%" : "--", colCV, y);
          doc.text(String(n), colN, y);
          const valStr = vals.map(v => v.toFixed(3)).join(", ");
          const trimmed = valStr.length > 40 ? valStr.substring(0, 37) + "..." : valStr;
          doc.text(trimmed, colVals, y);
          y += 3.5;
        }
        y += 3;
      }

      // Outlier test results
      if (pdfSections.outlierResults && grubbsResults) {
        if (y > 220) { doc.addPage(); y = margin; }
        addSection("Grubbs' Outlier Test (a=" + grubbsAlpha + ")");
        addKeyVal("Total outliers", grubbsResults.totalOutliers);
        y += 2;
        doc.setFontSize(7); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
        doc.text("Conc", margin, y); doc.text("n", margin + 22, y); doc.text("Outliers", margin + 32, y); doc.text("G stat", margin + 56, y); doc.text("G crit", margin + 76, y); doc.text("Result", margin + 96, y);
        y += 1;
        doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
        doc.setFont("helvetica","normal"); doc.setTextColor(...gray); doc.setFontSize(7);
        for (const gr of grubbsResults.groupResults) {
          if (y > 275) { doc.addPage(); y = margin; }
          const hasOutlier = gr.tested && gr.outlierCount > 0;
          if (hasOutlier) {
            // Highlight row background
            doc.setFillColor(255, 235, 235);
            doc.rect(margin - 1, y - 2.5, cw + 2, 3.5, "F");
            doc.setTextColor(180, 40, 40);
            doc.setFont("helvetica","bold");
          } else {
            doc.setTextColor(...gray);
            doc.setFont("helvetica","normal");
          }
          doc.text(gr.x.toString(), margin, y);
          doc.text(String(gr.n), margin + 22, y);
          if (!gr.tested) {
            doc.setTextColor(...gray); doc.setFont("helvetica","normal");
            doc.text("n<3, skipped", margin + 32, y);
          } else {
            const maxG = gr.result.details ? Math.max(...gr.result.details.map(d => d.g)) : 0;
            doc.text(String(gr.outlierCount), margin + 32, y);
            doc.text(maxG.toFixed(4), margin + 56, y);
            doc.text(gr.result.gCrit ? gr.result.gCrit.toFixed(4) : "--", margin + 76, y);
            doc.text(hasOutlier ? "OUTLIER" : "Pass", margin + 96, y);
          }
          y += 3.5;
          // List individual outlier values below the row
          if (hasOutlier && gr.result.outliers) {
            doc.setFontSize(6); doc.setTextColor(180, 40, 40); doc.setFont("helvetica","italic");
            const olVals = gr.result.outliers.map(o => o.value.toFixed(4) + " (G=" + o.g.toFixed(3) + ")").join(", ");
            doc.text("  Flagged: " + olVals, margin + 4, y);
            y += 3;
            doc.setFontSize(7);
          }
        }
        // Reset
        doc.setTextColor(...gray); doc.setFont("helvetica","normal");
        y += 3;
      }

      // Footer on last page
      const pages = doc.getNumberOfPages();
      for (let i = 1; i <= pages; i++) {
        doc.setPage(i);
        doc.setFontSize(7); doc.setTextColor(160,160,160); doc.setFont("helvetica","normal");
        doc.text("assaycurvefit.com  |  Page " + i + " of " + pages, pw / 2, 290, { align: "center" });
      }

      doc.save("bioassay_" + activeModel + "_report.pdf");
    } catch (e) {
      console.error("PDF generation failed:", e);
      setError("PDF generation failed: " + e.message);
    } finally {
      setPdfGenerating(false);
      setShowPdfModal(false);
    }
  }, [parsedData, fitResult, activeModel, comparison, grubbsResults, grubbsAlpha, pdfSections, fixedParams, t]);

  const interpolate = useCallback((targetY) => {
    if (!fitResult) return null;
    const [A, B, C, D] = fitResult.params;
    if (activeModel === "4PL") {
      const ratio = (A - targetY) / (targetY - D);
      if (ratio <= 0) return null;
      return C * Math.pow(ratio, 1 / B);
    }
    // 5PL: numerical inverse via bisection in log-space
    const modelFn = model5PL;
    let lo = 1e-15, hi = 1e15;
    // Determine curve direction: evaluate at low and high x
    const yLo = modelFn(lo, fitResult.params);
    const yHi = modelFn(hi, fitResult.params);
    const increasing = yHi > yLo;
    for (let i = 0; i < 100; i++) {
      const mid = Math.sqrt(lo * hi);
      const yMid = modelFn(mid, fitResult.params);
      if ((increasing && yMid > targetY) || (!increasing && yMid < targetY)) hi = mid;
      else lo = mid;
    }
    return Math.sqrt(lo * hi);
  }, [fitResult, activeModel]);

  const [interpY, setInterpY] = useState("");
  const [interpResult, setInterpResult] = useState(null);

  // ── Multi-Compound CSV state ──────────────────────────────────
  const [multiData, setMultiData] = useState(null);
  const [multiIndex, setMultiIndex] = useState(0);
  const [multiCsvError, setMultiCsvError] = useState(null);
  const multiCsvRef = useRef(null);

  // Helper: reset all fit-related state, load a compound's data, and auto-fit
  const loadCompound = useCallback((compound) => {
    const csv = compoundToCSV(compound);
    setRawData(csv);
    setParsedData(null);
    setFitResult(null);
    setComparison(null);
    setError(null);
    setGrubbsResults(null);
    setShowOutliers(false);
    setSelectedGrubbsGroup(null);
    setExcludedIndices(new Set());
    setBgStats(null);
    // Run fit inline with the csv string directly — avoids stale rawData state
    try {
      const { xData, yData } = parseData(csv);
      if (xData.length < 4) return;
      const fit4 = fitModel(xData, yData, model4PL, false);
      const fit5 = xData.length >= 5 ? fitModel(xData, yData, model5PL, true) : null;
      setParsedData({ xData, yData, yRaw: yData, bgSubtracted: 0, normMin: 0, normMax: 1, normalized: false });
      if (!fit4 && !fit5) return;
      if (!fit5 || Math.abs(fit5.params[4] - 1) < 0.05) {
        setActiveModel("4PL"); setFitResult(fit4);
        setComparison({ fit4PL: fit4, fit5PL: fit5 || null, selected: "4PL", reason: "Auto" });
      } else {
        const deltaAICc = (fit4?.aicc ?? Infinity) - fit5.aicc;
        const selected = (deltaAICc > 2 && fit4.bic - fit5.bic > 0) ? "5PL" : "4PL";
        setActiveModel(selected);
        setFitResult(selected === "5PL" ? fit5 : fit4);
        setComparison({ fit4PL: fit4, fit5PL: fit5, selected, reason: "Auto" });
      }
    } catch { /* silently skip on malformed data */ }
  }, [parseData]);

  // When multiIndex changes, swap the textarea content
  useEffect(() => {
    if (!multiData) return;
    loadCompound(multiData[multiIndex]);
  }, [multiIndex, multiData]); // intentionally not including loadCompound to avoid stale closure loop

  const handleMultiCsv = useCallback((file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const compounds = parsePrismCSV(e.target.result);
        setMultiData(compounds);
        setMultiIndex(0);
        setMultiCsvError(null);
        // Immediately load first compound
        loadCompound(compounds[0]);
      } catch (err) {
        setMultiCsvError(err.message);
        setMultiData(null);
      }
    };
    reader.readAsText(file);
  }, [loadCompound]);

  useEffect(() => {
    if (!multiData) return;
    const handler = (e) => {
      if (e.key === "ArrowLeft") setMultiIndex(i => Math.max(0, i - 1));
      if (e.key === "ArrowRight") setMultiIndex(i => Math.min(multiData.length - 1, i + 1));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [multiData]);

  return (
    <div style={{
      minHeight: "100vh",
      background: t.bg,
      color: t.text,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
      padding: isMobile ? "12px" : "24px",
      transition: "background 0.3s, color 0.3s",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        textarea { font-family: 'JetBrains Mono', monospace; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${t.scrollTrack}; }
        ::-webkit-scrollbar-thumb { background: ${t.scrollThumb}; border-radius: 3px; }
        html, body { overflow-x: hidden; }
        @media (max-width: 767px) {
          textarea { font-size: 11px !important; }
          select { font-size: 11px !important; }
        }
      `}</style>

      {/* Header */}
      <div style={{ maxWidth: 1200, margin: "0 auto 24px", display: "flex", justifyContent: "space-between", alignItems: isMobile ? "center" : "flex-start", flexWrap: "wrap", gap: 12 }}>
        <div>
          <h1 style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: isMobile ? 22 : 28,
            fontWeight: 700,
            background: "linear-gradient(135deg, #3b9eff, #a855f7, #00e6b4)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: "-0.5px",
            display: "flex",
            alignItems: "baseline",
            gap: 8,
          }}>
            Bioassay Curve Fitter
            <span style={{ fontSize: 10, fontWeight: 400, opacity: 0.5, WebkitTextFillColor: t.textDim, background: "none" }}>
              {typeof __APP_VERSION__ !== "undefined" ? __APP_VERSION__ : "dev"}
            </span>
          </h1>
          <p style={{ fontSize: 12, color: t.textMuted, marginTop: 4 }}>
            4-Parameter & 5-Parameter Logistic Regression | Levenberg-Marquardt Optimization
          </p>
        </div>
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          style={{
            padding: "6px 12px",
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 6,
            color: t.textMuted,
            fontSize: 10,
            cursor: "pointer",
            fontFamily: "'JetBrains Mono', monospace",
            display: "flex",
            alignItems: "center",
            gap: 6,
            transition: "all 0.2s",
          }}
        >
          <span style={{ fontSize: 14 }}>{theme === "dark" ? "☀️" : "🌙"}</span>
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", display: "grid", gridTemplateColumns: isMobile ? "1fr" : "340px 1fr", gap: 20, alignItems: "start" }}>
        {/* Left Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16, order: isMobile ? 2 : 1 }}>
          {/* Data Input */}
          <div style={{
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 10,
            padding: 16,
          }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: t.label, marginBottom: 8, textTransform: "uppercase", letterSpacing: 1, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 }}>
              <span style={{ whiteSpace: "nowrap" }}>Data Input</span>
              <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
              <select
                onChange={(e) => {
                  if (e.target.value && EXAMPLE_DATASETS[e.target.value]) {
                    setRawData(EXAMPLE_DATASETS[e.target.value]);
                    setParsedData(null);
                    setFitResult(null);
                    setComparison(null);
                    setError(null);
                    setGrubbsResults(null);
                    setShowOutliers(false);
                    setSelectedGrubbsGroup(null);
                    setExcludedIndices(new Set());
                    setBgStats(null);
                    setFixedMin("");
                    setFixedMax("");
                  }
                  e.target.value = "";
                }}
                style={{
                  padding: "2px 4px",
                  background: t.input,
                  border: `1px solid ${t.inputBorder}`,
                  borderRadius: 4,
                  color: t.textDim,
                  fontSize: 9,
                  fontFamily: "'JetBrains Mono', monospace",
                  outline: "none",
                  cursor: "pointer",
                  width: 82,
                }}
              >
                <option value="">Examples…</option>
                {Object.keys(EXAMPLE_DATASETS).map(name => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
              {/* Multi-Compound CSV upload button */}
              <button
                onClick={() => multiCsvRef.current && multiCsvRef.current.click()}
                title="Load a Prism-style multi-compound CSV (duplicate headers = replicates)"
                style={{
                  padding: "2px 8px",
                  background: multiData ? "rgba(0,230,180,0.12)" : t.btnInactive,
                  border: `1px solid ${multiData ? "rgba(0,230,180,0.3)" : t.inputBorder}`,
                  borderRadius: 4,
                  color: multiData ? (t.teal || "#00e6b4") : t.textDim,
                  fontSize: 9,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                  whiteSpace: "nowrap",
                  fontWeight: multiData ? 700 : 400,
                }}
              >
                {multiData ? `Multi CSV ✓` : "Multi CSV"}
              </button>
              <input
                ref={multiCsvRef}
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={e => { handleMultiCsv(e.target.files[0]); e.target.value = ""; }}
              />
              </div>
            </div>
            <textarea
              value={rawData}
              onChange={(e) => {
                setRawData(e.target.value);
                setParsedData(null);
                setFitResult(null);
                setComparison(null);
                setError(null);
                setGrubbsResults(null);
                setShowOutliers(false);
                setSelectedGrubbsGroup(null);
                setExcludedIndices(new Set());
                setBgStats(null);
              }}
              placeholder="Concentration,Response&#10;0.01,0.5&#10;0.1,1.2&#10;..."
              style={{
                width: "100%",
                height: 200,
                background: t.input,
                border: `1px solid ${t.inputBorder}`,
                borderRadius: 6,
                color: t.text,
                fontSize: 11,
                padding: 10,
                resize: "vertical",
                outline: "none",
              }}
            />
            <p style={{ fontSize: 9, color: t.textDim, marginTop: 6 }}>
              CSV/TSV format. First column = concentration, additional columns = replicates. Comma-formatted numbers (e.g. 47,189.7) supported.
            </p>
            {parsedData && (
              <p style={{ fontSize: 9, color: t.teal, marginTop: 4 }}>
                Parsed: {parsedData.xData.length} data points across {new Set(parsedData.xData).size} concentrations
                {parsedData.bgSubtracted ? ` (bg: −${parsedData.bgSubtracted.toFixed(1)})` : ""}
                {parsedData.normalized ? " (normalized 0-100%)" : ""}
              </p>
            )}
          </div>

          {multiCsvError && (
            <div style={{
              padding: "10px 12px",
              background: "rgba(255,80,80,0.08)",
              border: "1px solid rgba(255,80,80,0.25)",
              borderRadius: 6,
              color: t.red,
              fontSize: 10,
            }}>
              Multi CSV: {multiCsvError}
            </div>
          )}

          {/* Background Subtraction */}
          <div style={{
            background: t.panel,
            border: `1px solid ${bgEnabled ? "rgba(168,85,247,0.2)" : "rgba(60,100,160,0.15)"}`,
            borderRadius: 10,
            padding: bgEnabled ? 16 : 0,
            overflow: "hidden",
            transition: "all 0.2s",
          }}>
            <button
              onClick={() => setBgEnabled(!bgEnabled)}
              style={{
                width: "100%",
                padding: bgEnabled ? "0 0 10px 0" : "12px 16px",
                background: "transparent",
                border: "none",
                color: bgEnabled ? "rgba(168,85,247,0.9)" : "rgba(160,190,230,0.5)",
                fontSize: 11,
                fontWeight: 600,
                cursor: "pointer",
                fontFamily: "'JetBrains Mono', monospace",
                textTransform: "uppercase",
                letterSpacing: 1,
                textAlign: "left",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>Background Subtraction</span>
              <span style={{ fontSize: 10, opacity: 0.6 }}>{bgEnabled ? "▾" : "▸"}</span>
            </button>
            {bgEnabled && (
              <>
                <textarea
                  value={bgRawData}
                  onChange={(e) => setBgRawData(e.target.value)}
                  placeholder={"Paste background response values\ne.g. 2150.3  2089.1  2201.5\nor one per line"}
                  style={{
                    width: "100%",
                    height: 60,
                    background: t.input,
                    border: "1px solid rgba(168,85,247,0.15)",
                    borderRadius: 6,
                    color: t.text,
                    fontSize: 11,
                    padding: 10,
                    resize: "vertical",
                    outline: "none",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                />
                <p style={{ fontSize: 9, color: t.textDim, marginTop: 6 }}>
                  Response values only (no concentrations). Mean is subtracted from all data before fitting.
                </p>
                {bgStats && (
                  <div style={{
                    marginTop: 8,
                    padding: "6px 10px",
                    background: "rgba(168,85,247,0.06)",
                    border: "1px solid rgba(168,85,247,0.12)",
                    borderRadius: 6,
                    fontSize: 10,
                    color: "rgba(190,170,230,0.7)",
                    display: "flex",
                    gap: 12,
                  }}>
                    <span>n={bgStats.n}</span>
                    <span>Mean: <span style={{ color: t.purple, fontWeight: 600 }}>{bgStats.mean.toFixed(1)}</span></span>
                    {bgStats.n > 1 && <span>SD: {bgStats.sd.toFixed(1)}</span>}
                    {bgStats.n > 1 && <span>%CV: {(bgStats.sd / bgStats.mean * 100).toFixed(1)}%</span>}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Model Selection */}
          <div style={{
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 10,
            padding: 16,
          }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: t.label, marginBottom: 10, textTransform: "uppercase", letterSpacing: 1 }}>
              Model
            </div>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              {["Auto", "1PL", "2PL", "3PL", "4PL", "5PL"].map(m => {
                const colors = {
                  Auto: { active: t.teal, bg: t.tealBg, border: t.tealBorder },
                  "1PL": { active: t.orange, bg: t.orangeBg, border: t.orangeBorder },
                  "2PL": { active: t.orange, bg: t.orangeBg, border: t.orangeBorder },
                  "3PL": { active: t.blue, bg: t.blueBg, border: t.blueBorder },
                  "4PL": { active: t.blue, bg: t.blueBg, border: t.blueBorder },
                  "5PL": { active: t.purple, bg: t.purpleBg, border: t.purpleBorder },
                };
                const c = colors[m];
                return (
                  <button
                    key={m}
                    onClick={() => setModelType(m)}
                    style={{
                      flex: modelType === m ? 2 : 1,
                      minWidth: 40,
                      padding: "8px 0",
                      background: modelType === m ? c.bg : t.btnInactive,
                      border: `1px solid ${modelType === m ? c.border : "rgba(60,100,160,0.15)"}`,
                      borderRadius: 6,
                      color: modelType === m ? c.active : "rgba(160,190,230,0.5)",
                      fontSize: 12,
                      fontWeight: 600,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      transition: "all 0.2s",
                    }}
                  >
                    {m}
                  </button>
                );
              })}
            </div>
            <div style={{ marginTop: 10, fontSize: 10, color: t.textDim, lineHeight: 1.6 }}>
              {modelType === "Auto"
                ? "Fits 4PL and 5PL; selects best via AICc with parsimony preference"
                : modelType === "1PL"
                  ? "Fix asymptotes and Hill slope; fit EC50 only (1 free parameter)"
                  : modelType === "2PL"
                    ? "Fix asymptotes; fit Hill slope and EC50 (2 free parameters)"
                    : modelType === "3PL"
                      ? "Fix Hill slope; fit asymptotes and EC50 (3 free parameters)"
                      : modelType === "4PL"
                        ? "y = D + (A−D) / (1 + (x/C)^B)"
                        : "y = Bot + (Top−Bot) / (1 + (EC50/x)^Hill)^S"}
            </div>

            {/* Constraint inputs for 1PL, 2PL, 3PL */}
            {["1PL", "2PL", "3PL"].includes(modelType) && (
              <div style={{
                marginTop: 10,
                padding: "10px 12px",
                background: "rgba(255,180,50,0.06)",
                border: "1px solid rgba(255,180,50,0.15)",
                borderRadius: 6,
              }}>
                <div style={{ fontSize: 9, color: t.orange, fontWeight: 600, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  Fixed Parameters
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {(modelType === "1PL" || modelType === "2PL") && (
                    <>
                      <div style={{ flex: 1, minWidth: 60 }}>
                        <label style={{ fontSize: 8, color: t.textDim, display: "block", marginBottom: 3 }}>Min (A)</label>
                        <input
                          type="number"
                          value={fixedMin}
                          onChange={(e) => setFixedMin(e.target.value)}
                          placeholder="e.g. 0"
                          style={{
                            width: "100%",
                            padding: "5px 8px",
                            background: t.input,
                            border: `1px solid ${fixedMin === "" ? "rgba(255,180,50,0.3)" : t.inputBorder}`,
                            borderRadius: 4,
                            color: t.text,
                            fontSize: 11,
                            fontFamily: "'JetBrains Mono', monospace",
                            outline: "none",
                          }}
                        />
                      </div>
                      <div style={{ flex: 1, minWidth: 60 }}>
                        <label style={{ fontSize: 8, color: t.textDim, display: "block", marginBottom: 3 }}>Max (D)</label>
                        <input
                          type="number"
                          value={fixedMax}
                          onChange={(e) => setFixedMax(e.target.value)}
                          placeholder="e.g. 100"
                          style={{
                            width: "100%",
                            padding: "5px 8px",
                            background: t.input,
                            border: `1px solid ${fixedMax === "" ? "rgba(255,180,50,0.3)" : t.inputBorder}`,
                            borderRadius: 4,
                            color: t.text,
                            fontSize: 11,
                            fontFamily: "'JetBrains Mono', monospace",
                            outline: "none",
                          }}
                        />
                      </div>
                    </>
                  )}
                  {(modelType === "1PL" || modelType === "3PL") && (
                    <div style={{ flex: 1, minWidth: 60 }}>
                      <label style={{ fontSize: 8, color: t.textDim, display: "block", marginBottom: 3 }}>Hill slope (B)</label>
                      <input
                        type="number"
                        value={fixedHill}
                        onChange={(e) => setFixedHill(e.target.value)}
                        placeholder="1"
                        style={{
                          width: "100%",
                          padding: "5px 8px",
                          background: t.input,
                          border: `1px solid ${t.inputBorder}`,
                          borderRadius: 4,
                          color: t.text,
                          fontSize: 11,
                          fontFamily: "'JetBrains Mono', monospace",
                          outline: "none",
                        }}
                      />
                    </div>
                  )}
                </div>
                <p style={{ fontSize: 8, color: t.textDim, marginTop: 6 }}>
                  {modelType === "3PL" ? "Hill slope is fixed; asymptotes are fitted from data"
                    : modelType === "2PL" ? "Asymptotes are fixed; Hill slope and EC50 are fitted"
                    : "All parameters fixed except EC50"}
                </p>
              </div>
            )}

            {/* Normalize toggle */}
            <div style={{
              marginTop: 10,
              paddingTop: 10,
              borderTop: `1px solid ${t.panelBorder}`,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}>
              <button
                onClick={() => setNormalize(!normalize)}
                style={{
                  width: 36, height: 20,
                  borderRadius: 10,
                  border: `1px solid ${normalize ? t.tealBorder : "rgba(60,100,160,0.15)"}`,
                  background: normalize ? t.tealBg : t.btnInactive,
                  cursor: "pointer",
                  position: "relative",
                  transition: "all 0.2s",
                  padding: 0,
                }}
              >
                <div style={{
                  width: 14, height: 14,
                  borderRadius: 7,
                  background: normalize ? (t.teal || "#00e6b4") : "rgba(140,170,210,0.3)",
                  position: "absolute",
                  top: 2,
                  left: normalize ? 18 : 2,
                  transition: "all 0.2s",
                }} />
              </button>
              <span style={{ fontSize: 10, color: normalize ? t.teal : t.textDim }}>
                Normalize (0-100%)
              </span>
            </div>
            {normalize && (
              <p style={{ fontSize: 8, color: t.textDim, marginTop: 4 }}>
                Responses scaled to 0-100% using raw min/max before fitting
              </p>
            )}
          </div>

          {/* Model Comparison Panel (Auto mode) */}
          {comparison && comparison.fit5PL && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.panelBorder}`,
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: t.label, marginBottom: 10, textTransform: "uppercase", letterSpacing: 1 }}>
                Model Comparison
              </div>
              
              {/* Comparison table */}
              <div style={{ fontSize: 10 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4, marginBottom: 4 }}>
                  <span style={{ color: t.textDim }}></span>
                  <span style={{ color: t.blue, fontWeight: 600, textAlign: "center" }}>4PL</span>
                  <span style={{ color: t.purple, fontWeight: 600, textAlign: "center" }}>5PL</span>
                </div>
                {[
                  { label: "R²", v4: comparison.fit4PL.r2.toFixed(6), v5: comparison.fit5PL.r2.toFixed(6) },
                  { label: "AICc", v4: comparison.fit4PL.aicc.toFixed(1), v5: comparison.fit5PL.aicc.toFixed(1) },
                  { label: "BIC", v4: comparison.fit4PL.bic.toFixed(1), v5: comparison.fit5PL.bic.toFixed(1) },
                  { label: "SSR", v4: comparison.fit4PL.ssr.toExponential(3), v5: comparison.fit5PL.ssr.toExponential(3) },
                ].map((row, idx) => (
                  <div key={idx} style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: 4,
                    padding: "4px 0",
                    borderTop: "1px solid rgba(60,100,160,0.06)",
                  }}>
                    <span style={{ color: t.textMuted }}>{row.label}</span>
                    <span style={{ textAlign: "center", color: comparison.selected === "4PL" && row.label === "AICc" ? "#00e6b4" : "#c8daf0" }}>{row.v4}</span>
                    <span style={{ textAlign: "center", color: comparison.selected === "5PL" && row.label === "AICc" ? "#00e6b4" : "#c8daf0" }}>{row.v5}</span>
                  </div>
                ))}
                {comparison.fit5PL && (
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: 4,
                    padding: "4px 0",
                    borderTop: "1px solid rgba(60,100,160,0.06)",
                  }}>
                    <span style={{ color: t.textMuted }}>S param</span>
                    <span style={{ textAlign: "center", color: t.textFaint }}>—</span>
                    <span style={{ textAlign: "center", color: Math.abs(comparison.fit5PL.params[4] - 1) < 0.05 ? "#ffb432" : "#c8daf0" }}>
                      {comparison.fit5PL.params[4].toFixed(4)}
                    </span>
                  </div>
                )}
              </div>

              {/* Selection result */}
              <div style={{
                marginTop: 10,
                padding: "8px 10px",
                background: comparison.selected === "4PL" ? "rgba(59,158,255,0.08)" : "rgba(168,85,247,0.08)",
                border: `1px solid ${comparison.selected === "4PL" ? "rgba(59,158,255,0.2)" : "rgba(168,85,247,0.2)"}`,
                borderRadius: 6,
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: comparison.selected === "4PL" ? t.blue : t.purple, marginBottom: 2 }}>
                  ▸ {comparison.selected} Selected
                </div>
                <div style={{ fontSize: 9, color: t.textMuted, lineHeight: 1.5 }}>
                  {comparison.reason}
                </div>
              </div>

              {/* Toggle to view the other model */}
              <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                {["4PL", "5PL"].map(m => (
                  <button
                    key={m}
                    onClick={() => {
                      const fit = m === "4PL" ? comparison.fit4PL : comparison.fit5PL;
                      if (fit) { setActiveModel(m); setFitResult(fit); }
                    }}
                    style={{
                      flex: 1,
                      padding: "6px 0",
                      background: activeModel === m ? "rgba(0,230,180,0.1)" : t.btnInactive,
                      border: `1px solid ${activeModel === m ? "rgba(0,230,180,0.3)" : "rgba(60,100,160,0.1)"}`,
                      borderRadius: 4,
                      color: activeModel === m ? "#00e6b4" : "rgba(160,190,230,0.4)",
                      fontSize: 9,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      textTransform: "uppercase",
                      letterSpacing: 0.5,
                    }}
                  >
                    View {m}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Fit Button */}
          <button
            onClick={runFit}
            style={{
              padding: "14px 0",
              background: "linear-gradient(135deg, rgba(59,158,255,0.2), rgba(0,230,180,0.2))",
              border: "1px solid rgba(59,158,255,0.3)",
              borderRadius: 8,
              color: t.blue,
              fontSize: 14,
              fontWeight: 700,
              cursor: "pointer",
              fontFamily: "'Space Grotesk', sans-serif",
              letterSpacing: 0.5,
              transition: "all 0.2s",
            }}
            onMouseOver={(e) => {
              e.target.style.background = "linear-gradient(135deg, rgba(59,158,255,0.3), rgba(0,230,180,0.3))";
              e.target.style.borderColor = "rgba(59,158,255,0.5)";
            }}
            onMouseOut={(e) => {
              e.target.style.background = "linear-gradient(135deg, rgba(59,158,255,0.2), rgba(0,230,180,0.2))";
              e.target.style.borderColor = "rgba(59,158,255,0.3)";
            }}
          >
            FIT MODEL
          </button>

          {error && (
            <div style={{
              padding: 12,
              background: "rgba(255,80,80,0.1)",
              border: "1px solid rgba(255,80,80,0.3)",
              borderRadius: 6,
              color: t.red,
              fontSize: 11,
            }}>
              {error}
            </div>
          )}

          {/* Results */}
          {fitResult && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.panelBorder}`,
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: t.label, marginBottom: 12, textTransform: "uppercase", letterSpacing: 1, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>Fit Parameters</span>
                <span style={{ color: activeModel === "5PL" ? t.purple : ["1PL","2PL"].includes(activeModel) ? t.orange : t.blue, fontSize: 10 }}>{activeModel}</span>
              </div>
              {paramLabels.map((label, i) => (
                <div key={i} style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "6px 0",
                  borderBottom: i < paramLabels.length - 1 ? "1px solid rgba(60,100,160,0.08)" : "none",
                }}>
                  <span style={{ fontSize: 11, color: t.labelDim, display: "flex", alignItems: "center", gap: 4 }}>
                    {label}
                    {fixedParams.has(i) && (
                      <span style={{
                        fontSize: 7, padding: "1px 4px", borderRadius: 3,
                        background: "rgba(255,180,50,0.12)", border: "1px solid rgba(255,180,50,0.25)",
                        color: t.orange, fontWeight: 600, textTransform: "uppercase",
                      }}>fixed</span>
                    )}
                  </span>
                  <span style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: i === 2 ? t.orange : t.teal,
                  }}>
                    {Math.abs(fitResult.params[i]) < 0.01 || Math.abs(fitResult.params[i]) > 10000
                      ? fitResult.params[i].toExponential(4)
                      : fitResult.params[i].toFixed(4)}
                  </span>
                </div>
              ))}

              {/* Biological EC50 for 5PL */}
              {activeModel === "5PL" && fitResult.bioEC50 && (
                <div style={{
                  marginTop: 8,
                  padding: "8px 0",
                  borderTop: `1px solid ${t.panelBorder}`,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 10, color: t.labelDim }}>Parametric EC50</span>
                    <span style={{ fontSize: 11, fontWeight: 600, color: t.orange }}>
                      {fitResult.params[2] < 0.01 || fitResult.params[2] > 10000
                        ? fitResult.params[2].toExponential(3)
                        : fitResult.params[2].toPrecision(4)}
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 10, color: t.labelDim }}>Biological EC50</span>
                    <span style={{ fontSize: 11, fontWeight: 600, color: t.teal }}>
                      {fitResult.bioEC50 < 0.01 || fitResult.bioEC50 > 10000
                        ? fitResult.bioEC50.toExponential(3)
                        : fitResult.bioEC50.toPrecision(4)}
                    </span>
                  </div>
                  <p style={{ fontSize: 8, color: t.textDim, marginTop: 4 }}>
                    Biological EC50 = concentration at half-maximal response
                  </p>
                </div>
              )}

              <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${t.panelBorder}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: t.labelDim }}>R²</span>
                  <span style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: fitResult.r2 > 0.99 ? t.teal : fitResult.r2 > 0.95 ? t.orange : t.red,
                  }}>
                    {fitResult.r2.toFixed(6)}
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: t.labelDim }}>RMSE</span>
                  <span style={{ fontSize: 12, color: t.text }}>{fitResult.rmse.toFixed(6)}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 11, color: t.labelDim }}>SSR</span>
                  <span style={{ fontSize: 12, color: t.text }}>{fitResult.ssr.toFixed(6)}</span>
                </div>
                {fitResult.aicc !== undefined && (
                  <>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                      <span style={{ fontSize: 11, color: t.labelDim }}>AICc</span>
                      <span style={{ fontSize: 12, color: t.text }}>{fitResult.aicc.toFixed(2)}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                      <span style={{ fontSize: 11, color: t.labelDim }}>BIC</span>
                      <span style={{ fontSize: 12, color: t.text }}>{fitResult.bic.toFixed(2)}</span>
                    </div>
                  </>
                )}
              </div>

              {/* Interpolation */}
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${t.panelBorder}` }}>
                <div style={{ fontSize: 10, color: t.textMuted, marginBottom: 6 }}>
                  INTERPOLATE: Response → Concentration
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <input
                    type="number"
                    value={interpY}
                    onChange={(e) => setInterpY(e.target.value)}
                    placeholder="Response value"
                    style={{
                      flex: 1,
                      padding: "6px 8px",
                      background: t.input,
                      border: `1px solid ${t.inputBorder}`,
                      borderRadius: 4,
                      color: t.text,
                      fontSize: 11,
                      fontFamily: "'JetBrains Mono', monospace",
                      outline: "none",
                    }}
                  />
                  <button
                    onClick={() => {
                      const val = parseFloat(interpY);
                      if (!isNaN(val)) setInterpResult(interpolate(val));
                    }}
                    style={{
                      padding: "6px 12px",
                      background: "rgba(255,180,50,0.15)",
                      border: "1px solid rgba(255,180,50,0.3)",
                      borderRadius: 4,
                      color: t.orange,
                      fontSize: 10,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    CALC
                  </button>
                </div>
                {interpResult !== null && interpResult !== undefined && (
                  <div style={{ marginTop: 6, fontSize: 11, color: t.orange }}>
                    Concentration: {interpResult.toExponential(4)}
                  </div>
                )}
              </div>

              {/* Actions */}
              <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
                <button
                  onClick={() => setShowResiduals(!showResiduals)}
                  style={{
                    flex: 1,
                    padding: "8px 0",
                    background: showResiduals ? "rgba(0,230,180,0.1)" : t.btnInactive,
                    border: `1px solid ${showResiduals ? "rgba(0,230,180,0.3)" : "rgba(60,100,160,0.15)"}`,
                    borderRadius: 4,
                    color: showResiduals ? "#00e6b4" : "rgba(160,190,230,0.5)",
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  Residuals
                </button>
                <button
                  onClick={exportCSV}
                  style={{
                    flex: 1,
                    padding: "8px 0",
                    background: t.btnInactive,
                    border: `1px solid ${t.panelBorder}`,
                    borderRadius: 4,
                    color: t.textMuted,
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  Export CSV
                </button>
                <button
                  onClick={() => setShowPdfModal(true)}
                  style={{
                    flex: 1,
                    padding: "8px 0",
                    background: t.btnInactive,
                    border: `1px solid ${t.panelBorder}`,
                    borderRadius: 4,
                    color: t.textMuted,
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  PDF Report
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Charts */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16, order: isMobile ? 1 : 2 }}>
          {/* Multi-Compound nav bar — shown above chart when a multi-compound CSV is loaded */}
          {multiData && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.tealBorder || "rgba(0,230,180,0.25)"}`,
              borderRadius: 10,
              padding: "12px 16px",
              display: "flex",
              alignItems: "center",
              gap: 8,
              flexWrap: "wrap",
            }}>
              {/* Compound label */}
              <span style={{ fontSize: 11, fontWeight: 700, color: t.teal, textTransform: "uppercase", letterSpacing: 1, whiteSpace: "nowrap" }}>
                Multi CSV
              </span>
              <span style={{ fontSize: 11, fontWeight: 600, color: t.text, whiteSpace: "nowrap" }}>
                {multiData[multiIndex].name}
              </span>
              <span style={{ fontSize: 10, color: t.textDim, whiteSpace: "nowrap" }}>
                n={multiData[multiIndex].nReplicates} · {multiData[multiIndex].points.length} conc
              </span>

              {/* Spacer */}
              <div style={{ flex: 1, minWidth: 8 }} />

              {/* Prev button */}
              <button
                onClick={() => setMultiIndex(i => Math.max(0, i - 1))}
                disabled={multiIndex === 0}
                style={{
                  width: 28, height: 28, flexShrink: 0,
                  background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`,
                  borderRadius: 5,
                  color: multiIndex === 0 ? t.textFaint : t.textMuted,
                  fontSize: 14,
                  cursor: multiIndex === 0 ? "not-allowed" : "pointer",
                  fontFamily: "monospace",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  lineHeight: 1,
                }}
              >←</button>

              {/* Dropdown */}
              <select
                value={multiIndex}
                onChange={e => setMultiIndex(Number(e.target.value))}
                style={{
                  padding: "4px 8px",
                  background: t.input,
                  border: `1px solid ${t.inputBorder}`,
                  borderRadius: 5,
                  color: t.text,
                  fontSize: 11,
                  fontFamily: "'JetBrains Mono', monospace",
                  outline: "none",
                  cursor: "pointer",
                  maxWidth: 160,
                }}
              >
                {multiData.map((c, i) => (
                  <option key={i} value={i}>{c.name}</option>
                ))}
              </select>

              {/* Counter */}
              <span style={{ fontSize: 10, color: t.textDim, whiteSpace: "nowrap", minWidth: 36 }}>
                {multiIndex + 1}/{multiData.length}
              </span>

              {/* Next button */}
              <button
                onClick={() => setMultiIndex(i => Math.min(multiData.length - 1, i + 1))}
                disabled={multiIndex === multiData.length - 1}
                style={{
                  width: 28, height: 28, flexShrink: 0,
                  background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`,
                  borderRadius: 5,
                  color: multiIndex === multiData.length - 1 ? t.textFaint : t.textMuted,
                  fontSize: 14,
                  cursor: multiIndex === multiData.length - 1 ? "not-allowed" : "pointer",
                  fontFamily: "monospace",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  lineHeight: 1,
                }}
              >→</button>

              {/* Close */}
              <button
                onClick={() => { setMultiData(null); setMultiCsvError(null); }}
                style={{
                  padding: "4px 10px",
                  background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`,
                  borderRadius: 5,
                  color: t.textMuted,
                  fontSize: 9,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                  whiteSpace: "nowrap",
                }}
              >
                ✕ Close
              </button>
            </div>
          )}

          {/* Chart panel */}
          <div style={{
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 10,
            padding: 16,
          }}>
            {/* Point view toggle + export buttons */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10, flexWrap: "wrap", gap: 6 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {parsedData && hasReplicates && (
                  <>
                    <div style={{ display: "flex", gap: 4 }}>
                      {[
                        { key: "individual", label: "Individual Points" },
                        { key: "errorbars", label: "Error Bars" },
                      ].map(opt => (
                        <button
                          key={opt.key}
                          onClick={() => setPointView(opt.key)}
                          style={{
                            padding: "5px 10px",
                            background: pointView === opt.key ? "rgba(0,230,180,0.12)" : t.btnInactive,
                            border: `1px solid ${pointView === opt.key ? "rgba(0,230,180,0.3)" : "rgba(60,100,160,0.1)"}`,
                            borderRadius: 4,
                            color: pointView === opt.key ? "#00e6b4" : "rgba(160,190,230,0.4)",
                            fontSize: 9,
                            fontWeight: 600,
                            cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace",
                            textTransform: "uppercase",
                            letterSpacing: 0.5,
                            transition: "all 0.15s",
                          }}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                    {pointView === "errorbars" && (
                      <div style={{ display: "flex", gap: 4 }}>
                        {[
                          { key: "sd", label: "±SD" },
                          { key: "sem", label: "±SEM" },
                        ].map(opt => (
                          <button
                            key={opt.key}
                            onClick={() => setErrorBarType(opt.key)}
                            style={{
                              padding: "5px 8px",
                              background: errorBarType === opt.key ? "rgba(255,180,50,0.12)" : t.btnInactive,
                              border: `1px solid ${errorBarType === opt.key ? "rgba(255,180,50,0.3)" : "rgba(60,100,160,0.1)"}`,
                              borderRadius: 4,
                              color: errorBarType === opt.key ? "#ffb432" : "rgba(160,190,230,0.4)",
                              fontSize: 9,
                              fontWeight: 600,
                              cursor: "pointer",
                              fontFamily: "'JetBrains Mono', monospace",
                              transition: "all 0.15s",
                            }}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
              {fitResult && (
                <div style={{ display: "flex", gap: 4 }}>
                  {["PNG", "JPEG"].map(fmt => (
                    <button
                      key={fmt}
                      onClick={() => exportImage(fmt.toLowerCase())}
                      style={{
                        padding: "4px 8px",
                        background: t.btnInactive,
                        border: `1px solid rgba(60,100,160,0.1)`,
                        borderRadius: 4,
                        color: "rgba(160,190,230,0.4)",
                        fontSize: 8,
                        cursor: "pointer",
                        fontFamily: "'JetBrains Mono', monospace",
                        textTransform: "uppercase",
                        letterSpacing: 0.5,
                        transition: "all 0.15s",
                      }}
                    >
                      {fmt}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div ref={chartContainerRef} style={{ position: "relative" }}>
              <canvas
                ref={mainCanvasRef}
                style={{
                  width: "100%",
                  height: isMobile ? (showResiduals ? 240 : 320) : (showResiduals ? 340 : 480),
                  borderRadius: 6,
                  cursor: "crosshair",
                }}
              />
              <div
                ref={tooltipRef}
                style={{
                  display: "none",
                  position: "absolute",
                  top: 0,
                  left: 0,
                  padding: "4px 8px",
                  background: t.tooltip,
                  border: `1px solid ${t.tooltipBorder}`,
                  borderRadius: 4,
                  fontSize: 10,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: t.text,
                  pointerEvents: "none",
                  whiteSpace: "nowrap",
                  zIndex: 10,
                  backdropFilter: "blur(4px)",
                }}
              />
            </div>
          </div>

          {showResiduals && fitResult && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.panelBorder}`,
              borderRadius: 10,
              padding: 16,
            }}>
              <canvas
                ref={residCanvasRef}
                style={{
                  width: "100%",
                  height: isMobile ? 100 : 140,
                  borderRadius: 6,
                }}
              />
            </div>
          )}

          {/* Grubbs Outlier Test Panel */}
          {parsedData && hasReplicates && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.panelBorder}`,
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: t.label, textTransform: "uppercase", letterSpacing: 1 }}>
                  Grubbs' Outlier Test
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span style={{ fontSize: 9, color: t.textDim }}>α =</span>
                  <select
                    value={grubbsAlpha}
                    onChange={(e) => setGrubbsAlpha(parseFloat(e.target.value))}
                    style={{
                      padding: "3px 6px",
                      background: t.input,
                      border: `1px solid ${t.inputBorder}`,
                      borderRadius: 4,
                      color: t.text,
                      fontSize: 10,
                      fontFamily: "'JetBrains Mono', monospace",
                      outline: "none",
                    }}
                  >
                    <option value={0.01}>0.01</option>
                    <option value={0.05}>0.05</option>
                    <option value={0.10}>0.10</option>
                  </select>
                  <button
                    onClick={runGrubbs}
                    style={{
                      padding: "5px 12px",
                      background: "rgba(255,80,106,0.12)",
                      border: "1px solid rgba(255,80,106,0.3)",
                      borderRadius: 4,
                      color: t.red,
                      fontSize: 9,
                      fontWeight: 600,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      textTransform: "uppercase",
                      letterSpacing: 0.5,
                    }}
                  >
                    Run Test
                  </button>
                </div>
              </div>

              {grubbsResults && (
                <>
                  {/* Summary bar */}
                  <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "6px 10px", marginBottom: 10, borderRadius: 6,
                    background: grubbsResults.totalOutliers > 0 ? "rgba(255,80,106,0.08)" : "rgba(0,230,180,0.06)",
                    border: `1px solid ${grubbsResults.totalOutliers > 0 ? "rgba(255,80,106,0.15)" : "rgba(0,230,180,0.15)"}`,
                    flexWrap: "wrap", gap: 6,
                  }}>
                    <span style={{ fontSize: 10, color: grubbsResults.totalOutliers > 0 ? "#ff6b8a" : "#00e6b4" }}>
                      {grubbsResults.totalOutliers > 0
                        ? `${grubbsResults.totalOutliers} outlier${grubbsResults.totalOutliers > 1 ? "s" : ""} detected across ${grubbsResults.groupResults.filter(g => g.outlierCount > 0).length} group${grubbsResults.groupResults.filter(g => g.outlierCount > 0).length > 1 ? "s" : ""}`
                        : "No outliers detected at α=" + grubbsAlpha}
                    </span>
                    <div style={{ display: "flex", gap: 4 }}>
                      {grubbsResults.totalOutliers > 0 && (
                        <button onClick={excludeAllOutliers} style={{
                          padding: "3px 8px", background: "rgba(255,80,106,0.15)", border: "1px solid rgba(255,80,106,0.25)",
                          borderRadius: 3, color: t.red, fontSize: 8, cursor: "pointer",
                          fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                        }}>Exclude All</button>
                      )}
                      {excludedIndices.size > 0 && (
                        <>
                          <button onClick={clearExclusions} style={{
                            padding: "3px 8px", background: "rgba(140,170,210,0.08)", border: "1px solid rgba(140,170,210,0.15)",
                            borderRadius: 3, color: t.labelDim, fontSize: 8, cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                          }}>Clear All</button>
                          <button onClick={refitWithoutExcluded} style={{
                            padding: "3px 8px", background: "rgba(59,158,255,0.15)", border: "1px solid rgba(59,158,255,0.3)",
                            borderRadius: 3, color: t.blue, fontSize: 8, fontWeight: 700, cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                          }}>Refit ({excludedIndices.size} excl.)</button>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Concentration group list */}
                  <div style={{ display: "flex", flexDirection: isMobile ? "column" : "row", gap: 10 }}>
                    {/* Left: clickable concentration list */}
                    <div style={{ minWidth: isMobile ? "auto" : 120, maxHeight: isMobile ? 150 : 260, overflowY: "auto", display: "flex", flexDirection: isMobile ? "row" : "column", flexWrap: isMobile ? "wrap" : "nowrap", gap: 2 }}>
                      {grubbsResults.groupResults.map((g, gi) => {
                        const key = g.x.toString();
                        const isSelected = selectedGrubbsGroup === key;
                        const hasOutlier = g.outlierCount > 0;
                        const groupExcluded = g.indices ? g.indices.some(idx => excludedIndices.has(idx)) : false;
                        return (
                          <button
                            key={gi}
                            onClick={() => setSelectedGrubbsGroup(isSelected ? null : key)}
                            style={{
                              padding: "5px 8px",
                              background: isSelected ? "rgba(59,158,255,0.12)" : t.btnInactive,
                              border: `1px solid ${isSelected ? "rgba(59,158,255,0.3)" : hasOutlier ? "rgba(255,80,106,0.15)" : "rgba(60,100,160,0.08)"}`,
                              borderRadius: 4,
                              color: hasOutlier ? "#ff6b8a" : groupExcluded ? "rgba(255,180,50,0.7)" : "rgba(160,190,230,0.6)",
                              fontSize: 9,
                              cursor: "pointer",
                              fontFamily: "'JetBrains Mono', monospace",
                              textAlign: "left",
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                              gap: 6,
                              transition: "all 0.1s",
                            }}
                          >
                            <span>{g.x < 0.01 || g.x >= 10000 ? g.x.toExponential(2) : g.x.toPrecision(4)}</span>
                            <span style={{ fontSize: 8, opacity: 0.6 }}>
                              n={g.n}
                              {hasOutlier && ` ⚠${g.outlierCount}`}
                            </span>
                          </button>
                        );
                      })}
                    </div>

                    {/* Right: detail view for selected group */}
                    <div style={{ flex: 1, minHeight: 80 }}>
                      {selectedGrubbsGroup ? (() => {
                        const gResult = grubbsResults.groupResults.find(g => g.x.toString() === selectedGrubbsGroup);
                        if (!gResult) return null;
                        const grouped = groupedData.find(g => g.x.toString() === selectedGrubbsGroup);
                        if (!grouped) return null;

                        return (
                          <div>
                            <div style={{ fontSize: 10, color: t.labelDim, marginBottom: 6 }}>
                              Conc: <span style={{ color: t.text, fontWeight: 600 }}>{gResult.x < 0.01 || gResult.x >= 10000 ? gResult.x.toExponential(3) : gResult.x.toPrecision(5)}</span>
                              {gResult.tested && gResult.result && (
                                <span style={{ marginLeft: 10 }}>
                                  G<sub>crit</sub>: <span style={{ color: t.orange }}>{gResult.result.gCrit.toFixed(3)}</span>
                                </span>
                              )}
                            </div>

                            {!gResult.tested && (
                              <div style={{ fontSize: 9, color: t.textDim, fontStyle: "italic" }}>
                                n={gResult.n}: need n≥3 for Grubbs' test
                              </div>
                            )}

                            {gResult.tested && gResult.result && (
                              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                                {/* Column header */}
                                <div style={{ display: "grid", gridTemplateColumns: "28px 1fr 60px 60px 40px", gap: 4, padding: "2px 0", borderBottom: "1px solid rgba(60,100,160,0.1)" }}>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)" }}></span>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)" }}>Value</span>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)", textAlign: "right" }}>G stat</span>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)", textAlign: "right" }}>Deviation</span>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)", textAlign: "center" }}>Flag</span>
                                </div>
                                {gResult.result.details.map((d, di) => {
                                  const globalIdx = grouped.indices[d.index];
                                  const isExcl = excludedIndices.has(globalIdx);
                                  return (
                                    <div
                                      key={di}
                                      onClick={() => toggleExclusion(globalIdx)}
                                      style={{
                                        display: "grid",
                                        gridTemplateColumns: "28px 1fr 60px 60px 40px",
                                        gap: 4,
                                        padding: "4px 0",
                                        borderBottom: "1px solid rgba(60,100,160,0.04)",
                                        cursor: "pointer",
                                        opacity: isExcl ? 0.4 : 1,
                                        textDecoration: isExcl ? "line-through" : "none",
                                        transition: "opacity 0.15s",
                                      }}
                                    >
                                      <span style={{
                                        width: 18, height: 18, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center",
                                        background: isExcl ? "rgba(255,80,106,0.2)" : "rgba(0,230,180,0.1)",
                                        border: `1px solid ${isExcl ? "rgba(255,80,106,0.3)" : "rgba(0,230,180,0.2)"}`,
                                        fontSize: 10,
                                      }}>
                                        {isExcl ? "✕" : "✓"}
                                      </span>
                                      <span style={{ fontSize: 11, color: t.text, fontFamily: "'JetBrains Mono', monospace" }}>
                                        {d.value.toFixed(1)}
                                      </span>
                                      <span style={{
                                        fontSize: 10, textAlign: "right",
                                        color: d.isOutlier ? "#ff6b8a" : "rgba(160,190,230,0.5)",
                                        fontWeight: d.isOutlier ? 700 : 400,
                                      }}>
                                        {d.g.toFixed(3)}
                                      </span>
                                      <span style={{
                                        fontSize: 10, textAlign: "right",
                                        color: d.deviation > 0 ? "rgba(0,230,180,0.6)" : "rgba(255,140,180,0.6)",
                                      }}>
                                        {d.deviation > 0 ? "+" : ""}{d.deviation.toFixed(1)}
                                      </span>
                                      <span style={{ fontSize: 9, textAlign: "center", color: d.isOutlier ? "#ff6b8a" : "rgba(100,140,180,0.3)" }}>
                                        {d.isOutlier ? "OUT" : "—"}
                                      </span>
                                    </div>
                                  );
                                })}
                                {/* Group stats */}
                                <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid rgba(60,100,160,0.1)", fontSize: 9, color: t.textDim, display: "flex", gap: 12 }}>
                                  <span>Mean: <span style={{ color: t.text }}>{gResult.result.mean.toFixed(1)}</span></span>
                                  <span>SD: <span style={{ color: t.text }}>{gResult.result.sd.toFixed(1)}</span></span>
                                  <span>%CV: <span style={{ color: gResult.result.sd / gResult.result.mean * 100 > 20 ? "#ffb432" : "#c8daf0" }}>{(gResult.result.sd / gResult.result.mean * 100).toFixed(1)}%</span></span>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })() : (
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 80, color: "rgba(140,170,210,0.25)", fontSize: 10 }}>
                          Click a concentration to inspect replicates
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {!parsedData && (
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: 300,
              color: t.textFaint,
              fontSize: 13,
            }}>
              Enter data and click FIT MODEL to begin
            </div>
          )}
        </div>
      </div>

      {/* PDF Report Modal */}
      {showPdfModal && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
          display: "flex", alignItems: "center", justifyContent: "center",
          zIndex: 1000,
        }} onClick={() => setShowPdfModal(false)}>
          <div style={{
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 12,
            padding: 24,
            width: 380,
            maxWidth: "90vw",
            maxHeight: "80vh",
            overflowY: "auto",
          }} onClick={(e) => e.stopPropagation()}>
            <div style={{ fontSize: 14, fontWeight: 700, color: t.text, marginBottom: 16 }}>
              PDF Report
            </div>
            <p style={{ fontSize: 9, color: t.textDim, marginBottom: 14, lineHeight: 1.5 }}>
              Select sections to include in the exported report.
            </p>

            {/* Sections */}
            {[
              { key: "modelInfo", label: "Model & Parameters", bold: true },
            ].map(({ key, label, bold }) => (
              <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections[key]}
                  onChange={() => setPdfSections(prev => ({ ...prev, [key]: !prev[key] }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text, fontWeight: bold ? 600 : 400 }}>{label}</span>
              </label>
            ))}
            {pdfSections.modelInfo && (
              <div style={{ marginLeft: 24, marginBottom: 8 }}>
                {[
                  { key: "modelParams", label: "Parameter values (A, B, C, D / Bottom, Hill, EC50, Top, S)" },
                  { key: "paramEC50", label: "EC50" },
                  ...(activeModel === "5PL" && fitResult && fitResult.bioEC50 ? [{ key: "paramBioEC50", label: "Biological EC50" }] : []),
                ].map(({ key, label }) => (
                  <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={pdfSections[key]}
                      onChange={() => setPdfSections(prev => ({ ...prev, [key]: !prev[key] }))}
                      style={{ accentColor: t.teal }}
                    />
                    <span style={{ fontSize: 9, color: t.textMuted }}>{label}</span>
                  </label>
                ))}
              </div>
            )}

            {[
              { key: "plot", label: "Fitted Curve Plot" },
              { key: "rawData", label: "Raw Data Table" },
            ].map(({ key, label }) => (
              <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections[key]}
                  onChange={() => setPdfSections(prev => ({ ...prev, [key]: !prev[key] }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>{label}</span>
              </label>
            ))}

            {/* Fit Parameters - expandable */}
            <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={pdfSections.fitParams}
                onChange={() => setPdfSections(prev => ({ ...prev, fitParams: !prev.fitParams }))}
                style={{ accentColor: t.teal }}
              />
              <span style={{ fontSize: 10, color: t.text, fontWeight: 600 }}>Goodness of Fit Statistics</span>
            </label>
            {pdfSections.fitParams && (
              <div style={{ marginLeft: 24, marginBottom: 8 }}>
                {[
                  { key: "paramR2", label: "R\u00B2" },
                  { key: "paramRMSE", label: "RMSE" },
                  { key: "paramSSR", label: "SSR" },
                  { key: "paramAIC", label: "AIC" },
                  { key: "paramAICc", label: "AICc" },
                  { key: "paramBIC", label: "BIC" },
                  { key: "paramConverged", label: "Converged / N / K" },
                  { key: "paramNK", label: "N data points / K parameters" },
                ].map(({ key, label }) => (
                  <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={pdfSections[key]}
                      onChange={() => setPdfSections(prev => ({ ...prev, [key]: !prev[key] }))}
                      style={{ accentColor: t.teal }}
                    />
                    <span style={{ fontSize: 9, color: t.textMuted }}>{label}</span>
                  </label>
                ))}
              </div>
            )}

            {/* Model comparison */}
            {comparison && comparison.fit5PL && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections.modelComparison}
                  onChange={() => setPdfSections(prev => ({ ...prev, modelComparison: !prev.modelComparison }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>Model Comparison (4PL vs 5PL)</span>
              </label>
            )}

            {/* Background */}
            {parsedData && parsedData.bgSubtracted > 0 && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections.backgroundInfo}
                  onChange={() => setPdfSections(prev => ({ ...prev, backgroundInfo: !prev.backgroundInfo }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>Background Subtraction</span>
              </label>
            )}

            {/* Normalization */}
            {parsedData && parsedData.normalized && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections.normalizationInfo}
                  onChange={() => setPdfSections(prev => ({ ...prev, normalizationInfo: !prev.normalizationInfo }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>Normalization Info</span>
              </label>
            )}

            {/* Outlier results */}
            {grubbsResults && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections.outlierResults}
                  onChange={() => setPdfSections(prev => ({ ...prev, outlierResults: !prev.outlierResults }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>Grubbs' Outlier Test Results</span>
              </label>
            )}

            {/* Actions */}
            <div style={{ display: "flex", gap: 8, marginTop: 18 }}>
              <button
                onClick={() => setShowPdfModal(false)}
                style={{
                  flex: 1, padding: "10px 0",
                  background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`,
                  borderRadius: 6,
                  color: t.textMuted,
                  fontSize: 10, fontWeight: 600,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                Cancel
              </button>
              <button
                onClick={generatePdfReport}
                disabled={pdfGenerating}
                style={{
                  flex: 2, padding: "10px 0",
                  background: pdfGenerating ? "rgba(0,230,180,0.05)" : "rgba(0,230,180,0.12)",
                  border: `1px solid rgba(0,230,180,0.3)`,
                  borderRadius: 6,
                  color: pdfGenerating ? t.textDim : "#00e6b4",
                  fontSize: 10, fontWeight: 700,
                  cursor: pdfGenerating ? "wait" : "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                  textTransform: "uppercase",
                  letterSpacing: 0.5,
                }}
              >
                {pdfGenerating ? "Generating..." : "Generate PDF"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}