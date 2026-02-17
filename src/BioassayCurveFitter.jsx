import { useState, useCallback, useMemo, useRef, useEffect } from "react";

// ── Numerical Engine ──────────────────────────────────────────────
// 4PL: y = D + (A - D) / (1 + (x/C)^B)
// 5PL: y = D + (A - D) / (1 + (x/C)^B)^E

function model4PL(x, params) {
  const [A, B, C, D] = params;
  if (x <= 0) return A;
  return D + (A - D) / (1 + Math.pow(x / C, B));
}

function model5PL(x, params) {
  const [A, B, C, D, E] = params;
  if (x <= 0) return A;
  return D + (A - D) / Math.pow(1 + Math.pow(x / C, B), E);
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
    
    // Keep C positive
    if (modelFn === model4PL && newParams[2] <= 0) newParams[2] = Math.abs(newParams[2]) || 1e-6;
    if (modelFn === model5PL) {
      if (newParams[2] <= 0) newParams[2] = Math.abs(newParams[2]) || 1e-6;
      if (newParams[4] <= 0) newParams[4] = Math.abs(newParams[4]) || 0.1;
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
  if (is5PL) return [A, B, C, D, 1];
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

  return { ...best, r2, rmse, yPred, aicc, bic, aic, k, n };
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
function drawChart(canvas, xData, yData, fitResult, modelType, options = {}) {
  const { pointView = "individual", errorBarType = "sd", outlierIndices = null, excludedIndices: exclSet = null } = options;
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
    const modelFn = modelType === "4PL" ? model4PL : model5PL;
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
  canvas._chartMeta = { pad, plotW, plotH, xMin, xMax, yMin, yMax, W, H, toCanvasX, toCanvasY };

  // Background
  ctx.fillStyle = "#0a0f1a";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "rgba(100,140,180,0.08)";
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
  ctx.strokeStyle = "rgba(140,170,210,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = "rgba(160,190,230,0.6)";
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
  ctx.fillStyle = "rgba(180,210,240,0.7)";
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
    const modelFn = modelType === "4PL" ? model4PL : model5PL;
    ctx.strokeStyle = modelType === "4PL" ? "#3b9eff" : "#a855f7";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = modelType === "4PL" ? "rgba(59,158,255,0.4)" : "rgba(168,85,247,0.4)";
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
        ctx.strokeStyle = "rgba(255,180,50,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + plotW, cy); ctx.stroke();
        ctx.setLineDash([]);
        
        // EC50 marker
        ctx.fillStyle = "rgba(255,180,50,0.9)";
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "rgba(255,180,50,0.7)";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(`EC50: ${ec50.toExponential(3)}`, cx + 10, cy - 8);
      }
    }
  }

  // Data points
  if (pointView === "errorbars") {
    // Error bar view: mean ± SD or SEM per concentration
    grouped.forEach((g) => {
      if (g.x <= 0) return;
      const lx = Math.log10(g.x);
      const cx = toCanvasX(lx);
      const cyMean = toCanvasY(g.mean);
      const errVal = errorBarType === "sem" ? g.sem : g.sd;
      const cyHi = toCanvasY(g.mean + errVal);
      const cyLo = toCanvasY(g.mean - errVal);

      // Error bar line
      if (g.n > 1) {
        ctx.strokeStyle = "rgba(0,230,180,0.55)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx, cyHi); ctx.lineTo(cx, cyLo); ctx.stroke();
        // Caps
        const capW = 4;
        ctx.beginPath(); ctx.moveTo(cx - capW, cyHi); ctx.lineTo(cx + capW, cyHi); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx - capW, cyLo); ctx.lineTo(cx + capW, cyLo); ctx.stroke();
      }

      // Mean point glow
      const grad = ctx.createRadialGradient(cx, cyMean, 0, cx, cyMean, 14);
      grad.addColorStop(0, "rgba(0,230,180,0.35)");
      grad.addColorStop(1, "rgba(0,230,180,0)");
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(cx, cyMean, 14, 0, Math.PI * 2); ctx.fill();

      // Mean point
      ctx.fillStyle = "#00e6b4";
      ctx.strokeStyle = "rgba(0,230,180,0.6)";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cyMean, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

      // n label
      if (g.n > 1) {
        ctx.fillStyle = "rgba(160,190,230,0.35)";
        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(`n=${g.n}`, cx + 8, cyMean + 3);
      }
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
        ctx.fillStyle = "#ff506a";
        ctx.strokeStyle = "rgba(255,80,106,0.4)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        // Strikethrough line
        ctx.strokeStyle = "rgba(255,80,106,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy); ctx.stroke();
        ctx.globalAlpha = 1;
      } else if (isOutlier) {
        // Outlier flagged (red glow + X)
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 12);
        grad.addColorStop(0, "rgba(255,80,100,0.3)");
        grad.addColorStop(1, "rgba(255,80,100,0)");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#ff506a";
        ctx.strokeStyle = "rgba(255,80,106,0.6)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        const s = 6;
        ctx.strokeStyle = "rgba(255,80,106,0.7)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx - s, cy - s); ctx.lineTo(cx + s, cy + s); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx + s, cy - s); ctx.lineTo(cx - s, cy + s); ctx.stroke();
      } else {
        // Normal point
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 12);
        grad.addColorStop(0, "rgba(0,230,180,0.3)");
        grad.addColorStop(1, "rgba(0,230,180,0)");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#00e6b4";
        ctx.strokeStyle = "rgba(0,230,180,0.5)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
    });
  }
}

// ── Residuals chart ───────────────────────────────────────────────
function drawResiduals(canvas, xData, yData, fitResult, modelType) {
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

  const modelFn = modelType === "4PL" ? model4PL : model5PL;
  const res = xData.map((x, i) => yData[i] - modelFn(x, fitResult.params));
  const logX = xData.filter(x => x > 0).map(x => Math.log10(x));

  let xMin = Math.floor(Math.min(...logX)) - 0.5;
  let xMax = Math.ceil(Math.max(...logX)) + 0.5;
  const rMax = Math.max(Math.abs(Math.min(...res)), Math.abs(Math.max(...res))) * 1.3 || 1;

  const toCanvasX = (lx) => pad.left + ((lx - xMin) / (xMax - xMin)) * plotW;
  const toCanvasY = (r) => pad.top + ((rMax - r) / (2 * rMax)) * plotH;

  ctx.fillStyle = "#0a0f1a";
  ctx.fillRect(0, 0, W, H);

  // Zero line
  ctx.strokeStyle = "rgba(255,180,50,0.3)";
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
    const color = res[i] >= 0 ? "#00e6b4" : "#ff6b8a";
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    
    // Line to zero
    ctx.strokeStyle = color + "44";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, zeroY); ctx.stroke();
  });

  ctx.fillStyle = "rgba(160,190,230,0.6)";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("Residuals", pad.left + plotW / 2, H - 6);
}

// ── Sample Data ───────────────────────────────────────────────────
const SAMPLE_DATA = `Concentration,Response
0.001,0.05
0.003,0.08
0.01,0.12
0.03,0.25
0.1,0.52
0.3,1.15
1,2.45
3,3.82
10,4.55
30,4.88
100,5.02
300,5.10`;

// ── Main Component ────────────────────────────────────────────────
export default function BioassayCurveFitter() {
  const [rawData, setRawData] = useState(SAMPLE_DATA);
  const [modelType, setModelType] = useState("Auto");
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

      setParsedData({ xData, yData, yRaw, bgSubtracted: bgSub ? bgSub.mean : 0 });

      if (modelType === "Auto") {
        // Fit both models and compare
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
        // Check if 5PL E parameter is close to 1 (meaning 5PL collapses to 4PL)
        const eParam = fit5.params[4];
        const eNearOne = Math.abs(eParam - 1) < 0.05;

        let selected, reason;
        if (eNearOne) {
          selected = "4PL";
          reason = `5PL asymmetry parameter E≈${eParam.toFixed(3)} (near 1.0); extra parameter not justified`;
        } else if (deltaAICc > 2 && deltaBIC > 0) {
          selected = "5PL";
          reason = `5PL preferred: ΔAICc=${deltaAICc.toFixed(1)} (>2 threshold), ΔBIC=${deltaBIC.toFixed(1)}`;
        } else if (deltaAICc < -2) {
          selected = "4PL";
          reason = `4PL preferred: ΔAICc=${deltaAICc.toFixed(1)} favors simpler model`;
        } else {
          // Within 2 AICc units: prefer simpler model (parsimony)
          selected = "4PL";
          reason = `Models within ΔAICc=${deltaAICc.toFixed(1)}; 4PL preferred by parsimony`;
        }

        setComparison({ fit4PL: fit4, fit5PL: fit5, selected, reason });
        setActiveModel(selected);
        setFitResult(selected === "4PL" ? fit4 : fit5);
      } else {
        // Manual model selection
        if (modelType === "5PL" && xData.length < 5) { setError("Need at least 5 data points for 5PL"); return; }
        const modelFn = modelType === "4PL" ? model4PL : model5PL;
        const result = fitModel(xData, yData, modelFn, modelType === "5PL");
        if (!result) { setError("Fitting failed to converge. Check your data."); return; }
        setActiveModel(modelType);
        setFitResult(result);
        setComparison(null);
      }
    } catch (e) {
      setError("Error: " + e.message);
    }
  }, [rawData, modelType, parseData, bgEnabled, bgRawData, parseBgValues]);

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
      drawChart(mainCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, { pointView, errorBarType, outlierIndices: chartOutlierIndices, excludedIndices });
    }
  }, [parsedData, fitResult, activeModel, pointView, errorBarType, chartOutlierIndices, excludedIndices]);

  useEffect(() => {
    if (residCanvasRef.current && parsedData && fitResult && showResiduals) {
      drawResiduals(residCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel);
    }
  }, [parsedData, fitResult, activeModel, showResiduals]);

  // Resize handler
  useEffect(() => {
    const handleResize = () => {
      if (mainCanvasRef.current && parsedData) {
        drawChart(mainCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, { pointView, errorBarType, outlierIndices: chartOutlierIndices, excludedIndices });
      }
      if (residCanvasRef.current && parsedData && fitResult && showResiduals) {
        drawResiduals(residCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [parsedData, fitResult, activeModel, showResiduals, pointView, errorBarType, chartOutlierIndices, excludedIndices]);

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

      // Check if mouse is within plot area
      if (mx < pad.left || mx > pad.left + plotW || my < pad.top || my > pad.top + plotH) {
        tooltip.style.display = "none";
        return;
      }

      // Convert mouse position to data coordinates
      const logXMouse = xMin + (mx - pad.left) / plotW * (xMax - xMin);
      const yMouse = yMax - (my - pad.top) / plotH * (yMax - yMin);

      // Check proximity to data points first (within 12px)
      let nearestDist = Infinity;
      let nearestInfo = null;
      const hitRadius = 12;

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

      // If no data point nearby, show curve value
      if (!nearestInfo && fitResult) {
        const modelFn = activeModel === "4PL" ? model4PL : model5PL;
        const xVal = Math.pow(10, logXMouse);
        const yFit = modelFn(xVal, fitResult.params);
        const cyFit = meta.toCanvasY(yFit);
        if (Math.abs(my - cyFit) < 20) {
          nearestInfo = { type: "curve", x: xVal, y: yFit };
        }
      }

      if (nearestInfo) {
        const fmtX = nearestInfo.x < 0.01 || nearestInfo.x >= 10000
          ? nearestInfo.x.toExponential(3) : nearestInfo.x.toPrecision(4);
        const fmtY = Math.abs(nearestInfo.y) < 0.01 || Math.abs(nearestInfo.y) >= 100000
          ? nearestInfo.y.toExponential(3) : nearestInfo.y.toFixed(1);

        let label = nearestInfo.type === "curve" ? "Fit" : "Data";
        tooltip.innerHTML = `<span style="color:${nearestInfo.type === "curve" ? (activeModel === "4PL" ? "#3b9eff" : "#a855f7") : "#00e6b4"}">${label}</span>&nbsp; x: ${fmtX}&nbsp; y: ${fmtY}`;
        
        // Position tooltip near cursor but keep within bounds
        const containerRect = chartContainerRef.current ? chartContainerRef.current.getBoundingClientRect() : rect;
        let tx = e.clientX - containerRect.left + 14;
        let ty = e.clientY - containerRect.top - 28;
        // Clamp to container
        const tw = tooltip.offsetWidth || 160;
        if (tx + tw > containerRect.width - 8) tx = e.clientX - containerRect.left - tw - 10;
        if (ty < 4) ty = 4;
        tooltip.style.left = tx + "px";
        tooltip.style.top = ty + "px";
        tooltip.style.display = "block";
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
  }, [parsedData, fitResult, activeModel]);

  const paramLabels = activeModel === "4PL"
    ? ["A (min)", "B (slope)", "C (EC50)", "D (max)"]
    : ["A (min)", "B (slope)", "C (EC50)", "D (max)", "E (asymmetry)"];

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
    } else {
      const modelFn = modelType === "4PL" ? model4PL : model5PL;
      const result = fitModel(xFiltered, yFiltered, modelFn, modelType === "5PL");
      if (!result) { setError("Fitting failed"); return; }
      setActiveModel(modelType);
      setFitResult(result);
    }
  }, [parsedData, excludedIndices, modelType]);

  const exportCSV = useCallback(() => {
    if (!parsedData || !fitResult) return;
    const modelFn = activeModel === "4PL" ? model4PL : model5PL;
    const hasBg = parsedData.bgSubtracted > 0;
    let csv = "";
    if (hasBg) csv += `# Background subtracted: ${parsedData.bgSubtracted.toFixed(2)}\n`;
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

  const interpolate = useCallback((targetY) => {
    if (!fitResult) return null;
    const [A, B, C, D] = fitResult.params;
    if (activeModel === "4PL") {
      const ratio = (A - targetY) / (targetY - D);
      if (ratio <= 0) return null;
      return C * Math.pow(ratio, 1 / B);
    }
    // 5PL: numerical inverse
    const E = fitResult.params[4];
    const modelFn = model5PL;
    let lo = 1e-15, hi = 1e15;
    for (let i = 0; i < 100; i++) {
      const mid = Math.sqrt(lo * hi);
      const yMid = modelFn(mid, fitResult.params);
      if ((B > 0 && yMid > targetY) || (B < 0 && yMid < targetY)) hi = mid;
      else lo = mid;
    }
    return Math.sqrt(lo * hi);
  }, [fitResult, activeModel]);

  const [interpY, setInterpY] = useState("");
  const [interpResult, setInterpResult] = useState(null);

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(160deg, #060a12 0%, #0c1324 40%, #0a1020 100%)",
      color: "#c8daf0",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
      padding: "24px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        textarea { font-family: 'JetBrains Mono', monospace; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
        ::-webkit-scrollbar-thumb { background: rgba(100,140,200,0.3); border-radius: 3px; }
      `}</style>

      {/* Header */}
      <div style={{ maxWidth: 1200, margin: "0 auto 24px" }}>
        <h1 style={{
          fontFamily: "'Space Grotesk', sans-serif",
          fontSize: 28,
          fontWeight: 700,
          background: "linear-gradient(135deg, #3b9eff, #a855f7, #00e6b4)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          letterSpacing: "-0.5px",
        }}>
          Bioassay Curve Fitter
        </h1>
        <p style={{ fontSize: 12, color: "rgba(160,190,230,0.5)", marginTop: 4 }}>
          4-Parameter & 5-Parameter Logistic Regression | Levenberg-Marquardt Optimization
        </p>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", display: "grid", gridTemplateColumns: "340px 1fr", gap: 20, alignItems: "start" }}>
        {/* Left Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Data Input */}
          <div style={{
            background: "rgba(12,20,40,0.8)",
            border: "1px solid rgba(60,100,160,0.15)",
            borderRadius: 10,
            padding: 16,
          }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "rgba(160,190,230,0.7)", marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
              Data Input
            </div>
            <textarea
              value={rawData}
              onChange={(e) => setRawData(e.target.value)}
              placeholder="Concentration,Response&#10;0.01,0.5&#10;0.1,1.2&#10;..."
              style={{
                width: "100%",
                height: 200,
                background: "rgba(6,10,20,0.8)",
                border: "1px solid rgba(60,100,160,0.2)",
                borderRadius: 6,
                color: "#c8daf0",
                fontSize: 11,
                padding: 10,
                resize: "vertical",
                outline: "none",
              }}
            />
            <p style={{ fontSize: 9, color: "rgba(140,170,210,0.4)", marginTop: 6 }}>
              CSV/TSV format. First column = concentration, additional columns = replicates. Comma-formatted numbers (e.g. 47,189.7) supported.
            </p>
            {parsedData && (
              <p style={{ fontSize: 9, color: "#00e6b4", marginTop: 4 }}>
                Parsed: {parsedData.xData.length} data points across {new Set(parsedData.xData).size} concentrations
                {parsedData.bgSubtracted ? ` (bg: −${parsedData.bgSubtracted.toFixed(1)})` : ""}
              </p>
            )}
          </div>

          {/* Background Subtraction */}
          <div style={{
            background: "rgba(12,20,40,0.8)",
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
                    background: "rgba(6,10,20,0.8)",
                    border: "1px solid rgba(168,85,247,0.15)",
                    borderRadius: 6,
                    color: "#c8daf0",
                    fontSize: 11,
                    padding: 10,
                    resize: "vertical",
                    outline: "none",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                />
                <p style={{ fontSize: 9, color: "rgba(140,170,210,0.4)", marginTop: 6 }}>
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
                    <span>Mean: <span style={{ color: "#a855f7", fontWeight: 600 }}>{bgStats.mean.toFixed(1)}</span></span>
                    {bgStats.n > 1 && <span>SD: {bgStats.sd.toFixed(1)}</span>}
                    {bgStats.n > 1 && <span>%CV: {(bgStats.sd / bgStats.mean * 100).toFixed(1)}%</span>}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Model Selection */}
          <div style={{
            background: "rgba(12,20,40,0.8)",
            border: "1px solid rgba(60,100,160,0.15)",
            borderRadius: 10,
            padding: 16,
          }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: "rgba(160,190,230,0.7)", marginBottom: 10, textTransform: "uppercase", letterSpacing: 1 }}>
              Model
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {["Auto", "4PL", "5PL"].map(m => {
                const colors = { Auto: { active: "#00e6b4", bg: "rgba(0,230,180,0.15)", border: "rgba(0,230,180,0.4)" }, "4PL": { active: "#3b9eff", bg: "rgba(59,158,255,0.15)", border: "rgba(59,158,255,0.4)" }, "5PL": { active: "#a855f7", bg: "rgba(168,85,247,0.15)", border: "rgba(168,85,247,0.4)" } };
                const c = colors[m];
                return (
                  <button
                    key={m}
                    onClick={() => setModelType(m)}
                    style={{
                      flex: 1,
                      padding: "10px 0",
                      background: modelType === m ? c.bg : "rgba(6,10,20,0.6)",
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
            <div style={{ marginTop: 10, fontSize: 10, color: "rgba(140,170,210,0.4)", lineHeight: 1.6 }}>
              {modelType === "Auto"
                ? "Fits both models; selects best via AICc with parsimony preference"
                : modelType === "4PL"
                  ? "y = D + (A−D) / (1 + (x/C)^B)"
                  : "y = D + (A−D) / (1 + (x/C)^B)^E"}
            </div>
          </div>

          {/* Model Comparison Panel (Auto mode) */}
          {comparison && comparison.fit5PL && (
            <div style={{
              background: "rgba(12,20,40,0.8)",
              border: "1px solid rgba(60,100,160,0.15)",
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "rgba(160,190,230,0.7)", marginBottom: 10, textTransform: "uppercase", letterSpacing: 1 }}>
                Model Comparison
              </div>
              
              {/* Comparison table */}
              <div style={{ fontSize: 10 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4, marginBottom: 4 }}>
                  <span style={{ color: "rgba(140,170,210,0.4)" }}></span>
                  <span style={{ color: "#3b9eff", fontWeight: 600, textAlign: "center" }}>4PL</span>
                  <span style={{ color: "#a855f7", fontWeight: 600, textAlign: "center" }}>5PL</span>
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
                    <span style={{ color: "rgba(160,190,230,0.5)" }}>{row.label}</span>
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
                    <span style={{ color: "rgba(160,190,230,0.5)" }}>E param</span>
                    <span style={{ textAlign: "center", color: "rgba(140,170,210,0.3)" }}>—</span>
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
                <div style={{ fontSize: 11, fontWeight: 700, color: comparison.selected === "4PL" ? "#3b9eff" : "#a855f7", marginBottom: 2 }}>
                  ▸ {comparison.selected} Selected
                </div>
                <div style={{ fontSize: 9, color: "rgba(160,190,230,0.5)", lineHeight: 1.5 }}>
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
                      background: activeModel === m ? "rgba(0,230,180,0.1)" : "rgba(6,10,20,0.6)",
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
              color: "#3b9eff",
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
              color: "#ff6b8a",
              fontSize: 11,
            }}>
              {error}
            </div>
          )}

          {/* Results */}
          {fitResult && (
            <div style={{
              background: "rgba(12,20,40,0.8)",
              border: "1px solid rgba(60,100,160,0.15)",
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "rgba(160,190,230,0.7)", marginBottom: 12, textTransform: "uppercase", letterSpacing: 1, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>Fit Parameters</span>
                <span style={{ color: activeModel === "4PL" ? "#3b9eff" : "#a855f7", fontSize: 10 }}>{activeModel}</span>
              </div>
              {paramLabels.map((label, i) => (
                <div key={i} style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "6px 0",
                  borderBottom: i < paramLabels.length - 1 ? "1px solid rgba(60,100,160,0.08)" : "none",
                }}>
                  <span style={{ fontSize: 11, color: "rgba(160,190,230,0.6)" }}>{label}</span>
                  <span style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: i === 2 ? "#ffb432" : "#00e6b4",
                  }}>
                    {Math.abs(fitResult.params[i]) < 0.01 || Math.abs(fitResult.params[i]) > 10000
                      ? fitResult.params[i].toExponential(4)
                      : fitResult.params[i].toFixed(4)}
                  </span>
                </div>
              ))}

              <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid rgba(60,100,160,0.15)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: "rgba(160,190,230,0.6)" }}>R²</span>
                  <span style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: fitResult.r2 > 0.99 ? "#00e6b4" : fitResult.r2 > 0.95 ? "#ffb432" : "#ff6b8a",
                  }}>
                    {fitResult.r2.toFixed(6)}
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: "rgba(160,190,230,0.6)" }}>RMSE</span>
                  <span style={{ fontSize: 12, color: "#c8daf0" }}>{fitResult.rmse.toFixed(6)}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 11, color: "rgba(160,190,230,0.6)" }}>SSR</span>
                  <span style={{ fontSize: 12, color: "#c8daf0" }}>{fitResult.ssr.toFixed(6)}</span>
                </div>
                {fitResult.aicc !== undefined && (
                  <>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                      <span style={{ fontSize: 11, color: "rgba(160,190,230,0.6)" }}>AICc</span>
                      <span style={{ fontSize: 12, color: "#c8daf0" }}>{fitResult.aicc.toFixed(2)}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                      <span style={{ fontSize: 11, color: "rgba(160,190,230,0.6)" }}>BIC</span>
                      <span style={{ fontSize: 12, color: "#c8daf0" }}>{fitResult.bic.toFixed(2)}</span>
                    </div>
                  </>
                )}
              </div>

              {/* Interpolation */}
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid rgba(60,100,160,0.15)" }}>
                <div style={{ fontSize: 10, color: "rgba(160,190,230,0.5)", marginBottom: 6 }}>
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
                      background: "rgba(6,10,20,0.8)",
                      border: "1px solid rgba(60,100,160,0.2)",
                      borderRadius: 4,
                      color: "#c8daf0",
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
                      color: "#ffb432",
                      fontSize: 10,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    CALC
                  </button>
                </div>
                {interpResult !== null && interpResult !== undefined && (
                  <div style={{ marginTop: 6, fontSize: 11, color: "#ffb432" }}>
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
                    background: showResiduals ? "rgba(0,230,180,0.1)" : "rgba(6,10,20,0.6)",
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
                    background: "rgba(6,10,20,0.6)",
                    border: "1px solid rgba(60,100,160,0.15)",
                    borderRadius: 4,
                    color: "rgba(160,190,230,0.5)",
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  Export CSV
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Charts */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{
            background: "rgba(12,20,40,0.8)",
            border: "1px solid rgba(60,100,160,0.15)",
            borderRadius: 10,
            padding: 16,
          }}>
            {/* Point view toggle */}
            {parsedData && hasReplicates && (
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
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
                        background: pointView === opt.key ? "rgba(0,230,180,0.12)" : "rgba(6,10,20,0.5)",
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
                  <div style={{ display: "flex", gap: 4, marginLeft: 4 }}>
                    {[
                      { key: "sd", label: "±SD" },
                      { key: "sem", label: "±SEM" },
                    ].map(opt => (
                      <button
                        key={opt.key}
                        onClick={() => setErrorBarType(opt.key)}
                        style={{
                          padding: "5px 8px",
                          background: errorBarType === opt.key ? "rgba(255,180,50,0.12)" : "rgba(6,10,20,0.5)",
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
              </div>
            )}
            <div ref={chartContainerRef} style={{ position: "relative" }}>
              <canvas
                ref={mainCanvasRef}
                style={{
                  width: "100%",
                  height: showResiduals ? 340 : 480,
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
                  background: "rgba(10,16,30,0.92)",
                  border: "1px solid rgba(80,120,180,0.25)",
                  borderRadius: 4,
                  fontSize: 10,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: "#c8daf0",
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
              background: "rgba(12,20,40,0.8)",
              border: "1px solid rgba(60,100,160,0.15)",
              borderRadius: 10,
              padding: 16,
            }}>
              <canvas
                ref={residCanvasRef}
                style={{
                  width: "100%",
                  height: 140,
                  borderRadius: 6,
                }}
              />
            </div>
          )}

          {/* Grubbs Outlier Test Panel */}
          {parsedData && hasReplicates && (
            <div style={{
              background: "rgba(12,20,40,0.8)",
              border: "1px solid rgba(60,100,160,0.15)",
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: "rgba(160,190,230,0.7)", textTransform: "uppercase", letterSpacing: 1 }}>
                  Grubbs' Outlier Test
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span style={{ fontSize: 9, color: "rgba(140,170,210,0.4)" }}>α =</span>
                  <select
                    value={grubbsAlpha}
                    onChange={(e) => setGrubbsAlpha(parseFloat(e.target.value))}
                    style={{
                      padding: "3px 6px",
                      background: "rgba(6,10,20,0.8)",
                      border: "1px solid rgba(60,100,160,0.2)",
                      borderRadius: 4,
                      color: "#c8daf0",
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
                      color: "#ff6b8a",
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
                          borderRadius: 3, color: "#ff6b8a", fontSize: 8, cursor: "pointer",
                          fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                        }}>Exclude All</button>
                      )}
                      {excludedIndices.size > 0 && (
                        <>
                          <button onClick={clearExclusions} style={{
                            padding: "3px 8px", background: "rgba(140,170,210,0.08)", border: "1px solid rgba(140,170,210,0.15)",
                            borderRadius: 3, color: "rgba(160,190,230,0.6)", fontSize: 8, cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                          }}>Clear All</button>
                          <button onClick={refitWithoutExcluded} style={{
                            padding: "3px 8px", background: "rgba(59,158,255,0.15)", border: "1px solid rgba(59,158,255,0.3)",
                            borderRadius: 3, color: "#3b9eff", fontSize: 8, fontWeight: 700, cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                          }}>Refit ({excludedIndices.size} excl.)</button>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Concentration group list */}
                  <div style={{ display: "flex", gap: 10 }}>
                    {/* Left: clickable concentration list */}
                    <div style={{ minWidth: 120, maxHeight: 260, overflowY: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
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
                              background: isSelected ? "rgba(59,158,255,0.12)" : "rgba(6,10,20,0.4)",
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
                            <div style={{ fontSize: 10, color: "rgba(160,190,230,0.6)", marginBottom: 6 }}>
                              Conc: <span style={{ color: "#c8daf0", fontWeight: 600 }}>{gResult.x < 0.01 || gResult.x >= 10000 ? gResult.x.toExponential(3) : gResult.x.toPrecision(5)}</span>
                              {gResult.tested && gResult.result && (
                                <span style={{ marginLeft: 10 }}>
                                  G<sub>crit</sub>: <span style={{ color: "#ffb432" }}>{gResult.result.gCrit.toFixed(3)}</span>
                                </span>
                              )}
                            </div>

                            {!gResult.tested && (
                              <div style={{ fontSize: 9, color: "rgba(140,170,210,0.4)", fontStyle: "italic" }}>
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
                                      <span style={{ fontSize: 11, color: "#c8daf0", fontFamily: "'JetBrains Mono', monospace" }}>
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
                                <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid rgba(60,100,160,0.1)", fontSize: 9, color: "rgba(140,170,210,0.4)", display: "flex", gap: 12 }}>
                                  <span>Mean: <span style={{ color: "#c8daf0" }}>{gResult.result.mean.toFixed(1)}</span></span>
                                  <span>SD: <span style={{ color: "#c8daf0" }}>{gResult.result.sd.toFixed(1)}</span></span>
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
              color: "rgba(140,170,210,0.3)",
              fontSize: 13,
            }}>
              Enter data and click FIT MODEL to begin
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
