// ── Goodness-of-fit statistics ────────────────────────────────────
// Summary statistics computed from a converged fit, plus replicate grouping.

// Compute R²
export function rSquared(yData, yPred) {
  const yMean = yData.reduce((a, b) => a + b, 0) / yData.length;
  const ssTot = yData.reduce((s, y) => s + (y - yMean) ** 2, 0);
  const ssRes = yData.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0);
  return 1 - ssRes / ssTot;
}

// Information criteria for model comparison
// For nonlinear regression K is the number of fitted parameters PLUS ONE,
// because the residual variance is estimated too (Motulsky, GraphPad Prism
// regression guide; Burnham & Anderson 2002). Getting this wrong is harmless
// for AIC and BIC -- the offset cancels when two models are compared -- but
// not for AICc, whose correction term is nonlinear in K and would otherwise
// under-penalise the model with more parameters.
function effectiveK(k) {
  return k + 1;
}

export function computeAIC(n, k, ssr) {
  // k = number of fitted parameters, n = number of observations.
  // AIC = n * ln(SSR/n) + 2K   (additive constant dropped; comparison only)
  return n * Math.log(ssr / n) + 2 * effectiveK(k);
}

export function computeAICc(n, k, ssr) {
  // Corrected AIC for small sample sizes: AIC + 2K(K+1)/(n-K-1).
  const K = effectiveK(k);
  const aic = computeAIC(n, k, ssr);
  if (n - K - 1 <= 0) return Infinity;
  return aic + (2 * K * (K + 1)) / (n - K - 1);
}

export function computeBIC(n, k, ssr) {
  // BIC = n * ln(SSR/n) + K * ln(n)   (comparison only, as above)
  return n * Math.log(ssr / n) + effectiveK(k) * Math.log(n);
}

// Group replicate data by concentration, compute stats
export function groupByConcentration(xData, yData) {
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
