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
export function computeAIC(n, k, ssr) {
  // k = number of parameters, n = number of observations
  // AIC = n * ln(SSR/n) + 2k
  return n * Math.log(ssr / n) + 2 * k;
}

export function computeAICc(n, k, ssr) {
  // Corrected AIC for small sample sizes
  const aic = computeAIC(n, k, ssr);
  if (n - k - 1 <= 0) return Infinity;
  return aic + (2 * k * (k + 1)) / (n - k - 1);
}

export function computeBIC(n, k, ssr) {
  return n * Math.log(ssr / n) + k * Math.log(n);
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
