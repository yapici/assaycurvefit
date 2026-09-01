// ── Statistical distributions ─────────────────────────────────────
// Log-gamma, regularized incomplete beta, and the Student-t CDF/quantile.
// Used by the Grubbs outlier test and (from Phase 2 onward) parameter
// confidence intervals.

// Log-gamma function (Lanczos approximation)
export function lnGamma(z) {
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
export function betaIncomplete(x, a, b) {
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
export function tCDF(t, df) {
  const x = df / (df + t * t);
  const ib = betaIncomplete(x, df / 2, 0.5);
  if (t >= 0) return 1 - 0.5 * ib;
  return 0.5 * ib;
}

// Inverse t-distribution via bisection + Newton refinement
// Returns t such that P(T > t) = p (upper tail)
export function tInv(p, df) {
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

// ── F distribution ────────────────────────────────────────────────
// Backs the lack-of-fit and model-comparison F-tests. Expressed through the
// same regularized incomplete beta as the t-distribution above:
//
//   P(F <= f) = I_{d1 f / (d1 f + d2)}(d1/2, d2/2)

// CDF of the F-distribution: P(F <= f) for (df1, df2) degrees of freedom.
export function fCDF(f, df1, df2) {
  if (!isFinite(f) || f <= 0) return 0;
  if (df1 <= 0 || df2 <= 0) return NaN;
  const x = (df1 * f) / (df1 * f + df2);
  return betaIncomplete(x, df1 / 2, df2 / 2);
}

// Upper-tail p-value: P(F > f). This is the quantity an F-test reports, and
// computing it directly (rather than as 1 - fCDF) keeps the precision where it
// matters -- a decisive test has a p-value near zero, which is exactly where
// the subtraction would lose its significant digits.
export function fPValue(f, df1, df2) {
  if (!isFinite(f) || f <= 0) return 1;
  if (df1 <= 0 || df2 <= 0) return NaN;
  // I_x(a,b) = 1 - I_{1-x}(b,a), applied so the result is a direct evaluation.
  const x = df2 / (df1 * f + df2);
  return betaIncomplete(x, df2 / 2, df1 / 2);
}

// Inverse F: returns f such that P(F > f) = p (upper tail), by bisection.
export function fInv(p, df1, df2) {
  if (df1 <= 0 || df2 <= 0) return NaN;
  if (p <= 0) return Infinity;
  if (p >= 1) return 0;

  let lo = 0, hi = 2;
  // Expand until the upper tail beyond `hi` is smaller than the target.
  while (fPValue(hi, df1, df2) > p) {
    hi *= 2;
    if (hi > 1e12) return hi;
  }
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (fPValue(mid, df1, df2) > p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-12 * Math.max(1, hi)) break;
  }
  return (lo + hi) / 2;
}
