import { describe, it, expect } from "vitest";
import {
  parameterCovariance, parameterIntervals, correlationMatrix, backTransformLog10,
} from "../inference.js";
import { matInverse, matMul, matTranspose } from "../linalg.js";
import { tInv } from "../distributions.js";
import { fitModel, fitConstrainedModel } from "../lm.js";
import { model4PL, model5PL } from "../models.js";
import { make4PLData, make4PLDataGaussian } from "./fixtures.js";

describe("matInverse", () => {
  it("inverts a small matrix", () => {
    const A = [[4, 7], [2, 6]];
    const inv = matInverse(A);
    // Closed form: 1/det * [[d, -b], [-c, a]], det = 10
    expect(inv[0][0]).toBeCloseTo(0.6, 10);
    expect(inv[0][1]).toBeCloseTo(-0.7, 10);
    expect(inv[1][0]).toBeCloseTo(-0.2, 10);
    expect(inv[1][1]).toBeCloseTo(0.4, 10);
  });

  it("round-trips to the identity", () => {
    const A = [[4, -2, 1], [-2, 4, -2], [1, -2, 4]];
    const I = matMul(A, matInverse(A));
    I.forEach((row, i) => row.forEach((v, j) => {
      expect(v).toBeCloseTo(i === j ? 1 : 0, 9);
    }));
  });

  it("returns null for a singular matrix", () => {
    expect(matInverse([[1, 2], [2, 4]])).toBeNull();
  });
});

describe("parameterCovariance — validated against closed-form OLS", () => {
  // A model linear in its parameters, y = a + b*x, has an exact textbook
  // covariance. The general nonlinear machinery must reproduce it, which is
  // the strongest available check that the whole sigma^2 (J^T J)^-1 path is
  // right. For a linear model the Jacobian IS the design matrix.
  const x = [1, 2, 3, 4, 5, 6, 7, 8];
  const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1];
  const J = x.map(v => [1, v]); // d/da = 1, d/db = x

  // OLS solution and its residual sum of squares.
  const n = x.length;
  const xbar = x.reduce((s, v) => s + v, 0) / n;
  const ybar = y.reduce((s, v) => s + v, 0) / n;
  const Sxx = x.reduce((s, v) => s + (v - xbar) ** 2, 0);
  const Sxy = x.reduce((s, v, i) => s + (v - xbar) * (y[i] - ybar), 0);
  const b = Sxy / Sxx;
  const a = ybar - b * xbar;
  const ssr = y.reduce((s, v, i) => s + (v - (a + b * x[i])) ** 2, 0);

  it("reproduces the textbook slope and intercept standard errors", () => {
    const { cov, dof, syx } = parameterCovariance(J, ssr);
    expect(dof).toBe(n - 2);

    const s = Math.sqrt(ssr / (n - 2));
    expect(syx).toBeCloseTo(s, 12);

    const seSlope = s / Math.sqrt(Sxx);
    const seIntercept = s * Math.sqrt(1 / n + (xbar * xbar) / Sxx);

    expect(Math.sqrt(cov[1][1])).toBeCloseTo(seSlope, 12);
    expect(Math.sqrt(cov[0][0])).toBeCloseTo(seIntercept, 12);
  });

  it("reproduces the textbook intercept/slope covariance", () => {
    const { cov } = parameterCovariance(J, ssr);
    const s2 = ssr / (n - 2);
    expect(cov[0][1]).toBeCloseTo(-s2 * xbar / Sxx, 12);
    expect(cov[0][1]).toBeCloseTo(cov[1][0], 14); // symmetric
  });

  it("produces the textbook confidence interval on the slope", () => {
    const { cov, dof } = parameterCovariance(J, ssr);
    const { ci } = parameterIntervals([a, b], cov, dof);
    const s = Math.sqrt(ssr / (n - 2));
    const margin = tInv(0.025, n - 2) * (s / Math.sqrt(Sxx));
    expect(ci[1].lo).toBeCloseTo(b - margin, 10);
    expect(ci[1].hi).toBeCloseTo(b + margin, 10);
  });

  it("divides by n - p, not n", () => {
    // The single most common way to get this wrong. With n=8, p=2 the two
    // differ by a factor sqrt(8/6) = 1.155 -- not subtle, but silent.
    const { sigma2 } = parameterCovariance(J, ssr);
    expect(sigma2).toBeCloseTo(ssr / (n - 2), 12);
    expect(sigma2).not.toBeCloseTo(ssr / n, 6);
  });

  it("matches an explicit sigma^2 (J^T J)^-1 computation", () => {
    const { cov } = parameterCovariance(J, ssr);
    const explicit = matInverse(matMul(matTranspose(J), J))
      .map(row => row.map(v => v * (ssr / (n - 2))));
    cov.forEach((row, i) => row.forEach((v, j) => {
      expect(v).toBeCloseTo(explicit[i][j], 12);
    }));
  });
});

describe("parameterCovariance — degenerate inputs", () => {
  it("returns null when there are no residual degrees of freedom", () => {
    const J = [[1, 1], [1, 2]]; // n = p = 2
    expect(parameterCovariance(J, 0.5)).toBeNull();
  });

  it("returns null when parameters are not identifiable (singular J^T J)", () => {
    // Second column is a multiple of the first: the two params are aliased.
    const J = [[1, 2], [2, 4], [3, 6], [4, 8]];
    expect(parameterCovariance(J, 1)).toBeNull();
  });

  it("returns null for an empty Jacobian", () => {
    expect(parameterCovariance([], 1)).toBeNull();
  });

  it("rejects negative or non-finite weights", () => {
    const J = [[1, 1], [1, 2], [1, 3], [1, 4]];
    expect(parameterCovariance(J, 1, [1, 1, -1, 1])).toBeNull();
    expect(parameterCovariance(J, 1, [1, 1, NaN, 1])).toBeNull();
  });

  it("with unit weights reproduces the unweighted result exactly", () => {
    const J = [[1, 1], [1, 2], [1, 3], [1, 4]];
    const un = parameterCovariance(J, 2);
    const wt = parameterCovariance(J, 2, [1, 1, 1, 1]);
    expect(wt.cov[0][0]).toBeCloseTo(un.cov[0][0], 14);
    expect(wt.cov[1][1]).toBeCloseTo(un.cov[1][1], 14);
  });
});

describe("parameterIntervals", () => {
  const cov = [[4, 0], [0, 9]]; // SEs of 2 and 3
  const params = [10, 20];

  it("reports SE as the square root of the variance", () => {
    const { se } = parameterIntervals(params, cov, 20);
    expect(se[0]).toBeCloseTo(2, 12);
    expect(se[1]).toBeCloseTo(3, 12);
  });

  it("centres a symmetric interval on the estimate", () => {
    const { ci } = parameterIntervals(params, cov, 20);
    expect((ci[0].lo + ci[0].hi) / 2).toBeCloseTo(10, 10);
    expect((ci[1].lo + ci[1].hi) / 2).toBeCloseTo(20, 10);
  });

  it("widens the interval as the confidence level rises", () => {
    const at95 = parameterIntervals(params, cov, 20, 0.05).ci[0];
    const at99 = parameterIntervals(params, cov, 20, 0.01).ci[0];
    expect(at99.hi - at99.lo).toBeGreaterThan(at95.hi - at95.lo);
  });

  it("widens the interval as degrees of freedom fall", () => {
    const wide = parameterIntervals(params, cov, 3).ci[0];
    const tight = parameterIntervals(params, cov, 200).ci[0];
    expect(wide.hi - wide.lo).toBeGreaterThan(tight.hi - tight.lo);
  });

  it("reports null rather than NaN for a non-positive variance", () => {
    const bad = [[-1, 0], [0, NaN]];
    const { se, ci } = parameterIntervals(params, bad, 20);
    expect(se[0]).toBeNull();
    expect(ci[0]).toBeNull();
    expect(se[1]).toBeNull();
  });
});

describe("correlationMatrix", () => {
  it("has a unit diagonal", () => {
    const c = correlationMatrix([[4, 1], [1, 9]]);
    expect(c[0][0]).toBeCloseTo(1, 12);
    expect(c[1][1]).toBeCloseTo(1, 12);
  });

  it("normalises the off-diagonal by the two standard deviations", () => {
    const c = correlationMatrix([[4, 3], [3, 9]]);
    expect(c[0][1]).toBeCloseTo(3 / (2 * 3), 12); // 0.5
  });

  it("is symmetric and bounded by 1 in magnitude", () => {
    const c = correlationMatrix([[4, -5.9], [-5.9, 9]]);
    expect(c[0][1]).toBeCloseTo(c[1][0], 14);
    expect(Math.abs(c[0][1])).toBeLessThanOrEqual(1);
  });
});

describe("backTransformLog10", () => {
  const tCrit = 2;

  it("exponentiates the estimate", () => {
    expect(backTransformLog10(-9, 0.1, tCrit).value).toBeCloseTo(1e-9, 20);
  });

  it("produces an interval that is symmetric in log space, not linear space", () => {
    const { value, ci } = backTransformLog10(-9, 0.1, tCrit);
    // Symmetric on the log scale...
    expect(Math.log10(ci.lo) + Math.log10(ci.hi)).toBeCloseTo(2 * -9, 10);
    // ...and therefore asymmetric on the linear scale.
    expect(value - ci.lo).not.toBeCloseTo(ci.hi - value, 12);
    expect(ci.hi - value).toBeGreaterThan(value - ci.lo);
  });

  it("never produces a negative lower bound, however wide the interval", () => {
    // A linear `estimate +/- t*SE` would go negative here; the log form cannot.
    const { ci } = backTransformLog10(-9, 5, tCrit);
    expect(ci.lo).toBeGreaterThan(0);
  });

  it("applies the delta method for the linear-scale SE", () => {
    const { se } = backTransformLog10(2, 0.05, tCrit);
    expect(se).toBeCloseTo(100 * Math.LN10 * 0.05, 10); // d(10^t)/dt = 10^t ln10
  });

  it("returns nulls when the log-space SE is unavailable", () => {
    const r = backTransformLog10(-9, null, tCrit);
    expect(r.value).toBeCloseTo(1e-9, 20);
    expect(r.se).toBeNull();
    expect(r.ci).toBeNull();
  });
});

describe("fitModel — reported uncertainty", () => {
  const truth = [0, 1.2, 1, 100];

  it("reports dof, syx, se, ci and the log-space EC50", () => {
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 3, seed: 7 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.dof).toBe(xData.length - 4);
    expect(fit.syx).toBeCloseTo(Math.sqrt(fit.ssr / fit.dof), 12);
    expect(fit.se).toHaveLength(4);
    expect(fit.ci).toHaveLength(4);
    expect(fit.logEC50.value).toBeCloseTo(Math.log10(fit.params[2]), 9);
    expect(fit.logEC50.se).toBeGreaterThan(0);
  });

  it("gives an EC50 interval that brackets the estimate asymmetrically", () => {
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 3, seed: 11 });
    const fit = fitModel(xData, yData, model4PL, false);
    const { lo, hi } = fit.ci[2];
    expect(lo).toBeGreaterThan(0);
    expect(lo).toBeLessThan(fit.params[2]);
    expect(hi).toBeGreaterThan(fit.params[2]);
    // Geometric, not arithmetic, centring.
    expect(Math.sqrt(lo * hi)).toBeCloseTo(fit.params[2], 6);
  });

  it("tightens every interval as the sample grows", () => {
    const small = make4PLDataGaussian({ params: truth, sd: 3, nReps: 2, seed: 3 });
    const large = make4PLDataGaussian({ params: truth, sd: 3, nReps: 12, seed: 3 });
    const a = fitModel(small.xData, small.yData, model4PL, false);
    const b = fitModel(large.xData, large.yData, model4PL, false);
    expect(b.logEC50.se).toBeLessThan(a.logEC50.se);
    expect(b.se[1]).toBeLessThan(a.se[1]);
  });

  it("tightens every interval as the noise falls", () => {
    const noisy = make4PLDataGaussian({ params: truth, sd: 8, seed: 5 });
    const clean = make4PLDataGaussian({ params: truth, sd: 1, seed: 5 });
    const a = fitModel(noisy.xData, noisy.yData, model4PL, false);
    const b = fitModel(clean.xData, clean.yData, model4PL, false);
    expect(b.logEC50.se).toBeLessThan(a.logEC50.se);
  });

  it("scales the EC50 SE with the concentration unit but not the log SE", () => {
    // The log-space SE is a property of the curve, not of the units used.
    const base = make4PLDataGaussian({ params: truth, sd: 3, seed: 9 });
    const scaled = {
      xData: base.xData.map(v => v * 1e-9),
      yData: base.yData,
    };
    const a = fitModel(base.xData, base.yData, model4PL, false);
    const b = fitModel(scaled.xData, scaled.yData, model4PL, false);
    expect(b.logEC50.se).toBeCloseTo(a.logEC50.se, 6);
    expect(b.ci[2].lo / 1e-9).toBeCloseTo(a.ci[2].lo, 6);
  });

  it("reports a correlation matrix with a unit diagonal", () => {
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 3, seed: 13 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.correlation).toHaveLength(4);
    for (let i = 0; i < 4; i++) expect(fit.correlation[i][i]).toBeCloseTo(1, 9);
  });

  it("degrades gracefully when there are too few points for inference", () => {
    // 4 points, 4 parameters: a fit exists but carries no degrees of freedom.
    const xData = [0.1, 1, 10, 100];
    const yData = [5, 25, 75, 95];
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.params).toHaveLength(4);
    expect(fit.dof).toBeNull();
    expect(fit.se).toBeNull();
    expect(fit.ci).toBeNull();
  });

  it("reports uncertainty for the 5PL over its 5 parameters", () => {
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 2, seed: 17 });
    const fit = fitModel(xData, yData, model5PL, true);
    expect(fit.se).toHaveLength(5);
    expect(fit.dof).toBe(xData.length - 5);
  });
});

describe("fitConstrainedModel — reported uncertainty", () => {
  const truth = [0, 1, 1, 100];

  it("reports no SE for a parameter that was held fixed", () => {
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 2, seed: 21 });
    const fit = fitConstrainedModel(xData, yData, { 1: 1.0 }); // 3PL
    expect(fit.se).toHaveLength(4);
    expect(fit.se[1]).toBeNull();  // Hill was fixed, not estimated
    expect(fit.ci[1]).toBeNull();
    expect(fit.se[2]).toBeGreaterThan(0); // EC50 was free
  });

  it("uses the free-parameter count for degrees of freedom", () => {
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 2, seed: 23 });
    const fit = fitConstrainedModel(xData, yData, { 1: 1.0 });
    expect(fit.dof).toBe(xData.length - 3);
  });

  it("maps SEs onto the right slots for a 2PL", () => {
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 2, seed: 27 });
    const fit = fitConstrainedModel(xData, yData, { 0: 0, 3: 100 });
    expect(fit.se[0]).toBeNull();          // A fixed
    expect(fit.se[3]).toBeNull();          // D fixed
    expect(fit.se[1]).toBeGreaterThan(0);  // Hill free
    expect(fit.se[2]).toBeGreaterThan(0);  // EC50 free
    expect(fit.dof).toBe(xData.length - 2);
  });

  it("still back-transforms the EC50 interval when the EC50 is the only free param", () => {
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 2, seed: 29 });
    const fit = fitConstrainedModel(xData, yData, { 0: 0, 1: 1, 3: 100 }); // 1PL
    expect(fit.dof).toBe(xData.length - 1);
    expect(fit.ci[2].lo).toBeGreaterThan(0);
    expect(Math.sqrt(fit.ci[2].lo * fit.ci[2].hi)).toBeCloseTo(fit.params[2], 6);
  });

  it("buys little precision by fixing a parameter already at its true value", () => {
    // Data generated with Hill exactly 1.0, then fitted with Hill pinned to
    // 1.0. Spending a parameter on something already known is close to free:
    // the constraint raises SSR (and so sigma^2) while returning a degree of
    // freedom, and the two effects very nearly cancel. Note this does NOT
    // reliably tighten the interval -- the constrained SE can land marginally
    // either side of the unconstrained one.
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 2, seed: 31 });
    const free = fitModel(xData, yData, model4PL, false);
    const fixed = fitConstrainedModel(xData, yData, { 1: 1.0 });
    const ratio = fixed.logEC50.se / free.logEC50.se;
    expect(ratio).toBeGreaterThan(0.9);
    expect(ratio).toBeLessThan(1.1);
  });

  it("inflates the residual spread when a parameter is pinned to a wrong value", () => {
    // The constraint has to actually bite: pinning Hill far from the truth
    // must show up as a worse fit.
    const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 2, seed: 31 });
    const right = fitConstrainedModel(xData, yData, { 1: 1.0 });
    const wrong = fitConstrainedModel(xData, yData, { 1: 4.0 });
    expect(wrong.syx).toBeGreaterThan(right.syx * 1.5);
    expect(wrong.logEC50.se).toBeGreaterThan(right.logEC50.se);
  });
});

describe("confidence interval coverage", () => {
  // The real test of an interval: over many datasets, a nominal 95% interval
  // should contain the true value about 95% of the time. This is what
  // distinguishes a correct SE from one that merely looks plausible.
  const truth = [0, 1.2, 1, 100];
  const TRIALS = 300;

  function coverage(paramIndex, pick) {
    let covered = 0, usable = 0;
    for (let s = 0; s < TRIALS; s++) {
      const { xData, yData } = make4PLDataGaussian({
        params: truth, sd: 4, nConc: 8, nReps: 3, seed: 1000 + s,
      });
      const fit = fitModel(xData, yData, model4PL, false);
      if (!fit || !fit.ci) continue;
      const ci = pick(fit);
      if (!ci) continue;
      usable++;
      if (truth[paramIndex] >= ci.lo && truth[paramIndex] <= ci.hi) covered++;
    }
    return { rate: covered / usable, usable };
  }

  it("covers the true EC50 at close to the nominal 95%", () => {
    const { rate, usable } = coverage(2, f => f.ci[2]);
    expect(usable).toBeGreaterThan(TRIALS * 0.9);
    // Binomial SE at n=300 is ~1.3%, so allow a generous band around 0.95.
    // A wrong divisor (n vs n-p) or a missing t-quantile lands well outside it.
    expect(rate).toBeGreaterThan(0.90);
    expect(rate).toBeLessThan(0.99);
  });

  it("covers the true Hill slope at close to the nominal 95%", () => {
    const { rate } = coverage(1, f => f.ci[1]);
    expect(rate).toBeGreaterThan(0.90);
    expect(rate).toBeLessThan(0.99);
  });

  it("covers the true upper asymptote at close to the nominal 95%", () => {
    const { rate } = coverage(3, f => f.ci[3]);
    expect(rate).toBeGreaterThan(0.90);
    expect(rate).toBeLessThan(0.99);
  });
});
