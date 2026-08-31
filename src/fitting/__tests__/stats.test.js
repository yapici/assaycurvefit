import { describe, it, expect } from "vitest";
import {
  rSquared, computeAIC, computeAICc, computeBIC, groupByConcentration,
} from "../stats.js";

describe("rSquared", () => {
  it("is 1 for a perfect fit", () => {
    expect(rSquared([1, 2, 3], [1, 2, 3])).toBe(1);
  });

  it("is 0 when the model does no better than the mean", () => {
    expect(rSquared([1, 2, 3], [2, 2, 2])).toBeCloseTo(0, 12);
  });

  it("goes negative when the model is worse than the mean", () => {
    // Worth stating explicitly: for nonlinear regression R^2 is not the
    // "fraction of variance explained" and is not bounded below by 0.
    expect(rSquared([1, 2, 3], [10, 10, 10])).toBeLessThan(0);
  });
});

describe("information criteria", () => {
  // Comparison-only forms; the additive constant common to all models is dropped:
  //   AIC  = n ln(SSR/n) + 2K
  //   BIC  = n ln(SSR/n) + K ln(n)
  //   AICc = AIC + 2K(K+1)/(n-K-1)
  // K = k + 1 throughout: the residual variance is an estimated parameter too.
  const n = 20, k = 4, ssr = 10;

  it("computes AIC from the documented formula", () => {
    expect(computeAIC(n, k, ssr)).toBeCloseTo(n * Math.log(ssr / n) + 2 * (k + 1), 10);
  });

  it("computes BIC from the documented formula", () => {
    expect(computeBIC(n, k, ssr)).toBeCloseTo(n * Math.log(ssr / n) + (k + 1) * Math.log(n), 10);
  });

  it("computes AICc as AIC plus the small-sample correction", () => {
    const K = k + 1;
    const expected = computeAIC(n, k, ssr) + (2 * K * (K + 1)) / (n - K - 1);
    expect(computeAICc(n, k, ssr)).toBeCloseTo(expected, 10);
  });

  it("penalises extra parameters at fixed SSR", () => {
    expect(computeAIC(n, 5, ssr)).toBeGreaterThan(computeAIC(n, 4, ssr));
    expect(computeBIC(n, 5, ssr)).toBeGreaterThan(computeBIC(n, 4, ssr));
    expect(computeAICc(n, 5, ssr)).toBeGreaterThan(computeAICc(n, 4, ssr));
  });

  it("rewards a lower SSR at fixed parameter count", () => {
    expect(computeAIC(n, k, 5)).toBeLessThan(computeAIC(n, k, 20));
  });

  it("returns Infinity from AICc when the sample cannot support the parameters", () => {
    // n - K - 1 <= 0 with K = 5, so n = 6 is already too few for a 4PL.
    expect(computeAICc(6, 4, ssr)).toBe(Infinity);
    expect(computeAICc(5, 4, ssr)).toBe(Infinity);
    expect(Number.isFinite(computeAICc(10, 4, ssr))).toBe(true);
  });

  it("AICc converges to AIC as n grows", () => {
    const big = 100000;
    expect(computeAICc(big, k, ssr) - computeAIC(big, k, ssr)).toBeLessThan(0.01);
  });
});

describe("information criteria — variance-parameter regression", () => {
  // Regression guard. For nonlinear regression K must be the parameter count
  // PLUS ONE, because the residual variance is also estimated (Motulsky,
  // GraphPad Prism regression guide; Burnham & Anderson 2002). Passing K = k
  // is harmless for AIC/BIC -- the offset cancels in a difference -- but not
  // for AICc, whose correction is nonlinear in K. The old code under-penalised
  // the 5PL: at n = 16 the 4PL -> 5PL AICc step was 4.36 instead of 5.33, and
  // at n = 12 it was 6.29 instead of 8.80. The "Auto" selector compares against
  // a fixed dAICc > 2 threshold, so this tipped it toward the 5PL too readily,
  // and most so on the small datasets where the correction matters most.
  it("penalises the 4PL -> 5PL step correctly at small n", () => {
    const n = 16;
    const correct = (k) => {
      const K = k + 1;
      return n * Math.log(10 / n) + 2 * K + (2 * K * (K + 1)) / (n - K - 1);
    };
    const expectedDelta = correct(5) - correct(4);
    const actualDelta = computeAICc(n, 5, 10) - computeAICc(n, 4, 10);
    expect(expectedDelta).toBeCloseTo(5.333, 3);
    expect(actualDelta).toBeCloseTo(expectedDelta, 6);
  });

  it("penalises the extra parameter harder as n shrinks", () => {
    const delta = (n) => computeAICc(n, 5, 10) - computeAICc(n, 4, 10);
    expect(delta(12)).toBeCloseTo(8.8, 1);
    expect(delta(16)).toBeCloseTo(5.333, 2);
    expect(delta(48)).toBeCloseTo(2.62, 2);
    expect(delta(12)).toBeGreaterThan(delta(48));
  });
});

describe("groupByConcentration", () => {
  const xData = [1, 1, 1, 10, 10, 10];
  const yData = [2, 4, 6, 20, 30, 40];

  it("groups replicates by x and sorts ascending", () => {
    const g = groupByConcentration(xData, yData);
    expect(g.map(v => v.x)).toEqual([1, 10]);
    expect(g[0].values).toEqual([2, 4, 6]);
  });

  it("computes the mean and the sample (n-1) standard deviation", () => {
    const g = groupByConcentration(xData, yData);
    expect(g[0].mean).toBeCloseTo(4, 12);
    expect(g[0].sd).toBeCloseTo(2, 12); // sqrt(((2-4)^2+(4-4)^2+(6-4)^2)/2)
  });

  it("computes SEM as sd/sqrt(n)", () => {
    const g = groupByConcentration(xData, yData);
    expect(g[0].sem).toBeCloseTo(2 / Math.sqrt(3), 12);
  });

  it("retains original indices so exclusions can map back to raw rows", () => {
    const g = groupByConcentration(xData, yData);
    expect(g[0].indices).toEqual([0, 1, 2]);
    expect(g[1].indices).toEqual([3, 4, 5]);
  });

  it("reports zero spread for a singleton group rather than NaN", () => {
    const g = groupByConcentration([5], [42]);
    expect(g[0].n).toBe(1);
    expect(g[0].sd).toBe(0);
    expect(g[0].sem).toBe(0);
  });

  it("keeps unsorted input in ascending x order", () => {
    const g = groupByConcentration([100, 1, 10], [1, 2, 3]);
    expect(g.map(v => v.x)).toEqual([1, 10, 100]);
  });
});
