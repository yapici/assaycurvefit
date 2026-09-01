import { describe, it, expect } from "vitest";
import { fCDF, fPValue, fInv, tCDF } from "../distributions.js";
import { lackOfFitTest, describeLackOfFit } from "../lackoffit.js";
import { groupByConcentration } from "../stats.js";
import { fitModel, fitConstrainedModel } from "../lm.js";
import { model4PL, model5PL } from "../models.js";
import { make4PLData, make4PLDataGaussian, eval4PLExported } from "./fixtures.js";

describe("F distribution", () => {
  it("is a proper CDF: 0 at the origin, rising to 1", () => {
    expect(fCDF(0, 3, 10)).toBe(0);
    expect(fCDF(-1, 3, 10)).toBe(0);
    expect(fCDF(1e9, 3, 10)).toBeCloseTo(1, 6);
    let prev = 0;
    for (const f of [0.1, 0.5, 1, 2, 5, 20]) {
      const v = fCDF(f, 3, 10);
      expect(v).toBeGreaterThan(prev);
      prev = v;
    }
  });

  it("agrees with the t-distribution, since F(1, df) is t(df) squared", () => {
    // The strongest available cross-check: it validates the F path against an
    // implementation already pinned by its own tests, through a different
    // parameterisation of the same incomplete beta.
    for (const df of [3, 8, 25]) {
      for (const t of [0.4, 1.0, 2.3, 4.1]) {
        const twoTailed = 2 * (1 - tCDF(t, df));
        expect(fPValue(t * t, 1, df)).toBeCloseTo(twoTailed, 10);
      }
    }
  });

  it("puts half the mass either side of 1 when the degrees of freedom match", () => {
    // F(d, d) is reciprocal-symmetric, so its median is exactly 1.
    for (const df of [2, 6, 30]) {
      expect(fCDF(1, df, df)).toBeCloseTo(0.5, 8);
    }
  });

  it("reproduces textbook critical values", () => {
    // Standard 5% F table entries.
    expect(fInv(0.05, 3, 20)).toBeCloseTo(3.098, 3);
    expect(fInv(0.05, 1, 10)).toBeCloseTo(4.965, 3);
    expect(fInv(0.05, 5, 15)).toBeCloseTo(2.901, 3);
    expect(fInv(0.01, 4, 12)).toBeCloseTo(5.412, 3);
  });

  it("round-trips through its own inverse", () => {
    for (const p of [0.001, 0.01, 0.05, 0.2, 0.5]) {
      const f = fInv(p, 4, 17);
      expect(fPValue(f, 4, 17)).toBeCloseTo(p, 9);
    }
  });

  it("sums the two tails to one", () => {
    for (const f of [0.3, 1, 2.5, 9]) {
      expect(fCDF(f, 5, 9) + fPValue(f, 5, 9)).toBeCloseTo(1, 10);
    }
  });
});

describe("lackOfFitTest — decomposition arithmetic", () => {
  const params = [0, 1.2, 10, 100];
  const { xData, yData } = make4PLDataGaussian({ params, nConc: 8, nReps: 4, sd: 3, seed: 7 });
  const yPred = xData.map(x => eval4PLExported(x, params));

  it("splits the residual sum of squares exactly", () => {
    const r = lackOfFitTest(xData, yData, yPred, 4);
    expect(r.applicable).toBe(true);
    const ssr = yData.reduce((s, y, i) => s + (y - yPred[i]) ** 2, 0);
    expect(r.ssPureError + r.ssLackOfFit).toBeCloseTo(ssr, 8);
    expect(r.ssr).toBeCloseTo(ssr, 10);
  });

  it("splits the degrees of freedom exactly", () => {
    const r = lackOfFitTest(xData, yData, yPred, 4);
    const m = groupByConcentration(xData, yData).length;
    expect(r.dfPureError).toBe(xData.length - m);
    expect(r.dfLackOfFit).toBe(m - 4);
    expect(r.dfPureError + r.dfLackOfFit).toBe(xData.length - 4);
  });

  it("reproduces the pooled within-group variance in closed form", () => {
    // With balanced groups the pure-error mean square is exactly the mean of
    // the per-group sample variances. Pins the whole SSPE / (n - m) path.
    const r = lackOfFitTest(xData, yData, yPred, 4);
    const groups = groupByConcentration(xData, yData);
    const pooled = groups.reduce((s, g) => s + g.sd ** 2, 0) / groups.length;
    expect(r.msPureError).toBeCloseTo(pooled, 10);
    expect(r.sdPureError).toBeCloseTo(Math.sqrt(pooled), 10);
  });

  it("computes F as the ratio of the two mean squares", () => {
    const r = lackOfFitTest(xData, yData, yPred, 4);
    expect(r.F).toBeCloseTo(r.msLackOfFit / r.msPureError, 10);
    expect(r.pValue).toBeCloseTo(fPValue(r.F, r.dfLackOfFit, r.dfPureError), 12);
  });

  it("never reports a negative lack-of-fit sum of squares", () => {
    // On a near-perfect fit SSR - SSPE cancels to the noise floor. Summing the
    // group terms keeps it non-negative where the subtraction would not.
    const exact = make4PLData({ params, nConc: 8, nReps: 4, noiseAmp: 0 });
    const perfect = exact.xData.map(x => eval4PLExported(x, params));
    const jittered = exact.yData.map((y, i) => y + (i % 2 ? 1e-9 : -1e-9));
    const r = lackOfFitTest(exact.xData, jittered, perfect, 4);
    expect(r.ssLackOfFit).toBeGreaterThanOrEqual(0);
  });
});

describe("lackOfFitTest — does it actually detect misfit?", () => {
  it("stays quiet when the model is the one that generated the data", () => {
    const params = [0, 1.2, 10, 100];
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 8, nReps: 4, sd: 4, seed: 11 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.lackOfFit.applicable).toBe(true);
    expect(fit.lackOfFit.significant).toBe(false);
    // Under a correct model the two mean squares estimate the same variance,
    // so their ratio should sit near 1 rather than merely below the critical
    // value.
    expect(fit.lackOfFit.F).toBeLessThan(3);
  });

  it("holds its nominal false-positive rate across many seeded datasets", () => {
    // The test is only useful if a correct model passes it about 95% of the
    // time. Anything much higher means it has no power; much lower means it
    // would cry wolf on every good fit.
    const params = [0, 1.2, 10, 100];
    let flagged = 0;
    const trials = 200;
    for (let s = 0; s < trials; s++) {
      const { xData, yData } = make4PLDataGaussian({
        params, nConc: 8, nReps: 3, sd: 4, seed: 1000 + s,
      });
      const fit = fitModel(xData, yData, model4PL, false);
      if (fit.lackOfFit.applicable && fit.lackOfFit.significant) flagged++;
    }
    const rate = flagged / trials;
    expect(rate).toBeGreaterThan(0.01);
    expect(rate).toBeLessThan(0.12);
  });

  it("fires on a systematic deviation that R-squared hides", () => {
    // A precise assay with a real bump at one dose. R² stays excellent because
    // the bump is small against the full response range -- which is exactly
    // the situation the test exists for.
    const params = [0, 1.2, 10, 100];
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 9, nReps: 5, sd: 0.4, seed: 3 });
    const doses = [...new Set(xData)].sort((a, b) => a - b);
    const target = doses[4];
    const bumped = yData.map((y, i) => (xData[i] === target ? y + 6 : y));

    const fit = fitModel(xData, bumped, model4PL, false);
    expect(fit.r2).toBeGreaterThan(0.99);          // looks fine by the usual metric
    expect(fit.lackOfFit.applicable).toBe(true);
    expect(fit.lackOfFit.significant).toBe(true);  // but is not fine
    // And it localises the problem to the dose that was actually perturbed.
    expect(fit.lackOfFit.groups[0].x).toBeCloseTo(target, 12);
  });

  it("fires when an asymmetric curve is fitted with a symmetric model", () => {
    // 5PL truth, 4PL model: the misfit is structural rather than injected.
    const truth = [0, 1.1, 10, 100, 0.35]; // [Bottom, Hill, EC50, Top, S]
    const doses = [];
    for (let i = 0; i < 10; i++) doses.push(Math.pow(10, -1 + (4 * i) / 9));
    const xData = [], yData = [];
    let k = 0;
    for (const x of doses) {
      for (let r = 0; r < 5; r++) {
        xData.push(x);
        // Small deterministic noise; the point is the structural deviation.
        yData.push(model5PL(x, truth) + 0.3 * Math.sin(k++ * 2.399));
      }
    }
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.lackOfFit.applicable).toBe(true);
    expect(fit.lackOfFit.significant).toBe(true);
  });
});

describe("lackOfFitTest — when it cannot be run", () => {
  const params = [0, 1.2, 10, 100];

  it("declines without replicates", () => {
    const { xData, yData } = make4PLData({ params, nConc: 10, nReps: 1 });
    const yPred = xData.map(x => eval4PLExported(x, params));
    const r = lackOfFitTest(xData, yData, yPred, 4);
    expect(r.applicable).toBe(false);
    expect(r.reason).toMatch(/replicate/i);
    expect(r.F).toBeNull();
  });

  it("declines when the model can thread every concentration", () => {
    // 4 parameters, 4 distinct doses: zero lack-of-fit degrees of freedom.
    const { xData, yData } = make4PLData({ params, nConc: 4, nReps: 4, noiseAmp: 2 });
    const yPred = xData.map(x => eval4PLExported(x, params));
    const r = lackOfFitTest(xData, yData, yPred, 4);
    expect(r.applicable).toBe(false);
    expect(r.reason).toMatch(/more concentrations than parameters/i);
  });

  it("declines when replicates are identical, which means they are means", () => {
    const { xData, yData } = make4PLData({ params, nConc: 8, nReps: 3, noiseAmp: 0 });
    const yPred = xData.map(x => eval4PLExported(x, params) + 0.5);
    const r = lackOfFitTest(xData, yData, yPred, 4);
    expect(r.applicable).toBe(false);
    expect(r.reason).toMatch(/identical/i);
  });

  it("declines on missing input rather than throwing", () => {
    expect(lackOfFitTest(null, null, null, 4).applicable).toBe(false);
    expect(lackOfFitTest([], [], [], 4).applicable).toBe(false);
  });
});

describe("lackOfFitTest — weighted fits", () => {
  it("uses the weighted group mean, so the two sums of squares share a metric", () => {
    // Deliberately lopsided weights within one group: the unweighted mean
    // would leave a cross-term and break the SSR = SSPE + SSLOF identity.
    const params = [1, 1.2, 10, 100];
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 7, nReps: 4, sd: 3, seed: 5 });
    const yPred = xData.map(x => eval4PLExported(x, params));
    const weights = xData.map((_, i) => 0.2 + ((i * 7) % 5));

    const r = lackOfFitTest(xData, yData, yPred, 4, { weights });
    const wssr = yData.reduce((s, y, i) => s + weights[i] * (y - yPred[i]) ** 2, 0);
    expect(r.ssPureError + r.ssLackOfFit).toBeCloseTo(wssr, 8);
  });

  it("reduces to the unweighted test when every weight is one", () => {
    const params = [1, 1.2, 10, 100];
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 7, nReps: 4, sd: 3, seed: 5 });
    const yPred = xData.map(x => eval4PLExported(x, params));
    const plain = lackOfFitTest(xData, yData, yPred, 4);
    const unit = lackOfFitTest(xData, yData, yPred, 4, { weights: xData.map(() => 1) });
    expect(unit.F).toBeCloseTo(plain.F, 10);
  });
});

describe("lack-of-fit integration with the fitters", () => {
  const params = [0, 1.2, 10, 100];

  it("rides along on every fitModel result", () => {
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 8, nReps: 3, sd: 3, seed: 2 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.lackOfFit).toBeDefined();
    expect(fit.lackOfFit.applicable).toBe(true);
    expect(fit.lackOfFit.dfLackOfFit).toBe(8 - 4);
  });

  it("counts only the free parameters for a constrained fit", () => {
    // 3PL fixes the Hill slope, so it spends one fewer degree of freedom and
    // gets one more for lack of fit.
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 8, nReps: 3, sd: 3, seed: 2 });
    const fit = fitConstrainedModel(xData, yData, { 1: 1.0 });
    expect(fit.lackOfFit.applicable).toBe(true);
    expect(fit.lackOfFit.dfLackOfFit).toBe(8 - 3);
  });

  it("carries the weighted decomposition through a weighted fit", () => {
    const { xData, yData } = make4PLDataGaussian({
      params: [10, 1.2, 10, 100], nConc: 8, nReps: 3, sd: 3, seed: 4,
    });
    const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });
    expect(fit.weighting.applied).toBe("1/Y^2");
    expect(fit.lackOfFit.applicable).toBe(true);
    // The weighted SSR the fit minimised is what got partitioned.
    expect(fit.lackOfFit.ssr).toBeCloseTo(fit.wssr, 6);
  });
});

describe("describeLackOfFit", () => {
  it("passes the reason through when the test could not run", () => {
    const r = lackOfFitTest([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 4);
    expect(describeLackOfFit(r)).toBe(r.reason);
  });

  it("reports both verdicts with the statistic", () => {
    const params = [0, 1.2, 10, 100];
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 8, nReps: 4, sd: 4, seed: 11 });
    const good = fitModel(xData, yData, model4PL, false);
    expect(describeLackOfFit(good.lackOfFit)).toMatch(/No detectable lack of fit/);

    const doses = [...new Set(xData)].sort((a, b) => a - b);
    const bumped = yData.map((y, i) => (xData[i] === doses[4] ? y + 40 : y));
    const bad = fitModel(xData, bumped, model4PL, false);
    expect(describeLackOfFit(bad.lackOfFit)).toMatch(/deviates from the replicate means/);
  });

  it("returns null for a missing result", () => {
    expect(describeLackOfFit(null)).toBeNull();
  });
});
