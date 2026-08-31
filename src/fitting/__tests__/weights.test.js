import { describe, it, expect } from "vitest";
import {
  buildWeights, weightedSSR, weightsConverged, estimateVariancePower, WEIGHTING_TYPES,
} from "../weights.js";
import { fitModel } from "../lm.js";
import { model4PL } from "../models.js";
import { make4PLDataConstantCV, make4PLDataGaussian } from "./fixtures.js";

describe("buildWeights", () => {
  const yPred = [10, 20, 40, 80];
  const ctx = { yPred, xData: [1, 2, 3, 4], yData: [10, 20, 40, 80] };

  it("returns no weights for 'none'", () => {
    expect(buildWeights("none", ctx).weights).toBeNull();
    expect(buildWeights(undefined, ctx).weights).toBeNull();
  });

  it("computes 1/Y", () => {
    const { weights } = buildWeights("1/Y", ctx);
    expect(weights).toEqual([1 / 10, 1 / 20, 1 / 40, 1 / 80]);
  });

  it("computes 1/Y^2", () => {
    const { weights } = buildWeights("1/Y^2", ctx);
    expect(weights[0]).toBeCloseTo(1 / 100, 12);
    expect(weights[3]).toBeCloseTo(1 / 6400, 12);
  });

  it("down-weights high responses more steeply under 1/Y^2 than 1/Y", () => {
    const w1 = buildWeights("1/Y", ctx).weights;
    const w2 = buildWeights("1/Y^2", ctx).weights;
    expect(w1[0] / w1[3]).toBeCloseTo(8, 9);   // ratio of Y
    expect(w2[0] / w2[3]).toBeCloseTo(64, 9);  // ratio of Y^2
  });

  it("weights from the PREDICTED values, not the observed ones", () => {
    // Weighting on observed y biases the fit: a point that happened to read
    // low would be rewarded with a larger weight purely for being noisy.
    const { weights } = buildWeights("1/Y^2", {
      yPred: [10, 10, 10, 10],
      xData: ctx.xData,
      yData: [1, 5, 500, 9000], // wildly different observations
    });
    weights.forEach(w => expect(w).toBeCloseTo(1 / 100, 12));
  });

  it("refuses relative weighting when the fitted curve reaches zero", () => {
    const r = buildWeights("1/Y^2", { ...ctx, yPred: [0, 20, 40, 80] });
    expect(r.weights).toBeNull();
    expect(r.warning).toMatch(/strictly positive/);
  });

  it("applies, but cautions, when the baseline is positive yet near zero", () => {
    // Arithmetically fine, statistically strained: relative weighting assumes
    // a constant CV, and a real assay has a noise floor, so absolute error
    // does not shrink to nothing as the signal does. Distinguishing this from
    // the hard refusal above matters -- with a true baseline of zero, whether
    // the fitted value lands just above or below it is a coin flip, and the
    // same assay should not sometimes weight and sometimes refuse.
    const r = buildWeights("1/Y^2", { ...ctx, yPred: [0.05, 20, 40, 80] });
    expect(r.weights).not.toBeNull();
    expect(r.warning).toMatch(/baseline/);
    expect(r.warning).toMatch(/constant CV/);
  });

  it("reports no caution when the baseline is a healthy fraction of the peak", () => {
    const r = buildWeights("1/Y^2", { ...ctx, yPred: [10, 20, 40, 80] });
    expect(r.weights).not.toBeNull();
    expect(r.warning).toBeNull();
  });

  it("refuses relative weighting when the fitted curve goes negative", () => {
    const r = buildWeights("1/Y", { ...ctx, yPred: [-5, 20, 40, 80] });
    expect(r.weights).toBeNull();
    expect(r.warning).toMatch(/strictly positive/);
  });

  it("names normalisation as the likely cause, since that is the usual one", () => {
    const r = buildWeights("1/Y^2", { ...ctx, yPred: [0, 20, 40, 80] });
    expect(r.warning).toMatch(/normalised|background/i);
  });

  it("rejects an unknown weighting type rather than silently ignoring it", () => {
    const r = buildWeights("1/sqrt(Y)", ctx);
    expect(r.weights).toBeNull();
    expect(r.warning).toMatch(/Unknown/);
  });

  it("exposes the supported types", () => {
    expect(WEIGHTING_TYPES).toEqual(["none", "1/Y", "1/Y^2", "1/SD^2"]);
  });
});

describe("buildWeights — 1/SD^2", () => {
  const xData = [1, 1, 1, 10, 10, 10];
  const yData = [10, 12, 14, 50, 60, 70]; // SD 2 and 10
  const ctx = { yPred: yData, xData, yData };

  it("assigns 1/variance from the replicate spread at each dose", () => {
    const { weights } = buildWeights("1/SD^2", ctx);
    expect(weights[0]).toBeCloseTo(1 / 4, 10);   // sd = 2
    expect(weights[3]).toBeCloseTo(1 / 100, 10); // sd = 10
  });

  it("gives every replicate in a group the same weight", () => {
    const { weights } = buildWeights("1/SD^2", ctx);
    expect(weights[0]).toBe(weights[1]);
    expect(weights[1]).toBe(weights[2]);
  });

  it("refuses when a concentration has fewer than 3 replicates", () => {
    // With few replicates the SD is so noisy that the weights add variance
    // rather than removing it.
    const r = buildWeights("1/SD^2", {
      yPred: [1, 2, 3, 4], xData: [1, 1, 10, 10], yData: [10, 12, 50, 60],
    });
    expect(r.weights).toBeNull();
    expect(r.warning).toMatch(/at least 3 replicates/);
  });

  it("refuses when replicates within a group are identical (zero SD)", () => {
    const r = buildWeights("1/SD^2", {
      yPred: [1, 2, 3], xData: [1, 1, 1], yData: [7, 7, 7],
    });
    expect(r.weights).toBeNull();
    expect(r.warning).toMatch(/zero SD/);
  });

  it("is unaffected by the sign or magnitude of the response", () => {
    // Unlike relative weighting, 1/SD^2 works on a zero-crossing response.
    const r = buildWeights("1/SD^2", {
      yPred: [0, 0, 0, 0, 0, 0],
      xData, yData: [-10, -12, -14, 50, 60, 70],
    });
    expect(r.weights).not.toBeNull();
    expect(r.warning).toBeNull();
  });
});

describe("weightedSSR", () => {
  it("reduces to the plain sum of squares without weights", () => {
    expect(weightedSSR([1, 2, 3], null)).toBeCloseTo(14, 12);
  });

  it("scales each squared residual by its weight", () => {
    expect(weightedSSR([1, 2], [4, 0.5])).toBeCloseTo(4 * 1 + 0.5 * 4, 12);
  });
});

describe("weightsConverged", () => {
  it("is true for identical vectors", () => {
    expect(weightsConverged([1, 2, 3], [1, 2, 3])).toBe(true);
  });

  it("is false when any element moves appreciably", () => {
    expect(weightsConverged([1, 2, 3], [1, 2, 3.5])).toBe(false);
  });

  it("compares relatively, not absolutely", () => {
    // Huge weights differing in the 12th significant figure are converged;
    // tiny weights differing by a factor of two are not.
    expect(weightsConverged([1e9], [1e9 * (1 + 1e-12)])).toBe(true);
    expect(weightsConverged([1e-9], [2e-9])).toBe(false);
  });

  it("is false against a null or mismatched vector", () => {
    expect(weightsConverged(null, [1, 2])).toBe(false);
    expect(weightsConverged([1, 2], [1])).toBe(false);
  });
});

describe("fitModel with weighting", () => {
  // Ascending curve with a non-zero floor, so relative weights are defined.
  const truth = [10, 1.2, 1, 1000];

  it("is unweighted by default, preserving existing behaviour", () => {
    const { xData, yData } = make4PLDataConstantCV({ params: truth, seed: 1 });
    const a = fitModel(xData, yData, model4PL, false);
    const b = fitModel(xData, yData, model4PL, false, { weighting: "none" });
    expect(a.weighting.applied).toBe("none");
    expect(a.params[2]).toBeCloseTo(b.params[2], 12);
    expect(a.wssr).toBeNull();
  });

  it("reports which weighting was requested and which was applied", () => {
    const { xData, yData } = make4PLDataConstantCV({ params: truth, nReps: 4, seed: 3 });
    const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });
    expect(fit.weighting.requested).toBe("1/Y^2");
    expect(fit.weighting.applied).toBe("1/Y^2");
    expect(fit.weighting.warning).toBeNull();
    expect(fit.weighting.iterations).toBeGreaterThan(0);
    expect(fit.weighting.weights).toHaveLength(xData.length);
  });

  it("keeps ssr unweighted and reports the weighted objective separately", () => {
    // `ssr` stays on a scale comparable across weighting choices; `wssr` is
    // what was actually minimised.
    const { xData, yData } = make4PLDataConstantCV({ params: truth, nReps: 4, seed: 5 });
    const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });
    const manual = yData.reduce((s, y, i) => s + (y - fit.yPred[i]) ** 2, 0);
    expect(fit.ssr).toBeCloseTo(manual, 6);
    expect(fit.wssr).toBeGreaterThan(0);
    expect(fit.wssr).not.toBeCloseTo(fit.ssr, 3);
    expect(fit.rmse).toBeCloseTo(Math.sqrt(fit.ssr / fit.n), 9);
  });

  it("falls back to unweighted, with a reason, when the curve goes negative", () => {
    // A descending curve whose lower asymptote sits below zero -- the result
    // of background-subtracting past the baseline. Relative weights are
    // undefined there, so the fit must not silently proceed.
    const { xData, yData } = make4PLDataGaussian({ params: [100, 1.2, 1, -20], sd: 2, seed: 7 });
    const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });
    expect(fit.weighting.requested).toBe("1/Y^2");
    expect(fit.weighting.applied).toBe("none");
    expect(fit.weighting.warning).toMatch(/strictly positive/);
    // The fit itself is still perfectly usable, just unweighted.
    expect(fit.ci[2]).not.toBeNull();
  });

  it("never silently declines the weighting that was asked for", () => {
    // The invariant that must hold on every dataset: if the applied weighting
    // differs from the requested one, there is always a reason attached.
    for (let seed = 1; seed <= 30; seed++) {
      const { xData, yData } = make4PLDataConstantCV({
        params: truth, nReps: 4, cv: 0.15, seed,
      });
      const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });
      if (fit.weighting.applied !== fit.weighting.requested) {
        expect(fit.weighting.applied).toBe("none");
        expect(typeof fit.weighting.warning).toBe("string");
        expect(fit.weighting.warning.length).toBeGreaterThan(0);
      }
      expect(fit.ci[2].lo).toBeGreaterThan(0);
    }
  });

  it("flags every zero-baseline fit whose EC50 the weighting degraded", () => {
    // Relative weighting on a curve whose true baseline is zero measurably
    // biases the EC50: a handful of near-zero points acquire thousands of
    // times the weight of the plateau while their noise is absolute rather
    // than relative. Across these seeds the weighted EC50 lands as far off as
    // 0.47 against a truth of 1.0, where unweighted fits recover ~0.99.
    //
    // The engine does not silently refuse -- whether the fitted baseline lands
    // just above or just below zero is noise, and an assay that sometimes
    // weighted and sometimes did not would be worse than one that behaves
    // consistently. The guarantee is narrower and more useful: a fit that
    // weighting actually damaged never comes back unexplained.
    let damaged = 0;
    for (let seed = 1; seed <= 30; seed++) {
      const { xData, yData } = make4PLDataGaussian({ params: [0, 1.2, 1, 100], sd: 2, seed });
      const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });

      if (fit.weighting.applied === "none") {
        expect(fit.weighting.warning).toMatch(/strictly positive/);
        continue;
      }
      const off = Math.abs(Math.log10(fit.params[2]));
      if (off > Math.log10(1.2)) {           // EC50 wrong by more than 20%
        damaged++;
        expect(fit.weighting.warning).toMatch(/baseline/);
      }
      expect(fit.ci[2].lo).toBeGreaterThan(0);
    }
    expect(damaged).toBeGreaterThan(0); // the hazard is real on this data
  });

  it("judges the baseline against the noise, not a fraction of the peak", () => {
    // A baseline of 10 is well resolved when replicates scatter by 1.5 and not
    // resolved at all when they scatter by 20. Both have the same
    // baseline-to-peak ratio, so a peak-relative rule cannot separate them.
    const resolved = make4PLDataConstantCV({
      params: [10, 1.2, 1, 1000], nReps: 4, cv: 0.15, seed: 2,
    });
    const buried = make4PLDataGaussian({
      params: [10, 1.2, 1, 1000], nReps: 4, sd: 20, seed: 2,
    });
    const a = fitModel(resolved.xData, resolved.yData, model4PL, false, { weighting: "1/Y^2" });
    const b = fitModel(buried.xData, buried.yData, model4PL, false, { weighting: "1/Y^2" });
    expect(a.weighting.warning).toBeNull();
    expect(b.weighting.warning).toMatch(/replicate SDs of zero|strictly positive/);
  });

  it("points at the variance structure the data actually show", () => {
    // The engine reports what the replicates say, so a user who picked the
    // wrong scheme can see the mismatch rather than trusting a default.
    const { xData, yData } = make4PLDataGaussian({
      params: truth, nReps: 6, sd: 40, seed: 19,
    });
    const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });
    expect(fit.weighting.variance.recommended).toBe("none");
    expect(fit.weighting.variance.recommended).not.toBe(fit.weighting.requested);
  });

  it("falls back with a reason when 1/SD^2 has too few replicates", () => {
    const { xData, yData } = make4PLDataConstantCV({ params: truth, nReps: 2, seed: 9 });
    const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/SD^2" });
    expect(fit.weighting.applied).toBe("none");
    expect(fit.weighting.warning).toMatch(/at least 3 replicates/);
  });

  it("converges the IRLS loop rather than running to the iteration cap", () => {
    const { xData, yData } = make4PLDataConstantCV({ params: truth, nReps: 4, seed: 11 });
    const fit = fitModel(xData, yData, model4PL, false, {
      weighting: "1/Y^2", maxWeightIterations: 25,
    });
    expect(fit.weighting.iterations).toBeLessThan(25);
  });

  it("still reports standard errors and an EC50 interval", () => {
    const { xData, yData } = make4PLDataConstantCV({ params: truth, nReps: 4, seed: 13 });
    const fit = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });
    expect(fit.se[2]).toBeGreaterThan(0);
    expect(fit.ci[2].lo).toBeGreaterThan(0);
    expect(fit.ci[2].lo).toBeLessThan(fit.params[2]);
    expect(fit.ci[2].hi).toBeGreaterThan(fit.params[2]);
    expect(fit.dof).toBe(xData.length - 4);
  });

  it("shifts the fit toward the low end of the curve", () => {
    // The whole point of relative weighting: stop the high plateau, which
    // carries the largest absolute residuals, from dominating.
    const { xData, yData } = make4PLDataConstantCV({
      params: truth, nReps: 4, cv: 0.2, seed: 17,
    });
    const un = fitModel(xData, yData, model4PL, false);
    const wt = fitModel(xData, yData, model4PL, false, { weighting: "1/Y^2" });

    const lowEnd = xData.map((x, i) => (x < truth[2] ? i : -1)).filter(i => i >= 0);
    const relErr = (fit) => lowEnd.reduce(
      (s, i) => s + Math.abs((yData[i] - fit.yPred[i]) / fit.yPred[i]), 0,
    ) / lowEnd.length;

    expect(relErr(wt)).toBeLessThan(relErr(un));
  });
});

describe("weighting accuracy on heteroscedastic data", () => {
  // The claim from the ligand-binding literature is specific and testable:
  // when the coefficient of variation is constant, 1/Y^2 weighting recovers
  // the potency more accurately than unweighted least squares.
  const truth = [10, 1.2, 1, 1000];
  const TRIALS = 120;

  function meanLogError(weighting) {
    let total = 0, used = 0;
    for (let s = 0; s < TRIALS; s++) {
      const { xData, yData } = make4PLDataConstantCV({
        params: truth, nReps: 4, cv: 0.15, seed: 2000 + s,
      });
      const fit = fitModel(xData, yData, model4PL, false, { weighting });
      if (!fit) continue;
      used++;
      total += Math.abs(Math.log10(fit.params[2]) - Math.log10(truth[2]));
    }
    return total / used;
  }

  it("1/Y^2 recovers the EC50 more accurately than unweighted", () => {
    const unweighted = meanLogError("none");
    const relative = meanLogError("1/Y^2");
    expect(relative).toBeLessThan(unweighted);
  });

  it("orders the schemes by how well they match the variance structure", () => {
    // Variance here is proportional to Y^2, so 1/Y^2 should beat 1/Y, which
    // should in turn beat no weighting at all.
    const none = meanLogError("none");
    const inverseY = meanLogError("1/Y");
    const inverseY2 = meanLogError("1/Y^2");
    expect(inverseY).toBeLessThan(none);
    expect(inverseY2).toBeLessThan(inverseY);
  });

  it("does not help on homoscedastic data, where it is the wrong model", () => {
    // Weighting is a hypothesis about the variance, not a free improvement.
    // With constant noise, relative weighting should not beat unweighted.
    const err = (weighting) => {
      let total = 0;
      for (let s = 0; s < TRIALS; s++) {
        const { xData, yData } = make4PLDataGaussian({
          params: truth, nReps: 4, sd: 60, seed: 3000 + s,
        });
        const fit = fitModel(xData, yData, model4PL, false, { weighting });
        total += Math.abs(Math.log10(fit.params[2]) - Math.log10(truth[2]));
      }
      return total / TRIALS;
    };
    expect(err("1/Y^2")).toBeGreaterThan(err("none") * 0.98);
  });
});


describe("estimateVariancePower", () => {
  // Choosing a weighting scheme is choosing a hypothesis about how variance
  // scales with signal. With replicates that hypothesis is measurable:
  // regress log(SD) on log(mean) and read off the exponent.
  const truth = [10, 1.2, 1, 1000];

  it("recovers theta ~ 1 from constant-CV data and recommends 1/Y^2", () => {
    const { xData, yData } = make4PLDataConstantCV({
      params: truth, nReps: 6, cv: 0.15, seed: 1,
    });
    const r = estimateVariancePower(xData, yData);
    expect(r.theta).toBeGreaterThan(0.75);
    expect(r.theta).toBeLessThan(1.25);
    expect(r.recommended).toBe("1/Y^2");
    expect(r.groups).toBe(8);
  });

  it("recovers theta ~ 0 from constant-SD data and recommends no weighting", () => {
    const { xData, yData } = make4PLDataGaussian({
      params: truth, nReps: 6, sd: 40, seed: 1,
    });
    const r = estimateVariancePower(xData, yData);
    expect(Math.abs(r.theta)).toBeLessThan(0.25);
    expect(r.recommended).toBe("none");
  });

  it("separates the two regimes decisively", () => {
    const cv = make4PLDataConstantCV({ params: truth, nReps: 6, cv: 0.15, seed: 2 });
    const hom = make4PLDataGaussian({ params: truth, nReps: 6, sd: 40, seed: 2 });
    const a = estimateVariancePower(cv.xData, cv.yData);
    const b = estimateVariancePower(hom.xData, hom.yData);
    expect(a.theta - b.theta).toBeGreaterThan(0.5);
  });

  it("reports a standard error, so an imprecise estimate is visible as one", () => {
    const { xData, yData } = make4PLDataConstantCV({
      params: truth, nReps: 6, cv: 0.15, seed: 3,
    });
    const r = estimateVariancePower(xData, yData);
    expect(r.se).toBeGreaterThan(0);
    expect(r.se).toBeLessThan(0.5);
  });

  it("is more precise with more replicates", () => {
    const few = make4PLDataConstantCV({ params: truth, nReps: 3, cv: 0.15, seed: 4 });
    const many = make4PLDataConstantCV({ params: truth, nReps: 15, cv: 0.15, seed: 4 });
    const a = estimateVariancePower(few.xData, few.yData);
    const b = estimateVariancePower(many.xData, many.yData);
    expect(b.se).toBeLessThan(a.se);
  });

  it("declines when there are too few usable concentration groups", () => {
    const r = estimateVariancePower([1, 1, 1, 10, 10, 10], [1, 2, 3, 4, 5, 6]);
    expect(r.theta).toBeNull();
    expect(r.warning).toMatch(/at least 3 concentrations/);
  });

  it("declines when replicates are too thin to estimate an SD", () => {
    const { xData, yData } = make4PLDataConstantCV({ params: truth, nReps: 2, seed: 5 });
    const r = estimateVariancePower(xData, yData);
    expect(r.theta).toBeNull();
    expect(r.warning).toMatch(/3\+ replicates/);
  });

  it("declines without raw data rather than throwing", () => {
    expect(estimateVariancePower(null, null).theta).toBeNull();
  });

  it("is reported on the fit result regardless of which weighting was used", () => {
    const { xData, yData } = make4PLDataConstantCV({
      params: truth, nReps: 6, cv: 0.15, seed: 6,
    });
    const unweighted = fitModel(xData, yData, model4PL, false);
    // Even an unweighted fit tells the caller what the data suggest.
    expect(unweighted.weighting.applied).toBe("none");
    expect(unweighted.weighting.variance.recommended).toBe("1/Y^2");
    expect(unweighted.weighting.variance.theta).toBeGreaterThan(0.75);
  });
});
