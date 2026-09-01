import { describe, it, expect } from "vitest";
import {
  profileInterval, profileIntervals, bootstrapIntervals, fitModelWithIntervals,
} from "../resample.js";
import {
  fitModel, levenbergMarquardt, withLogParams, jacobian, sumSquaredResiduals,
} from "../lm.js";
import { parameterCovariance, parameterIntervals } from "../inference.js";
import { model4PL } from "../models.js";
import { tInv } from "../distributions.js";
import { make4PLDataGaussian, eval4PLExported } from "./fixtures.js";

/** Dataset over an explicit log10 dose window, so a plateau can be cut off deliberately. */
function dataOverWindow({ params, logLo, logHi, nConc = 8, nReps = 3, sd = 2, seed = 1 }) {
  const xData = [], yData = [];
  let k = seed;
  for (let i = 0; i < nConc; i++) {
    const x = Math.pow(10, logLo + ((logHi - logLo) * i) / (nConc - 1));
    for (let r = 0; r < nReps; r++) {
      xData.push(x);
      yData.push(eval4PLExported(x, params) + sd * Math.sin(k++ * 12.9898));
    }
  }
  return { xData, yData };
}

describe("profileInterval — validated against the linear case", () => {
  // For a model LINEAR in its parameters the likelihood really is a parabola,
  // so the profile interval and the Wald interval must agree exactly. This is
  // the strongest available check that the threshold, the search and the
  // bisection are all right: any error in any of them breaks the identity.
  const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const y = [2.3, 3.8, 6.4, 7.9, 10.2, 12.1, 13.6, 16.3, 18.1, 19.7];
  const linear = (xi, [a, b]) => a + b * xi;

  const fit = levenbergMarquardt(x, y, linear, [0, 1]);
  const J = x.map(v => [1, v]);
  const { cov, dof } = parameterCovariance(J, fit.ssr);
  const wald = parameterIntervals(fit.params, cov, dof);

  it("reproduces the Wald interval for both parameters", () => {
    for (let i = 0; i < 2; i++) {
      const prof = profileInterval({
        xData: x, yData: y, modelFn: linear, params: fit.params,
        index: i, ssr: fit.ssr, scale: wald.se[i], tolerance: 1e-10,
      });
      expect(prof.loBounded).toBe(true);
      expect(prof.hiBounded).toBe(true);
      expect(prof.lo).toBeCloseTo(wald.ci[i].lo, 6);
      expect(prof.hi).toBeCloseTo(wald.ci[i].hi, 6);
    }
  });

  it("is symmetric about the estimate, as a parabola must be", () => {
    const prof = profileInterval({
      xData: x, yData: y, modelFn: linear, params: fit.params,
      index: 1, ssr: fit.ssr, scale: wald.se[1], tolerance: 1e-10,
    });
    expect(fit.params[1] - prof.lo).toBeCloseTo(prof.hi - fit.params[1], 8);
  });

  it("uses the documented threshold, SSR_min (1 + t^2/(n-p))", () => {
    const t = tInv(0.025, dof);
    const prof = profileInterval({
      xData: x, yData: y, modelFn: linear, params: fit.params,
      index: 0, ssr: fit.ssr, scale: wald.se[0],
    });
    expect(prof.threshold).toBeCloseTo(fit.ssr * (1 + (t * t) / dof), 12);
  });

  it("lands on the threshold: refitting at the endpoint reproduces it", () => {
    const prof = profileInterval({
      xData: x, yData: y, modelFn: linear, params: fit.params,
      index: 1, ssr: fit.ssr, scale: wald.se[1], tolerance: 1e-10,
    });
    // Re-optimise the intercept with the slope pinned at the endpoint.
    const pinned = (xi, [a]) => a + prof.hi * xi;
    const r = levenbergMarquardt(x, y, pinned, [fit.params[0]]);
    expect(r.ssr / prof.threshold).toBeCloseTo(1, 5);
  });
});

describe("profileInterval — where it departs from Wald", () => {
  const params = [0, 1.2, 10, 100];

  it("agrees with Wald when the curve is fully bracketed", () => {
    // Well-conditioned data: the parabola is a good local description, so the
    // expensive interval should buy nothing. Anything else would mean the
    // profile search is biased.
    const { xData, yData } = dataOverWindow({ params, logLo: -2, logHi: 4, nConc: 10, sd: 2 });
    const fit = fitModelWithIntervals(xData, yData, model4PL, false, { intervals: "profile" });
    for (const i of [1, 3]) {
      const w = fit.ci[i], p = fit.profile.ci[i];
      expect(p.lo).toBeCloseTo(w.lo, 1);
      expect(p.hi).toBeCloseTo(w.hi, 1);
    }
  });

  it("departs from Wald further as the plateau is cut off harder", () => {
    // The claim the module rests on, tested as a trend rather than at one
    // point: while the curve is bracketed the parabola is a fair description
    // and the two intervals agree, and the agreement degrades monotonically as
    // the bottom plateau is pushed out of the dose range.
    const asymmetry = [];
    const widthRatio = [];
    for (const logLo of [0.3, 0.5, 0.7, 1.0]) {
      const { xData, yData } = dataOverWindow({ params, logLo, logHi: 4, nConc: 8, sd: 1 });
      const fit = fitModelWithIntervals(xData, yData, model4PL, false, { intervals: "profile" });
      const slot = fit.identifiability.geometry.zeroDoseSlot;
      const p = fit.profile.ci[slot];
      const b = fit.profile.bounded[slot];
      expect(b.lo && b.hi).toBe(true);
      asymmetry.push((fit.params[slot] - p.lo) / (p.hi - fit.params[slot]));
      widthRatio.push((p.hi - p.lo) / (fit.ci[slot].hi - fit.ci[slot].lo));
    }

    // Well bracketed: the expensive interval buys essentially nothing.
    expect(widthRatio[0]).toBeCloseTo(1, 1);
    expect(asymmetry[0]).toBeCloseTo(1, 0);

    // Badly bracketed: Wald is materially over-optimistic and mis-centred.
    expect(widthRatio.at(-1)).toBeGreaterThan(1.4);
    expect(asymmetry.at(-1)).toBeGreaterThan(3);

    for (let i = 1; i < asymmetry.length; i++) {
      expect(asymmetry[i]).toBeGreaterThan(asymmetry[i - 1]);
      expect(widthRatio[i]).toBeGreaterThanOrEqual(widthRatio[i - 1] - 1e-9);
    }
  });

  it("puts the extra width on the side the likelihood is flat", () => {
    // A plateau below the tested range can run a long way down and barely
    // change the curve, but cannot rise far without visibly missing the data.
    // Wald, being symmetric by construction, cannot express that at all.
    const { xData, yData } = dataOverWindow({ params, logLo: 1, logHi: 4, nConc: 8, sd: 1 });
    const fit = fitModelWithIntervals(xData, yData, model4PL, false, { intervals: "profile" });
    const slot = fit.identifiability.geometry.zeroDoseSlot;
    const est = fit.params[slot];
    const below = est - fit.profile.ci[slot].lo;
    const above = fit.profile.ci[slot].hi - est;
    expect(below / above).toBeGreaterThan(3);
    expect(est - fit.ci[slot].lo).toBeCloseTo(fit.ci[slot].hi - est, 6);
  });

  it("reports an endpoint as unbounded rather than inventing one", () => {
    // Doses entirely above the EC50: the data bound the bottom plateau from
    // above but say nothing about how far below it could lie.
    const { xData, yData } = dataOverWindow({ params, logLo: 2, logHi: 5, nConc: 6, sd: 0.5 });
    const fit = fitModel(xData, yData, model4PL, false);
    const log = withLogParams(model4PL, [2]);
    const prof = profileInterval({
      xData, yData, modelFn: log.modelFn, params: log.toLog(fit.params),
      index: 0, ssr: fit.ssr, scale: fit.se[0],
    });
    expect(prof.loBounded).toBe(false);
    expect(prof.lo).toBeNull();
    expect(prof.hiBounded).toBe(true);
    expect(prof.hi).toBeGreaterThan(fit.params[0]);
  });

  it("gives a strictly positive EC50 interval, asymmetric on the linear scale", () => {
    const { xData, yData } = dataOverWindow({ params, logLo: 1, logHi: 4, nConc: 7, sd: 3 });
    const fit = fitModelWithIntervals(xData, yData, model4PL, false, { intervals: "profile" });
    const ci = fit.profile.ci[2];
    expect(ci.lo).toBeGreaterThan(0);
    expect(ci.lo).toBeLessThan(fit.params[2]);
    expect(ci.hi).toBeGreaterThan(fit.params[2]);
  });

  it("back-transforms the EC50 from the log space it was profiled in", () => {
    const { xData, yData } = dataOverWindow({ params, logLo: -1, logHi: 3, nConc: 8, sd: 2 });
    const fit = fitModelWithIntervals(xData, yData, model4PL, false, { intervals: "profile" });

    const log = withLogParams(model4PL, [2]);
    const direct = profileInterval({
      xData, yData, modelFn: log.modelFn, params: log.toLog(fit.params),
      index: 2, ssr: fit.ssr, scale: fit.logEC50.se,
    });
    expect(fit.profile.ci[2].lo).toBeCloseTo(Math.pow(10, direct.lo), 8);
    expect(fit.profile.ci[2].hi).toBeCloseTo(Math.pow(10, direct.hi), 8);
  });
});

describe("profileIntervals", () => {
  it("returns one interval per parameter", () => {
    const params = [0, 1.2, 10, 100];
    const { xData, yData } = dataOverWindow({ params, logLo: -2, logHi: 4, nConc: 9 });
    const fit = fitModel(xData, yData, model4PL, false);
    const log = withLogParams(model4PL, [2]);
    const all = profileIntervals({
      xData, yData, modelFn: log.modelFn, params: log.toLog(fit.params),
      ssr: fit.ssr, se: fit.se,
    });
    expect(all).toHaveLength(4);
    all.forEach(r => expect(r).toHaveProperty("threshold"));
  });

  it("declines when there are no residual degrees of freedom", () => {
    const x = [1, 2], y = [1, 2];
    const linear = (xi, [a, b]) => a + b * xi;
    const r = profileInterval({ xData: x, yData: y, modelFn: linear, params: [0, 1], index: 0, ssr: 0 });
    expect(r.lo).toBeNull();
    expect(r.threshold).toBeNull();
  });
});

describe("bootstrapIntervals", () => {
  const params = [0, 1.2, 10, 100];
  const { xData, yData } = make4PLDataGaussian({ params, nConc: 9, nReps: 3, decades: 5, sd: 3, seed: 21 });
  const fit = fitModel(xData, yData, model4PL, false);
  const log = withLogParams(model4PL, [2]);
  const fitSpace = log.toLog(fit.params);
  const base = { xData, yData, modelFn: log.modelFn, params: fitSpace };

  it("is reproducible: the same seed gives the same interval", () => {
    const a = bootstrapIntervals({ ...base, nBoot: 120, seed: 7 });
    const b = bootstrapIntervals({ ...base, nBoot: 120, seed: 7 });
    expect(a.ci).toEqual(b.ci);
    expect(a.se).toEqual(b.se);
  });

  it("actually depends on the seed", () => {
    const a = bootstrapIntervals({ ...base, nBoot: 120, seed: 7 });
    const b = bootstrapIntervals({ ...base, nBoot: 120, seed: 8 });
    expect(a.ci[1].lo).not.toBeCloseTo(b.ci[1].lo, 10);
  });

  it("brackets the estimate", () => {
    const r = bootstrapIntervals({ ...base, nBoot: 300, seed: 3 });
    r.ci.forEach((c, i) => {
      expect(c.lo).toBeLessThan(fitSpace[i]);
      expect(c.hi).toBeGreaterThan(fitSpace[i]);
    });
  });

  it("agrees with the Wald standard error on well-conditioned data", () => {
    // The residual inflation by sqrt(n/(n-p)) is what makes this hold; without
    // it the bootstrap spread is systematically narrow.
    const r = bootstrapIntervals({ ...base, nBoot: 600, seed: 5 });
    const waldSe = [fit.se[0], fit.se[1], fit.logEC50.se, fit.se[3]];
    r.se.forEach((s, i) => {
      expect(s / waldSe[i]).toBeGreaterThan(0.7);
      expect(s / waldSe[i]).toBeLessThan(1.4);
    });
  });

  it("reports negligible bias when the estimator is well behaved", () => {
    const r = bootstrapIntervals({ ...base, nBoot: 600, seed: 5 });
    r.bias.forEach((b, i) => expect(Math.abs(b)).toBeLessThan(0.5 * r.se[i]));
  });

  it("stratifies within concentration groups when asked", () => {
    const r = bootstrapIntervals({ ...base, nBoot: 120, seed: 3, method: "stratified" });
    expect(r.method).toBe("stratified");
    expect(r.warning).toBeNull();
  });

  it("falls back with a warning when stratifying is impossible", () => {
    const single = make4PLDataGaussian({ params, nConc: 12, nReps: 1, decades: 5, sd: 3, seed: 4 });
    const f = fitModel(single.xData, single.yData, model4PL, false);
    const r = bootstrapIntervals({
      xData: single.xData, yData: single.yData, modelFn: log.modelFn,
      params: log.toLog(f.params), nBoot: 80, seed: 3, method: "stratified",
    });
    expect(r.method).toBe("residual");
    expect(r.warning).toMatch(/needs replicates/i);
  });

  it("declines when there are more parameters than observations", () => {
    const r = bootstrapIntervals({
      xData: [1, 2, 3], yData: [1, 2, 3], modelFn: model4PL, params: [0, 1, 2, 3], nBoot: 10,
    });
    expect(r.ci).toEqual([null, null, null, null]);
    expect(r.warning).toMatch(/more observations than parameters/i);
  });

  it("uses the type-7 quantile convention", () => {
    // Pin the percentile definition, so an interval computed today matches one
    // computed by R or NumPy on the same draws.
    const r = bootstrapIntervals({ ...base, nBoot: 200, seed: 11 });
    const col = r.draws.map(d => d[1]).sort((a, b) => a - b);
    const h = (col.length - 1) * 0.025;
    const lo = Math.floor(h), hi = Math.ceil(h);
    expect(r.ci[1].lo).toBeCloseTo(col[lo] + (h - lo) * (col[hi] - col[lo]), 12);
  });
});

describe("bootstrap coverage", () => {
  it("covers the true Hill slope at close to the nominal 95%", () => {
    // The check that matters for an interval: over many datasets, does it
    // contain the truth as often as it claims?
    const params = [0, 1.2, 10, 100];
    const log = withLogParams(model4PL, [2]);
    let covered = 0;
    const trials = 60;
    for (let s = 0; s < trials; s++) {
      const { xData, yData } = make4PLDataGaussian({
        params, nConc: 8, nReps: 3, decades: 5, sd: 4, seed: 500 + s,
      });
      const fit = fitModel(xData, yData, model4PL, false);
      const r = bootstrapIntervals({
        xData, yData, modelFn: log.modelFn, params: log.toLog(fit.params),
        nBoot: 150, seed: 90000 + s,
      });
      if (r.ci[1] && r.ci[1].lo <= params[1] && params[1] <= r.ci[1].hi) covered++;
    }
    expect(covered / trials).toBeGreaterThan(0.85);
    expect(covered / trials).toBeLessThanOrEqual(1);
  });
});

describe("fitModelWithIntervals", () => {
  const params = [0, 1.2, 10, 100];
  const { xData, yData } = make4PLDataGaussian({ params, nConc: 9, nReps: 3, decades: 5, sd: 3, seed: 13 });

  it("returns the ordinary fit result untouched", () => {
    const plain = fitModel(xData, yData, model4PL, false);
    const rich = fitModelWithIntervals(xData, yData, model4PL, false, { intervals: "profile" });
    expect(rich.params).toEqual(plain.params);
    expect(rich.r2).toBe(plain.r2);
    expect(rich.ci).toEqual(plain.ci);
    expect(rich.lackOfFit.F).toBeCloseTo(plain.lackOfFit.F, 12);
  });

  it("skips the work when asked for no intervals", () => {
    const r = fitModelWithIntervals(xData, yData, model4PL, false, { intervals: "none" });
    expect(r.profile).toBeUndefined();
    expect(r.bootstrap).toBeUndefined();
  });

  it("computes both when asked for both", () => {
    const r = fitModelWithIntervals(xData, yData, model4PL, false, {
      intervals: "both", bootstrapOptions: { nBoot: 100, seed: 2 },
    });
    expect(r.profile.ci).toHaveLength(4);
    expect(r.bootstrap.ci).toHaveLength(4);
    expect(r.bootstrap.nSuccess).toBeGreaterThan(50);
    expect(r.profile.evaluations).toBeGreaterThan(0);
  });

  it("carries the boundedness flags through to the caller", () => {
    const r = fitModelWithIntervals(xData, yData, model4PL, false, { intervals: "profile" });
    expect(r.profile.bounded).toHaveLength(4);
    r.profile.bounded.forEach(b => {
      expect(typeof b.lo).toBe("boolean");
      expect(typeof b.hi).toBe("boolean");
    });
  });

  it("passes weighting options through and conditions on the fitted weights", () => {
    const shifted = make4PLDataGaussian({
      params: [10, 1.2, 10, 100], nConc: 9, nReps: 3, decades: 5, sd: 3, seed: 6,
    });
    const r = fitModelWithIntervals(shifted.xData, shifted.yData, model4PL, false, {
      intervals: "profile", weighting: "1/Y^2",
    });
    expect(r.weighting.applied).toBe("1/Y^2");
    expect(r.profile.ci[2].lo).toBeGreaterThan(0);
    // Profiled against the weighted objective, which is what the fit minimised.
    expect(r.profile.threshold).toBeGreaterThan(r.wssr);
  });

  it("works for the 5PL, whose asymmetry exponent must stay positive", () => {
    const r = fitModelWithIntervals(xData, yData, (x, p) => {
      const [Bottom, Hill, EC50, Top, S] = p;
      if (x <= 0) return Hill > 0 ? Bottom : Top;
      return Bottom + (Top - Bottom) / Math.pow(1 + Math.pow(EC50 / x, Hill), S);
    }, true, { intervals: "profile" });
    expect(r.profile.ci).toHaveLength(5);
    expect(r.params[4]).toBeGreaterThan(0);
  });
});
