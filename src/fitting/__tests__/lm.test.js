import { describe, it, expect } from "vitest";
import {
  residuals, sumSquaredResiduals, jacobian, levenbergMarquardt,
  estimateInitialParams, fitModel, fitConstrainedModel,
} from "../lm.js";
import { model4PL, model5PL } from "../models.js";
import { make4PLData, grad4PL, SCALES } from "./fixtures.js";

describe("residuals / sumSquaredResiduals", () => {
  const p = [0, 1, 1, 100];

  it("returns observed minus predicted, in input order", () => {
    const r = residuals([1], [60], model4PL, p);
    expect(r[0]).toBeCloseTo(10, 9); // model gives 50 at the EC50
  });

  it("is zero for data lying exactly on the curve", () => {
    const xs = [0.1, 1, 10];
    const ys = xs.map(x => model4PL(x, p));
    residuals(xs, ys, model4PL, p).forEach(r => expect(Math.abs(r)).toBeLessThan(1e-12));
  });

  it("sums the squares of the residuals", () => {
    const xs = [0.1, 1, 10];
    const ys = xs.map(x => model4PL(x, p) + 2);
    expect(sumSquaredResiduals(xs, ys, model4PL, p)).toBeCloseTo(12, 9); // 3 * 2^2
  });
});

describe("jacobian", () => {
  it("matches the analytic 4PL gradient at unit scale", () => {
    const p = [0, 1.2, 1, 100];
    const xs = [0.1, 0.5, 1, 2, 10];
    const J = jacobian(xs, model4PL, p);
    xs.forEach((x, i) => {
      grad4PL(x, p).forEach((analytic, j) => {
        expect(J[i][j]).toBeCloseTo(analytic, 4);
      });
    });
  });

  it("has one row per observation and one column per parameter", () => {
    const J = jacobian([1, 2, 3], model4PL, [0, 1, 1, 100]);
    expect(J).toHaveLength(3);
    expect(J[0]).toHaveLength(4);
  });

  it("gives a zero gradient for the Hill slope exactly at the EC50", () => {
    // At x = C, ln(x/C) = 0, so dy/dB vanishes.
    const J = jacobian([1], model4PL, [0, 1.2, 1, 100]);
    expect(Math.abs(J[0][1])).toBeLessThan(1e-6);
  });
});

describe("jacobian — known defect (fixed in Phase 1)", () => {
  // The finite-difference step is an ABSOLUTE 1e-8 applied to every parameter.
  // Concentrations are stored in linear molar units, so a nanomolar EC50 is a
  // number like 1e-9 and the "small" perturbation is a 10x change. The
  // resulting column is not a derivative; it can even carry the wrong sign.
  it.fails("is accurate for the EC50 column at nanomolar scale", () => {
    const p = [0, 1.2, 1e-9, 100];
    const xs = [p[2] / 10, p[2], p[2] * 10];
    const J = jacobian(xs, model4PL, p);
    xs.forEach((x, i) => {
      const analytic = grad4PL(x, p)[2];
      expect(Math.abs((J[i][2] - analytic) / analytic)).toBeLessThan(1e-3);
    });
  });

  it("is accurate at micromolar scale and above (documents the boundary)", () => {
    for (const C of [1e-6, 1, 1e3]) {
      const p = [0, 1.2, C, 100];
      const xs = [C / 10, C, C * 10];
      const J = jacobian(xs, model4PL, p);
      xs.forEach((x, i) => {
        const analytic = grad4PL(x, p)[2];
        expect(Math.abs((J[i][2] - analytic) / analytic)).toBeLessThan(1e-3);
      });
    }
  });
});

describe("estimateInitialParams", () => {
  it("brackets the plateaus from the extreme 20% of the data", () => {
    const truth = [0, 1, 1, 100];
    const { xData, yData } = make4PLData({ params: truth });
    const init = estimateInitialParams(xData, yData, false);
    expect(init[0]).toBeLessThan(30);    // A near the low plateau
    expect(init[3]).toBeGreaterThan(70); // D near the high plateau
  });

  it("places the initial EC50 inside the tested concentration range", () => {
    const truth = [0, 1, 1, 100];
    const { xData, yData } = make4PLData({ params: truth });
    const init = estimateInitialParams(xData, yData, false);
    expect(init[2]).toBeGreaterThan(Math.min(...xData));
    expect(init[2]).toBeLessThan(Math.max(...xData));
  });

  it("returns 5 params with S = 1 for the 5PL", () => {
    const { xData, yData } = make4PLData({ params: [0, 1, 1, 100] });
    const init = estimateInitialParams(xData, yData, true);
    expect(init).toHaveLength(5);
    expect(init[4]).toBe(1);
  });

  it("always returns a strictly positive EC50 estimate", () => {
    const { xData, yData } = make4PLData({ params: [0, 1, 1e-7, 100] });
    expect(estimateInitialParams(xData, yData, false)[2]).toBeGreaterThan(0);
  });
});

describe("levenbergMarquardt", () => {
  it("converges to the truth from a perturbed start on noiseless data", () => {
    const truth = [0, 1.2, 1, 100];
    const { xData, yData } = make4PLData({ params: truth, nReps: 1 });
    const r = levenbergMarquardt(xData, yData, model4PL, [5, 1, 0.5, 90]);
    expect(r.converged).toBe(true);
    expect(r.params[2]).toBeCloseTo(1, 4);
    expect(r.ssr).toBeLessThan(1e-8);
  });

  it("never increases SSR relative to the starting point", () => {
    const truth = [0, 1.2, 1, 100];
    const { xData, yData } = make4PLData({ params: truth, noiseAmp: 3 });
    const start = [5, 1, 0.5, 90];
    const startSSR = sumSquaredResiduals(xData, yData, model4PL, start);
    expect(levenbergMarquardt(xData, yData, model4PL, start).ssr)
      .toBeLessThanOrEqual(startSSR);
  });

  it("respects the maxIter budget", () => {
    const { xData, yData } = make4PLData({ params: [0, 1.2, 1, 100], noiseAmp: 3 });
    const r = levenbergMarquardt(xData, yData, model4PL, [5, 1, 0.5, 90], { maxIter: 1 });
    expect(r.converged).toBe(false);
  });

  it("keeps the EC50 parameter positive", () => {
    const { xData, yData } = make4PLData({ params: [0, 1.2, 1, 100] });
    const r = levenbergMarquardt(xData, yData, model4PL, [5, 1, 0.5, 90]);
    expect(r.params[2]).toBeGreaterThan(0);
  });
});

describe("fitModel", () => {
  it("recovers 4PL parameters from noiseless data", () => {
    const truth = [0, 1.5, 2.5, 100];
    const { xData, yData } = make4PLData({ params: truth, nReps: 1 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.params[2]).toBeCloseTo(2.5, 4); // EC50
    expect(Math.abs(fit.params[1])).toBeCloseTo(1.5, 3); // |Hill|
    expect(fit.r2).toBeGreaterThan(0.9999);
  });

  it("recovers the EC50 within tolerance under noise", () => {
    const truth = [0, 1.2, 1, 100];
    const { xData, yData } = make4PLData({ params: truth, noiseAmp: 2 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(Math.abs(Math.log10(fit.params[2]) - Math.log10(1))).toBeLessThan(0.1);
  });

  it("reports n, k and the derived goodness-of-fit fields", () => {
    const { xData, yData } = make4PLData({ params: [0, 1.2, 1, 100], nConc: 8, nReps: 3 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.n).toBe(24);
    expect(fit.k).toBe(4);
    expect(fit.rmse).toBeCloseTo(Math.sqrt(fit.ssr / fit.n), 12);
    expect(fit.yPred).toHaveLength(24);
    expect(Number.isFinite(fit.aic)).toBe(true);
    expect(Number.isFinite(fit.bic)).toBe(true);
  });

  it("fits a 5PL with k = 5 and a biological EC50", () => {
    const { xData, yData } = make4PLData({ params: [0, 1.2, 1, 100], noiseAmp: 1 });
    const fit = fitModel(xData, yData, model5PL, true);
    expect(fit.k).toBe(5);
    expect(fit.bioEC50).toBeGreaterThan(0);
  });

  it("leaves bioEC50 null for the 4PL", () => {
    const { xData, yData } = make4PLData({ params: [0, 1.2, 1, 100] });
    expect(fitModel(xData, yData, model4PL, false).bioEC50).toBeNull();
  });

  it("fits a descending curve as readily as an ascending one", () => {
    const truth = [100, 1.2, 1, 0]; // A > D: response falls with dose
    const { xData, yData } = make4PLData({ params: truth, nReps: 1 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.params[2]).toBeCloseTo(1, 3);
    expect(fit.r2).toBeGreaterThan(0.9999);
  });
});

describe("fitConstrainedModel", () => {
  const truth = [0, 1, 1, 100];

  it("holds the Hill slope at 1 for a 3PL", () => {
    const { xData, yData } = make4PLData({ params: truth, noiseAmp: 1 });
    const fit = fitConstrainedModel(xData, yData, { 1: 1.0 });
    expect(fit.params[1]).toBe(1.0);
    expect(fit.k).toBe(3);
  });

  it("holds both plateaus for a 2PL", () => {
    const { xData, yData } = make4PLData({ params: truth, noiseAmp: 1 });
    const fit = fitConstrainedModel(xData, yData, { 0: 0, 3: 100 });
    expect(fit.params[0]).toBe(0);
    expect(fit.params[3]).toBe(100);
    expect(fit.k).toBe(2);
  });

  it("fits the EC50 alone for a 1PL", () => {
    const { xData, yData } = make4PLData({ params: truth, noiseAmp: 1 });
    const fit = fitConstrainedModel(xData, yData, { 0: 0, 1: 1, 3: 100 });
    expect(fit.k).toBe(1);
    expect(fit.params[2]).toBeCloseTo(1, 1);
  });

  it("returns a full 4-element param vector regardless of how many are free", () => {
    const { xData, yData } = make4PLData({ params: truth });
    expect(fitConstrainedModel(xData, yData, { 1: 1.0 }).params).toHaveLength(4);
  });

  it("cannot beat the unconstrained 4PL on SSR", () => {
    const { xData, yData } = make4PLData({ params: [0, 2.2, 1, 100], noiseAmp: 1 });
    const free = fitModel(xData, yData, model4PL, false);
    const fixed = fitConstrainedModel(xData, yData, { 1: 1.0 }); // wrong Hill
    expect(fixed.ssr).toBeGreaterThan(free.ssr);
  });
});

describe("scale invariance", () => {
  // The same curve expressed in different concentration units must produce the
  // same fit: identical Hill slope and plateaus, with the EC50 tracking the
  // unit change exactly. This is the property the absolute Jacobian step
  // breaks, and the clearest end-to-end statement of the defect.
  const truth = (s) => [0, 1.2, 1.0 * s, 100];

  function fitAtScale(s) {
    const { xData, yData } = make4PLData({ params: truth(s), noiseAmp: 2 });
    return fitModel(xData, yData, model4PL, false);
  }

  it("fits micromolar data identically to unit-scale data", () => {
    const a = fitAtScale(SCALES.unit);
    const b = fitAtScale(SCALES.micromolar);
    expect(b.params[2] / SCALES.micromolar).toBeCloseTo(a.params[2], 6);
    expect(b.params[1]).toBeCloseTo(a.params[1], 6);
    expect(b.r2).toBeCloseTo(a.r2, 9);
    expect(a.converged).toBe(true);
    expect(b.converged).toBe(true);
  });

  it.fails("fits nanomolar data identically to unit-scale data", () => {
    const a = fitAtScale(SCALES.unit);
    const b = fitAtScale(SCALES.nanomolar);
    expect(b.converged).toBe(true);
    expect(b.params[2] / SCALES.nanomolar).toBeCloseTo(a.params[2], 6);
    expect(b.params[1]).toBeCloseTo(a.params[1], 6);
  });

  it.fails("recovers a nanomolar EC50 to the same accuracy as a unit-scale one", () => {
    const fit = fitAtScale(SCALES.nanomolar);
    const err = Math.abs(Math.log10(fit.params[2]) - Math.log10(SCALES.nanomolar));
    expect(err).toBeLessThan(0.05); // ~0.116 today, vs ~0.023 at unit scale
  });
});
