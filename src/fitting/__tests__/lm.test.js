import { describe, it, expect } from "vitest";
import {
  residuals, sumSquaredResiduals, jacobian, levenbergMarquardt,
  estimateInitialParams, fitModel, fitConstrainedModel,
} from "../lm.js";
import { model4PL, model5PL } from "../models.js";
import { make4PLData, make4PLDataGaussian, grad4PL, SCALES } from "./fixtures.js";

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

describe("jacobian — step-size regression", () => {
  // Regression guard. The step used to be an ABSOLUTE 1e-8 applied to every
  // parameter. Concentrations are stored in linear molar units, so a
  // nanomolar EC50 is a number like 1e-9 and the "small" perturbation was a
  // 10x change: the column was not a derivative and could carry the wrong
  // sign. The step is now relative to each parameter's own magnitude.
  it("is accurate for the EC50 column at nanomolar scale", () => {
    const p = [0, 1.2, 1e-9, 100];
    const xs = [p[2] / 10, p[2], p[2] * 10];
    const J = jacobian(xs, model4PL, p);
    xs.forEach((x, i) => {
      const analytic = grad4PL(x, p)[2];
      expect(Math.abs((J[i][2] - analytic) / analytic)).toBeLessThan(1e-3);
    });
  });

  it("is accurate across nine decades of EC50", () => {
    for (const C of [1e-12, 1e-9, 1e-6, 1, 1e3]) {
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

  it("reports convergence for an exact fit that hits the precision floor", () => {
    // The SSR criterion only fires on an *improving* step. A fit that reaches
    // machine precision produces no further improvement, so convergence has to
    // come from the lambda ceiling instead. Before that existed, a numerically
    // perfect fit was reported as a failure.
    const truth = [0, 1.2, 1, 100];
    const { xData, yData } = make4PLData({ params: truth, nReps: 1 });
    const r = levenbergMarquardt(xData, yData, model4PL, [5, 1, 0.5, 90]);
    expect(r.ssr).toBeLessThan(1e-15);
    expect(r.converged).toBe(true);
  });

  it("does NOT claim convergence when no downhill step is ever found", () => {
    // Degenerate data: every y identical, started somewhere the optimiser
    // cannot improve on. Reaching the lambda ceiling without a single accepted
    // step is a genuine failure and must still report converged: false.
    const xData = [1, 2, 3, 4];
    const yData = [5, 5, 5, 5];
    const r = levenbergMarquardt(xData, yData, model4PL, [5, 1, 1, 5]);
    expect(r.converged).toBe(false);
  });

  it("honours a custom constrain callback on every proposed step", () => {
    // constrain filters PROPOSED steps; the caller's initial vector is taken
    // as given. Start below the ceiling so the optimiser genuinely tries to
    // push the upper asymptote past it, and confirm it is held back.
    const { xData, yData } = make4PLData({ params: [0, 1.2, 1, 100], noiseAmp: 1 });
    const CEILING = 50;
    const r = levenbergMarquardt(xData, yData, model4PL, [0, 1.2, 1, 20], {
      constrain: (proposed) => proposed.map((v, i) => (i === 3 ? Math.min(v, CEILING) : v)),
    });
    expect(r.params[3]).toBeLessThanOrEqual(CEILING);
    // Unconstrained, the same start climbs to the true plateau near 100.
    const free = levenbergMarquardt(xData, yData, model4PL, [0, 1.2, 1, 20]);
    expect(free.params[3]).toBeGreaterThan(90);
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

describe("fitModel — canonical solution form", () => {
  // The 4PL has an exact two-fold symmetry: [A, B, C, D] and [D, -B, C, A]
  // are the same curve. Which one the optimiser lands on depends only on the
  // starting point, so without canonicalisation A and D are not stable
  // quantities and their confidence intervals are uninterpretable.
  const ascending = [0, 1.2, 1, 100];  // low response at low dose
  const descending = [100, 1.2, 1, 0]; // high response at low dose

  it("always returns a positive Hill slope", () => {
    for (const truth of [ascending, descending]) {
      for (let seed = 1; seed <= 20; seed++) {
        const { xData, yData } = make4PLDataGaussian({ params: truth, sd: 4, seed });
        expect(fitModel(xData, yData, model4PL, false).params[1]).toBeGreaterThan(0);
      }
    }
  });

  it("reports A as the response at zero dose and D at infinite dose", () => {
    const { xData, yData } = make4PLDataGaussian({ params: ascending, sd: 3, seed: 41 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(model4PL(1e-12, fit.params)).toBeCloseTo(fit.params[0], 6);
    expect(model4PL(1e12, fit.params)).toBeCloseTo(fit.params[3], 6);
  });

  it("puts A below D for ascending data and above D for descending data", () => {
    const asc = make4PLDataGaussian({ params: ascending, sd: 3, seed: 43 });
    const desc = make4PLDataGaussian({ params: descending, sd: 3, seed: 43 });
    const a = fitModel(asc.xData, asc.yData, model4PL, false);
    const d = fitModel(desc.xData, desc.yData, model4PL, false);
    expect(a.params[0]).toBeLessThan(a.params[3]);
    expect(d.params[0]).toBeGreaterThan(d.params[3]);
  });

  it("leaves the curve, and every fit statistic, untouched", () => {
    // Canonicalisation is a relabelling, not a refit.
    const { xData, yData } = make4PLDataGaussian({ params: ascending, sd: 3, seed: 47 });
    const fit = fitModel(xData, yData, model4PL, false);
    const mirrored = [fit.params[3], -fit.params[1], fit.params[2], fit.params[0]];
    for (const x of [0.01, 0.1, 1, 10, 100]) {
      expect(model4PL(x, mirrored)).toBeCloseTo(model4PL(x, fit.params), 9);
    }
    expect(fit.r2).toBeGreaterThan(0.9);
  });

  it("recovers the true parameters in canonical orientation", () => {
    const { xData, yData } = make4PLDataGaussian({ params: ascending, sd: 1, seed: 53 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.params[0]).toBeCloseTo(0, 0);
    expect(fit.params[1]).toBeCloseTo(1.2, 0);
    expect(fit.params[2]).toBeCloseTo(1, 1);
    expect(fit.params[3]).toBeCloseTo(100, 0);
  });

  it("does not canonicalise the 5PL, which has no such symmetry", () => {
    // Flipping the slope sign does not commute with the asymmetry exponent S,
    // so there is no equivalent relabelling to apply.
    const { xData, yData } = make4PLDataGaussian({ params: ascending, sd: 2, seed: 59 });
    const fit = fitModel(xData, yData, model5PL, true);
    expect(fit.params).toHaveLength(5);
    expect(Number.isFinite(fit.params[1])).toBe(true);
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

  it("fits nanomolar data identically to unit-scale data", () => {
    const a = fitAtScale(SCALES.unit);
    const b = fitAtScale(SCALES.nanomolar);
    expect(b.converged).toBe(true);
    expect(b.params[2] / SCALES.nanomolar).toBeCloseTo(a.params[2], 6);
    expect(b.params[1]).toBeCloseTo(a.params[1], 6);
  });

  it("recovers a nanomolar EC50 to the same accuracy as a unit-scale one", () => {
    const fit = fitAtScale(SCALES.nanomolar);
    const err = Math.abs(Math.log10(fit.params[2]) - Math.log10(SCALES.nanomolar));
    expect(err).toBeLessThan(0.05); // was ~0.116 with the absolute step
  });

  it("fits identically across nine decades of concentration units", () => {
    const ref = fitAtScale(1);
    for (const s of [1e-12, 1e-9, 1e-6, 1e-3, 1e3]) {
      const fit = fitAtScale(s);
      expect(fit.converged).toBe(true);
      expect(fit.params[2] / s).toBeCloseTo(ref.params[2], 6);
      expect(fit.params[1]).toBeCloseTo(ref.params[1], 6);
      expect(fit.params[0]).toBeCloseTo(ref.params[0], 6);
      expect(fit.params[3]).toBeCloseTo(ref.params[3], 6);
      expect(fit.r2).toBeCloseTo(ref.r2, 9);
    }
  });

  it("is invariant to the units of the RESPONSE axis too", () => {
    // Scaling y scales the plateaus and leaves EC50 and Hill untouched.
    const truth = [0, 1.2, 1, 100];
    const base = make4PLData({ params: truth, noiseAmp: 2 });
    const ref = fitModel(base.xData, base.yData, model4PL, false);
    for (const m of [1e-3, 1e3, 47189.7]) {
      const fit = fitModel(base.xData, base.yData.map(y => y * m), model4PL, false);
      expect(fit.params[2]).toBeCloseTo(ref.params[2], 6);
      expect(fit.params[1]).toBeCloseTo(ref.params[1], 5);
      expect(fit.params[3] / m).toBeCloseTo(ref.params[3], 4);
    }
  });
});
