import { describe, it, expect } from "vitest";
import { identifiabilityWarnings } from "../identifiability.js";
import { fitModel, fitConstrainedModel } from "../lm.js";
import { model4PL, model5PL } from "../models.js";
import { make4PLDataGaussian, eval4PLExported } from "./fixtures.js";

/** Build a dataset over an explicit log10 dose window rather than around the EC50. */
function dataOverWindow({ params, logLo, logHi, nConc = 8, nReps = 3, sd = 0.5, seed = 1 }) {
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

const codes = (result) => result.warnings.map(w => w.code);

describe("identifiabilityWarnings — plateau geometry", () => {
  const params = [0, 1.2, 10, 100]; // EC50 = 10, ascending, span 100

  it("is quiet when the dose range brackets both plateaus", () => {
    const { xData, yData } = dataOverWindow({ params, logLo: -2, logHi: 4, nConc: 10, nReps: 3 });
    const fit = fitModel(xData, yData, model4PL, false);
    const r = fit.identifiability;
    expect(r.warnings.filter(w => w.code.includes("plateau"))).toHaveLength(0);
    expect(r.geometry.gapLow).toBeLessThan(0.05);
    expect(r.geometry.gapHigh).toBeLessThan(0.05);
  });

  it("flags the bottom plateau when the low doses stop short of it", () => {
    // Starts at the EC50 itself: the curve is half way up at the lowest dose.
    const { xData, yData } = dataOverWindow({ params, logLo: 1, logHi: 4, nConc: 8, nReps: 3 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(codes(fit.identifiability)).toContain("bottom-plateau-unobserved");
    expect(codes(fit.identifiability)).not.toContain("top-plateau-unobserved");
  });

  it("flags the top plateau when the high doses stop short of it", () => {
    const { xData, yData } = dataOverWindow({ params, logLo: -2, logHi: 1, nConc: 8, nReps: 3 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(codes(fit.identifiability)).toContain("top-plateau-unobserved");
    expect(codes(fit.identifiability)).not.toContain("bottom-plateau-unobserved");
  });

  it("flags both ends when the window is narrow", () => {
    const { xData, yData } = dataOverWindow({ params, logLo: 0.6, logHi: 1.4, nConc: 6, nReps: 3 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(codes(fit.identifiability)).toContain("bottom-plateau-unobserved");
    expect(codes(fit.identifiability)).toContain("top-plateau-unobserved");
    expect(fit.identifiability.worst).toBe("severe");
  });

  it("measures the gap as the analytic fraction of the response span", () => {
    // For a 4PL the fraction traversed at dose x is u/(1+u), u = (x/C)^B,
    // independent of both asymptotes. Pins the geometry against closed form.
    const xData = [1, 1, 3, 3, 10, 10, 30, 30];
    const fit = { params, se: [1, 1, 1, 1], ci: null, correlation: null };
    const r = identifiabilityWarnings(fit, xData, model4PL);

    const frac = (x) => { const u = Math.pow(x / 10, 1.2); return u / (1 + u); };
    expect(r.geometry.gapLow).toBeCloseTo(frac(1), 12);
    expect(r.geometry.gapHigh).toBeCloseTo(1 - frac(30), 12);
  });

  it("recommends an extension that would actually bracket the plateau", () => {
    const xData = [2, 2, 6, 6, 20, 20, 60, 60];
    const fit = { params, se: [1, 1, 1, 1], ci: null, correlation: null };
    const r = identifiabilityWarnings(fit, xData, model4PL);
    const w = r.warnings.find(x => x.code === "bottom-plateau-unobserved");
    expect(w).toBeDefined();

    // Closed form: 5% of the span means u = 0.05/0.95, so x = C (1/19)^(1/B).
    const target = 10 * Math.pow(0.05 / 0.95, 1 / 1.2);
    const decades = Math.log10(2 / target);
    expect(w.message).toContain(`${decades.toFixed(1)} decades`);
  });

  it("resolves which slot holds which asymptote from the sign of the slope", () => {
    // A negative Hill slope swaps the roles of A and D. The check must follow
    // the curve, not the slot index.
    const ascending = [100, -1.2, 10, 0]; // y -> 0 at x -> 0, -> 100 at x -> inf
    const xData = [10, 10, 30, 30, 100, 100];
    const r = identifiabilityWarnings(
      { params: ascending, se: [1, 1, 1, 1] }, xData, model4PL,
    );
    expect(r.geometry.zeroDoseSlot).toBe(3);
    expect(r.geometry.infDoseSlot).toBe(0);
    expect(r.geometry.yZero).toBeCloseTo(0, 10);
    expect(r.geometry.yInf).toBeCloseTo(100, 10);
    // The unbracketed end is the bottom, which lives in slot 3 here.
    const w = r.warnings.find(x => x.code === "bottom-plateau-unobserved");
    expect(w.parameter).toBe(3);
  });

  it("says nothing about a plateau the caller fixed rather than estimated", () => {
    // A 2PL pins both asymptotes. They are assertions, not estimates, so there
    // is no identifiability question to raise about them.
    const { xData, yData } = dataOverWindow({ params, logLo: 0.6, logHi: 1.4, nConc: 6, nReps: 3 });
    const fit = fitConstrainedModel(xData, yData, { 0: 0, 3: 100 });
    expect(fit.se[0]).toBeNull();
    expect(fit.se[3]).toBeNull();
    expect(codes(fit.identifiability)).not.toContain("bottom-plateau-unobserved");
    expect(codes(fit.identifiability)).not.toContain("top-plateau-unobserved");
  });
});

describe("identifiabilityWarnings — potency", () => {
  const params = [0, 1.2, 10, 100];

  it("flags an EC50 outside the tested dose range", () => {
    const xData = [100, 100, 300, 300, 1000, 1000];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1] }, xData, model4PL);
    expect(codes(r)).toContain("ec50-outside-dose-range");
    expect(r.geometry.ec50InRange).toBe(false);
    expect(r.worst).toBe("severe");
  });

  it("does not flag an EC50 inside the range", () => {
    const xData = [1, 1, 10, 10, 100, 100];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1] }, xData, model4PL);
    expect(codes(r)).not.toContain("ec50-outside-dose-range");
    expect(r.geometry.ec50InRange).toBe(true);
  });

  it("reports the dose range in decades either side of the EC50", () => {
    const xData = [0.1, 1000];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1] }, xData, model4PL);
    expect(r.geometry.decadesBelowEC50).toBeCloseTo(2, 10);
    expect(r.geometry.decadesAboveEC50).toBeCloseTo(2, 10);
  });

  it("flags a confidence interval that only bounds the potency", () => {
    const r = identifiabilityWarnings(
      { params, se: [1, 1, 1, 1], ci: [null, null, { lo: 1, hi: 400 }, null] },
      [1, 10, 100], model4PL,
    );
    const w = r.warnings.find(x => x.code === "wide-ec50-interval");
    expect(w.severity).toBe("severe"); // 400-fold
    expect(w.value).toBeCloseTo(400, 10);
  });

  it("leaves a tight potency interval alone", () => {
    const r = identifiabilityWarnings(
      { params, se: [1, 1, 1, 1], ci: [null, null, { lo: 9, hi: 11 }, null] },
      [1, 10, 100], model4PL,
    );
    expect(codes(r)).not.toContain("wide-ec50-interval");
  });
});

describe("identifiabilityWarnings — parameter separability", () => {
  const params = [0, 1.2, 10, 100];
  const wide = [0.01, 0.1, 1, 10, 100, 1000, 10000];

  it("flags a near-collinear pair", () => {
    const correlation = [
      [1, 0.1, 0.2, 0.3],
      [0.1, 1, 0.995, 0.2],
      [0.2, 0.995, 1, 0.1],
      [0.3, 0.2, 0.1, 1],
    ];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1], correlation }, wide, model4PL);
    const w = r.warnings.find(x => x.code === "parameter-correlation");
    expect(w.severity).toBe("warning");
    expect(w.value).toBeCloseTo(0.995, 10);
  });

  it("escalates an almost perfectly collinear pair", () => {
    const correlation = [
      [1, 0, 0, 0], [0, 1, -0.9997, 0], [0, -0.9997, 1, 0], [0, 0, 0, 1],
    ];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1], correlation }, wide, model4PL);
    expect(r.warnings.find(x => x.code === "parameter-correlation").severity).toBe("severe");
  });

  it("ignores ordinary correlation, which every nonlinear fit has", () => {
    const correlation = [
      [1, 0.6, 0.7, 0.5], [0.6, 1, 0.8, 0.4], [0.7, 0.8, 1, 0.6], [0.5, 0.4, 0.6, 1],
    ];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1], correlation }, wide, model4PL);
    expect(codes(r)).not.toContain("parameter-correlation");
  });

  it("tolerates a NaN correlation entry without throwing", () => {
    const correlation = [
      [1, NaN, 0, 0], [NaN, 1, NaN, 0], [0, NaN, 1, 0], [0, 0, 0, 1],
    ];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1], correlation }, wide, model4PL);
    expect(codes(r)).not.toContain("parameter-correlation");
  });

  it("flags a Hill slope interval that includes zero", () => {
    const r = identifiabilityWarnings(
      { params, se: [1, 1, 1, 1], ci: [null, { lo: -0.4, hi: 2.9 }, null, null] },
      wide, model4PL,
    );
    const w = r.warnings.find(x => x.code === "slope-includes-zero");
    expect(w.severity).toBe("severe");
  });

  it("reports a fit whose covariance matrix was singular", () => {
    const r = identifiabilityWarnings(
      { params, se: null, ci: null, correlation: null, cov: null, dof: null },
      wide, model4PL,
    );
    expect(codes(r)).toContain("no-inference");
  });
});

describe("identifiabilityWarnings — design", () => {
  const params = [0, 1.2, 10, 100];

  it("flags a design with fewer doses than free parameters", () => {
    const xData = [1, 1, 10, 10, 100, 100];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1] }, xData, model4PL);
    const w = r.warnings.find(x => x.code === "too-few-doses");
    expect(w.severity).toBe("severe");
    expect(w.value).toBe(3);
  });

  it("flags a saturated design that cannot check its own model", () => {
    const xData = [0.01, 0.01, 1, 1, 10, 10, 1e4, 1e4];
    const r = identifiabilityWarnings({ params, se: [1, 1, 1, 1] }, xData, model4PL);
    expect(codes(r)).toContain("saturated-design");
  });

  it("counts free parameters, not display slots", () => {
    // A 1PL estimates only the EC50, so four doses is not a saturated design.
    const xData = [0.01, 0.01, 1, 1, 10, 10, 1e4, 1e4];
    const r = identifiabilityWarnings(
      { params, se: [null, null, 1, null] }, xData, model4PL,
    );
    expect(codes(r)).not.toContain("saturated-design");
    expect(codes(r)).not.toContain("too-few-doses");
  });
});

describe("identifiabilityWarnings — robustness", () => {
  it("returns a clean result for missing or empty input", () => {
    for (const bad of [null, undefined, {}, { params: null }]) {
      const r = identifiabilityWarnings(bad, [1, 2, 3], model4PL);
      expect(r.ok).toBe(true);
      expect(r.warnings).toEqual([]);
    }
    expect(identifiabilityWarnings({ params: [0, 1, 10, 100] }, [], model4PL).ok).toBe(true);
  });

  it("survives a dose list with no positive concentrations", () => {
    const r = identifiabilityWarnings({ params: [0, 1, 10, 100] }, [0, 0, 0], model4PL);
    expect(r.ok).toBe(true);
  });

  it("survives a degenerate curve with no response span", () => {
    const r = identifiabilityWarnings(
      { params: [50, 1.2, 10, 50], se: [1, 1, 1, 1] }, [1, 10, 100, 1000], model4PL,
    );
    expect(r.geometry).toBeNull(); // no span to take a fraction of
    expect(() => r.warnings.length).not.toThrow();
  });

  it("names 5PL parameters with the 5PL layout", () => {
    const params = [0, 1.1, 10, 100, 0.8]; // [Bottom, Hill, EC50, Top, S]
    const xData = [10, 10, 30, 30, 100, 100];
    const r = identifiabilityWarnings(
      { params, se: [1, 1, 1, 1, 1] }, xData, model5PL, { is5PL: true },
    );
    const w = r.warnings.find(x => x.code === "bottom-plateau-unobserved");
    expect(w.message).toMatch(/^Bottom is extrapolated/);
  });
});

describe("identifiability integration with the fitters", () => {
  it("rides along on every fit result", () => {
    const params = [0, 1.2, 10, 100];
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 10, nReps: 3, decades: 6, sd: 2, seed: 9 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.identifiability).toBeDefined();
    expect(fit.identifiability.ok).toBe(true);
    expect(fit.identifiability.worst).toBe("none");
  });

  it("is present on a constrained fit too", () => {
    const params = [0, 1.2, 10, 100];
    const { xData, yData } = make4PLDataGaussian({ params, nConc: 10, nReps: 3, decades: 6, sd: 2, seed: 9 });
    const fit = fitConstrainedModel(xData, yData, { 1: 1.0 });
    expect(fit.identifiability).toBeDefined();
    expect(fit.identifiability.geometry).not.toBeNull();
  });

  it("catches the case the standard errors alone would hide", () => {
    // Truncated range: the fit converges, R² is excellent, and every parameter
    // comes back with a finite standard error. The geometry is what shows that
    // the bottom plateau was never actually observed.
    const params = [0, 1.2, 10, 100];
    const { xData, yData } = dataOverWindow({ params, logLo: 0.8, logHi: 4, nConc: 8, nReps: 3, sd: 1 });
    const fit = fitModel(xData, yData, model4PL, false);
    expect(fit.r2).toBeGreaterThan(0.99);
    expect(fit.se.every(v => v !== null && isFinite(v))).toBe(true);
    expect(fit.identifiability.ok).toBe(false);
    expect(codes(fit.identifiability)).toContain("bottom-plateau-unobserved");
  });
});
