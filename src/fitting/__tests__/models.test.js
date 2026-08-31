import { describe, it, expect } from "vitest";
import {
  model4PL, model5PL, makeConstrainedModel, getModelFn, computeBiologicalEC50,
} from "../models.js";

describe("model4PL", () => {
  // y = D + (A - D) / (1 + (x/C)^B), params [A, B, C, D].
  // A is the asymptote where (x/C)^B -> 0 and D where it -> infinity; which
  // end of the dose range each sits at depends on the sign of B. See the
  // note in models.js.
  const p = [0, 1, 1, 100]; // B > 0, so: A at x -> 0, D at x -> inf (ascending)

  it("returns the midpoint response at x = EC50", () => {
    expect(model4PL(1, p)).toBe(50);
  });

  it("approaches A as x -> 0+ and D as x -> inf when the Hill slope is positive", () => {
    expect(model4PL(1e-12, p)).toBeCloseTo(0, 6);
    expect(model4PL(1e12, p)).toBeCloseTo(100, 6);
  });

  it("swaps which asymptote is which when the Hill slope is negative", () => {
    // A and D are not "low" and "high" plateaus; they are the asymptotes at
    // (x/C)^B -> 0 and -> inf respectively, and the sign of B decides which
    // end of the dose range each one sits at. Both signs arise in practice:
    // the app's own sample data fits to B = -1.33.
    const neg = [0, -1, 1, 100];
    expect(model4PL(1e-12, neg)).toBeCloseTo(100, 6);
    expect(model4PL(1e12, neg)).toBeCloseTo(0, 6);
  });

  it("returns the correct x -> 0+ limit at x <= 0 for either Hill sign", () => {
    // Regression guard: this used to return A unconditionally, which is the
    // wrong asymptote whenever B < 0. A zero-dose vehicle-control row is
    // legitimate input, so the wrong branch was reachable from the UI.
    expect(model4PL(0, p)).toBe(0);
    expect(model4PL(-5, p)).toBe(0);

    const neg = [0, -1, 1, 100];
    expect(model4PL(0, neg)).toBe(100);
    expect(model4PL(-5, neg)).toBe(100);
  });

  it("agrees with its own one-sided limit at x <= 0", () => {
    // Probe deep: a shallow slope approaches its asymptote slowly, so at
    // B = 0.5 an x of 1e-14 is still 9e-6 away from the limit.
    for (const B of [2, 1, 0.5, -0.5, -1, -2]) {
      const q = [7, B, 1, 93];
      expect(model4PL(0, q)).toBeCloseTo(model4PL(1e-30, q), 6);
    }
  });

  it("is constant in x when the Hill slope is zero", () => {
    const flat = [0, 0, 1, 100];
    expect(model4PL(0, flat)).toBeCloseTo(50, 9);
    expect(model4PL(1, flat)).toBeCloseTo(50, 9);
    expect(model4PL(1e6, flat)).toBeCloseTo(50, 9);
  });

  it("is symmetric about the EC50 in log-x space", () => {
    // A 4PL is point-symmetric around (log EC50, midpoint).
    const mid = (p[0] + p[3]) / 2;
    for (const f of [2, 10, 100]) {
      const above = model4PL(p[2] * f, p);
      const below = model4PL(p[2] / f, p);
      expect(above + below).toBeCloseTo(2 * mid, 9);
    }
  });

  it("is scale-equivariant: scaling x and EC50 together leaves y unchanged", () => {
    for (const s of [1e-6, 1e-9, 1e3]) {
      const scaled = [p[0], p[1], p[2] * s, p[3]];
      expect(model4PL(0.37 * s, scaled)).toBeCloseTo(model4PL(0.37, p), 9);
    }
  });

  it("steepens with the Hill slope", () => {
    // Ascending curve one decade above the EC50: a steeper slope has already
    // travelled further toward the upper asymptote.
    const shallow = model4PL(10, [0, 1, 1, 100]);
    const steep = model4PL(10, [0, 3, 1, 100]);
    expect(steep).toBeGreaterThan(shallow);
    // ...and correspondingly further from it one decade below.
    expect(model4PL(0.1, [0, 3, 1, 100])).toBeLessThan(model4PL(0.1, [0, 1, 1, 100]));
  });
});

describe("model5PL", () => {
  // y = Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)^S
  const p = [0, 1, 1, 100, 1]; // S = 1 degenerates to a 4PL

  it("reduces to the 4PL midpoint when the asymmetry factor S = 1", () => {
    expect(model5PL(1, p)).toBeCloseTo(50, 9);
  });

  it("returns the correct x -> 0+ limit at x <= 0 for either Hill sign", () => {
    // The 5PL uses (EC50/x), so the sign relationship is inverted relative to
    // the 4PL: a POSITIVE Hill puts Bottom at x -> 0.
    expect(model5PL(0, p)).toBe(0);
    expect(model5PL(-1, p)).toBe(0);

    const neg = [0, -1, 1, 100, 1];
    expect(model5PL(0, neg)).toBe(100);
    expect(model5PL(-1, neg)).toBe(100);
  });

  it("agrees with its own one-sided limit at x <= 0", () => {
    for (const Hill of [2, 1, 0.5, -0.5, -1, -2]) {
      const q = [7, Hill, 1, 93, 1.4];
      expect(model5PL(0, q)).toBeCloseTo(model5PL(1e-30, q), 6);
    }
  });

  it("shifts the half-maximal point away from the EC50 parameter when S != 1", () => {
    // This is the documented reason computeBiologicalEC50 exists: with S != 1
    // the `EC50` parameter is no longer where the response is half-maximal.
    const asym = [0, 1, 1, 100, 2];
    expect(model5PL(1, asym)).not.toBeCloseTo(50, 1);
  });
});

describe("makeConstrainedModel", () => {
  it("expands free params around fixed ones in 4PL index order", () => {
    // Fix Hill (index 1) to 1.0 -> a 3PL; free params are [A, C, D].
    const f = makeConstrainedModel({ 1: 1.0 });
    expect(f(1, [0, 1, 100])).toBe(model4PL(1, [0, 1, 1, 100]));
  });

  it("supports multiple fixed params (2PL: A and D fixed)", () => {
    const f = makeConstrainedModel({ 0: 0, 3: 100 });
    expect(f(2, [1.5, 1])).toBe(model4PL(2, [0, 1.5, 1, 100]));
  });

  it("is equivalent to the unconstrained 4PL when nothing is fixed", () => {
    const f = makeConstrainedModel({});
    expect(f(3, [0, 1, 1, 100])).toBe(model4PL(3, [0, 1, 1, 100]));
  });
});

describe("getModelFn", () => {
  it("selects the 5PL evaluator only for '5PL'", () => {
    expect(getModelFn("5PL")).toBe(model5PL);
  });

  it("maps every constrained variant onto the 4PL evaluator", () => {
    for (const m of ["1PL", "2PL", "3PL", "4PL", "Auto", undefined]) {
      expect(getModelFn(m)).toBe(model4PL);
    }
  });
});

describe("computeBiologicalEC50", () => {
  it("recovers the EC50 parameter when S = 1 (curve is symmetric)", () => {
    const p = [0, 1, 5, 100, 1];
    expect(computeBiologicalEC50(model5PL, p)).toBeCloseTo(5, 4);
  });

  it("finds the true half-maximal concentration when S != 1", () => {
    const p = [0, 1, 5, 100, 3];
    const bio = computeBiologicalEC50(model5PL, p);
    // By definition the response there is (Top + Bottom) / 2.
    expect(model5PL(bio, p)).toBeCloseTo(50, 4);
    // ...and it is genuinely different from the EC50 parameter.
    expect(bio).not.toBeCloseTo(5, 2);
  });

  it("works for ascending curves", () => {
    const p = [100, 1, 5, 0, 2]; // Bottom > Top
    const bio = computeBiologicalEC50(model5PL, p);
    expect(model5PL(bio, p)).toBeCloseTo(50, 4);
  });
});
