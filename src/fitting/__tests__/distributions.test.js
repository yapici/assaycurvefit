import { describe, it, expect } from "vitest";
import { lnGamma, betaIncomplete, tCDF, tInv } from "../distributions.js";

describe("lnGamma", () => {
  it("matches ln((n-1)!) at small integers", () => {
    expect(lnGamma(1)).toBeCloseTo(0, 10);            // 0! = 1
    expect(lnGamma(2)).toBeCloseTo(0, 10);            // 1! = 1
    expect(lnGamma(5)).toBeCloseTo(Math.log(24), 10); // 4! = 24
    expect(lnGamma(10)).toBeCloseTo(Math.log(362880), 9); // 9!
  });

  it("matches the half-integer value ln(sqrt(pi))", () => {
    expect(lnGamma(0.5)).toBeCloseTo(Math.log(Math.sqrt(Math.PI)), 10);
  });

  it("satisfies the recurrence lnGamma(z+1) = lnGamma(z) + ln(z)", () => {
    for (const z of [0.3, 1.7, 4.2, 11.9]) {
      expect(lnGamma(z + 1)).toBeCloseTo(lnGamma(z) + Math.log(z), 9);
    }
  });

  it("handles the reflection branch for z < 0.5", () => {
    // lnGamma(0.25) + lnGamma(0.75) = ln(pi / sin(pi/4))
    expect(lnGamma(0.25) + lnGamma(0.75))
      .toBeCloseTo(Math.log(Math.PI / Math.sin(Math.PI / 4)), 9);
  });

  it("stays finite for large arguments", () => {
    expect(Number.isFinite(lnGamma(1000))).toBe(true);
  });
});

describe("betaIncomplete", () => {
  it("is 0 at x <= 0 and 1 at x >= 1", () => {
    expect(betaIncomplete(0, 2, 3)).toBe(0);
    expect(betaIncomplete(-1, 2, 3)).toBe(0);
    expect(betaIncomplete(1, 2, 3)).toBe(1);
    expect(betaIncomplete(2, 2, 3)).toBe(1);
  });

  it("reduces to x for a = b = 1 (uniform distribution)", () => {
    for (const x of [0.1, 0.5, 0.9]) {
      expect(betaIncomplete(x, 1, 1)).toBeCloseTo(x, 9);
    }
  });

  it("is 1/2 at x = 1/2 for symmetric parameters", () => {
    expect(betaIncomplete(0.5, 2, 2)).toBeCloseTo(0.5, 9);
    expect(betaIncomplete(0.5, 7, 7)).toBeCloseTo(0.5, 9);
  });

  it("satisfies the symmetry I_x(a,b) = 1 - I_{1-x}(b,a)", () => {
    for (const [x, a, b] of [[0.3, 2, 5], [0.8, 4, 1.5], [0.05, 0.5, 3]]) {
      expect(betaIncomplete(x, a, b)).toBeCloseTo(1 - betaIncomplete(1 - x, b, a), 9);
    }
  });

  it("is monotonically increasing in x", () => {
    let prev = -1;
    for (let x = 0.05; x < 1; x += 0.05) {
      const v = betaIncomplete(x, 3, 4);
      expect(v).toBeGreaterThan(prev);
      prev = v;
    }
  });
});

describe("tCDF", () => {
  it("is 1/2 at t = 0 for any df", () => {
    for (const df of [1, 5, 30]) expect(tCDF(0, df)).toBeCloseTo(0.5, 9);
  });

  it("is symmetric: F(-t) = 1 - F(t)", () => {
    for (const [t, df] of [[1.5, 10], [2.8, 4], [0.3, 25]]) {
      expect(tCDF(-t, df)).toBeCloseTo(1 - tCDF(t, df), 9);
    }
  });

  it("matches the Cauchy closed form at df = 1", () => {
    // df=1 is the standard Cauchy: F(t) = 1/2 + atan(t)/pi
    for (const t of [0.5, 1, 3]) {
      expect(tCDF(t, 1)).toBeCloseTo(0.5 + Math.atan(t) / Math.PI, 8);
    }
  });

  it("approaches the normal CDF for large df", () => {
    // P(T <= 1.959964) -> 0.975 as df -> inf
    expect(tCDF(1.959964, 100000)).toBeCloseTo(0.975, 4);
  });
});

describe("tInv", () => {
  // tInv(p, df) is the UPPER-tail quantile: returns t with P(T > t) = p.
  // Reference values from standard Student-t tables.
  const table = [
    [0.025, 1, 12.706],
    [0.025, 5, 2.571],
    [0.025, 10, 2.228],
    [0.025, 30, 2.042],
    [0.05, 10, 1.812],
    [0.05, 20, 1.725],
    [0.005, 20, 2.845],
    [0.0025, 8, 3.833], // the value Grubbs' critical G leans on at n = 10
  ];

  it.each(table)("t(p=%f, df=%i) = %f", (p, df, expected) => {
    expect(tInv(p, df)).toBeCloseTo(expected, 2);
  });

  it("inverts tCDF", () => {
    for (const [p, df] of [[0.025, 12], [0.1, 7], [0.001, 40]]) {
      expect(tCDF(tInv(p, df), df)).toBeCloseTo(1 - p, 8);
    }
  });

  it("decreases with df at fixed p (heavier tails need larger t)", () => {
    expect(tInv(0.025, 1)).toBeGreaterThan(tInv(0.025, 10));
    expect(tInv(0.025, 10)).toBeGreaterThan(tInv(0.025, 100));
  });

  it("returns 0 at p >= 0.5 and Infinity for degenerate input", () => {
    expect(tInv(0.5, 10)).toBe(0);
    expect(tInv(0.9, 10)).toBe(0);
    expect(tInv(0.025, 0)).toBe(Infinity);
    expect(tInv(0, 10)).toBe(Infinity);
  });
});
