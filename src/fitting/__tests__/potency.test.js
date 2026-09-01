import { describe, it, expect } from "vitest";
import {
  fitCurvePair, parallelismFTest, parallelismEquivalence, relativePotencyFrom,
  relativePotency,
} from "../potency.js";
import { tInv } from "../distributions.js";
import { eval4PLExported, seededNormal } from "./fixtures.js";

/**
 * A dose-response curve with Gaussian noise, on an explicit dose grid so that
 * test and reference can be built on the same concentrations.
 */
function curve({ params, nConc = 8, nReps = 3, logLo = -1, logHi = 3, sd = 2, seed = 1 }) {
  const normal = seededNormal(seed);
  const xData = [], yData = [];
  for (let i = 0; i < nConc; i++) {
    const x = Math.pow(10, logLo + ((logHi - logLo) * i) / (nConc - 1));
    for (let r = 0; r < nReps; r++) {
      xData.push(x);
      yData.push(eval4PLExported(x, params) + sd * normal());
    }
  }
  return { xData, yData };
}

/** Reference at EC50 = 10, test shifted to be `rp` times as potent. */
function pairWithRP(rp, { sd = 2, slopeTest = 1.2, seedRef = 1, seedTest = 2 } = {}) {
  return {
    reference: curve({ params: [0, 1.2, 10, 100], sd, seed: seedRef }),
    test: curve({ params: [0, slopeTest, 10 / rp, 100], sd, seed: seedTest }),
  };
}

const LOOSE_BOUNDS = { slope: 0.5, lower: 15, upper: 15 };

describe("relative potency — recovering a known ratio", () => {
  it.each([0.25, 0.5, 2, 4, 10])("recovers a %sx potency ratio", (rp) => {
    const { reference, test } = pairWithRP(rp);
    const r = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    expect(r.ok).toBe(true);
    expect(r.potency.rp / rp).toBeGreaterThan(0.85);
    expect(r.potency.rp / rp).toBeLessThan(1.18);
    expect(r.potency.ci.lo).toBeLessThan(rp);
    expect(r.potency.ci.hi).toBeGreaterThan(rp);
  });

  it("inverts exactly when test and reference swap roles", () => {
    // Potency is a ratio, so relabelling which article is the standard must
    // reciprocate the answer and its interval, not merely approximate it.
    const { reference, test } = pairWithRP(4);
    const forward = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    const reverse = relativePotency(test, reference, { bounds: LOOSE_BOUNDS });

    expect(reverse.potency.rp).toBeCloseTo(1 / forward.potency.rp, 6);
    expect(reverse.potency.ci.lo).toBeCloseTo(1 / forward.potency.ci.hi, 6);
    expect(reverse.potency.ci.hi).toBeCloseTo(1 / forward.potency.ci.lo, 6);
    expect(reverse.potency.logRP).toBeCloseTo(-forward.potency.logRP, 8);
  });

  it("is invariant to the units the concentrations are expressed in", () => {
    const { reference, test } = pairWithRP(4);
    const scale = (d) => ({ xData: d.xData.map(x => x * 1e-9), yData: d.yData });
    const plain = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    const molar = relativePotency(scale(reference), scale(test), { bounds: LOOSE_BOUNDS });

    expect(molar.potency.rp).toBeCloseTo(plain.potency.rp, 6);
    expect(molar.potency.ci.lo).toBeCloseTo(plain.potency.ci.lo, 6);
    // The EC50s themselves do move, by exactly the factor applied.
    expect(molar.potency.ec50Reference / plain.potency.ec50Reference).toBeCloseTo(1e-9, 15);
  });

  it("reports potency as a percentage of the reference too", () => {
    const { reference, test } = pairWithRP(2);
    const r = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    expect(r.potency.percent).toBeCloseTo(r.potency.rp * 100, 10);
    expect(r.potency.percentCi.lo).toBeCloseTo(r.potency.ci.lo * 100, 10);
  });

  it("gives an interval that is symmetric in log space, not on the ratio", () => {
    // Formed as 10^(logRP +/- t*SE), so the multiplicative distance to each
    // endpoint matches while the additive distance does not.
    const { reference, test } = pairWithRP(4);
    const r = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    const { rp, ci } = r.potency;
    expect(rp / ci.lo).toBeCloseTo(ci.hi / rp, 6);
    expect(rp - ci.lo).not.toBeCloseTo(ci.hi - rp, 6);
    expect(ci.lo).toBeGreaterThan(0);
  });

  it("places the two EC50s exactly RP apart", () => {
    const { reference, test } = pairWithRP(4);
    const r = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    const { ec50Reference, ec50Test, rp } = r.potency;
    expect(ec50Reference / ec50Test).toBeCloseTo(rp, 8);
  });
});

describe("the two fitted models", () => {
  it("nests: the unconstrained fit can never fit worse than the parallel one", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    expect(pair.separate.ssr).toBeLessThanOrEqual(pair.parallel.ssr + 1e-9);
    expect(pair.separate.k).toBe(8);
    expect(pair.parallel.k).toBe(5);
    expect(pair.separate.n).toBe(reference.xData.length + test.xData.length);
  });

  it("canonicalises both curves to a positive Hill slope", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    expect(pair.separate.params[1]).toBeGreaterThan(0); // reference slope
    expect(pair.separate.params[5]).toBeGreaterThan(0); // test slope
    expect(pair.parallel.params[1]).toBeGreaterThan(0);
  });

  it("agrees with two independent single-curve fits when the data are parallel", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    const { reference: rf, test: tf } = pair.singleCurveFits;
    // The joint unconstrained fit is mathematically the two separate fits.
    expect(Math.pow(10, pair.separate.params[2])).toBeCloseTo(rf.params[2], 4);
    expect(Math.pow(10, pair.separate.params[6])).toBeCloseTo(tf.params[2], 4);
    expect(pair.separate.ssr).toBeCloseTo(rf.ssr + tf.ssr, 4);
  });

  it("gives the parallel fit more residual degrees of freedom", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    expect(pair.parallel.dof).toBe(pair.separate.dof + 3);
  });
});

describe("parallelism — the F-test", () => {
  it("does not reject parallelism for genuinely parallel curves", () => {
    const { reference, test } = pairWithRP(4);
    const f = parallelismFTest(fitCurvePair(reference, test));
    expect(f.applicable).toBe(true);
    expect(f.conclusion).toBe("parallel");
    expect(f.dfNum).toBe(3);
  });

  it("rejects parallelism when the slopes genuinely differ", () => {
    const { reference, test } = pairWithRP(4, { slopeTest: 3.0, sd: 1 });
    const f = parallelismFTest(fitCurvePair(reference, test));
    expect(f.conclusion).toBe("non-parallel");
    expect(f.pValue).toBeLessThan(0.05);
  });

  it("computes F from the two sums of squares and their degrees of freedom", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    const f = parallelismFTest(pair);
    const expected = ((pair.parallel.ssr - pair.separate.ssr) / 3)
      / (pair.separate.ssr / (pair.separate.n - 8));
    expect(f.F).toBeCloseTo(expected, 10);
  });

  it("carries the warning about its own inverted logic", () => {
    const { reference, test } = pairWithRP(4);
    const f = parallelismFTest(fitCurvePair(reference, test));
    expect(f.caveat).toMatch(/not evidence of parallelism/i);
  });

  it("rewards an imprecise assay, which is why it is not the default", () => {
    // Identical non-parallelism, two noise levels. The precise assay detects
    // it; the sloppy one passes. Demonstrating this is the argument for
    // equivalence testing, so it is worth pinning rather than asserting.
    const shared = { slopeTest: 2.2 };
    const precise = pairWithRP(4, { ...shared, sd: 0.5 });
    const sloppy = pairWithRP(4, { ...shared, sd: 12 });

    const fPrecise = parallelismFTest(fitCurvePair(precise.reference, precise.test));
    const fSloppy = parallelismFTest(fitCurvePair(sloppy.reference, sloppy.test));

    expect(fPrecise.conclusion).toBe("non-parallel");
    expect(fSloppy.conclusion).toBe("parallel"); // the worse assay "passes"
  });
});

describe("parallelism — equivalence testing", () => {
  it("refuses to reach a verdict without acceptance criteria", () => {
    const { reference, test } = pairWithRP(4);
    const e = parallelismEquivalence(fitCurvePair(reference, test), null);
    expect(e.applicable).toBe(false);
    expect(e.conclusion).toBe("no-criteria");
    expect(e.reason).toMatch(/product- and assay-specific/i);
    // The differences are still reported, just not judged.
    expect(e.parameters).toHaveLength(3);
    e.parameters.forEach(p => {
      expect(p.bounds).toBeNull();
      expect(p.passes).toBeNull();
      expect(Number.isFinite(p.estimate)).toBe(true);
    });
  });

  it("passes parallel curves against generous bounds", () => {
    const { reference, test } = pairWithRP(4);
    const e = parallelismEquivalence(fitCurvePair(reference, test), LOOSE_BOUNDS);
    expect(e.applicable).toBe(true);
    expect(e.conclusion).toBe("parallel");
    expect(e.failed).toEqual([]);
    expect(e.assessed).toEqual(["slope", "lower", "upper"]);
  });

  it("fails the slope criterion when the slopes differ", () => {
    const { reference, test } = pairWithRP(4, { slopeTest: 3.0, sd: 1 });
    const e = parallelismEquivalence(fitCurvePair(reference, test), LOOSE_BOUNDS);
    expect(e.conclusion).toBe("non-parallel");
    expect(e.failed).toContain("slope");
  });

  it("fails when the bounds are tighter than the assay can support", () => {
    // Parallel curves, but criteria narrower than the confidence interval.
    // Equivalence is not established, which is the correct answer: an assay
    // too imprecise to demonstrate similarity has not demonstrated it.
    const { reference, test } = pairWithRP(4);
    const e = parallelismEquivalence(fitCurvePair(reference, test), { slope: 0.001 });
    expect(e.conclusion).toBe("non-parallel");
    expect(e.failed).toEqual(["slope"]);
  });

  it("assesses only the parameters given bounds", () => {
    const { reference, test } = pairWithRP(4);
    const e = parallelismEquivalence(fitCurvePair(reference, test), { slope: 0.5 });
    expect(e.assessed).toEqual(["slope"]);
    expect(e.parameters.find(p => p.key === "lower").passes).toBeNull();
  });

  it("accepts asymmetric bounds as an explicit pair", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    const slope = parallelismEquivalence(pair, null).parameters.find(p => p.key === "slope");
    // A window that excludes the observed difference on one side only.
    const e = parallelismEquivalence(pair, { slope: [slope.estimate + 0.01, 5] });
    expect(e.parameters.find(p => p.key === "slope").bounds).toEqual({
      lo: slope.estimate + 0.01, hi: 5,
    });
    expect(e.conclusion).toBe("non-parallel");
  });

  it("uses a 90% interval at alpha = 0.05, as two one-sided tests require", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    const e = parallelismEquivalence(pair, LOOSE_BOUNDS, 0.05);
    expect(e.confidenceLevel).toBeCloseTo(0.9, 12);

    const slope = e.parameters.find(p => p.key === "slope");
    const t = tInv(0.05, pair.separate.dof);
    expect(slope.ci.hi - slope.ci.lo).toBeCloseTo(2 * t * slope.se, 10);
  });

  it("forms the difference as test minus reference", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    const e = parallelismEquivalence(pair, LOOSE_BOUNDS);
    const slope = e.parameters.find(p => p.key === "slope");
    expect(slope.estimate).toBeCloseTo(pair.separate.params[5] - pair.separate.params[1], 12);
    expect(slope.reference).toBeCloseTo(pair.separate.params[1], 12);
    expect(slope.test).toBeCloseTo(pair.separate.params[5], 12);
  });

  it("becomes more likely to pass as the assay gets more precise", () => {
    // The incentive the F-test gets backwards. Same truth, better data,
    // narrower interval, easier to demonstrate equivalence.
    const tight = { slope: 0.25, lower: 8, upper: 8 };
    const precise = pairWithRP(4, { sd: 0.5 });
    const sloppy = pairWithRP(4, { sd: 10 });
    expect(parallelismEquivalence(fitCurvePair(precise.reference, precise.test), tight).conclusion)
      .toBe("parallel");
    expect(parallelismEquivalence(fitCurvePair(sloppy.reference, sloppy.test), tight).conclusion)
      .toBe("non-parallel");
  });
});

describe("relativePotency — the whole analysis", () => {
  it("does not call potency reportable without acceptance criteria", () => {
    // The design decision worth pinning: an assay nobody set criteria for has
    // not passed, even though the F-test happens to say "parallel".
    const { reference, test } = pairWithRP(4);
    const r = relativePotency(reference, test);
    expect(r.ok).toBe(true);
    expect(r.fTest.conclusion).toBe("parallel");
    expect(r.reportable).toBe(false);
    expect(r.basis).toBe("none");
    expect(r.note).toMatch(/cannot establish parallelism on its own/i);
    // The number is still computed, so nobody has to recompute it by hand.
    expect(r.potency.rp).toBeGreaterThan(0);
  });

  it("marks potency reportable once equivalence is demonstrated", () => {
    const { reference, test } = pairWithRP(4);
    const r = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    expect(r.reportable).toBe(true);
    expect(r.basis).toBe("equivalence");
    expect(r.note).toBeNull();
  });

  it("withholds reportability when the curves are not parallel", () => {
    const { reference, test } = pairWithRP(4, { slopeTest: 3.0, sd: 1 });
    const r = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    expect(r.equivalence.conclusion).toBe("non-parallel");
    expect(r.reportable).toBe(false);
  });

  it("can use a profile-likelihood interval for the potency ratio", () => {
    const { reference, test } = pairWithRP(4);
    const wald = relativePotency(reference, test, { bounds: LOOSE_BOUNDS });
    const prof = relativePotency(reference, test, { bounds: LOOSE_BOUNDS, intervals: "profile" });
    expect(prof.potency.method).toBe("profile");
    expect(prof.potency.profile.loBounded).toBe(true);
    // Well-conditioned curves: the two should broadly agree.
    expect(prof.potency.ci.lo).toBeCloseTo(wald.potency.ci.lo, 1);
    expect(prof.potency.rp).toBeCloseTo(wald.potency.rp, 10);
  });

  it("recovers the ratio exactly from noiseless data, under every weighting", () => {
    // The clean check on the joint model itself: with no noise there is a
    // single exact answer, and the fit must find it whatever metric it
    // minimises. Pins the shift parameterisation, the group indexing and the
    // log10 handling all at once.
    const exact = (ec50) => {
      const xData = [], yData = [];
      for (let i = 0; i < 8; i++) {
        const x = Math.pow(10, -1 + (4 * i) / 7);
        for (let r = 0; r < 3; r++) {
          xData.push(x);
          yData.push(eval4PLExported(x, [10, 1.2, ec50, 100]));
        }
      }
      return { xData, yData };
    };
    for (const weighting of ["none", "1/Y", "1/Y^2"]) {
      const r = relativePotency(exact(10), exact(2.5), { bounds: LOOSE_BOUNDS, weighting });
      expect(r.potency.rp).toBeCloseTo(4, 6);
    }
  });

  it("applies one weight vector across both curves", () => {
    // The F-test compares two sums of squares, so both fits have to be
    // minimising the same thing. A single weight vector spanning the combined
    // dataset is what guarantees that.
    const reference = curve({ params: [10, 1.2, 10, 100], sd: 3, seed: 5 });
    const test = curve({ params: [10, 1.2, 2.5, 100], sd: 3, seed: 6 });
    const r = relativePotency(reference, test, { bounds: LOOSE_BOUNDS, weighting: "1/Y^2" });

    expect(r.pair.weighting.applied).toBe("1/Y^2");
    expect(r.pair.weights).toHaveLength(reference.xData.length + test.xData.length);
    expect(r.pair.weights.every(w => w > 0)).toBe(true);
    expect(r.fTest.applicable).toBe(true);
    expect(r.fTest.ssrSeparate).toBeLessThanOrEqual(r.fTest.ssrParallel + 1e-12);
  });

  it("recovers the ratio under weighting across repeated assays", () => {
    // Weighting is a variance model, not a bias correction: on any single
    // plate 1/Y^2 concentrates leverage on the baseline points and the potency
    // estimate scatters more than the unweighted one. What must hold is that
    // it is centred, so this checks the median across assays rather than
    // pinning one draw.
    const estimates = [];
    for (let s = 0; s < 9; s++) {
      const reference = curve({ params: [10, 1.2, 10, 100], sd: 3, seed: 100 + s });
      const test = curve({ params: [10, 1.2, 2.5, 100], sd: 3, seed: 200 + s });
      const r = relativePotency(reference, test, { bounds: LOOSE_BOUNDS, weighting: "1/Y^2" });
      estimates.push(r.potency.rp);
    }
    estimates.sort((a, b) => a - b);
    const median = estimates[Math.floor(estimates.length / 2)];
    expect(median).toBeGreaterThan(3.2);
    expect(median).toBeLessThan(5);
  });
});

describe("relativePotency — input validation", () => {
  const good = curve({ params: [0, 1.2, 10, 100] });

  it("rejects missing data", () => {
    expect(relativePotency(null, good).ok).toBe(false);
    expect(relativePotency(good, null).reason).toMatch(/missing xData/i);
    expect(relativePotency({}, good).reason).toMatch(/reference/i);
  });

  it("rejects mismatched lengths", () => {
    const bad = { xData: [1, 2, 3, 4], yData: [1, 2, 3] };
    expect(relativePotency(bad, good).reason).toMatch(/4 concentrations and 3 responses/);
  });

  it("rejects a curve with too few distinct concentrations", () => {
    const thin = { xData: [1, 1, 10, 10, 100, 100], yData: [1, 2, 3, 4, 5, 6] };
    const r = relativePotency(thin, good);
    expect(r.ok).toBe(false);
    expect(r.reason).toMatch(/3 distinct concentrations/);
  });

  it("names which curve is at fault", () => {
    const thin = { xData: [1, 10, 100], yData: [1, 2, 3] };
    expect(relativePotency(good, thin).reason).toMatch(/^The test curve/);
    expect(relativePotency(thin, good).reason).toMatch(/^The reference curve/);
  });
});

describe("relativePotencyFrom", () => {
  it("returns null for a missing pair rather than throwing", () => {
    expect(relativePotencyFrom(null)).toBeNull();
    expect(parallelismFTest(null)).toBeNull();
    expect(parallelismEquivalence(null)).toBeNull();
  });

  it("reads the ratio straight off the fitted shift parameter", () => {
    const { reference, test } = pairWithRP(4);
    const pair = fitCurvePair(reference, test);
    const p = relativePotencyFrom(pair);
    expect(p.logRP).toBeCloseTo(pair.parallel.params[4], 12);
    expect(p.rp).toBeCloseTo(Math.pow(10, pair.parallel.params[4]), 12);
    expect(p.seLog).toBeCloseTo(pair.parallel.se[4], 12);
  });
});
