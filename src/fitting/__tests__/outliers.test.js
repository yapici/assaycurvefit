import { describe, it, expect } from "vitest";
import { grubbsCriticalG, grubbsTest, runGrubbsAllGroups } from "../outliers.js";

describe("grubbsCriticalG", () => {
  // Published two-sided Grubbs critical values at alpha = 0.05.
  const table = [
    [3, 1.155], [4, 1.481], [5, 1.715], [6, 1.887],
    [10, 2.290], [20, 2.709], [30, 2.908],
  ];

  it.each(table)("G_crit(n=%i, alpha=0.05) = %f", (n, expected) => {
    expect(grubbsCriticalG(n, 0.05)).toBeCloseTo(expected, 2);
  });

  it("is stricter at alpha = 0.01 than at 0.05", () => {
    expect(grubbsCriticalG(10, 0.01)).toBeGreaterThan(grubbsCriticalG(10, 0.05));
  });

  it("increases with n", () => {
    expect(grubbsCriticalG(20, 0.05)).toBeGreaterThan(grubbsCriticalG(10, 0.05));
  });

  it("is Infinity below the n = 3 minimum, so nothing can be flagged", () => {
    expect(grubbsCriticalG(2, 0.05)).toBe(Infinity);
    expect(grubbsCriticalG(0, 0.05)).toBe(Infinity);
  });
});

describe("grubbsTest", () => {
  it("flags a single clear outlier", () => {
    const r = grubbsTest([10, 11, 10.5, 10.2, 10.8, 40], 0.05);
    expect(r.outliers).toHaveLength(1);
    expect(r.outliers[0].value).toBe(40);
  });

  it("flags nothing in tight, well-behaved data", () => {
    expect(grubbsTest([10, 10.1, 10.2, 9.9, 10.05], 0.05).outliers).toHaveLength(0);
  });

  it("computes G as |v - mean| / sd", () => {
    const values = [1, 2, 3, 4, 100];
    const r = grubbsTest(values, 0.05);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sd = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1));
    const g = r.details.find(d => d.value === 100).g;
    expect(g).toBeCloseTo(Math.abs(100 - mean) / sd, 10);
  });

  it("reports one detail row per input value, with signed deviation", () => {
    const r = grubbsTest([1, 2, 3, 4, 100], 0.05);
    expect(r.details).toHaveLength(5);
    expect(r.details[0].deviation).toBeLessThan(0);
    expect(r.details[4].deviation).toBeGreaterThan(0);
  });

  it("declines to test fewer than 3 values", () => {
    expect(grubbsTest([1, 99], 0.05).outliers).toHaveLength(0);
    expect(grubbsTest([1], 0.05).details).toHaveLength(0);
  });

  it("returns no outliers when every value is identical (sd = 0)", () => {
    // Guards against the 0/0 that would otherwise produce NaN comparisons.
    expect(grubbsTest([7, 7, 7, 7], 0.05).outliers).toHaveLength(0);
  });

  it("is more permissive at a smaller alpha", () => {
    const values = [10, 11, 10.5, 10.2, 10.8, 14];
    const at05 = grubbsTest(values, 0.05).outliers.length;
    const at001 = grubbsTest(values, 0.001).outliers.length;
    expect(at001).toBeLessThanOrEqual(at05);
  });
});

describe("runGrubbsAllGroups", () => {
  // Two concentrations, 4 replicates each; one spike in the second group.
  const xData = [1, 1, 1, 1, 10, 10, 10, 10];
  const yData = [10, 10.2, 9.8, 10.1, 50, 50.5, 49.5, 200];

  it("maps flagged points back to their original row indices", () => {
    const r = runGrubbsAllGroups(xData, yData, 0.05);
    expect([...r.outlierIndices]).toEqual([7]); // the 200
    expect(r.totalOutliers).toBe(1);
  });

  it("returns one result per concentration group", () => {
    const r = runGrubbsAllGroups(xData, yData, 0.05);
    expect(r.groupResults).toHaveLength(2);
    expect(r.groupResults.map(g => g.x)).toEqual([1, 10]);
    expect(r.groupResults[0].outlierCount).toBe(0);
    expect(r.groupResults[1].outlierCount).toBe(1);
  });

  it("marks groups with n < 3 as untested rather than testing them", () => {
    const r = runGrubbsAllGroups([1, 1, 5], [10, 12, 99], 0.05);
    const g1 = r.groupResults.find(g => g.x === 1);
    const g5 = r.groupResults.find(g => g.x === 5);
    expect(g1.tested).toBe(false);
    expect(g5.tested).toBe(false);
    expect(r.totalOutliers).toBe(0);
  });

  it("finds nothing in clean data", () => {
    const r = runGrubbsAllGroups(
      [1, 1, 1, 10, 10, 10],
      [10, 10.1, 9.9, 50, 50.1, 49.9],
      0.05,
    );
    expect(r.totalOutliers).toBe(0);
  });
});
