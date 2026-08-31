// ── Grubbs' Test for Outliers ─────────────────────────────────────
// Detects a single extreme value within each concentration group.
// Assumes approximate normality within a group and requires n >= 3.

import { tInv } from "./distributions.js";
import { groupByConcentration } from "./stats.js";

export function grubbsCriticalG(n, alpha) {
  if (n < 3) return Infinity;
  const df = n - 2;
  const p = alpha / (2 * n); // two-sided Grubbs uses alpha/(2n) for t lookup
  const t = tInv(p, df);
  const tSq = t * t;
  return ((n - 1) / Math.sqrt(n)) * Math.sqrt(tSq / (n - 2 + tSq));
}

export function grubbsTest(values, alpha = 0.05) {
  const n = values.length;
  if (n < 3) return { outliers: [], details: [] };
  
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1));
  
  if (sd < 1e-15) return { outliers: [], details: [] };
  
  const gCrit = grubbsCriticalG(n, alpha);
  const details = values.map((v, i) => {
    const g = Math.abs(v - mean) / sd;
    return { index: i, value: v, g, gCrit, isOutlier: g > gCrit, deviation: v - mean };
  });
  
  return {
    outliers: details.filter(d => d.isOutlier),
    details,
    mean,
    sd,
    gCrit,
  };
}

// Run Grubbs on all concentration groups, return per-group results + flat outlier index set
export function runGrubbsAllGroups(xData, yData, alpha = 0.05) {
  const grouped = groupByConcentration(xData, yData);
  const outlierIndices = new Set();
  const groupResults = [];

  for (const g of grouped) {
    if (g.n < 3) {
      groupResults.push({ x: g.x, n: g.n, tested: false, result: null, indices: g.indices, outlierCount: 0 });
      continue;
    }
    const result = grubbsTest(g.values, alpha);
    for (const ol of result.outliers) {
      outlierIndices.add(g.indices[ol.index]);
    }
    groupResults.push({
      x: g.x,
      n: g.n,
      tested: true,
      result,
      indices: g.indices,
      outlierCount: result.outliers.length,
    });
  }

  return { outlierIndices, groupResults, totalOutliers: outlierIndices.size };
}
