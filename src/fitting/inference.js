// ── Parameter uncertainty ─────────────────────────────────────────
// Standard errors and confidence intervals for a converged least-squares fit.
//
// This module deliberately takes a Jacobian matrix rather than a model
// function: inference is a property of the linearisation at the solution, not
// of the model, and keeping it that way avoids an import cycle with lm.js.

import { matInverse } from "./linalg.js";
import { tInv } from "./distributions.js";

/**
 * Covariance matrix of the fitted parameters.
 *
 * At the optimum the model is locally linear, so the usual Gauss-Newton
 * approximation applies:
 *
 *   cov = sigma^2 * (J^T W J)^-1,    sigma^2 = SSR_w / (n - p)
 *
 * where W is diagonal with the fitting weights (all 1 when unweighted) and p
 * is the number of FREE parameters. Note the divisor is n - p, not n: the
 * residuals have already had p degrees of freedom removed by the fit.
 *
 * @param {number[][]} J   Jacobian at the solution, n x p, in FITTING space.
 * @param {number} ssr     Sum of squared residuals, weighted if W is given.
 * @param {number[]} [weights] Per-observation weights; omit for unweighted.
 * @returns {{cov: number[][], dof: number, sigma2: number, syx: number}|null}
 *   null when the fit cannot support inference: no residual degrees of
 *   freedom, or a singular J^T W J (parameters not identifiable from the data).
 */
export function parameterCovariance(J, ssr, weights = null) {
  const n = J.length;
  if (n === 0) return null;
  const p = J[0].length;
  const dof = n - p;
  if (dof <= 0) return null;

  // J^T W J, accumulated directly so the weights never need a dense matrix.
  const JtWJ = Array.from({ length: p }, () => new Array(p).fill(0));
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    if (!isFinite(w) || w < 0) return null;
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) {
        JtWJ[a][b] += w * J[i][a] * J[i][b];
      }
    }
  }

  const inv = matInverse(JtWJ);
  if (!inv) return null;

  const sigma2 = ssr / dof;
  const cov = inv.map(row => row.map(v => sigma2 * v));
  return { cov, dof, sigma2, syx: Math.sqrt(sigma2) };
}

/**
 * Standard errors and Wald confidence intervals from a covariance matrix.
 *
 * The interval is params[i] +/- t(alpha/2, dof) * SE. This is exact only for a
 * model linear in its parameters; for a nonlinear model it is a linearisation
 * at the solution and is optimistic when a parameter is poorly determined --
 * which is the usual case for a plateau the data do not reach. Forming the
 * EC50 interval in log space (see lm.js) removes the worst of that skew.
 *
 * @returns {{se: (number|null)[], ci: ({lo:number,hi:number}|null)[], tCrit: number}}
 */
export function parameterIntervals(params, cov, dof, alpha = 0.05) {
  const tCrit = tInv(alpha / 2, dof);
  const se = [];
  const ci = [];
  for (let i = 0; i < params.length; i++) {
    const v = cov[i][i];
    // A negative or non-finite variance means the normal equations were too
    // ill-conditioned to trust. Report "unknown" rather than a bogus number.
    if (!isFinite(v) || v < 0) {
      se.push(null);
      ci.push(null);
      continue;
    }
    const s = Math.sqrt(v);
    se.push(s);
    ci.push({ lo: params[i] - tCrit * s, hi: params[i] + tCrit * s });
  }
  return { se, ci, tCrit };
}

/**
 * Correlation matrix derived from a covariance matrix.
 *
 * Near-unit off-diagonal entries are the signature of parameters the data
 * cannot separate -- typically EC50 against a plateau the doses never reach.
 * The individual standard errors look finite in that situation; the
 * correlation is what reveals that they are not independently meaningful.
 */
export function correlationMatrix(cov) {
  const p = cov.length;
  const sd = cov.map((row, i) => Math.sqrt(row[i]));
  return Array.from({ length: p }, (_, i) =>
    Array.from({ length: p }, (_, j) => {
      const d = sd[i] * sd[j];
      return d > 0 && isFinite(d) ? cov[i][j] / d : NaN;
    }),
  );
}

/**
 * Back-transform a log10-space estimate to a linear confidence interval.
 *
 * A potency estimated as log10(EC50) has a symmetric interval in log space;
 * raising the endpoints to the power of ten gives the correct ASYMMETRIC
 * interval on the EC50 itself. Never form the linear interval as
 * `EC50 +/- t * SE` -- for a wide interval that can run negative.
 *
 * The returned `se` is the delta-method standard error on the linear scale,
 * provided for display only; the interval is the trustworthy quantity.
 */
export function backTransformLog10(logValue, logSe, tCrit) {
  const value = Math.pow(10, logValue);
  if (logSe == null || !isFinite(logSe)) {
    return { value, se: null, ci: null };
  }
  const margin = tCrit * logSe;
  return {
    value,
    se: value * Math.LN10 * logSe, // d(10^t)/dt = 10^t * ln(10)
    ci: { lo: Math.pow(10, logValue - margin), hi: Math.pow(10, logValue + margin) },
  };
}
