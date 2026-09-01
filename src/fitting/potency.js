// ── Relative potency and parallelism ──────────────────────────────
// A bioassay rarely measures potency in absolute units. What it measures is
// how much of a test article is needed to produce the same response as a known
// amount of a reference standard -- a ratio, reported as relative potency.
//
// That ratio only exists if the two curves have the same SHAPE. If the test
// article's dose-response is genuinely a different curve -- a different slope,
// a different maximum -- then there is no single number by which one dose axis
// maps onto the other, and the "potency ratio" you compute is an artefact of
// which part of the curve you happened to look at. Establishing parallelism is
// therefore a precondition for reporting potency, not a footnote to it.
//
// Under parallelism the two curves differ by a pure horizontal shift on the
// log-dose axis, which is why the model here fits log10(RP) directly as a
// parameter: the quantity of interest gets its own standard error and its own
// confidence interval, rather than being reconstructed afterwards from a ratio
// of two EC50s and their correlated errors.
//
//   RP = EC50(reference) / EC50(test)
//
// so RP > 1 means the test article is more potent -- less of it is needed.
//
// ── Two ways to judge parallelism, and why the default matters ────
//
// The classical approach is a significance test: fit the curves with and
// without a common shape, and F-test the difference. Its logic runs backwards.
// The null hypothesis is parallelism, so FAILING to reject it is what passes
// the assay -- which means a noisy, poorly-run assay passes easily, and a very
// precise one fails for deviations far too small to matter. It rewards bad
// data.
//
// USP <1032> recommends equivalence testing instead: state in advance how
// different the shapes are ALLOWED to be, then require the confidence interval
// on the observed difference to fall entirely inside those bounds. A precise
// assay now passes more easily, which is the right incentive, and the decision
// is tied to a difference that was judged relevant beforehand rather than to
// whatever the assay could resolve.
//
// The bounds are product- and assay-specific -- they come from historical data
// on the reference standard, not from this module. Without them, no
// equivalence verdict is possible, and this reports that rather than
// substituting a default.

import { model4PL } from "./models.js";
import { levenbergMarquardt, fitModel, jacobian } from "./lm.js";
import { parameterCovariance, parameterIntervals } from "./inference.js";
import { buildWeights, weightsConverged } from "./weights.js";
import { profileInterval } from "./resample.js";
import { fPValue, tInv } from "./distributions.js";

// Parameter layouts used below.
//   separate: [A_r, B_r, logC_r, D_r, A_t, B_t, logC_t, D_t]
//   parallel: [A, B, logC_ref, D, logRP]
const SEPARATE_K = 8;
const PARALLEL_K = 5;

const SHAPE_PARAMS = [
  { key: "slope", label: "Hill slope", ref: 1, test: 5 },
  { key: "lower", label: "Lower asymptote", ref: 0, test: 4 },
  { key: "upper", label: "Upper asymptote", ref: 3, test: 7 },
];

/**
 * Build the joint models over a combined test/reference dataset.
 *
 * Both models are indexed by observation number rather than by concentration:
 * the optimiser is handed 0..N-1 and the model closes over the actual doses and
 * the group labels. That is what lets a single least-squares problem span two
 * curves without changing the optimiser's (x, params) contract.
 */
function makeJointModels(reference, test) {
  const xs = [...reference.xData, ...test.xData];
  const ys = [...reference.yData, ...test.yData];
  const isTest = [
    ...reference.xData.map(() => 0),
    ...test.xData.map(() => 1),
  ];
  const index = xs.map((_, i) => i);

  // Every parameter free: the two curves share nothing but a residual variance.
  const separate = (i, p) => {
    const o = isTest[i] ? 4 : 0;
    return model4PL(xs[i], [p[o], p[o + 1], Math.pow(10, p[o + 2]), p[o + 3]]);
  };

  // Common shape, curves separated by a horizontal shift of log10(RP).
  const parallel = (i, [A, B, logCref, D, logRP]) => {
    const logC = logCref - (isTest[i] ? logRP : 0);
    return model4PL(xs[i], [A, B, Math.pow(10, logC), D]);
  };

  return { xs, ys, isTest, index, separate, parallel, n: xs.length };
}

/** [A, B, C, D] and [D, -B, C, A] are the same curve; pin the slope positive. */
function canon4(A, B, logC, D) {
  return B < 0 ? [D, -B, logC, A] : [A, B, logC, D];
}

function inferenceFor(index, ys, modelFn, params, ssr, weights, alpha) {
  const J = jacobian(index, modelFn, params);
  const cov = parameterCovariance(J, ssr, weights);
  if (!cov) return { cov: null, se: null, ci: null, dof: null, sigma2: null, tCrit: null };
  const { se, ci, tCrit } = parameterIntervals(params, cov.cov, cov.dof, alpha);
  return { cov: cov.cov, se, ci, dof: cov.dof, sigma2: cov.sigma2, tCrit };
}

/**
 * Fit test and reference together, both with and without a common shape.
 *
 * Weights, when requested, are derived once from the separate fit and then held
 * fixed for the parallel fit. That is not a shortcut: the F-test below compares
 * two sums of squares, and they are only comparable if both were minimised
 * under the same metric. Letting each fit choose its own weights would compare
 * numbers in different units.
 */
export function fitCurvePair(reference, test, options = {}) {
  const { weighting = "none", alpha = 0.05, maxWeightIterations = 10 } = options;
  const joint = makeJointModels(reference, test);
  const { index, ys, n } = joint;

  // Start from independent single-curve fits, which already handle multi-start
  // and canonicalisation.
  const refFit = fitModel(reference.xData, reference.yData, model4PL, false);
  const testFit = fitModel(test.xData, test.yData, model4PL, false);
  if (!refFit || !testFit) return null;

  const [Ar, Br, , Dr] = refFit.params;
  const [At, Bt, , Dt] = testFit.params;
  const logCr = Math.log10(refFit.params[2]);
  const logCt = Math.log10(testFit.params[2]);

  const separateInit = [Ar, Br, logCr, Dr, At, Bt, logCt, Dt];
  const parallelInit = [
    (Ar + At) / 2, (Br + Bt) / 2, logCr, (Dr + Dt) / 2, logCr - logCt,
  ];

  const runSeparate = (w) => levenbergMarquardt(index, ys, joint.separate, separateInit, { weights: w });
  const runParallel = (w, init) => levenbergMarquardt(index, ys, joint.parallel, init, { weights: w });

  let weights = null;
  let weightWarning = null;
  let separate = runSeparate(null);

  if (weighting && weighting !== "none") {
    for (let iter = 0; iter < maxWeightIterations; iter++) {
      const predicted = index.map(i => joint.separate(i, separate.params));
      const built = buildWeights(weighting, { yPred: predicted, xData: joint.xs, yData: ys });
      weightWarning = built.warning;
      if (!built.weights) { weights = null; break; }
      if (weightsConverged(weights, built.weights)) break;
      weights = built.weights;
      separate = runSeparate(weights);
    }
  }

  let parallel = runParallel(weights, parallelInit);
  // A second start from the separate fit's own shape averages, in case the
  // first landed badly; the parallel model is the more constrained of the two
  // and benefits from the extra attempt.
  const alt = runParallel(weights, [
    separate.params[0], separate.params[1], separate.params[2], separate.params[3],
    separate.params[2] - separate.params[6],
  ]);
  if (alt.ssr < parallel.ssr) parallel = alt;

  // Canonicalise both so that "lower asymptote" means the same thing in each,
  // and so the shape comparisons below line up parameter for parameter.
  const sp = separate.params;
  const separateParams = [
    ...canon4(sp[0], sp[1], sp[2], sp[3]),
    ...canon4(sp[4], sp[5], sp[6], sp[7]),
  ];
  const pp = parallel.params;
  const parallelParams = [...canon4(pp[0], pp[1], pp[2], pp[3]), pp[4]];

  return {
    joint, weights, weightWarning,
    weighting: { requested: weighting || "none", applied: weights ? weighting : "none" },
    separate: {
      params: separateParams, ssr: separate.ssr, converged: separate.converged,
      k: SEPARATE_K, n,
      ...inferenceFor(index, ys, joint.separate, separateParams, separate.ssr, weights, alpha),
    },
    parallel: {
      params: parallelParams, ssr: parallel.ssr, converged: parallel.converged,
      k: PARALLEL_K, n,
      ...inferenceFor(index, ys, joint.parallel, parallelParams, parallel.ssr, weights, alpha),
    },
    singleCurveFits: { reference: refFit, test: testFit },
  };
}

/**
 * Classical parallelism test: does dropping the common-shape constraint help?
 *
 * Reported because it is what most existing SOPs still specify, and flagged
 * because its logic is inverted. Passing means "we could not prove the curves
 * differ", which is easiest to achieve with a poor assay. Prefer the
 * equivalence result whenever acceptance criteria exist.
 */
export function parallelismFTest(pair, alpha = 0.05) {
  if (!pair) return null;
  const { separate, parallel } = pair;
  const dfNum = separate.k - parallel.k;
  const dfDen = separate.n - separate.k;
  if (dfNum <= 0 || dfDen <= 0) {
    return {
      applicable: false,
      reason: `Not enough data to compare the two models (${separate.n} points, ` +
        `${separate.k} parameters in the unconstrained fit).`,
      F: null, pValue: null, conclusion: null,
    };
  }

  const F = ((parallel.ssr - separate.ssr) / dfNum) / (separate.ssr / dfDen);
  const pValue = fPValue(F, dfNum, dfDen);
  return {
    applicable: true, reason: null,
    F, pValue, dfNum, dfDen, alpha,
    ssrParallel: parallel.ssr, ssrSeparate: separate.ssr,
    significant: pValue < alpha,
    conclusion: pValue < alpha ? "non-parallel" : "parallel",
    caveat:
      "A non-significant result is not evidence of parallelism -- it is a " +
      "failure to detect non-parallelism, which an imprecise assay achieves " +
      "easily. Use equivalence bounds where acceptance criteria exist.",
  };
}

/**
 * Equivalence-based parallelism, per USP <1032>.
 *
 * For each shape parameter, form the difference between test and reference and
 * its confidence interval, then require that interval to lie entirely within
 * the pre-specified equivalence bounds. Because the two curves are fitted to
 * disjoint data, their estimates are independent and the variance of the
 * difference is the sum of the variances -- but the residual variance is
 * pooled across both, which is why this reads the differences off the joint
 * unconstrained fit rather than off two separate ones.
 *
 * The interval is a (1 - 2*alpha) interval, not (1 - alpha): the decision is
 * two one-sided tests at alpha each, so at alpha = 0.05 the conventional 90%
 * interval is what must fall inside the bounds.
 *
 * @param {object} bounds Per-parameter acceptance criteria, keyed by
 *   "slope" / "lower" / "upper". Each is either a symmetric half-width (a
 *   number) or an explicit [lo, hi] pair, in that parameter's own units.
 *   Parameters with no entry are not assessed.
 */
export function parallelismEquivalence(pair, bounds = null, alpha = 0.05) {
  if (!pair) return null;
  const { separate } = pair;

  if (!separate.cov || separate.dof == null) {
    return {
      applicable: false, conclusion: "not-estimable",
      reason: "Standard errors are unavailable for the unconstrained fit, so " +
        "no interval can be placed on the shape differences.",
      parameters: [],
    };
  }

  // Two one-sided tests at alpha => a (1 - 2 alpha) two-sided interval.
  const tCrit = tInv(alpha, separate.dof);

  const parameters = SHAPE_PARAMS.map(({ key, label, ref, test }) => {
    const estimate = separate.params[test] - separate.params[ref];
    const varDiff = separate.cov[test][test] + separate.cov[ref][ref]
      - 2 * separate.cov[test][ref];
    const se = varDiff > 0 ? Math.sqrt(varDiff) : null;
    const ci = se == null ? null : { lo: estimate - tCrit * se, hi: estimate + tCrit * se };

    const raw = bounds ? bounds[key] : undefined;
    let limit = null;
    if (typeof raw === "number") limit = { lo: -Math.abs(raw), hi: Math.abs(raw) };
    else if (Array.isArray(raw) && raw.length === 2) limit = { lo: raw[0], hi: raw[1] };

    let passes = null;
    if (limit && ci) passes = ci.lo >= limit.lo && ci.hi <= limit.hi;

    return {
      key, label, estimate, se, ci, bounds: limit, passes,
      reference: separate.params[ref], test: separate.params[test],
    };
  });

  const assessed = parameters.filter(p => p.bounds !== null);
  if (assessed.length === 0) {
    return {
      applicable: false, conclusion: "no-criteria",
      reason:
        "No equivalence bounds were supplied. Acceptance criteria for slope " +
        "and asymptote differences are product- and assay-specific -- they are " +
        "derived from historical reference-standard data during assay " +
        "validation, so there is no defensible default to fall back on.",
      parameters, confidenceLevel: 1 - 2 * alpha, alpha,
    };
  }

  const undecided = assessed.filter(p => p.passes === null);
  const failed = assessed.filter(p => p.passes === false);
  return {
    applicable: true, reason: null,
    conclusion: undecided.length ? "not-estimable" : failed.length ? "non-parallel" : "parallel",
    parameters, assessed: assessed.map(p => p.key),
    failed: failed.map(p => p.key),
    confidenceLevel: 1 - 2 * alpha, alpha, dof: separate.dof,
  };
}

/**
 * Relative potency with a confidence interval.
 *
 * Read off the parallel model's shift parameter, which is fitted as
 * log10(RP) directly. That matters for the interval: a ratio of potencies is
 * multiplicative, so its sampling distribution is far closer to symmetric in
 * log space, and exponentiating the log-scale endpoints gives an interval that
 * is asymmetric on the ratio scale and cannot contain a negative potency.
 * Forming it as RP +/- t*SE would do neither.
 *
 * @param {object} [options]
 * @param {"wald"|"profile"} [options.intervals] Profile likelihood follows the
 *   real shape of the likelihood; worth the refits when the curves are not
 *   well bracketed.
 */
export function relativePotencyFrom(pair, options = {}) {
  const { alpha = 0.05, intervals = "wald" } = options;
  if (!pair) return null;
  const { parallel, joint, weights } = pair;
  const LOG_RP = 4;

  const logRP = parallel.params[LOG_RP];
  const rp = Math.pow(10, logRP);
  const seLog = parallel.se ? parallel.se[LOG_RP] : null;

  let ci = null;
  let method = "none";
  if (parallel.ci && parallel.ci[LOG_RP]) {
    const w = parallel.ci[LOG_RP];
    ci = { lo: Math.pow(10, w.lo), hi: Math.pow(10, w.hi) };
    method = "wald";
  }

  let profile = null;
  if (intervals === "profile") {
    const r = profileInterval({
      xData: joint.index, yData: joint.ys, modelFn: joint.parallel,
      params: parallel.params, index: LOG_RP, ssr: parallel.ssr,
      alpha, weights, scale: seLog,
    });
    profile = {
      lo: r.lo == null ? null : Math.pow(10, r.lo),
      hi: r.hi == null ? null : Math.pow(10, r.hi),
      loBounded: r.loBounded, hiBounded: r.hiBounded,
    };
    if (r.loBounded && r.hiBounded) { ci = { lo: profile.lo, hi: profile.hi }; method = "profile"; }
  }

  return {
    rp, logRP, seLog, ci, method, profile, alpha,
    // Potency is conventionally reported as a percentage of the reference.
    percent: rp * 100,
    percentCi: ci ? { lo: ci.lo * 100, hi: ci.hi * 100 } : null,
    ec50Reference: Math.pow(10, parallel.params[2]),
    ec50Test: Math.pow(10, parallel.params[2] - logRP),
  };
}

/**
 * Full relative-potency analysis: fit, judge parallelism, then report potency.
 *
 * The ordering is the point. `potency` is populated regardless, because a
 * number the analyst cannot see is a number they will recompute by hand, but
 * `reportable` says whether parallelism was actually established -- and it is
 * false, not true, when no acceptance criteria were supplied. An unjudged
 * assay is not a passed one.
 */
export function relativePotency(reference, test, options = {}) {
  const { alpha = 0.05, bounds = null, weighting = "none", intervals = "wald" } = options;

  const problem = validatePair(reference, test);
  if (problem) return { ok: false, reason: problem };

  const pair = fitCurvePair(reference, test, { weighting, alpha });
  if (!pair) return { ok: false, reason: "Neither curve could be fitted." };

  const fTest = parallelismFTest(pair, alpha);
  const equivalence = parallelismEquivalence(pair, bounds, alpha);
  const potency = relativePotencyFrom(pair, { alpha, intervals });

  const reportable = equivalence.applicable && equivalence.conclusion === "parallel";

  return {
    ok: true, reason: null,
    pair, fTest, equivalence, potency, reportable,
    // Which verdict a reader should act on, and why.
    basis: equivalence.applicable ? "equivalence" : "none",
    note: equivalence.applicable
      ? null
      : `${equivalence.reason} The F-test result is reported for reference, but ` +
        `it cannot establish parallelism on its own.`,
  };
}

function validatePair(reference, test) {
  for (const [name, d] of [["reference", reference], ["test", test]]) {
    if (!d || !Array.isArray(d.xData) || !Array.isArray(d.yData)) {
      return `The ${name} curve is missing xData/yData.`;
    }
    if (d.xData.length !== d.yData.length) {
      return `The ${name} curve has ${d.xData.length} concentrations and ` +
        `${d.yData.length} responses.`;
    }
    if (new Set(d.xData).size < 4) {
      return `The ${name} curve has ${new Set(d.xData).size} distinct ` +
        `concentrations; a 4PL needs at least 4.`;
    }
  }
  return null;
}
