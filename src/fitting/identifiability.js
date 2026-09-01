// ── Identifiability diagnostics ───────────────────────────────────
// A converged fit always reports four parameters with four standard errors.
// It does not tell you that two of them were never actually measured.
//
// The usual case: the dose range stops before the response flattens, so the
// "bottom plateau" is not an observation but an extrapolation -- the optimiser
// inferred where the curve WOULD level off, from the curvature it could see.
// The Wald standard error on that parameter is computed from the same
// linearisation as every other, so it comes back looking perfectly respectable,
// and the plateau gets read off the table as though it were measured.
//
// Worse, a plateau the data do not reach is not independent of the potency: the
// fitter can slide the asymptote out and the EC50 along with it and barely
// change the curve through the observed points. So the EC50's interval is
// understated too, and the correlation matrix -- not the standard errors --
// is what reveals it.
//
// These checks are geometric, computed from the fitted curve and the dose
// range, and are deliberately independent of the covariance machinery: they
// still work when the covariance matrix is singular, which is precisely when
// something has gone wrong.

// Fraction of the total response span that must remain untraversed for an
// asymptote to count as bracketed by the data. At 5% the curve is flat enough
// that the plateau is genuinely observed rather than inferred.
const PLATEAU_TOLERANCE = 0.05;
const PLATEAU_SEVERE = 0.15;

// Correlation above which two parameters are not separately determined.
const CORRELATION_WARN = 0.99;
const CORRELATION_SEVERE = 0.999;

// Fold-width of the EC50 confidence interval (hi/lo) worth remarking on.
const EC50_FOLD_WARN = 10;
const EC50_FOLD_SEVERE = 100;

const SLOT_NAMES_4PL = ["A (x -> 0 asymptote)", "Hill slope", "EC50", "D (x -> inf asymptote)"];
const SLOT_NAMES_5PL = ["Bottom", "Hill slope", "EC50", "Top", "S (asymmetry)"];

const HILL_INDEX = 1;
const EC50_INDEX = 2;

/**
 * Where a dose sits on the curve, as a fraction of the full response span.
 *
 * Computed from the model's own limits at x -> 0 and x -> infinity rather than
 * from the parameter vector, so it does not need to know which slot holds
 * which asymptote -- a question whose answer depends on the sign of the Hill
 * slope and differs between the 4PL and 5PL parameterisations.
 *
 * Returns 0 at the zero-dose asymptote and 1 at the infinite-dose one.
 */
function spanFraction(modelFn, params, x, yZero, span) {
  return (modelFn(x, params) - yZero) / span;
}

/**
 * Lowest dose at which the curve has travelled `target` of its span.
 *
 * Bisection in log-dose, because the answer spans orders of magnitude and the
 * relationship is a sigmoid in log space. Used to turn "your bottom plateau is
 * not bracketed" into "test about 1.8 decades lower", which is the form the
 * finding is actually actionable in.
 */
function doseAtFraction(modelFn, params, target, yZero, span, anchor) {
  let lo = anchor * 1e-12, hi = anchor * 1e12;
  const fLo = spanFraction(modelFn, params, lo, yZero, span);
  const fHi = spanFraction(modelFn, params, hi, yZero, span);
  if (!isFinite(fLo) || !isFinite(fHi)) return null;
  if (target <= fLo || target >= fHi) return null; // outside what bisection can reach
  for (let i = 0; i < 200; i++) {
    const mid = Math.sqrt(lo * hi);
    if (spanFraction(modelFn, params, mid, yZero, span) < target) lo = mid;
    else hi = mid;
  }
  return Math.sqrt(lo * hi);
}

/**
 * Flag parameters the data cannot actually pin down.
 *
 * @param {object} fit    A converged fit result (params, se, ci, correlation).
 * @param {number[]} xData Concentrations the fit was given.
 * @param {function} modelFn The model evaluator matching `fit.params`.
 * @param {object} [options]
 * @param {boolean} [options.is5PL] Selects the parameter naming only.
 * @returns {{ok, worst, warnings, geometry}}
 */
export function identifiabilityWarnings(fit, xData, modelFn, options = {}) {
  const { is5PL = false } = options;
  const warnings = [];
  const names = is5PL ? SLOT_NAMES_5PL : SLOT_NAMES_4PL;
  const add = (code, severity, parameter, message, value = null) =>
    warnings.push({ code, severity, parameter, message, value });

  const empty = { ok: true, worst: "none", warnings: [], geometry: null };
  if (!fit || !fit.params || !xData || xData.length === 0) return empty;

  const positive = xData.filter(x => x > 0);
  if (positive.length === 0) return empty;
  const xMin = Math.min(...positive);
  const xMax = Math.max(...positive);
  const distinctDoses = new Set(xData).size;

  const { params, se = null, ci = null, correlation = null } = fit;

  // A slot with a null standard error was fixed by the caller, not estimated.
  // Fixed parameters carry no identifiability question -- they were asserted.
  const wasEstimated = (i) => !se || se[i] !== null;

  // ── Geometry: does the dose range bracket the curve? ────────────
  const yZero = modelFn(0, params);
  const yInf = modelFn(Infinity, params);
  const span = yInf - yZero;

  let geometry = null;
  if (isFinite(yZero) && isFinite(yInf) && Math.abs(span) > 0) {
    const fracLow = spanFraction(modelFn, params, xMin, yZero, span);
    const fracHigh = spanFraction(modelFn, params, xMax, yZero, span);
    // Distance still to travel to each asymptote, as a fraction of the span.
    const gapLow = Math.abs(fracLow);
    const gapHigh = Math.abs(1 - fracHigh);

    // Which caller-visible slot holds each asymptote. The sign of the Hill
    // slope decides this, so it is resolved by value rather than assumed.
    const zeroSlot = Math.abs(params[0] - yZero) <= Math.abs(params[3] - yZero) ? 0 : 3;
    const infSlot = zeroSlot === 0 ? 3 : 0;

    const ec50 = params[EC50_INDEX];
    geometry = {
      xMin, xMax, distinctDoses,
      yZero, yInf, span,
      zeroDoseSlot: zeroSlot, infDoseSlot: infSlot,
      gapLow, gapHigh,
      decadesBelowEC50: ec50 > 0 ? Math.log10(ec50 / xMin) : null,
      decadesAboveEC50: ec50 > 0 ? Math.log10(xMax / ec50) : null,
      ec50InRange: ec50 >= xMin && ec50 <= xMax,
    };

    const plateau = (gap, slot, endLabel, doseLabel, extendFn) => {
      if (gap <= PLATEAU_TOLERANCE || !wasEstimated(slot)) return;
      const severity = gap >= PLATEAU_SEVERE ? "severe" : "warning";
      const needed = extendFn();
      const advice = needed !== null
        ? ` Extending the dose range about ${needed.toFixed(1)} ${
            needed === 1 ? "decade" : "decades"} ${doseLabel} would bracket it.`
        : "";
      add(
        `${endLabel}-plateau-unobserved`, severity, slot,
        `${names[slot]} is extrapolated, not measured: at the ${
          doseLabel === "lower" ? "lowest" : "highest"} dose tested the curve is ` +
        `still ${(gap * 100).toFixed(0)}% of the response span away from it. ` +
        `Its confidence interval reflects only the curvature the data show, so ` +
        `it understates the real uncertainty.${advice}`,
        gap,
      );
    };

    plateau(gapLow, zeroSlot, "bottom", "lower", () => {
      const target = doseAtFraction(modelFn, params, PLATEAU_TOLERANCE, yZero, span, Math.abs(params[EC50_INDEX]) || xMin);
      return target && target > 0 ? Math.log10(xMin / target) : null;
    });
    plateau(gapHigh, infSlot, "top", "higher", () => {
      const target = doseAtFraction(modelFn, params, 1 - PLATEAU_TOLERANCE, yZero, span, Math.abs(params[EC50_INDEX]) || xMax);
      return target && target > 0 ? Math.log10(target / xMax) : null;
    });

    if (!geometry.ec50InRange && wasEstimated(EC50_INDEX)) {
      add(
        "ec50-outside-dose-range", "severe", EC50_INDEX,
        `The fitted EC50 (${ec50.toPrecision(3)}) lies outside the tested dose ` +
        `range (${xMin.toPrecision(3)} to ${xMax.toPrecision(3)}). It is an ` +
        `extrapolation from the shoulder of the curve, not a measured potency.`,
        ec50,
      );
    }
  }

  // ── Design: enough distinct doses to determine the shape? ───────
  const nFree = se ? se.filter(v => v !== null).length : params.length;
  if (distinctDoses < nFree) {
    add(
      "too-few-doses", "severe", null,
      `${distinctDoses} distinct concentrations cannot determine ${nFree} free ` +
      `parameters. Fix one or more parameters, or use a simpler model.`,
      distinctDoses,
    );
  } else if (distinctDoses === nFree) {
    add(
      "saturated-design", "warning", null,
      `${distinctDoses} distinct concentrations for ${nFree} free parameters ` +
      `leaves no degrees of freedom to check the shape: the model can pass ` +
      `through every dose mean regardless of whether it is the right model.`,
      distinctDoses,
    );
  }

  // ── Correlation: are the parameters separable from each other? ──
  if (correlation) {
    for (let i = 0; i < correlation.length; i++) {
      for (let j = i + 1; j < correlation.length; j++) {
        const r = correlation[i][j];
        if (!isFinite(r)) continue;
        const a = Math.abs(r);
        if (a < CORRELATION_WARN) continue;
        add(
          "parameter-correlation", a >= CORRELATION_SEVERE ? "severe" : "warning", null,
          `Two fitted parameters are ${(a * 100).toFixed(2)}% correlated ` +
          `(r = ${r.toFixed(4)}). The data cannot separate them: a change in one ` +
          `can be absorbed almost entirely by the other, so their individual ` +
          `confidence intervals are not independently meaningful.`,
          r,
        );
      }
    }
  }

  // ── Intervals: is the potency itself pinned down? ───────────────
  if (ci && ci[EC50_INDEX] && wasEstimated(EC50_INDEX)) {
    const { lo, hi } = ci[EC50_INDEX];
    if (isFinite(lo) && isFinite(hi) && lo > 0) {
      const fold = hi / lo;
      if (fold >= EC50_FOLD_WARN) {
        add(
          "wide-ec50-interval", fold >= EC50_FOLD_SEVERE ? "severe" : "warning", EC50_INDEX,
          `The 95% confidence interval on the EC50 spans a ${fold.toPrecision(3)}-fold ` +
          `range (${lo.toPrecision(3)} to ${hi.toPrecision(3)}). The potency is ` +
          `bounded rather than determined.`,
          fold,
        );
      }
    }
  }

  if (ci && ci[HILL_INDEX] && wasEstimated(HILL_INDEX)) {
    const { lo, hi } = ci[HILL_INDEX];
    if (isFinite(lo) && isFinite(hi) && lo <= 0 && hi >= 0) {
      add(
        "slope-includes-zero", "severe", HILL_INDEX,
        `The confidence interval on the Hill slope includes zero ` +
        `(${lo.toPrecision(3)} to ${hi.toPrecision(3)}), so the data are ` +
        `consistent with no dose-response at all. Every other parameter is ` +
        `conditional on a relationship that has not been established.`,
        null,
      );
    }
  }

  // A singular covariance matrix is itself the finding: the normal equations
  // were rank-deficient, which is what "not identifiable" means exactly.
  if (fit.cov === null && fit.dof === null) {
    add(
      "no-inference", "severe", null,
      `Standard errors could not be computed: the normal equations at the ` +
      `solution are singular or there are no residual degrees of freedom. The ` +
      `parameters are not jointly identifiable from this data.`,
      null,
    );
  }

  const worst = warnings.some(w => w.severity === "severe") ? "severe"
    : warnings.length ? "warning" : "none";
  return { ok: warnings.length === 0, worst, warnings, geometry };
}
