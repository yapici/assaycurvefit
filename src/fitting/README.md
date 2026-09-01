# Fitting engine

The numerical core of AssayCurveFitter, extracted from `BioassayCurveFitter.jsx`
so it can be tested independently of React and the canvas renderers.

Import from the barrel, not from individual modules:

```js
import { fitModel, model4PL } from "./fitting/index.js";
```

## Modules

| Module | Contents |
| --- | --- |
| `models.js` | `model4PL`, `model5PL`, `makeConstrainedModel`, `getModelFn`, `computeBiologicalEC50` |
| `linalg.js` | `matMul`, `matTranspose`, `solveLU` — dense helpers for the normal equations |
| `lm.js` | `residuals`, `jacobian`, `levenbergMarquardt`, `estimateInitialParams`, `fitModel`, `fitConstrainedModel` |
| `stats.js` | `rSquared`, `computeAIC`/`AICc`/`BIC`, `groupByConcentration` |
| `distributions.js` | `lnGamma`, `betaIncomplete`, `tCDF`, `tInv`, `fCDF`, `fPValue`, `fInv` |
| `outliers.js` | `grubbsCriticalG`, `grubbsTest`, `runGrubbsAllGroups` |
| `inference.js` | `parameterCovariance`, `parameterIntervals`, `correlationMatrix`, `backTransformLog10` |
| `weights.js` | `buildWeights`, `weightedSSR`, `weightsConverged`, `estimateVariancePower` |
| `lackoffit.js` | `lackOfFitTest`, `describeLackOfFit` — is the model adequate? |
| `identifiability.js` | `identifiabilityWarnings` — are the parameters actually determined? |
| `resample.js` | `profileInterval`, `bootstrapIntervals`, `fitModelWithIntervals` |
| `potency.js` | `fitCurvePair`, `parallelismFTest`, `parallelismEquivalence`, `relativePotency` |

## Model conventions

The 4PL is

```
y = D + (A - D) / (1 + (x / C)^B)
```

with parameter vector `[A, B, C, D]`. **`A` is the asymptote where `(x/C)^B → 0`
and `D` the one where it → ∞.** Which end of the dose range each sits at depends
on the *sign* of the Hill slope `B`:

| | `x → 0` | `x → ∞` |
| --- | --- | --- |
| `B > 0` | `A` | `D` |
| `B < 0` | `D` | `A` |

So neither `A` nor `D` is reliably "the low plateau", and the UI's "min"/"max"
labels are accurate for only one sign. Both signs occur on ordinary data — the
app's own sample dataset fits to `B = -1.33` — because the two are mirror
solutions and the optimiser may converge on either.

The 5PL uses `(EC50 / x)` instead of `(x / C)`, which inverts the relationship
again: there a *positive* Hill puts `Bottom` at `x → 0`. The two models
therefore take differently-ordered parameter vectors with opposite slope-sign
conventions.

Both models return the true one-sided limit at `x ≤ 0` rather than a fixed
asymptote, so a zero-dose vehicle-control row is handled correctly.

The reduced models are the 4PL with parameters held fixed:

| Model | Fixed | Free |
| --- | --- | --- |
| 3PL | `B = 1` | A, C, D |
| 2PL | `A`, `D` | B, C |
| 1PL | `A`, `B`, `D` | C |

`computeBiologicalEC50` exists because in a 5PL with asymmetry `S ≠ 1` the
`EC50` parameter is no longer the concentration at half-maximal response; the
half-maximal point is found by bisection instead.

## Units and conditioning

`xData` holds **linear** concentrations, in whatever unit the user supplied.
Molar input means EC50 values around `1e-9`, while a luminescence plateau may
be `5e4` — nine orders of magnitude apart in the same parameter vector.

Two things keep that from wrecking the numerics, and both must stay:

1. **`jacobian` steps proportionally to each parameter**, `cbrt(ε)·|p|`. A
   fixed absolute step cannot serve both ends of that range: it perturbs a
   nanomolar EC50 by 10× and a large plateau by less than the resolution of
   the difference.
2. **`fitModel` optimises `log10(EC50)`**, via `withLogParams`, and
   back-transforms before returning. The returned `params` therefore still
   hold a linear EC50 — callers are unaffected. Fitting in log space makes the
   result invariant to the concentration unit, gives positivity for free
   (`10^t > 0`), and is the correct space in which to form a confidence
   interval on a potency.

A regression test asserts that the same curve fits identically from `1e-12` to
`1e3`. Do not reintroduce absolute-step numerics on the EC50.

## Optimiser contract

`levenbergMarquardt(xData, yData, modelFn, initialParams, options)` takes:

- `constrain(proposed, previous)` — filters each **proposed** step before it is
  evaluated; the caller's initial vector is taken as given. Use it to hold a
  parameter in a valid region. Callers fitting `log10(EC50)` need no constraint
  on it. Defaults to reverting non-finite entries.
- `lambdaMax` — damping ceiling. Reaching it ends the fit.

`converged` is true when either the relative SSR improvement drops below `tol`
or the damping ceiling is reached *after at least one accepted step*. The
second case matters: a fit that reaches the machine-precision floor produces no
further improvement and so can never satisfy the first criterion. A run that
never finds a single downhill step reports `converged: false`.

Do not identify a model by reference (`modelFn === model4PL`) inside the
optimiser. Models are routinely wrapped in closures for constrained fits and
for the log transform, so identity checks silently do nothing.

## Statistics

`computeAIC`/`AICc`/`BIC` use the comparison-only form that drops the additive
constant common to all models:

```
AIC  = n ln(SSR/n) + 2K
BIC  = n ln(SSR/n) + K ln(n)
AICc = AIC + 2K(K+1) / (n - K - 1)
```

`K` is the parameter count **plus one**, because the residual variance is also
estimated. Absolute values are meaningless; only differences between models
fitted to the same data are interpretable.

`rSquared` is provided because users expect it, but for nonlinear regression it
is not the fraction of variance explained, it is not bounded below by zero, and
it stays high for visibly poor fits. Prefer RMSE and the residual plot.

## Uncertainty

`fitModel` and `fitConstrainedModel` return `dof`, `syx`, `se`, `ci`, `cov`,
`correlation` and `logEC50` alongside the point estimates.

- `se` and `ci` align with the caller-visible `params`. A parameter held fixed
  by a constrained fit gets `null`, not a fabricated zero — it was not
  estimated, so it carries no uncertainty from this fit.
- The **EC50 interval is formed in log space and back-transformed**, so it is
  asymmetric on the linear scale and can never have a negative lower bound.
  `logEC50` carries the log-space estimate it derives from. Never rebuild it as
  `EC50 ± t·SE`.
- `syx` is `sqrt(SSR/dof)`, the residual standard error. This is not `rmse`,
  which divides by `n` — both are reported, and `syx` is the one to quote.
- `correlation` is where non-identifiability shows up. Individual SEs can look
  finite while two parameters are hopelessly confounded; a near-unit
  off-diagonal is the tell, and usually means the doses do not reach a plateau.
- Everything is `null` when the fit has no residual degrees of freedom or the
  normal equations are singular. Check before displaying.

Intervals are Wald intervals from the linearisation at the solution: exact for
a model linear in its parameters, and optimistic when a parameter is poorly
determined. The coverage tests measure how well that holds up.

## Weighting

Off by default. Pass `{ weighting }` to `fitModel`, one of `WEIGHTING_TYPES`:
`"none"`, `"1/Y"`, `"1/Y^2"`, `"1/SD^2"`.

Relative weights come from the **predicted** values, not the observed ones —
weighting on observed `y` rewards a point for having been noisy downward — so
the fit and the weights are solved together by iteratively reweighted least
squares, seeded from the converged unweighted fit.

The result carries a `weighting` object: what was `requested`, what was
`applied`, any `warning`, the IRLS `iterations`, the `weights` themselves, and
a `variance` diagnostic. Weighting is never silently dropped; if `applied`
differs from `requested` there is always a reason.

**Weighting is a hypothesis about the variance, not a free improvement.** On
constant-CV data `1/Y²` recovers the EC50 about 13% more accurately than
unweighted; on homoscedastic data it does not help at all. `estimateVariancePower`
measures which regime the data are in, by regressing log(SD) on log(mean)
across concentration groups:

| θ | variance | scheme |
| --- | --- | --- |
| ~0 | constant | `none` |
| ~0.5 | ∝ mean | `1/Y` |
| ~1 | ∝ mean² | `1/Y^2` |

It is reported on every fit, weighted or not, so a mismatch between the chosen
scheme and the data is visible. Treat it as a guide — with a handful of doses θ
is imprecise, and picking a weighting is an assay-development decision.

**Do not use relative weighting on normalised or background-subtracted data.**
The weights are undefined at `Ŷ ≤ 0` (a hard refusal) and unsound near a zero
baseline, where the constant-CV assumption says near-baseline points are almost
noiseless. They are not, and the consequence is measurable: on curves whose
true baseline is zero, `1/Y²` pulls the fitted EC50 to roughly 0.75× its true
value while unweighted fits of the same data recover it. The engine warns when
the fitted baseline is within three replicate SDs of zero — judged against the
noise measured near the baseline, since on heteroscedastic data the spread at
the top of the curve says nothing about the spread at the bottom.

`1/SD^2` needs at least three replicates per concentration; below that the SD
is itself so noisy that the weights add variance rather than removing it.

Note that `ssr` stays **unweighted** so it remains comparable across weighting
choices, while `wssr` is the weighted objective that was actually minimised.
The information criteria use the minimised objective, so **never compare AIC or
BIC between fits with different weighting**.

## Tests

```bash
npm test
```

Tests live in `__tests__/`. `fixtures.js` builds deterministic synthetic
datasets: a fixed sine pattern for "does it converge" checks, and a seeded
mulberry32 + Box-Muller generator for anything needing genuine Gaussian noise.
Both are reproducible, so results are stable across machines and runs.

Distribution and outlier helpers are checked against published Student-t and
Grubbs tables. `lm.test.js` checks the numerical Jacobian against the analytic
4PL gradient, and `scale invariance` asserts that the same curve expressed in
different concentration units produces the same fit.

The inference code is validated two ways. Against **closed-form OLS**: for a
model linear in its parameters the machinery must reproduce the textbook slope
and intercept standard errors exactly, which pins the whole `σ²(JᵀJ)⁻¹` path
including the `n−p` divisor. And by **simulated coverage**: over 300 seeded
Gaussian datasets, nominal 95% intervals cover the true value 95.0% (A), 94.0%
(Hill), 95.3% (EC50) and 94.7% (D) of the time.

The F distribution is cross-checked against the t implementation through
`F(1, df) = t(df)²`, and against published critical values. The lack-of-fit
test is checked for **calibration**, not just direction: over 200 seeded
datasets from the correct model it flags 1–12%, bracketing the nominal 5%.
The profile interval is pinned against the Wald interval on a model linear in
its parameters, where the two must agree *exactly*, and the joint potency model
against noiseless data, where the ratio must come back exact under every
weighting scheme.

Tests marked `it.fails(...)` document a known defect: they assert the *correct*
behaviour and are expected to fail until it is fixed. When you fix one, flip it
back to `it(...)` in the same commit.

## Adequacy and identifiability

`fitModel` attaches two diagnostics that answer questions the goodness-of-fit
numbers cannot.

`lackOfFit` partitions the residual sum of squares using the replicates.
Scatter between replicates at one concentration is pure measurement error and
owes nothing to the model, so whatever variation is left over is the curve's
fault:

```
SSR  =  SS(pure error)  +  SS(lack of fit)
```

The ratio of their mean squares is an F-statistic; under a correct model both
estimate the same variance and F ≈ 1. This is the question R² does not answer.
A precise assay can show R² = 0.999 and still fail decisively, because the
deviations, though small, are far larger than that assay's own noise. The
result carries a per-concentration breakdown sorted by contribution, so a
misfit is localised to a dose. It reports `applicable: false` with a reason
when the design cannot support it — no replicates, or no more concentrations
than parameters.

`identifiability` asks whether the parameters were determined at all. Its
checks are geometric rather than statistical, computed from the fitted curve
and the dose range, which means they still work when the covariance matrix is
singular — precisely the case where something has gone wrong. It flags a
plateau the doses never reach (and names how many decades would bracket it,
solved from the curve), an EC50 outside the tested range, near-collinear
parameter pairs, a Hill slope interval spanning zero, and a design with no
more doses than parameters. Parameters the caller fixed are skipped: they were
asserted, not estimated.

## Intervals beyond Wald

The standard errors in `inference.js` linearise the model at the solution. For
a model linear in its parameters that is exact; for the 4PL it is an
approximation that fails in one direction — **symmetric and too narrow**.

`fitModelWithIntervals` is a drop-in replacement for `fitModel` that adds
honest intervals:

```js
const fit = fitModelWithIntervals(x, y, model4PL, false, { intervals: "both" });
fit.profile.ci[2]        // asymmetric EC50 interval
fit.profile.bounded[0]   // { lo: false, hi: true } — unbounded below
fit.bootstrap.bias[1]    // is the estimator itself skewed here?
```

**Profile likelihood** walks each parameter away from its estimate,
re-optimising the others at every step, and finds where the sum of squares
crosses `SSR_min · (1 + t²/(n−p))`. It follows the real shape of the
likelihood, so it returns asymmetric intervals and can report an endpoint as
*unbounded* rather than inventing one. **Bootstrap** resamples residuals —
weighted ones under a weighted fit, inflated by `√(n/(n−p))` to undo
least-squares shrinkage — and reads the interval off the percentiles.

Both are opt-in, because both cost hundreds of refits. On a well-bracketed
curve they reproduce the Wald interval and buy nothing. As the bottom plateau
is pushed out of the dose range they diverge monotonically: at the extreme the
profile interval is many times wider and runs almost entirely to one side.

## Relative potency and parallelism

Relative potency exists only if the two curves have the same shape. Otherwise
no single factor maps one dose axis onto the other, and the ratio is an
artefact of which part of the curve was examined. Parallelism is a
precondition, not a footnote.

```js
const r = relativePotency(reference, test, {
  bounds: { slope: 0.3, lower: 10, upper: 10 },   // from historical data
});
r.potency.rp        // EC50(reference) / EC50(test); > 1 means more potent
r.potency.ci        // asymmetric on the ratio scale, always positive
r.reportable        // did parallelism actually pass?
```

The parallel model carries **log10(RP) as a fitted parameter**, so the quantity
of interest gets its own standard error rather than being rebuilt from two
correlated EC50s. Exponentiating its log-scale interval is what makes the
result asymmetric and strictly positive.

Two parallelism verdicts are reported:

- **F-test** — compares the common-shape fit against the unconstrained one.
  Included because SOPs still specify it, and flagged because its logic is
  inverted: the null hypothesis is parallelism, so *failing to reject* is what
  passes, and an imprecise assay passes easily. The tests pin this directly —
  identical non-parallelism, detected in a precise assay and missed in a noisy
  one.
- **Equivalence (USP <1032>)** — requires the 90% interval on each shape
  difference to fall inside pre-specified bounds. A more precise assay now
  passes more easily, which is the right incentive.

`reportable` keys off equivalence. **Without acceptance criteria there is no
verdict, and potency is not reportable even when the F-test says "parallel"** —
criteria come from historical reference-standard data during validation, so
there is no defensible default to substitute. The ratio is still computed, so
nobody has to recompute it by hand.

Weights, when requested, are derived once and held fixed across both fits: the
F-test compares two sums of squares, which are only comparable if both were
minimised under the same metric.

## Not yet implemented

- **5PL relative potency.** `potency.js` fits 4PL curve pairs only; the 5PL's
  asymmetry exponent would need to join the common-shape constraint.
- **Non-sigmoid parallel-line and parallel-ratio models** for assays whose
  usable range is linear rather than sigmoid.
- **ROUT outlier detection**, which identifies outliers against the fitted
  curve rather than within concentration groups as Grubbs does.
- **Equivalence bounds derived from historical data** — the module consumes
  bounds but has nowhere to store or accumulate the reference-standard history
  they should come from.
