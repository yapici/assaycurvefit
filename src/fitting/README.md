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
| `distributions.js` | `lnGamma`, `betaIncomplete`, `tCDF`, `tInv` |
| `outliers.js` | `grubbsCriticalG`, `grubbsTest`, `runGrubbsAllGroups` |

## Model conventions

The 4PL is

```
y = D + (A - D) / (1 + (x / C)^B)
```

with parameter vector `[A, B, C, D]`. **`A` is the response as `x → 0` and `D`
the response as `x → ∞`** — `A` is not necessarily the smaller of the two. An
inhibition curve has `A > D`. The 5PL uses `(EC50 / x)` instead of `(x / C)`,
which flips the sign of the Hill slope relative to the 4PL; the two models
therefore take differently-ordered parameter vectors.

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

## Tests

```bash
npm test
```

Tests live in `__tests__/`. `fixtures.js` builds deterministic synthetic
datasets — there is no RNG, so golden values are stable across machines.

Distribution and outlier helpers are checked against published Student-t and
Grubbs tables. `lm.test.js` checks the numerical Jacobian against the analytic
4PL gradient, and `scale invariance` asserts that the same curve expressed in
different concentration units produces the same fit.

Tests marked `it.fails(...)` document a known defect: they assert the *correct*
behaviour and are expected to fail until it is fixed. When you fix one, flip it
back to `it(...)` in the same commit.

## Not yet implemented

The engine currently reports point estimates only. In rough priority order:

- **Parameter standard errors and confidence intervals.** Nothing here computes
  `σ²(JᵀJ)⁻¹`, so an EC50 comes back without an interval. `solveLU` and `tInv`
  are already present, and the log-EC50 parameterisation means the interval can
  be formed in log space and back-transformed to the correct asymmetric bounds.
- **Weighting.** All fitting is unweighted least squares. Assay response is
  typically heteroscedastic, and `1/Ŷ²` (relative) weighting is the usual
  default for ligand-binding work. Weight on the *predicted* value, iteratively
  reweighted, and disable it when the data have been normalised or background
  subtraction has pushed the low plateau near zero — the weights are undefined
  at `Ŷ = 0` and meaningless once the variance structure has been rescaled.
- **Replicate-based lack-of-fit F-test.** `groupByConcentration` already yields
  the per-group spread needed to split residual SS into pure error and
  lack-of-fit, which answers "is this model adequate?" far better than R².
- **Identifiability warnings** when the data do not bracket a plateau, so an
  extrapolated Top/Bottom cannot be mistaken for a measured one.
- **Relative potency and parallelism** (USP <1032>/<1034>): common-slope fits
  across a test and reference curve, equivalence-based parallelism, and a
  potency ratio with a confidence interval.
