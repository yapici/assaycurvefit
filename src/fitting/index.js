// ── Fitting engine barrel ─────────────────────────────────────────
// Single import surface for the numerical core. The React component
// should import from here, not from the individual modules, so that
// internal reorganisation stays invisible to the UI layer.

export { model4PL, model5PL, makeConstrainedModel, getModelFn, computeBiologicalEC50 } from "./models.js";
export { matMul, matTranspose, solveLU } from "./linalg.js";
export {
  residuals, sumSquaredResiduals, jacobian, withLogParams, levenbergMarquardt,
  estimateInitialParams, fitModel, fitConstrainedModel,
} from "./lm.js";
export { rSquared, computeAIC, computeAICc, computeBIC, groupByConcentration } from "./stats.js";
export { lnGamma, betaIncomplete, tCDF, tInv } from "./distributions.js";
export { grubbsCriticalG, grubbsTest, runGrubbsAllGroups } from "./outliers.js";
