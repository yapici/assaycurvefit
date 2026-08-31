// ── Dose-response model functions ─────────────────────────────────
// Pure, dependency-free model definitions shared by the fitter and the
// chart renderers. Extracted from BioassayCurveFitter.jsx (v1.5.0) with
// no behavioural change.
//
// Parameter vectors:
//   4PL: [A, B, C, D]                  (A = y at x->0, B = Hill, C = EC50, D = y at x->inf)
//   5PL: [Bottom, Hill, EC50, Top, S]
// The two orderings differ historically; `getModelFn` picks the right
// evaluator and callers must pass the matching vector.
//
// IMPORTANT — asymptote convention. In `y = D + (A - D) / (1 + (x/C)^B)`,
// A is the response as x -> 0 and D the response as x -> infinity. So A is
// NOT necessarily the smaller value:
//   A < D  ->  ascending curve  (agonist / activation)
//   A > D  ->  descending curve (inhibition; A is the uninhibited signal)
// The UI labels these "Min"/"Max", which matches only for ascending curves.
// The Hill slope B also carries a sign, and the 5PL uses (EC50/x) rather
// than (x/C), which flips the sign of Hill between the two models.

// 4PL: y = D + (A - D) / (1 + (x/C)^B)
// 5PL: y = Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)^S
// 3PL: 4PL with B fixed to 1 (Hill slope = 1)
// 2PL: 4PL with A and D fixed (fit B and C only)
// 1PL: 4PL with A, B, D fixed (fit C / EC50 only)

export function model4PL(x, params) {
  const [A, B, C, D] = params;
  if (x <= 0) return A;
  return D + (A - D) / (1 + Math.pow(x / C, B));
}

export function model5PL(x, params) {
  const [Bottom, Hill, EC50, Top, S] = params;
  if (x <= 0) return Bottom;
  return Bottom + (Top - Bottom) / Math.pow(1 + Math.pow(EC50 / x, Hill), S);
}

// Create a constrained model function that wraps the full 4PL
// fixedMap: object mapping param index -> fixed value, e.g. { 1: 1.0 } fixes B=1
export function makeConstrainedModel(fixedMap) {
  return function constrainedModel(x, freeParams) {
    // Expand free params into full 4PL params
    const fullParams = [0, 0, 0, 0];
    let freeIdx = 0;
    for (let i = 0; i < 4; i++) {
      if (i in fixedMap) {
        fullParams[i] = fixedMap[i];
      } else {
        fullParams[i] = freeParams[freeIdx++];
      }
    }
    return model4PL(x, fullParams);
  };
}

// Get the drawing/evaluation model function for a given model type
// 1PL, 2PL, 3PL, 4PL all use model4PL (with full 4-param vector)
export function getModelFn(mType) {
  return mType === "5PL" ? model5PL : model4PL;
}

// Compute biological EC50 for 5PL: concentration where response = (Top + Bottom) / 2
export function computeBiologicalEC50(modelFn, params) {
  const [Bottom, Hill, EC50, Top, S] = params;
  const targetY = (Top + Bottom) / 2;
  // Bisection in log-space
  let lo = 1e-15, hi = 1e15;
  const yLo = modelFn(lo, params);
  const yHi = modelFn(hi, params);
  const increasing = yHi > yLo;
  for (let i = 0; i < 100; i++) {
    const mid = Math.sqrt(lo * hi);
    const yMid = modelFn(mid, params);
    if ((increasing && yMid > targetY) || (!increasing && yMid < targetY)) hi = mid;
    else lo = mid;
  }
  return Math.sqrt(lo * hi);
}
