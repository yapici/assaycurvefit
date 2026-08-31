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
// A is the asymptote where (x/C)^B -> 0 and D the one where it -> infinity.
// Which end of the dose range that corresponds to depends on the SIGN of the
// Hill slope B, so neither A nor D is reliably "the low plateau":
//
//   B > 0:  y -> A as x -> 0,        y -> D as x -> infinity
//   B < 0:  y -> D as x -> 0,        y -> A as x -> infinity
//
// The optimiser is free to converge on either sign (the two are mirror
// solutions), so both cases occur in practice on ordinary data. The UI labels
// these "min"/"max", which is accurate only for one sign.
//
// The 5PL uses (EC50/x) rather than (x/C), which inverts the relationship
// again: there a POSITIVE Hill gives Bottom at x -> 0.

// 4PL: y = D + (A - D) / (1 + (x/C)^B)
// 5PL: y = Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)^S
// 3PL: 4PL with B fixed to 1 (Hill slope = 1)
// 2PL: 4PL with A and D fixed (fit B and C only)
// 1PL: 4PL with A, B, D fixed (fit C / EC50 only)

export function model4PL(x, params) {
  const [A, B, C, D] = params;
  // x <= 0 is outside the model's domain (log-x is undefined, and a zero-dose
  // control row is legitimate input). Return the true x -> 0+ limit, which
  // depends on the SIGN of the Hill slope: (x/C)^B tends to 0 for B > 0, to
  // infinity for B < 0, and to 1 for B = 0.
  if (x <= 0) {
    if (B > 0) return A;
    if (B < 0) return D;
    return D + (A - D) / 2; // B = 0: the model is constant in x
  }
  return D + (A - D) / (1 + Math.pow(x / C, B));
}

export function model5PL(x, params) {
  const [Bottom, Hill, EC50, Top, S] = params;
  // Same reasoning as model4PL. Here the ratio is (EC50/x), which tends to
  // infinity as x -> 0+, so the sign of Hill selects the opposite asymptote.
  if (x <= 0) {
    if (Hill > 0) return Bottom;
    if (Hill < 0) return Top;
    return Bottom + (Top - Bottom) / Math.pow(2, S); // Hill = 0: constant in x
  }
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
