// ── Dense linear algebra ──────────────────────────────────────────
// Minimal matrix helpers backing the Levenberg-Marquardt normal equations.
// Matrices are arrays of row arrays. No external dependencies.

// Matrix operations
export function matMul(A, B) {
  const m = A.length, n = B[0].length, k = B.length;
  const C = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      for (let l = 0; l < k; l++)
        C[i][j] += A[i][l] * B[l][j];
  return C;
}

export function matTranspose(A) {
  const m = A.length, n = A[0].length;
  const T = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      T[j][i] = A[i][j];
  return T;
}

// Solve Ax = b using LU decomposition with partial pivoting
export function solveLU(A, b) {
  const n = A.length;
  const LU = A.map(row => [...row]);
  const P = Array.from({ length: n }, (_, i) => i);
  
  for (let k = 0; k < n; k++) {
    let maxVal = 0, maxIdx = k;
    for (let i = k; i < n; i++) {
      if (Math.abs(LU[i][k]) > maxVal) {
        maxVal = Math.abs(LU[i][k]);
        maxIdx = i;
      }
    }
    if (maxVal < 1e-15) return null;
    if (maxIdx !== k) {
      [LU[k], LU[maxIdx]] = [LU[maxIdx], LU[k]];
      [P[k], P[maxIdx]] = [P[maxIdx], P[k]];
    }
    for (let i = k + 1; i < n; i++) {
      LU[i][k] /= LU[k][k];
      for (let j = k + 1; j < n; j++) {
        LU[i][j] -= LU[i][k] * LU[k][j];
      }
    }
  }

  const pb = P.map(i => b[i]);
  const y = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    y[i] = pb[i];
    for (let j = 0; j < i; j++) y[i] -= LU[i][j] * y[j];
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = y[i];
    for (let j = i + 1; j < n; j++) x[i] -= LU[i][j] * x[j];
    x[i] /= LU[i][i];
  }
  return x;
}

/**
 * Invert a square matrix by solving A x = e_i for each basis vector.
 * Returns null if A is singular, matching solveLU's contract.
 *
 * Only used on the small (<= 5x5) J^T J from the normal equations, so the
 * O(n^4) cost of repeated LU solves is irrelevant.
 */
export function matInverse(A) {
  const n = A.length;
  const columns = [];
  for (let i = 0; i < n; i++) {
    const e = new Array(n).fill(0);
    e[i] = 1;
    const col = solveLU(A, e);
    if (!col) return null;
    columns.push(col); // solveLU(A, e_i) is the i-th COLUMN of the inverse
  }
  return matTranspose(columns);
}
