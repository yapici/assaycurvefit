import { describe, it, expect } from "vitest";
import { matMul, matTranspose, solveLU } from "../linalg.js";

describe("matTranspose", () => {
  it("transposes a rectangular matrix", () => {
    expect(matTranspose([[1, 2, 3], [4, 5, 6]])).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it("is an involution", () => {
    const A = [[1, 2], [3, 4], [5, 6]];
    expect(matTranspose(matTranspose(A))).toEqual(A);
  });
});

describe("matMul", () => {
  it("multiplies conformable matrices", () => {
    expect(matMul([[1, 2], [3, 4]], [[5, 6], [7, 8]])).toEqual([[19, 22], [43, 50]]);
  });

  it("acts as identity when multiplied by I", () => {
    const A = [[1, 2], [3, 4]];
    expect(matMul(A, [[1, 0], [0, 1]])).toEqual(A);
  });

  it("handles non-square shapes (2x3 * 3x2 -> 2x2)", () => {
    const C = matMul([[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]]);
    expect(C).toEqual([[4, 5], [10, 11]]);
  });

  it("produces a symmetric Gram matrix for J^T J", () => {
    const J = [[1, 2], [3, 4], [5, 6]];
    const G = matMul(matTranspose(J), J);
    expect(G[0][1]).toBe(G[1][0]);
  });
});

describe("solveLU", () => {
  it("solves a small well-conditioned system", () => {
    // 2x + y = 5 ; x + 3y = 10  ->  x = 1, y = 3
    const x = solveLU([[2, 1], [1, 3]], [5, 10]);
    expect(x[0]).toBeCloseTo(1, 10);
    expect(x[1]).toBeCloseTo(3, 10);
  });

  it("returns the right-hand side unchanged for the identity matrix", () => {
    const x = solveLU([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [7, 8, 9]);
    expect(x[0]).toBeCloseTo(7, 12);
    expect(x[1]).toBeCloseTo(8, 12);
    expect(x[2]).toBeCloseTo(9, 12);
  });

  it("requires partial pivoting (leading zero pivot)", () => {
    // Without row swapping this divides by zero on the first pivot.
    const x = solveLU([[0, 1], [1, 0]], [2, 3]);
    expect(x[0]).toBeCloseTo(3, 10);
    expect(x[1]).toBeCloseTo(2, 10);
  });

  it("returns null for a singular matrix rather than NaNs", () => {
    // This is the signal levenbergMarquardt relies on to bump lambda.
    expect(solveLU([[1, 2], [2, 4]], [1, 2])).toBeNull();
  });

  it("round-trips: A * solveLU(A, b) === b", () => {
    const A = [[4, -2, 1], [-2, 4, -2], [1, -2, 4]];
    const b = [11, -16, 17];
    const x = solveLU(A, b);
    const back = A.map(row => row.reduce((s, v, j) => s + v * x[j], 0));
    back.forEach((v, i) => expect(v).toBeCloseTo(b[i], 9));
  });
});
