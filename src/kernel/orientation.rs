// SPDX-License-Identifier: MIT
//
// Copyright (c) 2025 Alexandre Severino
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::geometry::point::Point;
use crate::numeric::scalar::Scalar;
use std::{
    array::from_fn,
    ops::{Add, Mul, Neg, Sub},
};

/// Returns:
/// - >0 if counter-clockwise
/// - <0 if clockwise
/// - =0 if collinear
pub fn orient<T, const N: usize>(p: &[Point<T, N>]) -> T
where
    T: Scalar + Neg<Output = T>,
    for<'a> &'a T: Add<&'a T, Output = T> + Sub<&'a T, Output = T> + Mul<&'a T, Output = T>,
{
    assert!(p.len() == N + 1, "need N+1 points");
    // 1. Build an NxN matrix of the difference vectors  (p[i+1] – p[0])
    let mut m: [[T; N]; N] =
        from_fn(|row| from_fn(|col| &p[row + 1].coords[col] - &p[0].coords[col]));

    // 2. Fraction-free Gaussian elimination → determinant in `det`
    let mut det = T::one(); // running ±product of pivots
    for k in 0..N {
        // pivot = m[k][k];  if zero, determinant is zero
        if m[k][k].abs() == T::zero() {
            return T::zero();
        }
        for i in (k + 1)..N {
            for j in (k + 1)..N {
                //   m[i][j] = m[i][j] * pivot - m[i][k] * m[k][j];
                m[i][j] = &(&m[i][j] * &m[k][k]) - &(&m[i][k] * &m[k][j]);
            }
        }
        det = det * m[k][k].clone();
    }
    det
}
