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

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{
    geometry::{
        point::{Point, PointOps},
        vector::{Vector, VectorOps},
    },
    numeric::scalar::Scalar,
};

pub const EPS: f64 = 1e-10;

#[inline(always)]
pub fn f64_next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == -0.0 {
        return f64::MIN_POSITIVE;
    } // smallest positive
    let mut bits = x.to_bits();
    if x >= 0.0 {
        bits += 1;
    } else {
        bits -= 1;
    }
    f64::from_bits(bits)
}

#[inline(always)]
pub fn f64_next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::MIN_POSITIVE;
    } // smallest negative
    let mut bits = x.to_bits();
    if x > 0.0 {
        bits -= 1;
    } else {
        bits += 1;
    }
    f64::from_bits(bits)
}

pub fn barycentric_coords<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> Option<(T, T, T)>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let v0 = (b - a).as_vector();
    let v1 = (c - a).as_vector();
    let v2 = (p - a).as_vector();

    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);

    let denom = &d00 * &d11 - &d01 * &d01;
    if denom.is_zero() {
        return None; // degenerate triangle
    }

    let v = (&d11 * &d20 - &d01 * &d21) / denom.clone(); // coeff of B
    let w = (&d00 * &d21 - &d01 * &d20) / denom; // coeff of C
    let u = &(&T::one() - &v) - &w; // coeff of A

    // Optional exact sanity check:
    // debug_assert_eq!(&(&u + &v) + &w, T::one());

    Some((u, v, w))
}
