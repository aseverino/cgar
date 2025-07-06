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

use num_traits::ToPrimitive;

use crate::geometry::Point2;
use crate::geometry::point::Point;
use crate::geometry::segment::Segment;
use crate::numeric::scalar::Scalar;
use crate::operations::{Abs, Pow, Sqrt, Zero};
use std::ops::{Add, Div, Mul, Sub};

pub fn are_equal<T: Scalar, const N: usize>(p1: &Point<T, N>, p2: &Point<T, N>, eps: &T) -> bool
where
    for<'a> &'a T: Sub<&'a T, Output = T> + Mul<&'a T, Output = T>,
{
    for i in 0..N {
        if (&p1.coords[i] - &p2.coords[i]).abs() >= *eps {
            return false;
        }
    }

    return true;
}

pub fn are_collinear<T, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
    eps: &T,
) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    for i in 0..N {
        let ui = &b.coords[i] - &a.coords[i];
        let vi = &c.coords[i] - &a.coords[i];

        if ui.abs() > *eps {
            // first non-zero component gives the candidate scale factor
            let r = &vi / &ui;

            // every remaining coordinate must satisfy vj = r * uj
            for j in (i + 1)..N {
                let uj = &b.coords[j] - &a.coords[j];
                let vj = &c.coords[j] - &a.coords[j];
                if (&vj - &(&uj * &r)).abs() > *eps {
                    return false; // breaks proportionality
                }
            }
            return true; // all coordinates match
        } else if vi.abs() > *eps {
            return false; // ui ≈ 0 but vi isn’t ⇒ not collinear
        }
    }
    // all ui ≈ 0  ⇒  A and B coincide; collinear iff C coincides too
    true
}

pub fn is_point_on_segment<T, const N: usize>(p: &Point<T, N>, seg: &Segment<T, N>, eps: &T) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
    T: PartialOrd, // needed for comparisons inside the loop
{
    // 1.  If P, A, B are not collinear, P cannot lie on AB
    if !are_collinear(p, &seg.a, &seg.b, eps) {
        return false;
    }

    // 2.  For every coordinate axis, P must lie between A and B
    for i in 0..N {
        let ai = &seg.a.coords[i];
        let bi = &seg.b.coords[i];

        // min_i, max_i with ±eps tolerance
        let (min_i, max_i) = if ai < bi { (ai, bi) } else { (bi, ai) };
        let lo = min_i - eps;
        let hi = max_i + eps;

        let pi = &p.coords[i];
        if pi < &lo || pi > &hi {
            return false; // outside on some axis ⇒ not on segment
        }
    }

    true
}
