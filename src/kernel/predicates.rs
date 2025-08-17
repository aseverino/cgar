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

use crate::geometry::point::PointOps;
use crate::geometry::segment::Segment;
use crate::geometry::vector::VectorOps;
use crate::geometry::{point::Point, vector::Vector};
use crate::numeric::scalar::Scalar;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrianglePoint {
    Off,
    On,
    In,
}

pub fn are_equal<T: Scalar, const N: usize>(p1: &Point<T, N>, p2: &Point<T, N>) -> bool
where
    for<'a> &'a T: Sub<&'a T, Output = T> + Mul<&'a T, Output = T>,
{
    for i in 0..N {
        if !(&p1.coords[i] - &p2.coords[i]).abs().is_zero() {
            return false;
        }
    }

    return true;
}

pub fn are_collinear<T, const N: usize>(a: &Point<T, N>, b: &Point<T, N>, c: &Point<T, N>) -> bool
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

        if ui.abs().is_positive() {
            // first non-zero component gives the candidate scale factor
            let r = &vi / &ui;

            // every remaining coordinate must satisfy vj = r * uj
            for j in (i + 1)..N {
                let uj = &b.coords[j] - &a.coords[j];
                let vj = &c.coords[j] - &a.coords[j];
                if (&vj - &(&uj * &r)).abs().is_positive() {
                    return false; // breaks proportionality
                }
            }
            return true; // all coordinates match
        } else if vi.abs().is_positive() {
            return false; // ui ≈ 0 but vi isn’t ⇒ not collinear
        }
    }
    // all ui ≈ 0  ⇒  A and B coincide; collinear iff C coincides too
    true
}

pub fn triangle_is_degenerate<T: Scalar, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> bool
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Coincidence or collinearity → zero area
    are_equal(a, b) || are_equal(a, c) || are_equal(b, c) || are_collinear(a, b, c)
}

pub fn is_point_on_segment<T, const N: usize>(p: &Point<T, N>, seg: &Segment<T, N>) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
    T: PartialOrd, // needed for comparisons inside the loop
{
    // 1.  If P, A, B are not collinear, P cannot lie on AB
    if !are_collinear(p, &seg.a, &seg.b) {
        return false;
    }

    // 2.  For every coordinate axis, P must lie between A and B
    for i in 0..N {
        let ai = &seg.a.coords[i];
        let bi = &seg.b.coords[i];

        // min_i, max_i bounds
        let (min_i, max_i) = if ai <= bi { (ai, bi) } else { (bi, ai) };

        let pi = &p.coords[i];
        if pi < min_i || pi > max_i {
            return false; // outside on some axis ⇒ not on segment
        }
    }

    true
}

pub fn point_u_on_segment<T: Scalar + PartialOrd, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    p: &Point<T, N>,
) -> Option<T>
where
    Point<T, N>: PointOps<T, N, Vector = crate::geometry::vector::Vector<T, N>>,
    crate::geometry::vector::Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Direction and offset
    let ab = (b - a).as_vector();
    let ap = (p - a).as_vector();

    // Degenerate segment?
    let ab2 = ab.dot(&ab);
    if ab2.is_zero() {
        return if are_equal(a, p) {
            Some(T::zero())
        } else if are_equal(b, p) {
            Some(T::one())
        } else {
            None
        };
    }

    // Must be collinear with AB
    if !are_collinear(a, b, p) {
        return None;
    }

    // Parametric coordinate along AB
    let u = ap.dot(&ab) / ab2;
    if u < T::zero() || u > T::one() {
        None
    } else {
        Some(u)
    }
}

pub fn point_in_or_on_triangle<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> TrianglePoint
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Barycentric setup
    let v0 = (c - a).as_vector();
    let v1 = (b - a).as_vector();
    let v2 = (p - a).as_vector();

    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot11 = v1.dot(&v1);
    let dot12 = v1.dot(&v2);

    let denom = &dot00 * &dot11 - &dot01 * &dot01;

    // Degenerate triangle (zero/near-zero area)
    if denom.abs() < T::tolerance() {
        return TrianglePoint::Off;
    }

    let inv = T::one() / denom;
    let u = &(&dot11 * &dot02 - &dot01 * &dot12) * &inv;
    let v = &(&dot00 * &dot12 - &dot01 * &dot02) * &inv;
    let sum_uv = &u + &v;
    let w = &T::one() - &sum_uv;

    // Classification with tolerance
    let e = T::tolerance();
    let neg_e = e.clone().neg();

    // Outside if any barycentric is below -eps or u+v exceeds 1+eps
    if u < neg_e || v < neg_e || sum_uv > &T::one() + &e {
        return TrianglePoint::Off;
    }

    // On if any barycentric is within eps of the boundary
    if u <= e || v <= e || w <= e {
        return TrianglePoint::On;
    }

    TrianglePoint::In
}
