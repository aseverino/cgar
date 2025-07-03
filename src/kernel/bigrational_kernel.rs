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

use crate::geometry::{Point2, Segment2, Vector2};
use crate::kernel::{Kernel, are_collinear, are_equal, is_point_on_segment, orient2d};
use crate::numeric::cgar_rational::CgarRational;
use crate::operations::Abs;
use std::ops::{Mul, Sub};

/// Kernel using rug::Rational (arbitrary-precision rationals backed by GMP)
pub struct BigRationalKernel;

impl Kernel for BigRationalKernel {
    type FT = CgarRational;
    type Point2 = Point2<CgarRational>;
    type Segment2 = Segment2<CgarRational>;
    type Vector2 = Vector2<CgarRational>;

    fn orient2d(a: &Self::Point2, b: &Self::Point2, c: &Self::Point2) -> CgarRational {
        let dx1 = &b.x - &a.x;
        let dy1 = &b.y - &a.y;
        let dx2 = &c.x - &a.x;
        let dy2 = &c.y - &a.y;
        &(&dx1 * &dy2) - &(&dy1 * &dx2)
    }

    fn are_equal(a: &Self::Point2, b: &Self::Point2, eps: CgarRational) -> bool {
        let dx = &a.x - &b.x;
        let dy = &a.y - &b.y;
        dx.abs() < eps && dy.abs() < eps
    }

    fn are_collinear(
        a: &Self::Point2,
        b: &Self::Point2,
        c: &Self::Point2,
        eps: CgarRational,
    ) -> bool {
        Self::orient2d(&a, &b, &c).abs() < eps
    }

    fn is_point_on_segment(p: &Self::Point2, s: &Self::Segment2, eps: CgarRational) -> bool {
        if !Self::are_collinear(&s.a, &s.b, &p, eps.clone()) {
            return false;
        }

        let px = &p.x;
        let py = &p.y;
        let ax = &s.a.x;
        let ay = &s.a.y;
        let bx = &s.b.x;
        let by = &s.b.y;

        let (min_x, max_x) = if ax < bx {
            (ax.clone(), bx.clone())
        } else {
            (bx.clone(), ax.clone())
        };
        let (min_y, max_y) = if ay < by {
            (ay.clone(), by.clone())
        } else {
            (by.clone(), ay.clone())
        };

        px >= &(&min_x - &eps)
            && px <= &(&max_x + &eps)
            && py >= &(&min_y - &eps)
            && py <= &(&max_y + &eps)
    }
}
