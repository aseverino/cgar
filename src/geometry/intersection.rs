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

use crate::operations::{Abs, Pow, Sqrt, Zero};
use std::ops::{Add, Div, Mul, Sub};

use crate::{
    geometry::{Point2, Segment2},
    kernel::{are_collinear, orient2d},
};

#[derive(Debug, Clone, PartialEq)]
pub enum SegmentIntersection<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    None,
    Point(Point2<T>),
    Overlapping(Segment2<T>),
}

pub fn segment_segment_intersection<T>(
    seg1: &Segment2<T>,
    seg2: &Segment2<T>,
    eps: T,
) -> SegmentIntersection<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let a = &seg1.a;
    let b = &seg1.b;
    let c = &seg2.a;
    let d = &seg2.b;

    let o1 = orient2d(&a, &b, &c);
    let o2 = orient2d(&a, &b, &d);
    let o3 = orient2d(&c, &d, &a);
    let o4 = orient2d(&c, &d, &b);

    let zero = T::zero();

    let intersecting = (&o1 * &o2) <= zero && (&o3 * &o4) <= zero;

    if intersecting {
        if o1.abs() > eps || o2.abs() > eps || o3.abs() > eps || o4.abs() > eps {
            // Proper intersection — compute the intersection point
            let (x1, y1) = (&a.x, &a.y);
            let (x2, y2) = (&b.x, &b.y);
            let (x3, y3) = (&c.x, &c.y);
            let (x4, y4) = (&d.x, &d.y);

            let denom = &(&(x1 - x2) * &(y3 - y4)) - &(&(y1 - y2) * &(x3 - x4));
            if denom.abs() < eps {
                return SegmentIntersection::None; // Parallel but not overlapping
            }

            let px_num = &(&(&(x1 * y2) - &(y1 * x2)) * &(x3 - x4))
                - &(&(x1 - x2) * &(&(x3 * y4) - &(y3 * x4)));
            let py_num = &(&(&(x1 * y2) - &(y1 * x2)) * &(y3 - y4))
                - &(&(y1 - y2) * &(&(x3 * y4) - &(y3 * x4)));

            let px = &px_num / &denom;
            let py = &py_num / &denom;

            return SegmentIntersection::Point(Point2::new(px, py));
        }

        // Collinear case
        if are_collinear(&a, &b, &c, &eps) {
            // Compute overlapping segment
            let mut pts = [a, b, c, d];
            pts.sort_by(|p1, p2| {
                p1.x.partial_cmp(&p2.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(p1.y.partial_cmp(&p2.y).unwrap_or(std::cmp::Ordering::Equal))
            });

            let s = Segment2::new(&pts[1], &pts[2]);
            return SegmentIntersection::Overlapping(s);
        }
    }

    SegmentIntersection::None
}
