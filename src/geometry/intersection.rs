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
    T: Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Clone
        + PartialOrd
        + Abs
        + Pow
        + Sqrt,
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
    T: Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Clone
        + PartialOrd
        + Zero
        + Abs
        + Pow
        + Sqrt,
{
    let a = seg1.a.clone();
    let b = seg1.b.clone();
    let c = seg2.a.clone();
    let d = seg2.b.clone();

    let o1 = orient2d(&a, &b, &c);
    let o2 = orient2d(&a, &b, &d);
    let o3 = orient2d(&c, &d, &a);
    let o4 = orient2d(&c, &d, &b);

    let zero = T::zero();

    let intersecting = (o1.clone() * o2.clone()) <= zero && (o3.clone() * o4.clone()) <= zero;

    if intersecting {
        if o1.abs() > eps || o2.abs() > eps || o3.abs() > eps || o4.abs() > eps {
            // Proper intersection — compute the intersection point
            let (x1, y1) = (a.x, a.y).clone();
            let (x2, y2) = (b.x, b.y).clone();
            let (x3, y3) = (c.x, c.y).clone();
            let (x4, y4) = (d.x, d.y).clone();

            let denom = (x1.clone() - x2.clone()) * (y3.clone() - y4.clone())
                - (y1.clone() - y2.clone()) * (x3.clone() - x4.clone());
            if denom.abs() < eps {
                return SegmentIntersection::None; // Parallel but not overlapping
            }

            let px_num = (x1.clone() * y2.clone() - y1.clone() * x2.clone())
                * (x3.clone() - x4.clone())
                - (x1.clone() - x2.clone()) * (x3.clone() * y4.clone() - y3.clone() * x4.clone());
            let py_num = (x1.clone() * y2.clone() - y1.clone() * x2.clone())
                * (y3.clone() - y4.clone())
                - (y1.clone() - y2.clone()) * (x3.clone() * y4.clone() - y3.clone() * x4.clone());

            let px = px_num.clone() / denom.clone();
            let py = py_num.clone() / denom.clone();

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
