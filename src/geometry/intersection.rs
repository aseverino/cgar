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
use std::hash::Hash;

use crate::{
    geometry::{Point2, Point3, Segment2, Segment3, Vector3, point::PointOps, vector::VectorOps},
    kernel::are_collinear_3,
    operations::{Abs, Pow, Sqrt, Zero},
};
use std::ops::{Add, Div, Mul, Sub};

use crate::kernel::{are_collinear_2, orient2d};

#[derive(Debug, Clone)]
pub enum SegmentIntersection2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32> + Zero,
    Point2<T>: PartialEq,
    Segment2<T>: PartialEq,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    None,
    Point(Point2<T>),
    Overlapping(Segment2<T>),
}

impl<T> PartialEq for SegmentIntersection2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32> + Zero,
    Point2<T>: PartialEq,
    Segment2<T>: PartialEq,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SegmentIntersection2::None, SegmentIntersection2::None) => true,
            (SegmentIntersection2::Point(p1), SegmentIntersection2::Point(p2)) => p1 == p2,
            (SegmentIntersection2::Overlapping(s1), SegmentIntersection2::Overlapping(s2)) => {
                s1 == s2
            }
            _ => false,
        }
    }
}

pub fn segment_segment_intersection_2<T>(
    seg1: &Segment2<T>,
    seg2: &Segment2<T>,
    eps: T,
) -> SegmentIntersection2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero + ToPrimitive + From<i32> + Zero,
    Point2<T>: PointOps<T, T> + PartialEq,
    Segment2<T>: PartialEq,
    SegmentIntersection2<T>: PartialEq,
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
                return SegmentIntersection2::None; // Parallel but not overlapping
            }

            let px_num = &(&(&(x1 * y2) - &(y1 * x2)) * &(x3 - x4))
                - &(&(x1 - x2) * &(&(x3 * y4) - &(y3 * x4)));
            let py_num = &(&(&(x1 * y2) - &(y1 * x2)) * &(y3 - y4))
                - &(&(y1 - y2) * &(&(x3 * y4) - &(y3 * x4)));

            let px = &px_num / &denom;
            let py = &py_num / &denom;

            return SegmentIntersection2::Point(Point2::new(px, py));
        }

        // Collinear case
        if are_collinear_2(&a, &b, &c, &eps) {
            // Compute overlapping segment
            let mut pts = [a, b, c, d];
            pts.sort_by(|p1, p2| {
                p1.x.partial_cmp(&p2.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(p1.y.partial_cmp(&p2.y).unwrap_or(std::cmp::Ordering::Equal))
            });

            let s = Segment2::new(&pts[1], &pts[2]);
            return SegmentIntersection2::Overlapping(s);
        }
    }

    SegmentIntersection2::None
}

#[derive(Debug, Clone)]
pub enum SegmentIntersection3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32> + Zero,
    Point3<T>: PointOps<T, Vector3<T>> + PartialEq,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    None,
    Point(Point3<T>),
    Overlapping(Segment3<T>),
}

impl<T> PartialEq for SegmentIntersection3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32> + Zero,
    Point3<T>: PartialEq,
    Segment3<T>: PartialEq,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SegmentIntersection3::None, SegmentIntersection3::None) => true,
            (SegmentIntersection3::Point(p1), SegmentIntersection3::Point(p2)) => p1 == p2,
            (SegmentIntersection3::Overlapping(s1), SegmentIntersection3::Overlapping(s2)) => {
                s1 == s2
            }
            _ => false,
        }
    }
}

pub fn segment_segment_intersection_3<T>(
    seg1: &Segment3<T>,
    seg2: &Segment3<T>,
    eps: T,
) -> SegmentIntersection3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero + ToPrimitive + From<i32> + Zero,
    Point3<T>: PointOps<T, Vector3<T>> + Eq + Hash,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let p1 = &seg1.a;
    let p2 = &seg1.b;
    let q1 = &seg2.a;
    let q2 = &seg2.b;

    // Direction vectors
    let d1 = Vector3 {
        x: &p2.x - &p1.x,
        y: &p2.y - &p1.y,
        z: &p2.z - &p1.z,
    };
    let d2 = Vector3 {
        x: &q2.x - &q1.x,
        y: &q2.y - &q1.y,
        z: &q2.z - &q1.z,
    };

    let r = Vector3 {
        x: &p1.x - &q1.x,
        y: &p1.y - &q1.y,
        z: &p1.z - &q1.z,
    };

    // Dot products
    let a = d1.dot(&d1); // squared length of d1
    let b = d1.dot(&d2);
    let c = d2.dot(&d2); // squared length of d2
    let d = d1.dot(&r);
    let e = d2.dot(&r);

    let denom = &(&a * &c) - &(&b * &b);

    let zero = T::zero();

    let (s, t) = if denom.abs() > eps {
        let s = &(&(&b * &e) - &(&c * &d)) / &denom;
        let t = &(&(&a * &e) - &(&b * &d)) / &denom;
        (s, t)
    } else {
        // Segments are parallel — handle collinear case
        if are_collinear_3(p1, p2, q1, &eps) {
            // Sort points by x (then y then z)
            let mut pts = [p1, p2, q1, q2];
            pts.sort_by(|p1, p2| {
                p1.x.partial_cmp(&p2.x)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(p1.y.partial_cmp(&p2.y).unwrap_or(std::cmp::Ordering::Equal))
                    .then(p1.z.partial_cmp(&p2.z).unwrap_or(std::cmp::Ordering::Equal))
            });
            return SegmentIntersection3::Overlapping(Segment3 {
                a: pts[1].clone(),
                b: pts[2].clone(),
            });
        }

        return SegmentIntersection3::None;
    };

    // Compute closest points on each segment
    let s_clamped = if s < zero {
        zero.clone()
    } else if s > T::from(1) {
        T::from(1)
    } else {
        s
    };

    let t_clamped = if t < zero {
        zero.clone()
    } else if t > T::from(1) {
        T::from(1)
    } else {
        t
    };

    let closest_p = Point3 {
        x: &p1.x + &(&d1.x * &s_clamped),
        y: &p1.y + &(&d1.y * &s_clamped),
        z: &p1.z + &(&d1.z * &s_clamped),
    };

    let closest_q = Point3 {
        x: &q1.x + &(&d2.x * &t_clamped),
        y: &q1.y + &(&d2.y * &t_clamped),
        z: &q1.z + &(&d2.z * &t_clamped),
    };

    let dx = &closest_p.x - &closest_q.x;
    let dy = &closest_p.y - &closest_q.y;
    let dz = &closest_p.z - &closest_q.z;

    let dist_squared = &(&(&dx * &dx) + &(&dy * &dy)) + &(&dz * &dz);

    if dist_squared < &eps * &eps {
        SegmentIntersection3::Point(closest_p)
    } else {
        SegmentIntersection3::None
    }
}
