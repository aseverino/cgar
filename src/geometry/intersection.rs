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

use crate::{
    geometry::{
        Point2, Point3, Segment2, Segment3, Vector3, point::Point, spatial_element::SpatialElement,
        vector::VectorOps,
    },
    kernel::{are_collinear, orient},
    numeric::scalar::Scalar,
};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub enum SegmentIntersection2<T>
where
    T: Scalar,
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
    T: Scalar,
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
    T: Scalar,
    //SegmentIntersection2<T>: SpatialElement<T>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let a = &seg1.a;
    let b = &seg1.b;
    let c = &seg2.a;
    let d = &seg2.b;

    let o1 = orient(&[a.clone(), b.clone(), c.clone()]);
    let o2 = orient(&[a.clone(), b.clone(), d.clone()]);
    let o3 = orient(&[c.clone(), d.clone(), a.clone()]);
    let o4 = orient(&[c.clone(), d.clone(), b.clone()]);

    let zero = T::zero();

    let intersecting = (&o1 * &o2) <= zero && (&o3 * &o4) <= zero;

    if intersecting {
        if o1.abs() > eps || o2.abs() > eps || o3.abs() > eps || o4.abs() > eps {
            // Proper intersection — compute the intersection point
            let (x1, y1) = (&a[0], &a[1]);
            let (x2, y2) = (&b[0], &b[1]);
            let (x3, y3) = (&c[0], &c[1]);
            let (x4, y4) = (&d[0], &d[1]);

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

            return SegmentIntersection2::Point(Point::<T, 2>::from_vals([px, py]));
        }

        // Collinear case
        if are_collinear(&a, &b, &c, &eps) {
            // Compute overlapping segment
            let mut pts = [a, b, c, d];
            pts.sort_by(|p1, p2| {
                p1[0]
                    .partial_cmp(&p2[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(
                        p1[1]
                            .partial_cmp(&p2[1])
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
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
    T: Scalar,
{
    None,
    Point(Point3<T>),
    Overlapping(Segment3<T>),
}

impl<T> PartialEq for SegmentIntersection3<T>
where
    T: Scalar,
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
    T: Scalar,
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
    let d1 = Vector3::from(Point3::from_vals([
        &p2[0] - &p1[0],
        &p2[1] - &p1[1],
        &p2[2] - &p1[2],
    ]));
    let d2 = Vector3::from(Point3::from_vals([
        &q2[0] - &q1[0],
        &q2[1] - &q1[1],
        &q2[2] - &q1[2],
    ]));

    let r = Vector3::from(Point3::from_vals([
        &p1[0] - &q1[0],
        &p1[1] - &q1[1],
        &p1[2] - &q1[2],
    ]));

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
        if are_collinear(p1, p2, q1, &eps) {
            // Sort points by x (then y then z)
            let mut pts = [p1, p2, q1, q2];
            pts.sort_by(|p1, p2| {
                p1[0]
                    .partial_cmp(&p2[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(
                        p1[1]
                            .partial_cmp(&p2[1])
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
                    .then(
                        p1[2]
                            .partial_cmp(&p2[2])
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
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

    let closest_p = Point3::from_vals([
        &p1[0] + &(&d1[0] * &s_clamped),
        &p1[1] + &(&d1[1] * &s_clamped),
        &p1[2] + &(&d1[2] * &s_clamped),
    ]);

    let closest_q = Point3::from_vals([
        &q1[0] + &(&d2[0] * &t_clamped),
        &q1[1] + &(&d2[1] * &t_clamped),
        &q1[2] + &(&d2[2] * &t_clamped),
    ]);

    let dx = &closest_p[0] - &closest_q[0];
    let dy = &closest_p[1] - &closest_q[1];
    let dz = &closest_p[2] - &closest_q[2];

    let dist_squared = &(&(&dx * &dx) + &(&dy * &dy)) + &(&dz * &dz);

    if dist_squared < &eps * &eps {
        SegmentIntersection3::Point(closest_p)
    } else {
        SegmentIntersection3::None
    }
}
