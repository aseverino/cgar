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

use crate::geometry::point::Point3;
use crate::geometry::segment::Segment3;
use crate::geometry::vector::{Vector3, VectorOps};
use crate::geometry::{Point2, Segment2};
use crate::operations::{Abs, Pow, Sqrt, Zero};
use std::f64::EPSILON;
use std::ops::{Add, Div, Mul, Sub};

/// Determines whether two points are equal within a small tolerance.
pub fn are_equal_2<T>(p1: &Point2<T>, p2: &Point2<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    (&p1.x - &p2.x).abs() < *eps && (&p1.y - &p2.y).abs() < *eps
}

/// Checks if three points are collinear using the area of the triangle formula.
pub fn are_collinear_2<T>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let area = &(&(&b.x - &a.x) * &(&c.y - &a.y)) - &(&(&b.y - &a.y) * &(&c.x - &a.x));
    area.abs() < *eps
}

/// Checks if point `p` lies on segment `seg`.
pub fn is_point_on_segment_2<T>(p: &Point2<T>, seg: &Segment2<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32> + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if !are_collinear_2(&seg.a, &seg.b, &p, &eps) {
        return false;
    }

    let min_x = if &seg.a.x < &seg.b.x {
        &seg.a.x
    } else {
        &seg.b.x
    } - eps;

    let max_x = if &seg.a.x > &seg.b.x {
        &seg.a.x
    } else {
        &seg.b.x
    } + eps;

    let min_y = if &seg.a.y < &seg.b.y {
        &seg.a.y
    } else {
        &seg.b.y
    } - eps;

    let max_y = if &seg.a.y > &seg.b.y {
        &seg.a.y
    } else {
        &seg.b.y
    } + eps;

    //let max_x = seg.a.x.max(seg.b.x) + eps;
    //let min_y = seg.a.y.min(seg.b.y) - eps;
    //let max_y = seg.a.y.max(seg.b.y) + eps;

    p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y
}

/// Determines whether two points are equal within a small tolerance.
pub fn are_equal_3<T>(p1: &Point3<T>, p2: &Point3<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    (&p1.x - &p2.x).abs() < *eps && (&p1.y - &p2.y).abs() < *eps && (&p1.z - &p2.z).abs() < *eps
}

/// Checks if three points are collinear using the area of the triangle formula.
pub fn are_collinear_3<T>(a: &Point3<T>, b: &Point3<T>, c: &Point3<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32> + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let ab = Vector3 {
        x: &b.x - &a.x,
        y: &b.y - &a.y,
        z: &b.z - &a.z,
    };

    let ac = Vector3 {
        x: &c.x - &a.x,
        y: &c.y - &a.y,
        z: &c.z - &a.z,
    };

    let cross = ab.cross(&ac);
    cross.norm().abs() < *eps
}

/// Checks if point `p` lies on segment `seg`.
pub fn is_point_on_segment_3<T>(p: &Point3<T>, seg: &Segment3<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + ToPrimitive + From<i32> + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if !are_collinear_3(&seg.a, &seg.b, p, eps) {
        return false;
    }

    let min_x = if &seg.a.x < &seg.b.x {
        &seg.a.x
    } else {
        &seg.b.x
    } - eps;
    let max_x = if &seg.a.x > &seg.b.x {
        &seg.a.x
    } else {
        &seg.b.x
    } + eps;

    let min_y = if &seg.a.y < &seg.b.y {
        &seg.a.y
    } else {
        &seg.b.y
    } - eps;
    let max_y = if &seg.a.y > &seg.b.y {
        &seg.a.y
    } else {
        &seg.b.y
    } + eps;

    let min_z = if &seg.a.z < &seg.b.z {
        &seg.a.z
    } else {
        &seg.b.z
    } - eps;
    let max_z = if &seg.a.z > &seg.b.z {
        &seg.a.z
    } else {
        &seg.b.z
    } + eps;

    p.x >= min_x && p.x <= max_x && p.y >= min_y && p.y <= max_y && p.z >= min_z && p.z <= max_z
}
