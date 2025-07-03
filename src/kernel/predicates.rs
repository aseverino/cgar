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

use crate::geometry::{Point2, Segment2};
use crate::operations::{Abs, Pow, Sqrt};
use std::f64::EPSILON;
use std::ops::{Add, Div, Mul, Sub};

/// Determines whether two points are equal within a small tolerance.
pub fn are_equal<T>(p1: &Point2<T>, p2: &Point2<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    (&p1.x - &p2.x).abs() < *eps && (&p1.y - &p2.y).abs() < *eps
}

/// Checks if three points are collinear using the area of the triangle formula.
pub fn are_collinear<T>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let area = &(&(&b.x - &a.x) * &(&c.y - &a.y)) - &(&(&b.y - &a.y) * &(&c.x - &a.x));
    area.abs() < *eps
}

/// Checks if point `p` lies on segment `seg`.
pub fn is_point_on_segment<T>(p: &Point2<T>, seg: &Segment2<T>, eps: &T) -> bool
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if !are_collinear(&seg.a, &seg.b, &p, &eps) {
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
