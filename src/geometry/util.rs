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

use std::ops::{Add, Div, Mul, Sub};

use crate::{
    geometry::{
        point::{Point, PointOps},
        spatial_element::SpatialElement,
    },
    numeric::scalar::Scalar,
};

pub const EPS: f64 = 1e-10;

pub fn triangle_area_2d<T: Scalar>(p0: &Point<T, 2>, p1: &Point<T, 2>, p2: &Point<T, 2>) -> T
where
    for<'a> &'a T: Sub<&'a T, Output = T> + Mul<&'a T, Output = T>,
{
    let v1 = [&p1[0] - &p0[0], &p1[1] - &p0[1]];
    let v2 = [&p2[0] - &p0[0], &p2[1] - &p0[1]];
    let cross = &v1[0] * &v2[1] - &v1[1] * &v2[0];
    cross.abs() / T::from(2.0)
}

/// 2D segment intersection (including overlap)
pub fn segment_intersect_2d<T: Scalar>(
    a0: &Point<T, 2>,
    a1: &Point<T, 2>,
    b0: &Point<T, 2>,
    b1: &Point<T, 2>,
) -> Option<(Point<T, 2>, Point<T, 2>)>
where
    Point<T, 2>: SpatialElement<T, 2>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Mul<&'a T, Output = T>,
{
    // 2D segment intersection including overlap
    // direction vectors
    let da = [&a1[0] - &a0[0], &a1[1] - &a0[1]];
    let db = [&b1[0] - &b0[0], &b1[1] - &b0[1]];

    // cross of directions
    let denom = &da[0] * &db[1] - &da[1] * &db[0];

    // 2D segment intersection including overlap
    // direction vectors
    let da = [&a1[0] - &a0[0], &a1[1] - &a0[1]];
    let db = [&b1[0] - &b0[0], &b1[1] - &b0[1]];
    // cross of directions
    let denom = &da[0] * &db[1] - &da[1] * &db[0];
    if denom.is_zero() {
        // parallel or colinear
        // check colinearity via cross of (b0 - a0) and da
        let diff = [&b0[0] - &a0[0], &b0[1] - &a0[1]];
        let cross = &diff[0] * &da[1] - &diff[1] * &da[0];
        if cross.is_zero() {
            // colinear: project onto A's parameter t
            let len2 = &da[0] * &da[0] + &da[1] * &da[1];
            if !len2.is_positive() {
                return None;
            }
            let t0 = &(&diff[0] * &da[0] + &diff[1] * &da[1]) / &len2;
            let t1 = &(&(&b1[0] - &a0[0]) * &da[0] + &(&b1[1] - &a0[1]) * &da[1]) / &len2;
            let (tmin, tmax) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
            let start = tmin.max(0.0.into());
            let end = tmax.min(1.0.into());
            if start <= end {
                let p_start = Point::<T, 2>::from_vals([
                    &a0[0] + &(&da[0] * &start),
                    &a0[1] + &(&da[1] * &start),
                ]);
                let p_end = Point::<T, 2>::from_vals([
                    &a0[0] + &(&da[0] * &end),
                    &a0[1] + &(&da[1] * &end),
                ]);
                return Some((p_start, p_end));
            }
        }
        return None;
    }
    // lines intersect at single point, solve via cross ratios
    let diff = [&b0[0] - &a0[0], &b0[1] - &a0[1]];
    let s = &(&diff[0] * &db[1] - &diff[1] * &db[0]) / &denom;
    let u = &(&diff[0] * &da[1] - &diff[1] * &da[0]) / &denom;

    if s >= (-EPS).into()
        && s <= (1.0 + EPS).into()
        && u >= (-EPS).into()
        && u <= (1.0 + EPS).into()
    {
        let ix = &a0[0] + &(&s * &da[0]);
        let iy = &a0[1] + &(&s * &da[1]);
        return Some((
            Point::<T, 2>::from_vals([ix.clone(), iy.clone()]),
            Point::<T, 2>::from_vals([ix, iy]),
        ));
    }
    None
}
