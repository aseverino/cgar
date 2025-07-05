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

use crate::geometry::vector::VectorOps;
use crate::geometry::{Point2, Point3, Vector3};
use crate::numeric::scalar::Scalar;
use std::ops::{Add, Div, Mul, Sub};

/// Returns:
/// - >0 if counter-clockwise
/// - <0 if clockwise
/// - =0 if collinear
pub fn orient2d<T>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>) -> T
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    &(&(&b.x - &a.x) * &(&c.y - &a.y)) - &(&(&b.y - &a.y) * &(&c.x - &a.x))
}

pub fn orient3d<T>(a: &Point3<T>, b: &Point3<T>, c: &Point3<T>, d: &Point3<T>) -> T
where
    T: Scalar,
    Vector3<T>: VectorOps<T, Vector3<T>>,
    for<'a> &'a T: Sub<&'a T, Output = T>,
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

    let ad = Vector3 {
        x: &d.x - &a.x,
        y: &d.y - &a.y,
        z: &d.z - &a.z,
    };

    ab.cross(&ac).dot(&ad)
}
