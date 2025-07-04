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

use super::Point2;
use crate::{
    geometry::{Point3, Vector2, Vector3, point::PointOps, spatial_element::SpatialElement},
    numeric::scalar::Scalar,
    operations::{Abs, Pow, Sqrt, Zero},
};
use num_traits::ToPrimitive;
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Sub};

pub trait SegmentOps<T: Scalar, C>: Sized {
    type Point: SpatialElement<T> + PointOps<T, C>;

    fn a(&self) -> &Self::Point;
    fn b(&self) -> &Self::Point;

    fn length(&self) -> T {
        self.a().distance_to(self.b())
    }

    fn midpoint(&self) -> Self::Point;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment2<T>
where
    T: Scalar,
{
    pub a: Point2<T>,
    pub b: Point2<T>,
}

impl<T> Segment2<T>
where
    T: Scalar,
{
    pub fn new(a: &Point2<T>, b: &Point2<T>) -> Self {
        Self {
            a: a.clone(),
            b: b.clone(),
        }
    }
}

impl<T> SegmentOps<T, T> for Segment2<T>
where
    T: Scalar,
    Point2<T>: SpatialElement<T> + PointOps<T, T>,
    for<'c> &'c T: Add<&'c T, Output = T>
        + Sub<&'c T, Output = T>
        + Mul<&'c T, Output = T>
        + Div<&'c T, Output = T>,
{
    type Point = Point2<T>;

    fn a(&self) -> &Self::Point {
        &self.a
    }

    fn b(&self) -> &Self::Point {
        &self.b
    }

    fn midpoint(&self) -> Self::Point {
        Point2 {
            x: &(&self.a.x + &self.b.x) / &T::from(2),
            y: &(&self.a.y + &self.b.y) / &T::from(2),
        }
    }
}

/////////////
#[derive(Debug, Clone, PartialEq)]
pub struct Segment3<T>
where
    T: Scalar,
{
    pub a: Point3<T>,
    pub b: Point3<T>,
}

impl<T> Segment3<T>
where
    T: Scalar,
{
    pub fn new(a: &Point3<T>, b: &Point3<T>) -> Self {
        Self {
            a: a.clone(),
            b: b.clone(),
        }
    }
}

impl<T> SegmentOps<T, Vector3<T>> for Segment3<T>
where
    T: Scalar,
    Point3<T>: SpatialElement<T> + PointOps<T, Vector3<T>>,
    for<'c> &'c T: Add<&'c T, Output = T>
        + Sub<&'c T, Output = T>
        + Mul<&'c T, Output = T>
        + Div<&'c T, Output = T>,
{
    type Point = Point3<T>;

    fn a(&self) -> &Self::Point {
        &self.a
    }

    fn b(&self) -> &Self::Point {
        &self.b
    }

    fn midpoint(&self) -> Self::Point {
        Point3 {
            x: &(&self.a.x + &self.b.x) / &T::from(2),
            y: &(&self.a.y + &self.b.y) / &T::from(2),
            z: &(&self.a.z + &self.b.z) / &T::from(2),
        }
    }
}
