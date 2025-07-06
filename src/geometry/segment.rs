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
    geometry::{
        Point3, Vector2, Vector3,
        point::{Point, PointOps},
        spatial_element::SpatialElement,
    },
    numeric::scalar::Scalar,
    operations::{Abs, Pow, Sqrt, Zero},
};
use num_traits::ToPrimitive;
use std::hash::Hash;
use std::ops::{Add, Div, Mul, Sub};

pub trait SegmentOps<T: Scalar, const N: usize>: Sized {
    type Point: SpatialElement<T, N> + PointOps<T, N>;

    fn a(&self) -> &Self::Point;
    fn b(&self) -> &Self::Point;

    fn length(&self) -> T {
        self.a().distance_to(self.b())
    }

    fn midpoint(&self) -> Self::Point;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment<T: Scalar, const N: usize> {
    pub a: Point<T, N>,
    pub b: Point<T, N>,
}

impl<T: Scalar, const N: usize> Segment<T, N> {
    pub fn new(a: &Point<T, N>, b: &Point<T, N>) -> Self {
        Self {
            a: a.clone(),
            b: b.clone(),
        }
    }
}

impl<T: Scalar, const N: usize> SegmentOps<T, N> for Segment<T, N>
where
    T: Scalar,
    Point<T, N>: PointOps<T, N>,
    for<'c> &'c T: Add<&'c T, Output = T>
        + Sub<&'c T, Output = T>
        + Mul<&'c T, Output = T>
        + Div<&'c T, Output = T>,
{
    type Point = Point<T, N>;

    fn a(&self) -> &Self::Point {
        &self.a
    }

    fn b(&self) -> &Self::Point {
        &self.b
    }

    fn midpoint(&self) -> Self::Point {
        // Calculate the midpoint by averaging the coordinates of points a and b
        let coords = (0..N)
            .map(|i| {
                let a_coord = &self.a[i];
                let b_coord = &self.b[i];
                (a_coord + b_coord) / T::from(2)
            })
            .collect::<Vec<_>>();
        // Create a new Point with the calculated coordinates
        let a: [T; N] = coords.try_into().expect("Invalid length for Point");
        Point::from_vals(a)

        // Point::from_vals([
        //     &(&self.a[0] + &self.b[0]) / &T::from(2),
        //     &(&self.a[1] + &self.b[1]) / &T::from(2),
        // ])
    }
}

pub type Segment2<T> = Segment<T, 2>;
pub type Segment3<T> = Segment<T, 3>;
