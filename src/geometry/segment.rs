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
        point::{Point, PointOps},
        spatial_element::SpatialElement,
    },
    numeric::scalar::Scalar,
};
use std::ops::{Add, Div, Mul, Sub};

pub trait SegmentOps<T: Scalar, const N: usize>: Sized
where
    Point<T, N>: SpatialElement<T, N> + PointOps<T, N>,
{
    fn a(&self) -> &Point<T, N>;
    fn b(&self) -> &Point<T, N>;

    fn length(&self) -> T {
        self.a().distance_to(self.b())
    }

    fn midpoint(&self) -> Point<T, N>;
    fn is_point_on(&self, p: &Point<T, N>) -> bool;

    fn inverse(&self) -> Self;
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
    fn a(&self) -> &Point<T, N> {
        &self.a
    }

    fn b(&self) -> &Point<T, N> {
        &self.b
    }

    fn midpoint(&self) -> Point<T, N> {
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

    fn is_point_on(&self, p: &Point<T, N>) -> bool {
        let mut t_opt: Option<T> = None;

        for i in 0..N {
            let da = &p[i] - &self.a[i];
            let db = &self.b[i] - &self.a[i];

            if db != T::zero() {
                let t = &da / &db;
                if let Some(prev_t) = &t_opt {
                    if (&t - &prev_t).abs() > T::from(1e-10) {
                        return false;
                    }
                } else {
                    t_opt = Some(t);
                }
            } else if da != T::zero() {
                return false;
            }
        }

        if let Some(t) = &t_opt {
            t >= &T::zero() && t <= &T::one()
        } else {
            // a == b; degenerate segment
            p.iter().zip(self.a.iter()).all(|(p, a)| p == a)
        }
    }

    fn inverse(&self) -> Self {
        Self::new(self.b(), self.a())
    }
}

pub type Segment2<T> = Segment<T, 2>;
pub type Segment3<T> = Segment<T, 3>;
