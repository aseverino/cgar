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

use core::panic;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

use crate::geometry::spatial_element::SpatialElement;
use crate::mesh::point_trait::PointTrait;
use crate::numeric::cgar_f64::CgarF64;
use crate::numeric::cgar_rational::CgarRational;
use crate::numeric::scalar::Scalar;
use crate::{
    geometry::{Aabb, FromCoords, Vector2, point::PointOps},
    operations::{Pow, Sqrt, Zero},
};

#[derive(Debug, Clone)]
pub struct Point2<T>
where
    T: Scalar,
{
    pub x: T,
    pub y: T,
}

impl<T> Point2<T>
where
    T: Scalar,
{
    pub fn new<X, Y>(x: X, y: Y) -> Self
    where
        X: Into<T>,
        Y: Into<T>,
    {
        Self {
            x: x.into(),
            y: y.into(),
        }
    }
}

impl<'a, 'b> Add<&'b Point2<CgarF64>> for &'a Point2<CgarF64> {
    type Output = Point2<CgarF64>;
    fn add(self, rhs: &'b Point2<CgarF64>) -> Point2<CgarF64> {
        Point2 {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
        }
    }
}

impl Add for Point2<CgarRational> {
    type Output = Point2<CgarRational>;
    fn add(self, rhs: Point2<CgarRational>) -> Point2<CgarRational> {
        &self + &rhs
    }
}

impl<'a, 'b> Add<&'b Point2<CgarRational>> for &'a Point2<CgarRational> {
    type Output = Point2<CgarRational>;
    fn add(self, rhs: &'b Point2<CgarRational>) -> Point2<CgarRational> {
        Point2 {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
        }
    }
}

impl Add for Point2<CgarF64> {
    type Output = Point2<CgarF64>;
    fn add(self, rhs: Point2<CgarF64>) -> Point2<CgarF64> {
        &self + &rhs
    }
}

impl<'a, 'b> Sub<&'b Point2<CgarF64>> for &'a Point2<CgarF64> {
    type Output = Point2<CgarF64>;
    fn sub(self, rhs: &'b Point2<CgarF64>) -> Point2<CgarF64> {
        Point2 {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
        }
    }
}

impl<'a, 'b> Sub<&'b Point2<CgarRational>> for &'a Point2<CgarRational> {
    type Output = Point2<CgarRational>;
    fn sub(self, rhs: &'b Point2<CgarRational>) -> Point2<CgarRational> {
        Point2 {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
        }
    }
}

impl Sub for Point2<CgarF64> {
    type Output = Point2<CgarF64>;
    fn sub(self, rhs: Point2<CgarF64>) -> Point2<CgarF64> {
        &self - &rhs
    }
}

impl Sub for Point2<CgarRational> {
    type Output = Point2<CgarRational>;
    fn sub(self, rhs: Point2<CgarRational>) -> Point2<CgarRational> {
        &self - &rhs
    }
}

impl<T> Zero for Point2<T>
where
    T: Scalar,
{
    fn zero() -> Self {
        Point2 {
            x: T::zero(),
            y: T::zero(),
        }
    }
}

impl<T> PointOps<T, T> for Point2<T>
where
    T: Scalar,
    Vector2<T>: SpatialElement<T>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    type Vector = Vector2<T>;

    fn distance_to(&self, other: &Self) -> T {
        let a = &self.x - &other.x;
        let b = &self.y - &other.y;
        (&a.pow(2) + &b.pow(2)).sqrt()
    }

    fn sub(&self, other: &Self) -> Self {
        Self {
            x: &self.x - &other.x,
            y: &self.y - &other.y,
        }
    }

    fn as_vector(&self) -> Vector2<T> {
        Vector2 {
            x: self.x.clone(),
            y: self.y.clone(),
        }
    }

    fn add_vector(&self, v: &Vector2<T>) -> Self
    where
        T: Scalar,
    {
        Point2 {
            x: &self.x - &v.x,
            y: &self.y - &v.y,
        }
    }
}

impl<T: Clone> PointTrait<T> for Point2<T>
where
    T: Scalar,
{
    fn dimensions() -> usize {
        2
    }
    fn coord(&self, axis: usize) -> T {
        match axis {
            0 => self.x.clone(),
            1 => self.y.clone(),
            _ => panic!("Invalid axis"),
        }
    }
}

impl<T: Clone> FromCoords<T> for Point2<T>
where
    T: Scalar,
{
    fn from_coords(min_coords: Vec<T>, max_coords: Vec<T>) -> Aabb<T, Self> {
        // we know dimensions() == 2
        let min = Point2 {
            x: min_coords[0].clone(),
            y: min_coords[1].clone(),
        };
        let max = Point2 {
            x: max_coords[0].clone(),
            y: max_coords[1].clone(),
        };
        Aabb::new(min, max)
    }
}

impl<T> Hash for Point2<T>
where
    T: Scalar,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
    }
}

impl<T> PartialEq for Point2<T>
where
    T: Scalar,
{
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}

impl<T> PartialOrd for Point2<T>
where
    T: Scalar,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let x_cmp = self.x.partial_cmp(&other.x);
        if x_cmp.is_none() {
            return None;
        }
        let y_cmp = self.y.partial_cmp(&other.y);
        if y_cmp.is_none() {
            return None;
        }

        match (x_cmp, y_cmp) {
            (Some(x), Some(y)) => {
                if x == std::cmp::Ordering::Equal {
                    Some(y)
                } else {
                    Some(x)
                }
            }
            _ => None,
        }
    }
}

impl<T> Eq for Point2<T> where T: Scalar {}

impl<T> From<(T, T, T)> for Point2<T>
where
    T: Scalar,
{
    fn from(_coords: (T, T, T)) -> Self {
        panic!("Point2 only supports 2D coordinates, received 3D coordinates");
    }
}

impl SpatialElement<CgarF64> for Point2<CgarF64> {}
impl SpatialElement<CgarRational> for Point2<CgarRational> {}
