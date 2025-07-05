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

use crate::geometry::spatial_element::SpatialElement;
use crate::geometry::vector::VectorOps;
use crate::numeric::cgar_f64::CgarF64;
use crate::numeric::cgar_rational::CgarRational;
use crate::numeric::scalar::Scalar;
use crate::operations::{Sqrt, Zero};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Vector2<T>
where
    T: Scalar,
{
    pub x: T,
    pub y: T,
}

impl<T> Vector2<T>
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

impl<'a, 'b> Add<&'b Vector2<CgarF64>> for &'a Vector2<CgarF64> {
    type Output = Vector2<CgarF64>;
    fn add(self, rhs: &'b Vector2<CgarF64>) -> Vector2<CgarF64> {
        Vector2 {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
        }
    }
}

impl Add for Vector2<CgarRational> {
    type Output = Vector2<CgarRational>;
    fn add(self, rhs: Vector2<CgarRational>) -> Vector2<CgarRational> {
        &self + &rhs
    }
}

impl<'a, 'b> Add<&'b Vector2<CgarRational>> for &'a Vector2<CgarRational> {
    type Output = Vector2<CgarRational>;
    fn add(self, rhs: &'b Vector2<CgarRational>) -> Vector2<CgarRational> {
        Vector2 {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
        }
    }
}

impl Add for Vector2<CgarF64> {
    type Output = Vector2<CgarF64>;
    fn add(self, rhs: Vector2<CgarF64>) -> Vector2<CgarF64> {
        &self + &rhs
    }
}

impl<'a, 'b> Sub<&'b Vector2<CgarF64>> for &'a Vector2<CgarF64> {
    type Output = Vector2<CgarF64>;
    fn sub(self, rhs: &'b Vector2<CgarF64>) -> Vector2<CgarF64> {
        Vector2 {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
        }
    }
}

impl<'a, 'b> Sub<&'b Vector2<CgarRational>> for &'a Vector2<CgarRational> {
    type Output = Vector2<CgarRational>;
    fn sub(self, rhs: &'b Vector2<CgarRational>) -> Vector2<CgarRational> {
        Vector2 {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
        }
    }
}

impl Sub for Vector2<CgarF64> {
    type Output = Vector2<CgarF64>;
    fn sub(self, rhs: Vector2<CgarF64>) -> Vector2<CgarF64> {
        &self - &rhs
    }
}

impl Sub for Vector2<CgarRational> {
    type Output = Vector2<CgarRational>;
    fn sub(self, rhs: Vector2<CgarRational>) -> Vector2<CgarRational> {
        &self - &rhs
    }
}

impl<T> Zero for Vector2<T>
where
    T: Scalar,
{
    fn zero() -> Self {
        Vector2 {
            x: T::zero(),
            y: T::zero(),
        }
    }
}

impl<T> Hash for Vector2<T>
where
    T: Scalar,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
    }
}

impl<T> PartialOrd for Vector2<T>
where
    T: Scalar,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.x.partial_cmp(&other.x).and_then(|ord_x| {
            if ord_x == std::cmp::Ordering::Equal {
                self.y.partial_cmp(&other.y)
            } else {
                Some(ord_x)
            }
        })
    }
}

impl<T> PartialEq for Vector2<T>
where
    T: Scalar,
{
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}
impl<T> Eq for Vector2<T> where T: Scalar {}

impl<T> VectorOps<T, T> for Vector2<T>
where
    T: Scalar,
    for<'c> &'c T: Add<&'c T, Output = T>
        + Sub<&'c T, Output = T>
        + Mul<&'c T, Output = T>
        + Div<&'c T, Output = T>,
{
    fn dot(&self, other: &Vector2<T>) -> T {
        &(&self.x * &other.x) + &(&self.y * &other.y)
    }

    fn cross(&self, other: &Vector2<T>) -> T {
        &(&self.x * &other.y) - &(&self.y * &other.x)
    }

    fn norm(&self) -> T {
        (&(&self.x * &self.x) + &(&self.y * &self.y)).sqrt()
    }

    fn normalized(&self) -> Vector2<T> {
        let n = self.norm();
        Vector2 {
            x: &self.x / &n,
            y: &self.y / &n,
        }
    }

    fn scale(&self, s: &T) -> Self
    where
        T: Scalar,
    {
        Vector2 {
            x: &self.x * &s,
            y: &self.y * &s,
        }
    }
}

impl<T> From<(T, T, T)> for Vector2<T>
where
    T: Scalar,
{
    fn from(_coords: (T, T, T)) -> Self {
        panic!("Vector2 only supports 2D coordinates, received 3D coordinates");
    }
}

impl SpatialElement<CgarF64> for Vector2<CgarF64> {}
impl SpatialElement<CgarRational> for Vector2<CgarRational> {}
