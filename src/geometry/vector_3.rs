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

use crate::geometry::Point3;
use crate::geometry::vector::VectorOps;
use crate::numeric::cgar_f64::CgarF64;
use crate::numeric::cgar_rational::CgarRational;
use crate::numeric::scalar::Scalar;
use crate::operations::{Sqrt, Zero};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Vector3<T>
where
    T: Scalar,
{
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vector3<T>
where
    T: Scalar,
{
    pub fn new<X, Y, Z>(x: X, y: Y, z: Z) -> Self
    where
        X: Into<T>,
        Y: Into<T>,
        Z: Into<T>,
    {
        Self {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }
}

impl<'a, 'b> Add<&'b Vector3<CgarF64>> for &'a Vector3<CgarF64> {
    type Output = Vector3<CgarF64>;
    fn add(self, rhs: &'b Vector3<CgarF64>) -> Vector3<CgarF64> {
        Vector3 {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
            z: &self.z + &rhs.z,
        }
    }
}

impl Add for Vector3<CgarRational> {
    type Output = Vector3<CgarRational>;
    fn add(self, rhs: Vector3<CgarRational>) -> Vector3<CgarRational> {
        &self + &rhs
    }
}

impl<'a, 'b> Add<&'b Vector3<CgarRational>> for &'a Vector3<CgarRational> {
    type Output = Vector3<CgarRational>;
    fn add(self, rhs: &'b Vector3<CgarRational>) -> Vector3<CgarRational> {
        Vector3 {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
            z: &self.z + &rhs.z,
        }
    }
}

impl Add for Vector3<CgarF64> {
    type Output = Vector3<CgarF64>;
    fn add(self, rhs: Vector3<CgarF64>) -> Vector3<CgarF64> {
        &self + &rhs
    }
}

impl<'a, 'b> Sub<&'b Vector3<CgarF64>> for &'a Vector3<CgarF64> {
    type Output = Vector3<CgarF64>;
    fn sub(self, rhs: &'b Vector3<CgarF64>) -> Vector3<CgarF64> {
        Vector3 {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
            z: &self.z - &rhs.z,
        }
    }
}

impl<'a, 'b> Sub<&'b Vector3<CgarRational>> for &'a Vector3<CgarRational> {
    type Output = Vector3<CgarRational>;
    fn sub(self, rhs: &'b Vector3<CgarRational>) -> Vector3<CgarRational> {
        Vector3 {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
            z: &self.z - &rhs.z,
        }
    }
}

impl Sub for Vector3<CgarF64> {
    type Output = Vector3<CgarF64>;
    fn sub(self, rhs: Vector3<CgarF64>) -> Vector3<CgarF64> {
        &self - &rhs
    }
}

impl Sub for Vector3<CgarRational> {
    type Output = Vector3<CgarRational>;
    fn sub(self, rhs: Vector3<CgarRational>) -> Vector3<CgarRational> {
        &self - &rhs
    }
}

impl<T> Zero for Vector3<T>
where
    T: Scalar,
{
    fn zero() -> Self {
        Vector3 {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }
}

impl<T> Hash for Vector3<T>
where
    T: Scalar,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.z.hash(state);
    }
}

impl<T> PartialOrd for Vector3<T>
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
        let z_cmp = self.z.partial_cmp(&other.z);
        if z_cmp.is_none() {
            return None;
        }

        match (x_cmp, y_cmp, z_cmp) {
            (Some(x), Some(y), Some(z)) => {
                if x == std::cmp::Ordering::Equal && y == std::cmp::Ordering::Equal {
                    Some(z)
                } else if x == std::cmp::Ordering::Equal {
                    Some(y)
                } else {
                    Some(x)
                }
            }
            _ => None,
        }
    }
}

impl<T> PartialEq for Vector3<T>
where
    T: Scalar,
{
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}
impl<T> Eq for Vector3<T> where T: Scalar {}

impl<T> VectorOps<T, Vector3<T>> for Vector3<T>
where
    T: Scalar,
    for<'c> &'c T: Add<&'c T, Output = T>
        + Sub<&'c T, Output = T>
        + Mul<&'c T, Output = T>
        + Div<&'c T, Output = T>,
{
    fn dot(&self, other: &Vector3<T>) -> T {
        let a = &(&self.x * &other.x);
        let b = &(&self.y * &other.y);
        let c = &(&self.z * &other.z);
        &(&(&self.x * &other.x) + &(&self.y * &other.y)) + &(&self.z * &other.z)
    }

    fn cross(&self, other: &Vector3<T>) -> Vector3<T> {
        Vector3 {
            x: &(&self.y * &other.z) - &(&self.z * &other.y),
            y: &(&self.z * &other.x) - &(&self.x * &other.z),
            z: &(&self.x * &other.y) - &(&self.y * &other.x),
        }
    }

    fn norm(&self) -> T {
        (&(&(&self.x * &self.x) + &(&self.y * &self.y)) + &(&self.z * &self.z)).sqrt()
    }

    fn normalized(&self) -> Vector3<T> {
        let n = self.norm();
        Vector3 {
            x: &self.x / &n,
            y: &self.y / &n,
            z: &self.z / &n,
        }
    }

    fn scale(&self, s: &T) -> Self
    where
        T: Scalar,
    {
        Vector3 {
            x: &self.x * &s,
            y: &self.y * &s,
            z: &self.z * &s,
        }
    }
}

impl<T> From<Point3<T>> for Vector3<T>
where
    T: Scalar,
{
    fn from(point: Point3<T>) -> Self {
        Vector3 {
            x: point.x,
            y: point.y,
            z: point.z,
        }
    }
}

impl<T> From<(T, T, T)> for Vector3<T>
where
    T: Scalar,
{
    fn from(coords: (T, T, T)) -> Self {
        Vector3::new(coords.0, coords.1, coords.2)
    }
}
