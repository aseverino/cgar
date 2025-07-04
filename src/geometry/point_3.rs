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
        Vector3,
        aabb::{Aabb, FromCoords},
        point::PointOps,
    },
    numeric::cgar_rational::CgarRational,
    operations::{Abs, Pow, Sqrt, Zero},
};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
pub struct Point3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Point3 {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }
}

impl<T> PointOps<T, Vector3<T>> for Point3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    type Vector = Vector3<T>;
    fn distance_to(&self, other: &Self) -> T {
        let a = &(&self.x - &other.x).pow(2);
        let b = &(&self.y - &other.y).pow(2);
        let c = &(&self.z - &other.z).pow(2);
        let ab = a + b;

        (&ab + c).sqrt()
    }

    fn sub(&self, other: &Self) -> Self {
        Self {
            x: &self.x - &other.x,
            y: &self.y - &other.y,
            z: &self.z - &other.z,
        }
    }

    fn as_vector(&self) -> Vector3<T> {
        Vector3 {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    }

    fn add_vector(&self, v: &Vector3<T>) -> Self
    where
        T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        Point3 {
            x: &self.x + &v.x,
            y: &self.y + &v.y,
            z: &self.z + &v.z,
        }
    }
}

impl<T: Clone> FromCoords<T> for Point3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn from_coords(min_coords: Vec<T>, max_coords: Vec<T>) -> Aabb<T, Self> {
        let min = Point3 {
            x: min_coords[0].clone(),
            y: min_coords[1].clone(),
            z: min_coords[2].clone(),
        };
        let max = Point3 {
            x: max_coords[0].clone(),
            y: max_coords[1].clone(),
            z: max_coords[2].clone(),
        };
        Aabb::new(min, max)
    }
}

impl Hash for Point3<f64> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Convert each f64 to its raw u64 bits and feed that to the hasher
        state.write_u64(self.x.to_bits());
        state.write_u64(self.y.to_bits());
        state.write_u64(self.z.to_bits());
    }
}

impl PartialEq for Point3<f64> {
    fn eq(&self, other: &Self) -> bool {
        // Compare the raw bit patterns, so NaNs of the *same* bit‐pattern
        // match, and +0.0 vs -0.0 are distinguished if you care.
        self.x.to_bits() == other.x.to_bits()
            && self.y.to_bits() == other.y.to_bits()
            && self.z.to_bits() == other.z.to_bits()
    }
}
impl Eq for Point3<f64> {}

impl Hash for Point3<CgarRational> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        state.write_u8(b',');
        self.y.hash(state);
        state.write_u8(b',');
        self.z.hash(state);
    }
}
impl PartialEq for Point3<CgarRational> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}
impl Eq for Point3<CgarRational> {}

impl From<Vector3<f64>> for Point3<f64> {
    fn from(v: Vector3<f64>) -> Self {
        Point3::new(v.x, v.y, v.z)
    }
}
impl From<Vector3<CgarRational>> for Point3<CgarRational> {
    fn from(v: Vector3<CgarRational>) -> Self {
        Point3::new(v.x, v.y, v.z)
    }
}
