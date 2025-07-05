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
        spatial_element::SpatialElement,
    },
    mesh::point_trait::PointTrait,
    numeric::{cgar_f64::CgarF64, cgar_rational::CgarRational, scalar::Scalar},
    operations::{Pow, Sqrt, Zero},
};
use std::ops::{Div, Mul, Sub};
use std::{
    hash::{Hash, Hasher},
    ops::Add,
};

#[derive(Debug, Clone)]
pub struct Point3<T>
where
    T: Scalar,
{
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point3<T>
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

impl<'a, 'b> Add<&'b Point3<CgarF64>> for &'a Point3<CgarF64> {
    type Output = Point3<CgarF64>;
    fn add(self, rhs: &'b Point3<CgarF64>) -> Point3<CgarF64> {
        Point3 {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
            z: &self.z + &rhs.z,
        }
    }
}

impl Add for Point3<CgarRational> {
    type Output = Point3<CgarRational>;
    fn add(self, rhs: Point3<CgarRational>) -> Point3<CgarRational> {
        &self + &rhs
    }
}

impl<'a, 'b> Add<&'b Point3<CgarRational>> for &'a Point3<CgarRational> {
    type Output = Point3<CgarRational>;
    fn add(self, rhs: &'b Point3<CgarRational>) -> Point3<CgarRational> {
        Point3 {
            x: &self.x + &rhs.x,
            y: &self.y + &rhs.y,
            z: &self.z + &rhs.z,
        }
    }
}

impl Add for Point3<CgarF64> {
    type Output = Point3<CgarF64>;
    fn add(self, rhs: Point3<CgarF64>) -> Point3<CgarF64> {
        &self + &rhs
    }
}

impl<'a, 'b> Sub<&'b Point3<CgarF64>> for &'a Point3<CgarF64> {
    type Output = Point3<CgarF64>;
    fn sub(self, rhs: &'b Point3<CgarF64>) -> Point3<CgarF64> {
        Point3 {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
            z: &self.z - &rhs.z,
        }
    }
}

impl<'a, 'b> Sub<&'b Point3<CgarRational>> for &'a Point3<CgarRational> {
    type Output = Point3<CgarRational>;
    fn sub(self, rhs: &'b Point3<CgarRational>) -> Point3<CgarRational> {
        Point3 {
            x: &self.x - &rhs.x,
            y: &self.y - &rhs.y,
            z: &self.z - &rhs.z,
        }
    }
}

impl Sub for Point3<CgarF64> {
    type Output = Point3<CgarF64>;
    fn sub(self, rhs: Point3<CgarF64>) -> Point3<CgarF64> {
        &self - &rhs
    }
}

impl Sub for Point3<CgarRational> {
    type Output = Point3<CgarRational>;
    fn sub(self, rhs: Point3<CgarRational>) -> Point3<CgarRational> {
        &self - &rhs
    }
}

impl<T> Zero for Point3<T>
where
    T: Scalar,
{
    fn zero() -> Self {
        Point3 {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }
}

impl<T: Clone> PointTrait<T> for Point3<T>
where
    T: Scalar,
{
    fn dimensions() -> usize {
        3
    }
    fn coord(&self, axis: usize) -> T {
        match axis {
            0 => self.x.clone(),
            1 => self.y.clone(),
            2 => self.z.clone(),
            _ => panic!("Invalid axis"),
        }
    }
}

impl<T> PointOps<T, Vector3<T>> for Point3<T>
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    type Vector = Vector3<T>;

    fn distance_to(&self, other: &Self) -> T {
        let a = (&self.x - &other.x).pow(2);
        let b = (&self.y - &other.y).pow(2);
        let c = (&self.z - &other.z).pow(2);
        let ab = a + b;

        (ab + c).sqrt()
    }

    fn sub(&self, other: &Self) -> Self {
        let x = &self.x - &other.x;
        let y = &self.y - &other.y;
        let z = &self.z - &other.z;
        Self { x, y, z }
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
        T: Scalar,
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
    T: Scalar,
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

impl<T> Hash for Point3<T>
where
    T: Scalar,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.z.hash(state);
    }
}

impl<T> PartialEq for Point3<T>
where
    T: Scalar,
{
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl<T> PartialOrd for Point3<T>
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

impl<T> Eq for Point3<T> where T: Scalar {}

impl<T> From<(T, T, T)> for Point3<T>
where
    T: Scalar,
{
    fn from(coords: (T, T, T)) -> Self {
        Point3::new(coords.0, coords.1, coords.2)
    }
}

// impl From<Vector3<CgarF64>> for Point3<CgarF64> {
//     fn from(v: &Vector3<CgarF64>) -> Self {
//         Point3::new(v.x, v.y, v.z)
//     }
// }
// impl From<Vector3<CgarRational>> for Point3<CgarRational> {
//     fn from(v: Vector3<CgarRational>) -> Self {
//         Point3::new(v.x, v.y, v.z)
//     }
// }

impl SpatialElement<CgarF64> for Point3<CgarF64> {}
impl SpatialElement<CgarRational> for Point3<CgarRational> {}
