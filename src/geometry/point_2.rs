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

use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

use crate::numeric::cgar_rational::CgarRational;
use crate::{
    geometry::{Aabb, FromCoords, Vector2, point::PointOps},
    operations::{Abs, Pow, Sqrt, Zero},
};

#[derive(Debug, Clone)]
pub struct Point2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    pub x: T,
    pub y: T,
}

impl<T> Point2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Point2 {
            x: T::zero(),
            y: T::zero(),
        }
    }
}

impl<T> PointOps<T, T> for Point2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    type Vector = Vector2<T>;

    fn distance_to(&self, other: &Self) -> T {
        (&(&self.x - &other.x).pow(2) + &(&self.y - &other.y).pow(2)).sqrt()
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
        T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        Point2 {
            x: &self.x + &v.x,
            y: &self.y + &v.y,
        }
    }
}

impl<T: Clone> FromCoords<T> for Point2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
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

impl Hash for Point2<f64> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Convert each f64 to its raw u64 bits and feed that to the hasher
        state.write_u64(self.x.to_bits());
        state.write_u64(self.y.to_bits());
    }
}

impl PartialEq for Point2<f64> {
    fn eq(&self, other: &Self) -> bool {
        // Compare the raw bit patterns, so NaNs of the *same* bit‐pattern
        // match, and +0.0 vs -0.0 are distinguished if you care.
        self.x.to_bits() == other.x.to_bits() && self.y.to_bits() == other.y.to_bits()
    }
}
impl Eq for Point2<f64> {}

impl Hash for Point2<CgarRational> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        state.write_u8(b',');
        self.y.hash(state);
    }
}
impl PartialEq for Point2<CgarRational> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
}
impl Eq for Point2<CgarRational> {}
