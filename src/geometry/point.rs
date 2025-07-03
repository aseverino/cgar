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
    geometry::aabb::{Aabb, FromCoords},
    operations::{Abs, Pow, Sqrt},
};
use std::ops::{Add, Div, Mul, Sub};

use crate::geometry::Vector2;

pub trait PointOps<T>: Sized {
    fn distance_to(&self, other: &Self) -> T;
    fn sub(&self, other: &Self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T> PointOps<T> for Point2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn distance_to(&self, other: &Self) -> T {
        (&(&self.x - &other.x).pow(2) + &(&self.y - &other.y).pow(2)).sqrt()
    }

    fn sub(&self, other: &Self) -> Self {
        Self {
            x: &self.x - &other.x,
            y: &self.y - &other.y,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T> PointOps<T> for Point3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
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
