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

use crate::operations::{Abs, Pow, Sqrt, Zero};
use std::ops::{Add, Div, Mul, Sub};

pub trait VectorOps<T, C>: Sized {
    fn dot(&self, other: &Self) -> T;
    fn cross(&self, other: &Self) -> C;
    fn norm(&self) -> T;
    fn normalized(&self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2<T>
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

impl<T> Vector2<T>
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

impl<T> VectorOps<T, T> for Vector2<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vector3<T>
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
        Vector3 {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }
}

impl<T> VectorOps<T, Vector3<T>> for Vector3<T>
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn dot(&self, other: &Vector3<T>) -> T {
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
}
