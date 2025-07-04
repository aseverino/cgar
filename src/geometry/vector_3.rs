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

use crate::geometry::vector::VectorOps;
use crate::numeric::cgar_rational::CgarRational;
use crate::operations::{Abs, Pow, Sqrt, Zero};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone)]
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

    fn scale(&self, s: &T) -> Self
    where
        T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero,
        for<'a> &'a T: Add<&'a T, Output = T>
            + Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        Vector3 {
            x: &self.x * &s,
            y: &self.y * &s,
            z: &self.z * &s,
        }
    }
}

impl Hash for Vector3<f64> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Convert each f64 to its raw u64 bits and feed that to the hasher
        state.write_u64(self.x.to_bits());
        state.write_u64(self.y.to_bits());
        state.write_u64(self.z.to_bits());
    }
}

impl PartialEq for Vector3<f64> {
    fn eq(&self, other: &Self) -> bool {
        // Compare the raw bit patterns, so NaNs of the *same* bit‐pattern
        // match, and +0.0 vs -0.0 are distinguished if you care.
        self.x.to_bits() == other.x.to_bits()
            && self.y.to_bits() == other.y.to_bits()
            && self.z.to_bits() == other.z.to_bits()
    }
}
impl Eq for Vector3<f64> {}

impl Hash for Vector3<CgarRational> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        state.write_u8(b',');
        self.y.hash(state);
        state.write_u8(b',');
        self.z.hash(state);
    }
}
impl PartialEq for Vector3<CgarRational> {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}
impl Eq for Vector3<CgarRational> {}
