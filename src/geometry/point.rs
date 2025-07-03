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

use crate::operations::{Abs, Pow, Sqrt};
use std::ops::{Add, Div, Mul, Sub};

use crate::geometry::Vector2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2<T>
where
    T: Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Clone
        + PartialOrd
        + Abs
        + Pow
        + Sqrt,
{
    pub x: T,
    pub y: T,
}

impl<T> Point2<T>
where
    T: Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Clone
        + PartialOrd
        + Abs
        + Pow
        + Sqrt,
{
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point2<T>) -> T {
        ((self.x.clone() - other.x.clone()).pow(2) + (self.y.clone() - other.y.clone()).pow(2))
            .sqrt()
    }

    pub fn sub(&self, other: &Point2<T>) -> Vector2<T> {
        Vector2 {
            x: self.x.clone() - other.x.clone(),
            y: self.y.clone() - other.y.clone(),
        }
    }
}
