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

use super::Point2;
use crate::operations::{Abs, Pow, Sqrt};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, PartialEq)]
pub struct Segment2<T>
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
    pub a: Point2<T>,
    pub b: Point2<T>,
}

impl<T> Segment2<T>
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
    pub fn new(a: &Point2<T>, b: &Point2<T>) -> Self {
        Self {
            a: a.clone(),
            b: b.clone(),
        }
    }

    pub fn length(&self) -> T {
        self.a.distance_to(&self.b)
    }

    pub fn midpoint(&self) -> Point2<T> {
        Point2 {
            //x: (self.a.x + self.b.x) / 2.0,
            //y: (self.a.y + self.b.y) / 2.0,
            x: self.a.x.clone() + self.b.x.clone(),
            y: self.a.y.clone() + self.b.y.clone(),
        }
    }
}
