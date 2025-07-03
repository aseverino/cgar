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

use crate::geometry::Point2;
use crate::operations::{Abs, Pow, Sqrt};
use std::ops::{Add, Div, Mul, Sub};

/// Returns:
/// - >0 if counter-clockwise
/// - <0 if clockwise
/// - =0 if collinear
pub fn orient2d<T>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>) -> T
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    &(&(&b.x - &a.x) * &(&c.y - &a.y)) - &(&(&b.y - &a.y) * &(&c.x - &a.x))
}
#[cfg(test)]
mod tests {
    use crate::geometry::Point2;
    use crate::kernel::orient2d;

    #[test]
    fn ccw_test() {
        let a = Point2 { x: 0.0, y: 0.0 };
        let b = Point2 { x: 1.0, y: 0.0 };
        let c = Point2 { x: 0.0, y: 1.0 };

        assert!(orient2d(&a, &b, &c) > 0.0); // Counter-clockwise
    }
}
