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

use std::ops::{Add, Div, Index, Mul, Sub};

use crate::{geometry::point::Point, numeric::scalar::Scalar};

#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalEdge<T: Scalar, const N: usize> {
    pub a: Point<T, N>,
    pub b: Point<T, N>,
}

impl<T: Scalar, const N: usize> CanonicalEdge<T, N> {
    pub fn new(p1: &Point<T, N>, p2: &Point<T, N>) -> Self {
        if p1 < p2 {
            Self {
                a: p1.clone(),
                b: p2.clone(),
            }
        } else {
            Self {
                a: p2.clone(),
                b: p1.clone(),
            }
        }
    }

    pub fn parameter_t(&self, p: &Point<T, N>) -> T
    where
        Point<T, N>: Index<usize, Output = T>,
        for<'x> &'x T: Sub<&'x T, Output = T>
            + Mul<&'x T, Output = T>
            + Add<&'x T, Output = T>
            + Div<&'x T, Output = T>,
    {
        let edge_vec = &self.b - &self.a;
        let point_vec = p - &self.a;

        let edge_len2 = &(&edge_vec[0] * &edge_vec[0]) + &(&edge_vec[1] * &edge_vec[1]);

        if edge_len2.is_zero() {
            return T::zero();
        }

        let dot = &(&point_vec[0] * &edge_vec[0]) + &(&point_vec[1] * &edge_vec[1]);
        dot / edge_len2
    }
}
