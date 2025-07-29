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

use std::ops::{Add, Div, Mul, Sub};

use crate::{
    geometry::{
        point::{Point, PointOps},
        vector::{Vector, VectorOps},
    },
    numeric::scalar::Scalar,
};

use std::hash::{Hash, Hasher};

pub struct Plane<T: Scalar, const N: usize> {
    pub normal: Vector<T, N>,
    pub d: T,
}
impl<T: Scalar, const N: usize> Plane<T, N> {
    pub fn new(normal: Vector<T, N>, d: T) -> Self {
        Plane { normal, d }
    }

    pub fn from_points(p1: &Point<T, N>, p2: &Point<T, N>, p3: &Point<T, N>) -> Self
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let v1 = (p2 - p1).as_vector();
        let v2 = (p3 - p1).as_vector();
        let normal = v1.cross(&v2);
        let d = -normal.dot(&p1.as_vector());
        Plane::new(normal, d)
    }
}

impl<T: Scalar, const N: usize> Hash for Plane<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.normal.hash(state);
        self.d.hash(state);
    }
}

impl<T: Scalar, const N: usize> PartialEq for Plane<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.normal == other.normal && self.d == other.d
    }
}

impl<T: Scalar, const N: usize> PartialOrd for Plane<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.normal == other.normal {
            self.d.partial_cmp(&other.d)
        } else {
            self.normal.partial_cmp(&other.normal)
        }
    }
}

impl<T: Scalar, const N: usize> Eq for Plane<T, N> {}
