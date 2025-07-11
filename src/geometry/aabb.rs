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
    geometry::spatial_element::SpatialElement,
    numeric::{cgar_rational::CgarRational, scalar::Scalar},
    operations::Abs,
};
use std::{
    cmp::Ordering,
    ops::{Add, Mul, Sub},
};

/// An axis‐aligned bounding box in N dimensions.
#[derive(Clone, Debug)]
pub struct Aabb<T: Scalar, const N: usize, P: SpatialElement<T, N>> {
    pub min: P,
    pub max: P,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const N: usize, P: SpatialElement<T, N>> Aabb<T, N, P> {
    pub fn new(min: P, max: P) -> Self {
        Aabb {
            min,
            max,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn min(&self) -> &P {
        &self.min
    }

    /// Access the maximum corner.
    pub fn max(&self) -> &P {
        &self.max
    }

    /// Build the smallest AABB containing two points.
    pub fn from_points(a: &P, b: &P) -> Self {
        let mins = std::array::from_fn(|i| {
            let ai = a[i].clone();
            let bi = b[i].clone();
            if ai < bi { ai } else { bi }
        });
        let maxs = std::array::from_fn(|i| {
            let ai = a[i].clone();
            let bi = b[i].clone();
            if ai > bi { ai } else { bi }
        });
        Aabb::new(P::from_vals(mins), P::from_vals(maxs))
        // reconstruct P from coord slices
        //Aabb::new(mins, maxs.clone())
    }

    /// Expand this box to also contain `other`.
    pub fn union(&self, other: &Aabb<T, N, P>) -> Aabb<T, N, P> {
        let mins = std::array::from_fn(|i| {
            let a = self.min[i].clone();
            let b = other.min[i].clone();
            if a < b { a } else { b }
        });
        let maxs = std::array::from_fn(|i| {
            let a = self.max[i].clone();
            let b = other.max[i].clone();
            if a > b { a } else { b }
        });
        Aabb::new(P::from_vals(mins), P::from_vals(maxs))
    }

    /// Does this AABB intersect `other`?
    pub fn intersects(&self, other: &Aabb<T, N, P>) -> bool {
        for i in 0..N {
            let a_min = &self.min[i];
            let a_max = &self.max[i];
            let b_min = &other.min[i];
            let b_max = &other.max[i];
            if a_max < b_min || b_max < a_min {
                return false;
            }
        }
        true
    }

    /// Center coordinate along axis `i`.
    pub fn center(&self, i: usize) -> T
    where
        T: From<f64> + From<CgarRational>,
        for<'a> &'a T: Add<&'a T, Output = T> + Mul<&'a T, Output = T>,
    {
        let a = self.min[i].clone();
        let b = self.max[i].clone();
        &(&a + &b) * &T::from(0.5)
    }

    /// Length along axis `i`.
    fn extent(&self, i: usize) -> T
    where
        T: Abs,
        for<'a> &'a T: Sub<&'a T, Output = T>,
    {
        let a = self.min[i].clone();
        let b = self.max[i].clone();
        (&b - &a).abs()
    }

    /// Return the axis index with largest extent.
    pub fn longest_axis(&self) -> usize
    where
        T: Abs,
        for<'a> &'a T: Add<&'a T, Output = T> + Sub<&'a T, Output = T> + Mul<&'a T, Output = T>,
    {
        (0..N)
            .max_by(|&i, &j| {
                self.extent(i)
                    .partial_cmp(&self.extent(j))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap()
    }
}
