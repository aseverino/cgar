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
    pub fn from_points(a: &P, b: &P) -> Self
    where
        for<'a> &'a T: Sub<&'a T, Output = T>,
    {
        let mins = std::array::from_fn(|i| min_by_sign(&a[i], &b[i]));
        let maxs = std::array::from_fn(|i| max_by_sign(&a[i], &b[i]));
        Aabb::new(P::from_vals(mins), P::from_vals(maxs))
    }

    pub fn union(&self, other: &Aabb<T, N, P>) -> Aabb<T, N, P>
    where
        for<'a> &'a T: Sub<&'a T, Output = T>,
    {
        let mins = std::array::from_fn(|i| min_by_sign(&self.min[i], &other.min[i]));
        let maxs = std::array::from_fn(|i| max_by_sign(&self.max[i], &other.max[i]));
        Aabb::new(P::from_vals(mins), P::from_vals(maxs))
    }

    /// Does this AABB intersect `other`?
    pub fn intersects(&self, other: &Aabb<T, N, P>) -> bool
    where
        for<'a> &'a T: Sub<&'a T, Output = T>,
    {
        for i in 0..N {
            if (&self.max[i] - &other.min[i]).is_negative() {
                return false;
            }
            if (&other.max[i] - &self.min[i]).is_negative() {
                return false;
            }
        }
        true
    }

    /// Center coordinate along axis `i`.
    pub fn center(&self, i: usize) -> T
    where
        for<'a> &'a T: Add<&'a T, Output = T> + Mul<&'a T, Output = T>,
    {
        let half = T::from_num_den(1, 2);
        &(&self.min[i] + &self.max[i]) * &half
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
        for<'a> &'a T: Sub<&'a T, Output = T> + Add<&'a T, Output = T> + Mul<&'a T, Output = T>,
    {
        let mut best_i = 0usize;
        let mut best = self.extent(0);
        for i in 1..N {
            let e = self.extent(i);
            if (&e - &best).is_positive() {
                best_i = i;
                best = e;
            }
        }
        best_i
    }
}

#[inline(always)]
fn min_by_sign<T: Scalar>(a: &T, b: &T) -> T
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    if (a - b).is_negative() {
        a.clone()
    } else {
        b.clone()
    }
}
#[inline(always)]
fn max_by_sign<T: Scalar>(a: &T, b: &T) -> T
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    if (a - b).is_positive() {
        a.clone()
    } else {
        b.clone()
    }
}
