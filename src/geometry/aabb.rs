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

use crate::{mesh::point_trait::PointTrait, operations::Abs};
use std::{
    cmp::Ordering,
    ops::{Add, Mul, Sub},
};

/// An axis‐aligned bounding box in N dimensions.
#[derive(Clone, Debug)]
pub struct Aabb<T, P: PointTrait<T>> {
    pub min: P,
    pub max: P,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, P> Aabb<T, P>
where
    T: PartialOrd + Clone,
    P: PointTrait<T> + FromCoords<T>,
{
    pub fn new(min: P, max: P) -> Self {
        Aabb {
            min,
            max,
            _phantom: std::marker::PhantomData,
        }
    }
    /// Build the smallest AABB containing two points.
    pub fn from_points(a: &P, b: &P) -> Self {
        let dim = P::dimensions();
        let mut mins = Vec::with_capacity(dim);
        let mut maxs = Vec::with_capacity(dim);
        for i in 0..dim {
            let ai = a.coord(i);
            let bi = b.coord(i);
            mins.push(if ai.clone() < bi.clone() {
                ai.clone()
            } else {
                bi.clone()
            });
            maxs.push(if ai > bi { ai } else { bi });
        }
        // reconstruct P from coord slices
        P::from_coords(mins.clone(), maxs.clone())
    }

    /// Expand this box to also contain `other`.
    pub fn union(&self, other: &Aabb<T, P>) -> Aabb<T, P> {
        let dim = P::dimensions();
        let mut mins = Vec::with_capacity(dim);
        let mut maxs = Vec::with_capacity(dim);
        for i in 0..dim {
            let a = self.min.coord(i);
            let b = other.min.coord(i);
            mins.push(if a.clone() < b.clone() { a } else { b });

            let a = self.max.coord(i);
            let b = other.max.coord(i);
            maxs.push(if a.clone() > b.clone() { a } else { b });
        }
        P::from_coords(mins.clone(), maxs.clone())
    }

    /// Does this AABB intersect `other`?
    pub fn intersects(&self, other: &Aabb<T, P>) -> bool {
        let dim = P::dimensions();
        for i in 0..dim {
            let a_min = self.min.coord(i);
            let a_max = self.max.coord(i);
            let b_min = other.min.coord(i);
            let b_max = other.max.coord(i);
            if a_max < b_min || b_max < a_min {
                return false;
            }
        }
        true
    }

    /// Center coordinate along axis `i`.
    pub fn center(&self, i: usize) -> T
    where
        T: From<f64>,
        for<'a> &'a T: Add<&'a T, Output = T> + Mul<&'a T, Output = T>,
    {
        let a = self.min.coord(i).clone();
        let b = self.max.coord(i).clone();
        &(&a + &b) * &T::from(0.5)
    }

    /// Length along axis `i`.
    fn extent(&self, i: usize) -> T
    where
        T: Abs,
        for<'a> &'a T: Sub<&'a T, Output = T>,
    {
        let a = self.min.coord(i).clone();
        let b = self.max.coord(i).clone();
        (&b - &a).abs()
    }

    /// Return the axis index with largest extent.
    pub fn longest_axis(&self) -> usize
    where
        T: Abs,
        for<'a> &'a T: Add<&'a T, Output = T> + Sub<&'a T, Output = T> + Mul<&'a T, Output = T>,
    {
        let dim = P::dimensions();
        (0..dim)
            .max_by(|&i, &j| {
                self.extent(i)
                    .partial_cmp(&self.extent(j))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap()
    }
}

// helper to build P from coordinate vectors—add this to your Point2/Point3 implementations:
pub trait FromCoords<T> {
    fn from_coords(min_coords: Vec<T>, max_coords: Vec<T>) -> Aabb<T, Self>
    where
        Self: PointTrait<T> + Sized;
}
