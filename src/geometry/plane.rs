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

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{
    geometry::{
        point::{Point, PointOps},
        spatial_element::SpatialElement,
        vector::{Vector, VectorOps},
    },
    numeric::scalar::Scalar,
};

use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct Plane<T: Scalar, const N: usize> {
    pub normal: Vector<T, N>,
    pub d: T,
}
impl<T: Scalar, const N: usize> Plane<T, N>
where
    Vector<T, N>: VectorOps<T, N>,
{
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

    pub fn origin(&self) -> Point<T, 3>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>
            + Neg<Output = T>,
    {
        let zero = T::zero();
        let n = &self.normal;

        if n[2].abs().is_positive() {
            // solve for z: n·x = d → z = (d - n.x * 0 - n.y * 0) / n.z
            let z = &(&-(&self.d) / (&n[2]));
            Point::from_vals([zero.clone(), zero.clone(), z.clone()])
        } else if n[1].abs().is_positive() {
            let y = &-&self.d / &n[1];
            Point::from_vals([zero.clone(), y, zero.clone()])
        } else if n[0].abs().is_positive() {
            let x = &-&self.d / &n[0];
            Point::from_vals([x, zero.clone(), zero.clone()])
        } else {
            panic!("Invalid plane: zero normal");
        }
    }

    pub fn basis(&self) -> (Vector<T, N>, Vector<T, N>)
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        let u = self.normal.any_perpendicular();
        let v = self.normal.cross(&u);
        (u, v)
    }

    pub fn canonicalized(&self) -> Plane<T, N>
    where
        T: Scalar,
    {
        let mut plane = Plane::new(self.normal.clone(), self.d.clone());
        // Remove negative zeros
        for x in plane.normal.coords_mut() {
            if x.is_zero() {
                *x = T::zero();
            }
        }
        // Find first non-zero entry
        let mut flip = false;
        let mut scale = None;
        for x in plane.normal.coords_mut() {
            if !x.is_zero() {
                if x < &mut T::zero() {
                    flip = true;
                }
                scale = Some(x.abs());
                break;
            }
        }
        if flip {
            for x in plane.normal.coords_mut() {
                *x = -x.clone();
            }
            plane.d = -plane.d;
        }

        if let Some(scale) = scale {
            for x in plane.normal.coords_mut() {
                *x = x.clone() / scale.clone();
            }
            plane.d = plane.d / scale;
        }

        plane
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
