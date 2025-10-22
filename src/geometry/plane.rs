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
        vector::{Cross2, Cross3, Vector, VectorOps},
    },
    numeric::scalar::Scalar,
};

use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct Plane<T: Scalar, const N: usize> {
    pub normal: Vector<T, N>,
    pub d: T,
}

pub trait PlaneOps<T: Scalar, const N: usize> {
    fn basis(&self) -> (Vector<T, N>, Vector<T, N>);
    fn from_points(ps: [&Point<T, N>; N]) -> Self;
    fn origin(&self) -> Point<T, N>;
}

impl<T, const N: usize> Plane<T, N>
where
    T: Scalar + PartialOrd + Neg<Output = T>,
    Vector<T, N>: VectorOps<T, N>,
{
    pub fn new(normal: Vector<T, N>, d: T) -> Self {
        Plane { normal, d }
    }

    /// Canonical form: first non-zero of `normal` is +1, scaled accordingly.
    /// (Signed zeros in `normal` are collapsed.)
    pub fn canonicalized(&self) -> Plane<T, N> {
        let mut plane = Plane::new(self.normal.clone(), self.d.clone());

        // Remove signed zeros in the normal
        for x in plane.normal.coords_mut() {
            if x.is_zero() {
                *x = T::zero();
            }
        }

        // Find first non-zero coefficient
        if let Some(c) = plane.normal.coords().iter().find(|x| !x.is_zero()).cloned() {
            // scale by |c|
            let s = c.abs();
            if !s.is_zero() {
                for x in plane.normal.coords_mut() {
                    *x = x.clone() / s.clone();
                }
                plane.d = plane.d / s;
            }
            // flip if original first non-zero was negative
            if c < T::zero() {
                for x in plane.normal.coords_mut() {
                    *x = -x.clone();
                }
                plane.d = -plane.d;
            }
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

impl<T> PlaneOps<T, 3> for Plane<T, 3>
where
    T: Scalar + Neg<Output = T>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
    Point<T, 3>: PointOps<T, 3, Vector = Vector<T, 3>>,
    Vector<T, 3>: VectorOps<T, 3> + Cross3<T>,
{
    fn basis(&self) -> (Vector<T, 3>, Vector<T, 3>)
    where
        Vector<T, 3>: VectorOps<T, 3>,
    {
        let u = self.normal.any_perpendicular();
        let v = self.normal.cross(&u);
        (u, v)
    }

    fn origin(&self) -> Point<T, 3> {
        let n = &self.normal;
        let zero = T::zero();

        if !n[2].is_zero() {
            let z = &-&self.d / &n[2];
            Point::<T, 3>::from_vals([zero.clone(), zero.clone(), z])
        } else if !n[1].is_zero() {
            let y = &-&self.d / &n[1];
            Point::<T, 3>::from_vals([zero.clone(), y, zero.clone()])
        } else if !n[0].is_zero() {
            let x = &-&self.d / &n[0];
            Point::<T, 3>::from_vals([x, zero.clone(), zero])
        } else {
            panic!("Invalid plane: zero normal");
        }
    }

    fn from_points(ps: [&Point<T, 3>; 3]) -> Self {
        let v1 = (ps[1] - ps[0]).as_vector();
        let v2 = (ps[2] - ps[0]).as_vector();
        let n = v1.cross(&v2);
        let d = -n.dot(&ps[0].as_vector());
        Plane::new(n, d)
    }
}

impl<T> PlaneOps<T, 2> for Plane<T, 2>
where
    T: Scalar + Neg<Output = T>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
    Point<T, 2>: PointOps<T, 2, Vector = Vector<T, 2>>,
    Vector<T, 2>: VectorOps<T, 2>,
{
    fn basis(&self) -> (Vector<T, 2>, Vector<T, 2>) {
        // For 2D line, basis is the tangent vector (perpendicular to normal)
        let tangent = Vector::new([self.normal[1].clone(), -self.normal[0].clone()]);
        // Second basis vector is just the normal itself for completeness
        (tangent, self.normal.clone())
    }

    fn origin(&self) -> Point<T, 2> {
        let n = &self.normal;
        let zero = T::zero();

        if !n[1].is_zero() {
            let y = &-&self.d / &n[1];
            Point::<T, 2>::from_vals([zero.clone(), y])
        } else if !n[0].is_zero() {
            let x = &-&self.d / &n[0];
            Point::<T, 2>::from_vals([x, zero])
        } else {
            panic!("Invalid line: zero normal");
        }
    }

    fn from_points(ps: [&Point<T, 2>; 2]) -> Self {
        let e = (ps[1] - ps[0]).as_vector(); // (ex, ey)
        // +90° rotation gives an in-plane normal for the line
        let n = Vector::new([-e[1].clone(), e[0].clone()]);
        let d = -n.dot(&ps[0].as_vector());
        Plane::new(n, d)
    }
}

impl<T: Scalar, const N: usize> Eq for Plane<T, N> {}
