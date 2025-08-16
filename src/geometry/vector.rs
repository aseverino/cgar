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

use std::{
    array::{self, from_fn},
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign},
};

use crate::{
    geometry::{point::Point, spatial_element::SpatialElement},
    numeric::scalar::Scalar,
    operations::Zero,
};

pub trait VectorOps<T, const N: usize>: Sized
where
    T: Scalar,
{
    type Cross;

    fn dot(&self, other: &Self) -> T;
    fn cross(&self, other: &Self) -> Self::Cross;
    fn norm(&self) -> T;
    fn norm_squared(&self) -> T;
    fn normalized(&self) -> Self;
    fn scale(&self, s: &T) -> Self;
    fn any_perpendicular(&self) -> Vector<T, N>;
}

#[derive(Clone, Debug)]
pub struct Vector<T: Scalar, const N: usize>(pub Point<T, N>);

impl<T: Scalar, const N: usize> SpatialElement<T, N> for Vector<T, N> {
    fn new(coords: [T; N]) -> Vector<T, N> {
        Vector(Point { coords })
    }

    fn from_vals<V>(vals: [V; N]) -> Vector<T, N>
    where
        V: Into<T>,
    {
        Vector(Point::from_vals(vals.map(|v| v.into())))
    }

    fn coords(&self) -> &[T; N] {
        &self.0.coords
    }

    fn coords_mut(&mut self) -> &mut [T; N] {
        &mut self.0.coords
    }

    fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.coords.iter()
    }
}

impl<T: Scalar, const N: usize> Default for Vector<T, N> {
    fn default() -> Self {
        Vector::from(Point::default())
    }
}

// conveniences
impl<T: Scalar, const N: usize> From<Point<T, N>> for Vector<T, N> {
    fn from(p: Point<T, N>) -> Self {
        Vector(p)
    }
}
impl<T: Scalar, const N: usize> From<Vector<T, N>> for Point<T, N> {
    fn from(v: Vector<T, N>) -> Self {
        v.0
    }
}

impl<T: Scalar, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0.coords[i]
    }
}
impl<T: Scalar, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0.coords[i]
    }
}

impl<'a, 'b, T, const N: usize> Add<&'b Vector<T, N>> for &'a Vector<T, N>
where
    T: Scalar + for<'c> AddAssign<&'c T>,
{
    type Output = Vector<T, N>;
    fn add(self, rhs: &'b Vector<T, N>) -> Self::Output {
        let mut out = self.clone();
        for i in 0..N {
            out[i] += &rhs[i];
        }
        out
    }
}

impl<T, const N: usize> Add for Vector<T, N>
where
    T: Scalar,
    for<'a, 'b> &'a Vector<T, N>: Add<&'b Vector<T, N>, Output = Vector<T, N>>,
{
    type Output = Vector<T, N>;
    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        // call the ref-impl explicitly
        <&Vector<T, N> as Add<&Vector<T, N>>>::add(&self, &rhs)
    }
}

impl<'a, 'b, T, const N: usize> Sub<&'b Vector<T, N>> for &'a Vector<T, N>
where
    T: Scalar + for<'c> SubAssign<&'c T>,
{
    type Output = Vector<T, N>;
    fn sub(self, rhs: &'b Vector<T, N>) -> Self::Output {
        let mut out = self.clone();
        for i in 0..N {
            out[i] -= &rhs[i];
        }
        out
    }
}

impl<T, const N: usize> Sub for Vector<T, N>
where
    T: Scalar,
    for<'a, 'b> &'a Vector<T, N>: Sub<&'b Vector<T, N>, Output = Vector<T, N>>,
{
    type Output = Vector<T, N>;
    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        // call the ref-impl explicitly
        <&Vector<T, N> as Sub<&Vector<T, N>>>::sub(&self, &rhs)
    }
}

impl<T> VectorOps<T, 2> for Vector<T, 2>
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    type Cross = T; // <- a pseudo-scalar

    fn dot(&self, o: &Self) -> T {
        &self[0] * &o[0] + &self[1] * &o[1]
    }

    fn cross(&self, o: &Self) -> T {
        &self[0] * &o[1] - &self[1] * &o[0]
    }

    fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> T {
        &(&self[0] * &self[0]) + &(&self[1] * &self[1])
    }

    fn normalized(&self) -> Self {
        let n = self.norm();
        Self::from(Point::from_vals([&self[0] / &n, &self[1] / &n]))
    }

    fn scale(&self, s: &T) -> Self {
        Self::from(Point::from_vals([&self[0] * &s, &self[1] * &s]))
    }

    fn any_perpendicular(&self) -> Vector<T, 2> {
        unimplemented!("any_perpendicular is not implemented for 2D vectors");
    }
}

impl<T> VectorOps<T, 3> for Vector<T, 3>
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    type Cross = Vector<T, 3>;

    fn dot(&self, o: &Self) -> T {
        &self[0] * &o[0] + &self[1] * &o[1] + &self[2] * &o[2]
    }

    fn cross(&self, o: &Self) -> Self::Cross {
        Vector::from(Point::from_vals([
            &(&self[1] * &o[2]) - &(&self[2] * &o[1]),
            &(&self[2] * &o[0]) - &(&self[0] * &o[2]),
            &(&self[0] * &o[1]) - &(&self[1] * &o[0]),
        ]))
    }

    fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    fn norm_squared(&self) -> T {
        &(&(&self[0] * &self[0]) + &(&self[1] * &self[1])) + &(&self[2] * &self[2])
    }

    fn normalized(&self) -> Self {
        let n = self.norm();
        Self::from(Point::from_vals([
            &self[0] / &n,
            &self[1] / &n,
            &self[2] / &n,
        ]))
    }

    fn scale(&self, s: &T) -> Self {
        Self::from(Point::from_vals([
            &self[0] * &s,
            &self[1] * &s,
            &self[2] * &s,
        ]))
    }

    fn any_perpendicular(&self) -> Vector<T, 3> {
        if self[0].abs() < self[1].abs() && self[0].abs() < self[2].abs() {
            self.cross(&Vector::new([T::one(), T::zero(), T::zero()]))
        } else if self[1].abs() < self[2].abs() {
            self.cross(&Vector::new([T::zero(), T::one(), T::zero()]))
        } else {
            self.cross(&Vector::new([T::zero(), T::zero(), T::one()]))
        }
    }
}

impl<T, const N: usize> Zero for Vector<T, N>
where
    T: Scalar,
{
    fn zero() -> Self {
        Vector::from(Point {
            coords: array::from_fn(|_| T::zero()),
        })
    }
    fn is_zero(&self) -> bool {
        self.0.coords.iter().all(|c| c.is_zero())
    }
    fn is_positive(&self) -> bool {
        self.0.coords.iter().all(|c| c.is_positive())
    }
    fn is_negative(&self) -> bool {
        !self.is_positive() && !self.is_zero()
    }
    fn is_positive_or_zero(&self) -> bool {
        self.is_positive() || self.is_zero()
    }
    fn is_negative_or_zero(&self) -> bool {
        self.is_negative() || self.is_zero()
    }
}

impl<T: Scalar, const N: usize> Into<[T; N]> for Vector<T, N> {
    fn into(self) -> [T; N] {
        self.0.coords
    }
}

impl<T: Scalar, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(coords: [T; N]) -> Self {
        Vector(Point { coords })
    }
}

impl<T: Scalar, const N: usize> Hash for Vector<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for coord in &self.0.coords {
            coord.hash(state);
        }
    }
}

impl<T: Scalar, const N: usize> PartialEq for Vector<T, N> {
    fn eq(&self, other: &Self) -> bool {
        if self.0.coords.len() != other.0.coords.len() {
            return false;
        }
        // Compare each coordinate for equality
        for i in 0..N {
            if self.0.coords[i] != other.0.coords[i] {
                return false;
            }
        }
        true
    }
}

impl<T: Scalar, const N: usize> PartialOrd for Vector<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.0.coords.len() != other.0.coords.len() {
            return None;
        }
        for i in 0..N {
            match self.0.coords[i].partial_cmp(&other.0.coords[i]) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ordering) => return Some(ordering),
                None => return None,
            }
        }
        Some(std::cmp::Ordering::Equal)
    }
}
impl<T: Scalar, const N: usize> Eq for Vector<T, N> {}

impl<T: Scalar, const N: usize> Neg for Vector<T, N>
where
    for<'a> &'a T: Mul<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self {
        Vector::from(Point::from_vals(from_fn(|i| {
            &self[i] * &T::from_num_den(-1, 1)
        })))
    }
}

pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;
