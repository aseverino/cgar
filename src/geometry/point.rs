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
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Sub, SubAssign},
};

use crate::{
    geometry::{
        spatial_element::SpatialElement,
        vector::{Vector, VectorOps},
    },
    numeric::scalar::Scalar,
    operations::Zero,
};

#[derive(Clone, Debug)]
pub struct Point<T: Scalar, const N: usize> {
    pub coords: [T; N],
}

pub trait PointOps<T: Scalar, const N: usize>: Sized {
    type Vector: VectorOps<T, N>;

    // fn distance_to(&self, other: &Self) -> T;
    // fn distance_squared_to(&self, other: &Self) -> T;
    fn sub(&self, other: &Self) -> Self;
    fn as_point_2(&self) -> Point<T, 2>;
    fn as_point_3(&self) -> Point<T, 3>;
    fn as_vector(&self) -> Self::Vector;
    fn as_vector_2(&self) -> Vector<T, 2>;
    fn as_vector_3(&self) -> Vector<T, 3>;
    fn add_vector(&self, v: &Self::Vector) -> Self;
    fn vector_to(&self, other: &Self) -> Self::Vector;
    fn midpoint(&self, other: &Self) -> Self;
}

impl<T: Scalar, const N: usize> Default for Point<T, N> {
    fn default() -> Point<T, N> {
        Point {
            coords: array::from_fn(|_| T::default()),
        }
    }
}

impl<T: Scalar, const N: usize> SpatialElement<T, N> for Point<T, N> {
    type With<U: Scalar> = Point<U, N>;

    fn new(coords: [T; N]) -> Point<T, N> {
        Point { coords }
    }

    fn from_vals<V>(vals: [V; N]) -> Point<T, N>
    where
        V: Into<T>,
    {
        Point {
            coords: vals.map(|v| v.into()),
        }
    }

    fn coords(&self) -> &[T; N] {
        &self.coords
    }

    fn coords_mut(&mut self) -> &mut [T; N] {
        &mut self.coords
    }

    fn iter(&self) -> std::slice::Iter<'_, T> {
        self.coords.iter()
    }

    fn cast<U: Scalar>(&self) -> Self::With<U>
    where
        U: From<T>,
    {
        Point::<U, N>::from_vals(from_fn(|i| U::from(self.coords[i].clone())))
    }
}

impl<T: Scalar, const N: usize> Index<usize> for Point<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &Self::Output {
        &self.coords[i]
    }
}
impl<T: Scalar, const N: usize> IndexMut<usize> for Point<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.coords[i]
    }
}

impl<'a, 'b, T, const N: usize> Add<&'b Point<T, N>> for &'a Point<T, N>
where
    T: Scalar + for<'c> AddAssign<&'c T>,
{
    type Output = Point<T, N>;
    fn add(self, rhs: &'b Point<T, N>) -> Self::Output {
        let mut out = self.clone();
        for i in 0..N {
            out.coords[i] += &rhs.coords[i];
        }
        out
    }
}

impl<T, const N: usize> Add for Point<T, N>
where
    T: Scalar,
    for<'a, 'b> &'a Point<T, N>: Add<&'b Point<T, N>, Output = Point<T, N>>,
{
    type Output = Point<T, N>;
    fn add(self, rhs: Point<T, N>) -> Self::Output {
        // call the ref-impl explicitly
        <&Point<T, N> as Add<&Point<T, N>>>::add(&self, &rhs)
    }
}

impl<'a, 'b, T, const N: usize> Sub<&'b Point<T, N>> for &'a Point<T, N>
where
    T: Scalar + for<'c> SubAssign<&'c T>,
{
    type Output = Point<T, N>;
    fn sub(self, rhs: &'b Point<T, N>) -> Self::Output {
        let mut out = self.clone();
        for i in 0..N {
            out.coords[i] -= &rhs.coords[i];
        }
        out
    }
}

impl<T, const N: usize> Sub for Point<T, N>
where
    T: Scalar,
    for<'a, 'b> &'a Point<T, N>: Sub<&'b Point<T, N>, Output = Point<T, N>>,
{
    type Output = Point<T, N>;
    fn sub(self, rhs: Point<T, N>) -> Self::Output {
        // call the ref-impl explicitly
        <&Point<T, N> as Sub<&Point<T, N>>>::sub(&self, &rhs)
    }
}

impl<T, const N: usize> Zero for Point<T, N>
where
    T: Scalar,
{
    fn zero() -> Self {
        Point {
            coords: array::from_fn(|_| T::zero()),
        }
    }

    fn is_zero(&self) -> bool {
        self.coords.iter().all(|coord| coord.is_zero())
    }
    fn is_positive(&self) -> bool {
        self.coords.iter().all(|coord| coord.is_positive())
    }
    fn is_negative(&self) -> bool {
        self.coords.iter().all(|coord| coord.is_negative())
    }
    fn is_positive_or_zero(&self) -> bool {
        self.coords.iter().all(|coord| coord.is_positive_or_zero())
    }
    fn is_negative_or_zero(&self) -> bool {
        self.coords.iter().all(|coord| coord.is_negative_or_zero())
    }
}

impl<T> PointOps<T, 2> for Point<T, 2>
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    type Vector = Vector<T, 2>;

    // fn distance_to(&self, other: &Self) -> T {
    //     (self.distance_squared_to(other)).sqrt()
    // }

    // fn distance_squared_to(&self, other: &Self) -> T {
    //     let a = (&self[0] - &other[0]).pow(2);
    //     let b = (&self[1] - &other[1]).pow(2);

    //     a + b
    // }

    fn sub(&self, other: &Self) -> Self {
        let x = &self[0] - &other[0];
        let y = &self[1] - &other[1];
        Self { coords: [x, y] }
    }

    fn as_point_2(&self) -> Self {
        self.clone()
    }

    fn as_point_3(&self) -> Point<T, 3> {
        panic!("as_vector_3 is not implemented for 2D points");
    }

    fn as_vector(&self) -> Self::Vector {
        Vector(Point::<T, 2>::from_vals([self[0].clone(), self[1].clone()]))
    }

    fn as_vector_2(&self) -> Vector<T, 2> {
        Vector(Point::<T, 2>::from_vals([self[0].clone(), self[1].clone()]))
    }

    fn as_vector_3(&self) -> Vector<T, 3> {
        panic!("as_vector_3 is not implemented for 2D points");
    }

    fn add_vector(&self, v: &Self::Vector) -> Self
    where
        T: Scalar,
    {
        Point {
            coords: [&self[0] + &v[0], &self[1] + &v[1]],
        }
    }

    fn vector_to(&self, other: &Self) -> Self::Vector {
        Vector(Point::<T, 2>::from_vals([
            &other[0] - &self[0],
            &other[1] - &self[1],
        ]))
    }

    fn midpoint(&self, other: &Self) -> Self {
        let two: T = T::from(2);
        Self {
            coords: from_fn(|i| {
                let sum = &self.coords[i] + &other.coords[i];
                &sum / &two
            }),
        }
    }
}

impl<T> PointOps<T, 3> for Point<T, 3>
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    type Vector = Vector<T, 3>;

    // fn distance_to(&self, other: &Self) -> T {
    //     (self.distance_squared_to(other)).sqrt()
    // }

    // fn distance_squared_to(&self, other: &Self) -> T {
    //     let a = (&self[0] - &other[0]).pow(2);
    //     let b = (&self[1] - &other[1]).pow(2);
    //     let c = (&self[2] - &other[2]).pow(2);
    //     let ab = a + b;

    //     ab + c
    // }

    fn sub(&self, other: &Self) -> Self {
        let x = &self[0] - &other[0];
        let y = &self[1] - &other[1];
        let z = &self[2] - &other[2];
        Self { coords: [x, y, z] }
    }

    fn as_vector(&self) -> Self::Vector {
        Vector(Point::<T, 3>::from_vals([
            self[0].clone(),
            self[1].clone(),
            self[2].clone(),
        ]))
    }

    fn as_point_2(&self) -> Point<T, 2> {
        panic!("as_point_2 is not implemented for 3D points");
    }

    fn as_point_3(&self) -> Self {
        self.clone()
    }

    fn as_vector_2(&self) -> Vector<T, 2> {
        panic!("as_vector_2 is not implemented for 3D points");
    }

    fn as_vector_3(&self) -> Vector<T, 3> {
        Vector(Point::<T, 3>::from_vals([
            self[0].clone(),
            self[1].clone(),
            self[2].clone(),
        ]))
    }

    fn add_vector(&self, v: &Self::Vector) -> Self
    where
        T: Scalar,
    {
        Point {
            coords: [&self[0] + &v[0], &self[1] + &v[1], &self[2] + &v[2]],
        }
    }

    fn vector_to(&self, other: &Self) -> Self::Vector {
        Vector(Point::<T, 3>::from_vals([
            &other[0] - &self[0],
            &other[1] - &self[1],
            &other[2] - &self[2],
        ]))
    }

    fn midpoint(&self, other: &Self) -> Self {
        let two: T = T::from(2);
        Self {
            coords: from_fn(|i| {
                let sum = &self.coords[i] + &other.coords[i];
                &sum / &two
            }),
        }
    }
}

impl<T: Scalar, const N: usize> Into<[T; N]> for Point<T, N> {
    fn into(self) -> [T; N] {
        self.coords
    }
}

impl<T: Scalar, const N: usize> From<[T; N]> for Point<T, N> {
    fn from(coords: [T; N]) -> Self {
        Point { coords }
    }
}

impl<T: Scalar, const N: usize> Hash for Point<T, N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for coord in &self.coords {
            coord.hash(state);
        }
    }
}

impl<T: Scalar, const N: usize> PartialEq for Point<T, N> {
    fn eq(&self, other: &Self) -> bool {
        if self.coords.len() != other.coords.len() {
            return false;
        }
        // Compare each coordinate for equality
        for i in 0..N {
            if self.coords[i] != other.coords[i] {
                return false;
            }
        }
        true
    }
}

impl<T: Scalar, const N: usize> PartialOrd for Point<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.coords.len() != other.coords.len() {
            return None;
        }
        for i in 0..N {
            match self.coords[i].partial_cmp(&other.coords[i]) {
                Some(std::cmp::Ordering::Equal) => continue,
                Some(ordering) => return Some(ordering),
                None => return None,
            }
        }
        Some(std::cmp::Ordering::Equal)
    }
}

impl<T: Scalar, const N: usize> Eq for Point<T, N> {}

pub type Point2<T> = Point<T, 2>;
pub type Point3<T> = Point<T, 3>;
