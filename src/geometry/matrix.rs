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
    array,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign},
};

use crate::{geometry::point::Point, numeric::scalar::Scalar, operations::Zero};

// Bring the Vector type so we can multiply Matrix * Vector.
use crate::geometry::vector::Vector;

/// Generic row-major matrix R x C
#[derive(Clone, Debug)]
pub struct Matrix<T: Scalar, const R: usize, const C: usize>(pub [[T; C]; R]);

// ---------- Basics ----------
impl<T: Scalar, const R: usize, const C: usize> Matrix<T, R, C> {
    #[inline]
    pub fn new(data: [[T; C]; R]) -> Self {
        Matrix(data)
    }

    /// Matrix filled with a single value.
    #[inline]
    pub fn splat(val: T) -> Self {
        Matrix(array::from_fn(|_| array::from_fn(|_| val.clone())))
    }

    /// Cast element type.
    #[inline]
    pub fn cast<U: Scalar>(&self) -> Matrix<U, R, C>
    where
        U: From<T>,
    {
        Matrix(array::from_fn(|i| {
            array::from_fn(|j| U::from(self[i][j].clone()))
        }))
    }

    /// Transpose into C x R.
    #[inline]
    pub fn transpose(&self) -> Matrix<T, C, R> {
        Matrix(array::from_fn(|j| array::from_fn(|i| self[i][j].clone())))
    }

    /// Scale all entries by `s`.
    #[inline]
    pub fn scale(&self, s: &T) -> Self
    where
        for<'a> &'a T: Mul<&'a T, Output = T>,
    {
        Matrix(array::from_fn(|i| array::from_fn(|j| &self[i][j] * s)))
    }

    /// Build from rows.
    #[inline]
    pub fn from_rows(rows: [[T; C]; R]) -> Self {
        Matrix(rows)
    }

    /// Build from columns.
    #[inline]
    pub fn from_cols(cols: [[T; R]; C]) -> Self {
        let mut m = Matrix::<T, R, C>::zero();
        for i in 0..R {
            for j in 0..C {
                m[i][j] = cols[j][i].clone();
            }
        }
        m
    }

    /// Get a row as a Vector<T, C>.
    #[inline]
    pub fn row(&self, r: usize) -> Vector<T, C> {
        let mut v = Point {
            coords: array::from_fn(|_| T::zero()),
        };
        for j in 0..C {
            v.coords[j] = self[r][j].clone();
        }
        Vector::from(v)
    }

    /// Get a column as a Vector<T, R>.
    #[inline]
    pub fn col(&self, c: usize) -> Vector<T, R> {
        let mut v = Point {
            coords: array::from_fn(|_| T::zero()),
        };
        for i in 0..R {
            v.coords[i] = self[i][c].clone();
        }
        Vector::from(v)
    }

    /// Linear interpolation (element-wise): (1-u)*A + u*B
    #[inline]
    pub fn lerp(&self, other: &Self, u: &T) -> Self
    where
        for<'a> &'a T: Add<&'a T, Output = T> + Sub<&'a T, Output = T> + Mul<&'a T, Output = T>,
    {
        let one = T::one();
        let w0 = &one - u;
        let w1 = u;
        &self.scale(&w0) + &other.scale(w1)
    }

    /// Outer product: a (R) * bᵀ (C) => R x C
    #[inline]
    pub fn outer(a: &Vector<T, R>, b: &Vector<T, C>) -> Matrix<T, R, C>
    where
        for<'a> &'a T: Mul<&'a T, Output = T>,
    {
        Matrix(array::from_fn(|i| array::from_fn(|j| &a[i] * &b[j])))
    }
}

// ---------- Indexing ----------
impl<T: Scalar, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> {
    type Output = [T; C];
    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}
impl<T: Scalar, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0[i]
    }
}

// ---------- Zero ----------
impl<T: Scalar, const R: usize, const C: usize> Zero for Matrix<T, R, C> {
    #[inline]
    fn zero() -> Self {
        Matrix(array::from_fn(|_| array::from_fn(|_| T::zero())))
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|row| row.iter().all(|x| x.is_zero()))
    }
    #[inline]
    fn is_positive(&self) -> bool {
        self.0.iter().all(|row| row.iter().all(|x| x.is_positive()))
    }
    #[inline]
    fn is_negative(&self) -> bool {
        !self.is_positive() && !self.is_zero()
    }
    #[inline]
    fn is_positive_or_zero(&self) -> bool {
        self.is_positive() || self.is_zero()
    }
    #[inline]
    fn is_negative_or_zero(&self) -> bool {
        self.is_negative() || self.is_zero()
    }
}

// ---------- Add / Sub (by-ref primary impls) ----------
impl<'a, 'b, T, const R: usize, const C: usize> Add<&'b Matrix<T, R, C>> for &'a Matrix<T, R, C>
where
    T: Scalar + for<'c> AddAssign<&'c T>,
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn add(self, rhs: &'b Matrix<T, R, C>) -> Self::Output {
        let mut out = self.clone();
        for i in 0..R {
            for j in 0..C {
                out[i][j] += &rhs[i][j];
            }
        }
        out
    }
}
impl<T, const R: usize, const C: usize> Add for Matrix<T, R, C>
where
    T: Scalar,
    for<'a, 'b> &'a Matrix<T, R, C>: Add<&'b Matrix<T, R, C>, Output = Matrix<T, R, C>>,
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn add(self, rhs: Matrix<T, R, C>) -> Self::Output {
        <&Matrix<T, R, C> as Add<&Matrix<T, R, C>>>::add(&self, &rhs)
    }
}

impl<'a, 'b, T, const R: usize, const C: usize> Sub<&'b Matrix<T, R, C>> for &'a Matrix<T, R, C>
where
    T: Scalar + for<'c> SubAssign<&'c T>,
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn sub(self, rhs: &'b Matrix<T, R, C>) -> Self::Output {
        let mut out = self.clone();
        for i in 0..R {
            for j in 0..C {
                out[i][j] -= &rhs[i][j];
            }
        }
        out
    }
}
impl<T, const R: usize, const C: usize> Sub for Matrix<T, R, C>
where
    T: Scalar,
    for<'a, 'b> &'a Matrix<T, R, C>: Sub<&'b Matrix<T, R, C>, Output = Matrix<T, R, C>>,
{
    type Output = Matrix<T, R, C>;
    #[inline]
    fn sub(self, rhs: Matrix<T, R, C>) -> Self::Output {
        <&Matrix<T, R, C> as Sub<&Matrix<T, R, C>>>::sub(&self, &rhs)
    }
}
