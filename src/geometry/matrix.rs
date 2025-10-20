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
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign},
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

impl<T: Scalar, const R: usize, const C: usize> Default for Matrix<T, R, C> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

// ---------- Equality / Ordering / Hash ----------
impl<T: Scalar, const R: usize, const C: usize> PartialEq for Matrix<T, R, C> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        for i in 0..R {
            for j in 0..C {
                if self[i][j] != other[i][j] {
                    return false;
                }
            }
        }
        true
    }
}
impl<T: Scalar, const R: usize, const C: usize> Eq for Matrix<T, R, C> {}

impl<T: Scalar, const R: usize, const C: usize> PartialOrd for Matrix<T, R, C> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        for i in 0..R {
            for j in 0..C {
                match self[i][j].partial_cmp(&other[i][j]) {
                    Some(std::cmp::Ordering::Equal) => continue,
                    non_eq => return non_eq,
                }
            }
        }
        Some(std::cmp::Ordering::Equal)
    }
}

impl<T: Scalar, const R: usize, const C: usize> Hash for Matrix<T, R, C> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in 0..R {
            for j in 0..C {
                self[i][j].hash(state);
            }
        }
    }
}

// ---------- Neg ----------
impl<T: Scalar, const R: usize, const C: usize> Neg for Matrix<T, R, C>
where
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let minus_one = T::from_num_den(-1, 1);
        self.scale(&minus_one)
    }
}

// ---------- Matrix * Vector ----------
impl<'a, 'b, T, const R: usize, const C: usize> Mul<&'b Vector<T, C>> for &'a Matrix<T, R, C>
where
    T: Scalar,
    for<'x> &'x T: Add<&'x T, Output = T> + Mul<&'x T, Output = T>,
{
    type Output = Vector<T, R>;
    #[inline]
    fn mul(self, v: &'b Vector<T, C>) -> Self::Output {
        let mut out = Point {
            coords: array::from_fn(|_| T::zero()),
        };
        for i in 0..R {
            // dot(row_i, v)
            let mut acc = T::zero();
            for j in 0..C {
                let prod = &self[i][j] * &v[j];
                acc = &acc + &prod;
            }
            out.coords[i] = acc;
        }
        Vector::from(out)
    }
}

impl<T, const R: usize, const C: usize> Mul<Vector<T, C>> for Matrix<T, R, C>
where
    T: Scalar,
    for<'a, 'b> &'a Matrix<T, R, C>: Mul<&'b Vector<T, C>, Output = Vector<T, R>>,
{
    type Output = Vector<T, R>;
    #[inline]
    fn mul(self, v: Vector<T, C>) -> Self::Output {
        <&Matrix<T, R, C> as Mul<&Vector<T, C>>>::mul(&self, &v)
    }
}

// ---------- Matrix * Matrix ----------
impl<'a, 'b, T, const R: usize, const C: usize, const K: usize> Mul<&'b Matrix<T, C, K>>
    for &'a Matrix<T, R, C>
where
    T: Scalar,
{
    type Output = Matrix<T, R, K>;
    #[inline]
    fn mul(self, rhs: &'b Matrix<T, C, K>) -> Self::Output {
        let mut out = Matrix::<T, R, K>::zero();
        for i in 0..R {
            for k in 0..K {
                for j in 0..C {
                    out[i][k] += &(self[i][j].clone() * rhs[j][k].clone());
                }
            }
        }
        out
    }
}

impl<T, const R: usize, const C: usize, const K: usize> Mul<Matrix<T, C, K>> for Matrix<T, R, C>
where
    T: Scalar,
    for<'a, 'b> &'a Matrix<T, R, C>: Mul<&'b Matrix<T, C, K>, Output = Matrix<T, R, K>>,
{
    type Output = Matrix<T, R, K>;
    #[inline]
    fn mul(self, rhs: Matrix<T, C, K>) -> Self::Output {
        <&Matrix<T, R, C> as Mul<&Matrix<T, C, K>>>::mul(&self, &rhs)
    }
}

// ---------- Square-only helpers ----------
impl<T: Scalar, const N: usize> Matrix<T, N, N>
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    /// Identity matrix.
    #[inline]
    pub fn identity() -> Self {
        let mut m = Matrix::<T, N, N>::zero();
        for i in 0..N {
            m[i][i] = T::one();
        }
        m
    }

    pub fn det(&self) -> T {
        match N {
            0 => T::one(),
            1 => self[0][0].clone(),
            2 => &self[0][0] * &self[1][1] - &self[0][1] * &self[1][0],
            3 => {
                let a = &self[0][0];
                let b = &self[0][1];
                let c = &self[0][2];
                let d = &self[1][0];
                let e = &self[1][1];
                let f = &self[1][2];
                let g = &self[2][0];
                let h = &self[2][1];
                let i = &self[2][2];
                a * &(e * i - f * h) - b * &(d * i - f * g) + c * &(d * h - e * g)
            }
            _ => {
                let mut det = T::zero();
                for j in 0..N {
                    let minor_data = self.minor_data(0, j);
                    let minor_det = Self::det_from_data(&minor_data, N - 1);
                    let cofactor = if j % 2 == 0 {
                        T::one()
                    } else {
                        T::from_num_den(-1, 1)
                    };
                    det = &det + &(&(&self[0][j] * &cofactor) * &minor_det);
                }
                det
            }
        }
    }

    #[inline]
    fn minor_data(&self, row: usize, col: usize) -> Vec<Vec<T>> {
        let mut data = Vec::with_capacity(N - 1);
        for i in 0..N {
            if i == row {
                continue;
            }
            let mut row_data = Vec::with_capacity(N - 1);
            for j in 0..N {
                if j == col {
                    continue;
                }
                row_data.push(self[i][j].clone());
            }
            data.push(row_data);
        }
        data
    }

    fn det_from_data(data: &[Vec<T>], size: usize) -> T {
        match size {
            0 => T::one(),
            1 => data[0][0].clone(),
            2 => &data[0][0] * &data[1][1] - &data[0][1] * &data[1][0],
            _ => {
                let mut det = T::zero();
                for j in 0..size {
                    let mut minor_data = Vec::with_capacity(size - 1);
                    for i in 1..size {
                        let mut row_data = Vec::with_capacity(size - 1);
                        for k in 0..size {
                            if k == j {
                                continue;
                            }
                            row_data.push(data[i][k].clone());
                        }
                        minor_data.push(row_data);
                    }
                    let minor_det = Self::det_from_data(&minor_data, size - 1);
                    let cofactor = if j % 2 == 0 {
                        T::one()
                    } else {
                        T::from_num_den(-1, 1)
                    };
                    det = &det + &(&(&data[0][j] * &cofactor) * &minor_det);
                }
                det
            }
        }
    }
}

// ---------- 2x2 determinant / inverse ----------
impl<T> Matrix<T, 2, 2>
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    #[inline]
    pub fn inverse(&self) -> Option<Matrix<T, 2, 2>> {
        let d = self.det();
        if d.is_zero() {
            return None;
        }
        let inv_d = &T::one() / &d;
        let a = self[0][0].clone();
        let b = self[0][1].clone();
        let c = self[1][0].clone();
        let d_ = self[1][1].clone();

        let mut m = Matrix::<T, 2, 2>::zero();
        // 1/d * [ d -b; -c a ]
        m[0][0] = &d_ * &inv_d;
        m[0][1] = &(&T::from_num_den(-1, 1) * &b) * &inv_d;
        m[1][0] = &(&T::from_num_den(-1, 1) * &c) * &inv_d;
        m[1][1] = &a * &inv_d;
        Some(m)
    }
}

// ---------- 3x3 determinant / inverse ----------
impl<T> Matrix<T, 3, 3>
where
    T: Scalar,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    #[inline]
    pub fn inverse(&self) -> Option<Matrix<T, 3, 3>> {
        let det = self.det();
        if det.is_zero() {
            return None;
        }
        let inv_det = &T::one() / &det;

        // Cofactor matrix (then transpose -> adjugate)
        let m = &self.0;
        let c00 = &m[1][1] * &m[2][2] - &m[1][2] * &m[2][1];
        let c01 = -(&m[1][0] * &m[2][2] - &m[1][2] * &m[2][0]);
        let c02 = &m[1][0] * &m[2][1] - &m[1][1] * &m[2][0];

        let c10 = -(&m[0][1] * &m[2][2] - &m[0][2] * &m[2][1]);
        let c11 = &m[0][0] * &m[2][2] - &m[0][2] * &m[2][0];
        let c12 = -(&m[0][0] * &m[2][1] - &m[0][1] * &m[2][0]);

        let c20 = &m[0][1] * &m[1][2] - &m[0][2] * &m[1][1];
        let c21 = -(&m[0][0] * &m[1][2] - &m[0][2] * &m[1][0]);
        let c22 = &m[0][0] * &m[1][1] - &m[0][1] * &m[1][0];

        // adj(A) = cofactor(A)^T
        let adj = Matrix::new([[c00, c10, c20], [c01, c11, c21], [c02, c12, c22]]);

        Some(adj.scale(&inv_det))
    }
}

// ---------- Conversions ----------
impl<T: Scalar, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T, R, C> {
    #[inline]
    fn from(data: [[T; C]; R]) -> Self {
        Matrix::new(data)
    }
}
impl<T: Scalar, const R: usize, const C: usize> Into<[[T; C]; R]> for Matrix<T, R, C> {
    #[inline]
    fn into(self) -> [[T; C]; R] {
        self.0
    }
}

// ---------- Type aliases ----------
pub type Matrix2<T> = Matrix<T, 2, 2>;
pub type Matrix3<T> = Matrix<T, 3, 3>;
pub type Matrix4<T> = Matrix<T, 4, 4>;
pub type Matrix3x4<T> = Matrix<T, 3, 4>;
pub type Matrix4x3<T> = Matrix<T, 4, 3>;
