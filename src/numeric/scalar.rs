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

use num_traits::ToPrimitive;

use crate::{
    numeric::{cgar_f64::CgarF64, cgar_rational::CgarRational, lazy_exact::LazyExact},
    operations::*,
};

use std::{
    hash::Hash,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

use std::fmt::Debug;

pub trait FromRef<T> {
    fn from_ref(value: &T) -> Self;
}

pub trait RefInto<T> {
    fn ref_into(&self) -> T;
}

pub trait Scalar:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + Debug
    + Abs
    + Zero
    + One
    + Neg<Output = Self>
    + Eq
    + PartialEq
    + ToPrimitive
    + Hash
    + PartialOrd
    + From<i32>
    + From<f64>
    + From<rug::Rational>
    + From<CgarF64>
    + From<CgarRational>
    + From<LazyExact>
    + Into<CgarF64>
    + FromRef<CgarF64>
    + FromRef<CgarRational>
    + FromRef<LazyExact>
    + RefInto<CgarF64>
    + RefInto<CgarRational>
    + RefInto<LazyExact>
{
    fn min(self, other: Self) -> Self {
        if self < other { self } else { other }
    }
    fn max(self, other: Self) -> Self {
        if self > other { self } else { other }
    }

    fn default() -> Self {
        Self::from(0)
    }
    fn from_num_den(num: i32, den: i32) -> Self;

    fn cmp_ref(a: &Self, b: &Self) -> core::cmp::Ordering;

    fn tolerance() -> Self;

    fn tolerance_squared() -> Self;

    // CGAL-style: separate thresholds for different purposes
    fn point_merge_threshold() -> Self;

    fn edge_degeneracy_threshold() -> Self;

    fn area_degeneracy_threshold() -> Self;

    fn query_tolerance() -> Self;

    fn query_tolerance_squared() -> Self;

    fn point_merge_threshold_squared() -> Self;

    fn approx_eq(&self, other: &Self) -> bool;
    // fn acos(&self) -> Self;
}
