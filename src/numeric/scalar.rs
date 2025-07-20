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
use rug::Rational;

use crate::{
    numeric::{cgar_f64::CgarF64, cgar_rational::CgarRational},
    operations::*,
};

use std::{
    hash::Hash,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

use std::fmt::Debug;

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
    + Pow
    + Sqrt
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
{
    fn min(self, other: Self) -> Self {
        if self < other { self } else { other }
    }
    fn max(self, other: Self) -> Self {
        if self > other { self } else { other }
    }

    fn from_num_den(num: i32, den: i32) -> Self;
}
