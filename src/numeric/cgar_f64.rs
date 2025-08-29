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
    geometry::util::EPS,
    numeric::{
        cgar_rational::CgarRational,
        lazy_exact::LazyExact,
        scalar::{FromRef, RefInto, Scalar},
    },
    operations::{Abs, One, Zero},
};

use std::{
    cmp::Ordering,
    hash::Hash,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

#[derive(Clone, Debug)]
pub struct CgarF64(pub f64);

impl Scalar for CgarF64 {
    fn from_num_den(num: i32, den: i32) -> Self {
        CgarF64(num as f64 / den as f64)
    }

    fn tolerance() -> Self {
        return Self(EPS);
    }

    fn tolerance_squared() -> Self {
        let tol = Self::tolerance();
        CgarF64(tol.0 * tol.0)
    }

    // CGAL-style: separate thresholds for different purposes
    fn point_merge_threshold() -> Self {
        CgarF64(1e-6)
    }

    fn edge_degeneracy_threshold() -> Self {
        CgarF64(1e-5)
    }

    fn area_degeneracy_threshold() -> Self {
        CgarF64(1e-10) // Remove faces smaller than this
    }

    fn query_tolerance() -> Self {
        CgarF64(1e-10)
    }

    fn query_tolerance_squared() -> Self {
        let tol = Self::query_tolerance();
        CgarF64(tol.0 * tol.0)
    }

    fn point_merge_threshold_squared() -> Self {
        let tol = Self::point_merge_threshold();
        CgarF64(tol.0 * tol.0)
    }

    /// Returns -1, 0, or +1.
    fn sign(&self) -> i8 {
        if self.is_positive() {
            1
        } else if self.is_negative() {
            -1
        } else {
            0
        }
    }

    fn approx_eq(&self, other: &Self) -> bool {
        return (self.0 - other.0).abs() < EPS;
    }

    #[inline(always)]
    fn cmp_ref(a: &Self, b: &Self) -> Ordering {
        // total_cmp handles -0.0 and NaN deterministically
        a.0.total_cmp(&b.0)
    }
}

impl<'a, 'b> Add<&'b CgarF64> for &'a CgarF64 {
    type Output = CgarF64;

    fn add(self, rhs: &'b CgarF64) -> CgarF64 {
        // in‐place API on rug::Rational: result = self + rhs
        let mut result = self.0.clone();
        result += &rhs.0;
        CgarF64(result)
    }
}

impl Add for CgarF64 {
    type Output = CgarF64;
    fn add(self, rhs: CgarF64) -> CgarF64 {
        &self + &rhs // just borrow both and reuse the existing logic
    }
}

impl<'a, 'b> Sub<&'b CgarF64> for &'a CgarF64 {
    type Output = CgarF64;

    fn sub(self, rhs: &'b CgarF64) -> CgarF64 {
        // in‐place API on rug::Rational: result = self - rhs
        let mut result = self.0.clone();
        result -= &rhs.0;
        CgarF64(result)
    }
}

impl Sub for CgarF64 {
    type Output = CgarF64;
    fn sub(self, rhs: CgarF64) -> CgarF64 {
        &self - &rhs // just borrow both and reuse the existing logic
    }
}

impl<'a, 'b> Mul<&'b CgarF64> for &'a CgarF64 {
    type Output = CgarF64;

    fn mul(self, rhs: &'b CgarF64) -> CgarF64 {
        // in‐place API on rug::Rational: result = self * rhs
        let mut result = self.0.clone();
        result *= &rhs.0;
        CgarF64(result)
    }
}

impl Mul for CgarF64 {
    type Output = CgarF64;
    fn mul(self, rhs: CgarF64) -> CgarF64 {
        &self * &rhs // just borrow both and reuse the existing logic
    }
}

impl<'a, 'b> Div<&'b CgarF64> for &'a CgarF64 {
    type Output = CgarF64;

    fn div(self, rhs: &'b CgarF64) -> CgarF64 {
        // in‐place API on rug::Rational: result = self / rhs
        let mut result = self.0.clone();
        result /= &rhs.0;
        CgarF64(result)
    }
}

impl<'c> AddAssign<&'c CgarF64> for CgarF64 {
    fn add_assign(&mut self, rhs: &'c CgarF64) {
        self.0 += &rhs.0;
    }
}

impl<'d> SubAssign<&'d CgarF64> for CgarF64 {
    fn sub_assign(&mut self, rhs: &'d CgarF64) {
        self.0 -= &rhs.0;
    }
}

impl Div for CgarF64 {
    type Output = CgarF64;
    fn div(self, rhs: CgarF64) -> CgarF64 {
        &self / &rhs // just borrow both and reuse the existing logic
    }
}

impl From<i32> for CgarF64 {
    fn from(value: i32) -> Self {
        CgarF64(value as f64)
    }
}

impl From<f64> for CgarF64 {
    fn from(value: f64) -> Self {
        CgarF64(value)
    }
}

impl From<CgarF64> for f64 {
    fn from(value: CgarF64) -> Self {
        value.0
    }
}

impl From<rug::Rational> for CgarF64 {
    fn from(value: rug::Rational) -> Self {
        CgarF64(value.to_f64())
    }
}

impl From<CgarRational> for CgarF64 {
    fn from(value: CgarRational) -> Self {
        CgarF64(value.0.to_f64())
    }
}

impl From<LazyExact> for CgarF64 {
    fn from(value: LazyExact) -> Self {
        value.approx()
    }
}

impl ToPrimitive for CgarF64 {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0 as i64)
    }
    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }
    fn to_f32(&self) -> Option<f32> {
        Some(self.0 as f32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.0)
    }
}

impl FromRef<CgarF64> for CgarF64 {
    fn from_ref(value: &CgarF64) -> Self {
        value.clone()
    }
}

impl FromRef<CgarRational> for CgarF64 {
    fn from_ref(value: &CgarRational) -> Self {
        CgarF64(value.to_f64().unwrap_or(0.0))
    }
}

impl FromRef<LazyExact> for CgarF64 {
    fn from_ref(value: &LazyExact) -> Self {
        value.approx()
    }
}

impl RefInto<CgarF64> for CgarF64 {
    fn ref_into(&self) -> CgarF64 {
        self.clone()
    }
}

impl RefInto<CgarRational> for CgarF64 {
    fn ref_into(&self) -> CgarRational {
        CgarRational::from(self.0)
    }
}

impl RefInto<LazyExact> for CgarF64 {
    fn ref_into(&self) -> LazyExact {
        LazyExact::from_cgar_f64(self.clone())
    }
}

impl PartialEq for CgarF64 {
    fn eq(&self, other: &CgarF64) -> bool {
        self.0.to_bits() == other.0.to_bits() // Comparing with tolerance breaks the hashing contract
    }
}

impl PartialOrd for CgarF64 {
    fn partial_cmp(&self, other: &CgarF64) -> Option<std::cmp::Ordering> {
        let diff = self.0 - other.0;
        if diff.abs() < EPS {
            return Some(std::cmp::Ordering::Equal);
        }
        if diff > EPS {
            return Some(std::cmp::Ordering::Greater);
        }
        if diff < -EPS {
            return Some(std::cmp::Ordering::Less);
        }
        self.0.partial_cmp(&other.0)
    }
}

impl Hash for CgarF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl Eq for CgarF64 {}

impl crate::operations::Zero for CgarF64 {
    fn zero() -> Self {
        CgarF64(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0.abs() < EPS
    }

    fn is_positive(&self) -> bool {
        self.0 > EPS
    }
    fn is_negative(&self) -> bool {
        self.0 < -EPS
    }
    fn is_positive_or_zero(&self) -> bool {
        self.0 >= -EPS
    }
    fn is_negative_or_zero(&self) -> bool {
        self.0 <= EPS
    }
}

impl One for CgarF64 {
    fn one() -> Self {
        CgarF64(1.0)
    }
}

impl Abs for CgarF64 {
    fn abs(&self) -> Self {
        CgarF64(self.0.abs())
    }
}

impl Neg for CgarF64 {
    type Output = CgarF64;

    fn neg(self) -> CgarF64 {
        CgarF64(-self.0)
    }
}

impl<'a> Neg for &'a CgarF64 {
    type Output = CgarF64;

    fn neg(self) -> CgarF64 {
        CgarF64(-&self.0)
    }
}
