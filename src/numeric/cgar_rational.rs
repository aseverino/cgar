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
    numeric::{
        cgar_f64::CgarF64,
        lazy_exact::LazyExact,
        scalar::{FromRef, RefInto, Scalar},
    },
    operations::Abs,
};

use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

#[derive(Clone)]
pub struct CgarRational(pub Rational);

impl Default for CgarRational {
    fn default() -> Self {
        CgarRational(Rational::from((0, 1)))
    }
}

impl Scalar for CgarRational {
    fn from_num_den(num: i32, den: i32) -> Self {
        CgarRational(Rational::from((num, den)))
    }

    fn tolerance() -> Self {
        return 0.into(); // no tolerance for exact rational numbers
    }

    fn tolerance_squared() -> Self {
        let tol = Self::tolerance();
        &tol * &tol
    }

    // CGAL-style: separate thresholds for different purposes
    fn point_merge_threshold() -> Self {
        CgarRational::from_num_den(1, 1000000) // Merge points closer than this
    }

    fn edge_degeneracy_threshold() -> Self {
        CgarRational::from_num_den(1, 100000) // Remove edges shorter than this
    }

    fn area_degeneracy_threshold() -> Self {
        CgarRational::from_num_den(1, 10000000) // Remove faces smaller than this
    }

    fn query_tolerance() -> Self {
        CgarRational::from_num_den(1, 100000)
    }

    fn query_tolerance_squared() -> Self {
        let tol = Self::query_tolerance();
        &tol * &tol
    }

    fn point_merge_threshold_squared() -> Self {
        let tol = Self::point_merge_threshold();
        &tol * &tol
    }

    /// Returns -1, 0, or +1.
    fn sign(&self) -> i8 {
        if self.0.is_positive() {
            1
        } else if self.0.is_negative() {
            -1
        } else {
            0
        }
    }

    fn approx_eq(&self, other: &Self) -> bool {
        // For rational numbers, we can use exact equality
        self.0 == other.0
    }

    #[inline(always)]
    fn cmp_ref(a: &Self, b: &Self) -> Ordering {
        // (a.num/a.den) ? (b.num/b.den)  =>  a.num*b.den ? b.num*a.den
        (a.0.numer().clone() * b.0.denom().clone())
            .cmp(&(b.0.numer().clone() * a.0.denom().clone()))
    }

    fn as_f64_fast(&self) -> Option<f64> {
        Some(self.0.to_f64())
    }

    fn double_interval(&self) -> Option<(f64, f64)> {
        Some((self.0.to_f64(), self.0.to_f64()))
    }

    fn ball_center_f64(&self) -> f64 {
        self.0.to_f64()
    }
}

impl<'a, 'b> Add<&'b CgarRational> for &'a CgarRational {
    type Output = CgarRational;

    fn add(self, rhs: &'b CgarRational) -> CgarRational {
        // in‐place API on rug::Rational: result = self + rhs
        let mut result = self.0.clone();
        result += &rhs.0;
        CgarRational(result)
    }
}

impl Add for CgarRational {
    type Output = CgarRational;
    fn add(self, rhs: CgarRational) -> CgarRational {
        &self + &rhs // just borrow both and reuse the existing logic
    }
}

impl<'a, 'b> Sub<&'b CgarRational> for &'a CgarRational {
    type Output = CgarRational;

    fn sub(self, rhs: &'b CgarRational) -> CgarRational {
        // in‐place API on rug::Rational: result = self - rhs
        let mut result = self.0.clone();
        result -= &rhs.0;
        CgarRational(result)
    }
}

impl Sub for CgarRational {
    type Output = CgarRational;
    fn sub(self, rhs: CgarRational) -> CgarRational {
        &self - &rhs // just borrow both and reuse the existing logic
    }
}

impl<'a, 'b> Mul<&'b CgarRational> for &'a CgarRational {
    type Output = CgarRational;

    fn mul(self, rhs: &'b CgarRational) -> CgarRational {
        // in‐place API on rug::Rational: result = self * rhs
        let mut result = self.0.clone();
        result *= &rhs.0;
        CgarRational(result)
    }
}

impl Mul for CgarRational {
    type Output = CgarRational;
    fn mul(self, rhs: CgarRational) -> CgarRational {
        &self * &rhs // just borrow both and reuse the existing logic
    }
}

impl<'a, 'b> Div<&'b CgarRational> for &'a CgarRational {
    type Output = CgarRational;

    fn div(self, rhs: &'b CgarRational) -> CgarRational {
        // in‐place API on rug::Rational: result = self / rhs
        let mut result = self.0.clone();
        result /= &rhs.0;
        CgarRational(result)
    }
}

impl Div for CgarRational {
    type Output = CgarRational;
    fn div(self, rhs: CgarRational) -> CgarRational {
        &self / &rhs // just borrow both and reuse the existing logic
    }
}

impl<'c> AddAssign<&'c CgarRational> for CgarRational {
    fn add_assign(&mut self, rhs: &'c CgarRational) {
        self.0 += &rhs.0;
    }
}

impl<'d> SubAssign<&'d CgarRational> for CgarRational {
    fn sub_assign(&mut self, rhs: &'d CgarRational) {
        self.0 -= &rhs.0;
    }
}

impl From<i32> for CgarRational {
    fn from(value: i32) -> Self {
        CgarRational(Rational::from(value))
    }
}

impl From<f64> for CgarRational {
    fn from(value: f64) -> Self {
        CgarRational(Rational::from_f64(value).expect("Invalid f64 value"))
    }
}

impl From<CgarF64> for CgarRational {
    fn from(value: CgarF64) -> Self {
        CgarRational(Rational::from_f64(value.0).expect("Invalid f64 value"))
    }
}

impl From<rug::Rational> for CgarRational {
    fn from(value: rug::Rational) -> Self {
        CgarRational(value)
    }
}

impl From<rug::Float> for CgarRational {
    fn from(value: rug::Float) -> Self {
        CgarRational(value.to_rational().expect("Invalid Float value"))
    }
}

impl From<LazyExact> for CgarRational {
    fn from(value: LazyExact) -> Self {
        value.exact()
    }
}

impl ToPrimitive for CgarRational {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0.to_f64() as i64)
    }
    fn to_u64(&self) -> Option<u64> {
        Some(self.0.to_f64() as u64)
    }
    fn to_f32(&self) -> Option<f32> {
        Some(self.0.to_f64() as f32)
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.0.to_f64())
    }
}

impl FromRef<CgarF64> for CgarRational {
    fn from_ref(value: &CgarF64) -> Self {
        CgarRational(Rational::from_f64(value.0).expect("Invalid f64 value"))
    }
}

impl FromRef<CgarRational> for CgarRational {
    fn from_ref(value: &CgarRational) -> Self {
        CgarRational(value.0.clone())
    }
}

impl FromRef<LazyExact> for CgarRational {
    fn from_ref(value: &LazyExact) -> Self {
        value.exact()
    }
}

impl RefInto<CgarF64> for CgarRational {
    fn ref_into(&self) -> CgarF64 {
        CgarF64(self.0.to_f64())
    }
}

impl RefInto<CgarRational> for CgarRational {
    fn ref_into(&self) -> CgarRational {
        return self.clone();
    }
}

impl RefInto<LazyExact> for CgarRational {
    fn ref_into(&self) -> LazyExact {
        LazyExact::from_cgar_rational(self.clone())
    }
}

impl PartialEq for CgarRational {
    fn eq(&self, other: &CgarRational) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for CgarRational {
    fn partial_cmp(&self, other: &CgarRational) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Hash for CgarRational {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let num_bytes = self.0.numer().to_digits::<u8>(rug::integer::Order::MsfBe);
        let den_bytes = self.0.denom().to_digits::<u8>(rug::integer::Order::MsfBe);
        state.write(&num_bytes);
        state.write_u8(b'/');
        state.write(&den_bytes);
    }
}

impl Eq for CgarRational {}

impl crate::operations::Zero for CgarRational {
    fn zero() -> Self {
        CgarRational(Rational::from(0))
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn is_positive(&self) -> bool {
        self.0.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.0.is_negative()
    }

    fn is_positive_or_zero(&self) -> bool {
        !self.0.is_negative()
    }

    fn is_negative_or_zero(&self) -> bool {
        !self.0.is_positive()
    }
}

impl crate::operations::One for CgarRational {
    fn one() -> Self {
        CgarRational(Rational::from(1))
    }
}

impl Abs for CgarRational {
    fn abs(&self) -> Self {
        CgarRational(self.0.clone().abs())
    }
}

impl Neg for CgarRational {
    type Output = CgarRational;

    fn neg(self) -> CgarRational {
        CgarRational(-self.0)
    }
}

impl<'a> Neg for &'a CgarRational {
    type Output = CgarRational;

    fn neg(self) -> CgarRational {
        CgarRational(self.0.clone().neg())
    }
}

impl fmt::Debug for CgarRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print as f64, optionally with more precision
        write!(f, "{:.5}", self.0.to_f64())
    }
}
