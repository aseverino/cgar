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
use rug::{Rational, ops::Pow};

use crate::{
    numeric::{cgar_f64::CgarF64, scalar::Scalar},
    operations::{Abs, Sqrt},
};

use std::{
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Neg, Sub},
};

// use std::ops::{Add, Div, Mul, Sub};

// use num_traits::ToPrimitive;
// use rug::Rational;
// use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub struct CgarRational(pub Rational);

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
        // get references to the internally reduced numerator & denominator
        let num = self.0.numer();
        let den = self.0.denom();
        // serialize each to a decimal string (exact, no loss)
        // you could also use `to_string_radix(16)` if you prefer hex.
        num.to_string().hash(state);
        // you might want a separator so that 12/3 and 1/23 don’t collide:
        state.write_u8(b'/');
        den.to_string().hash(state);
    }
}

impl Eq for CgarRational {}

impl crate::operations::Zero for CgarRational {
    fn zero() -> Self {
        CgarRational(Rational::from(0))
    }
}

impl crate::operations::One for CgarRational {
    fn one() -> Self {
        CgarRational(Rational::from(1))
    }
}

impl Sqrt for CgarRational {
    fn sqrt(&self) -> Self {
        let sqrt_val = self.0.to_f64().sqrt();
        CgarRational::from(sqrt_val)
    }
}

impl crate::operations::Pow for CgarRational {
    fn pow(&self, exp: i32) -> Self {
        CgarRational(self.0.clone().pow(exp))
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

impl Scalar for CgarRational {}
