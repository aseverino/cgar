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

use std::ops::{Add, Div, Mul, Sub};

use rug::Rational;

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

impl<'a, 'b> Sub<&'b CgarRational> for &'a CgarRational {
    type Output = CgarRational;

    fn sub(self, rhs: &'b CgarRational) -> CgarRational {
        // in‐place API on rug::Rational: result = self - rhs
        let mut result = self.0.clone();
        result -= &rhs.0;
        CgarRational(result)
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

impl<'a, 'b> Div<&'b CgarRational> for &'a CgarRational {
    type Output = CgarRational;

    fn div(self, rhs: &'b CgarRational) -> CgarRational {
        // in‐place API on rug::Rational: result = self / rhs
        let mut result = self.0.clone();
        result /= &rhs.0;
        CgarRational(result)
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
