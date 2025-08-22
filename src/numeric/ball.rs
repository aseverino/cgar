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

#[derive(Copy, Clone, Debug)]
pub struct Ball {
    pub m: f64,
    pub r: f64,
} // value ∈ [m - r, m + r]

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // uses FMA when available (compile with -C target-cpu=native)
    let err = f64::mul_add(a, b, -p);
    (p, err)
}

impl Ball {
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        Ball { m: x, r: 0.0 }
    }
    #[inline]
    pub fn unknown() -> Self {
        Ball {
            m: 0.0,
            r: f64::INFINITY,
        }
    }

    #[inline]
    pub fn add(self, o: Self) -> Self {
        let (s, e) = two_sum(self.m, o.m);
        Ball {
            m: s,
            r: self.r + o.r + e.abs(),
        }
    }
    #[inline]
    pub fn sub(self, o: Self) -> Self {
        self.add(Ball { m: -o.m, r: o.r })
    }
    #[inline]
    pub fn neg(self) -> Self {
        Ball {
            m: -self.m,
            r: self.r,
        }
    }

    #[inline]
    pub fn mul(self, o: Self) -> Self {
        let (p, e) = two_prod(self.m, o.m);
        Ball {
            m: p,
            r: self.m.abs() * o.r + o.m.abs() * self.r + e.abs(),
        }
    }
    // Safe but conservative division; if denom straddles zero, return “unknown”
    #[inline]
    pub fn div(self, o: Self) -> Self {
        if o.m.abs() <= o.r {
            return Ball::unknown();
        }
        let denom = o.m.abs() * (o.m.abs() - o.r);
        if denom <= 0.0 {
            return Ball::unknown();
        }
        let m = self.m / o.m;
        let r = (self.m.abs() * o.r + o.m.abs() * self.r) / denom;
        Ball { m, r }
    }

    #[inline]
    pub fn sign_if_certain(self) -> Option<i8> {
        if self.r.is_infinite() {
            return None;
        }
        if self.m > self.r {
            Some(1)
        } else if self.m < -self.r {
            Some(-1)
        } else {
            None
        }
    }
}
