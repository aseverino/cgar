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

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use num_traits::ToPrimitive;
use once_cell::sync::OnceCell;
use rug::Rational;

use crate::numeric::scalar::Scalar;
use crate::numeric::{cgar_f64::CgarF64, cgar_rational::CgarRational};
use crate::operations::{Abs, Zero};

/// A lazily-evaluated scalar expression:
/// - Stores a cheap double-like approximation (CgarF64) eagerly
/// - Computes exact (CgarRational) only on demand, memoized
///
/// Design goals:
/// - Expression nodes are immutable and shared via Arc (DAG-friendly)
/// - Approx is always available quickly
/// - Exact is computed lazily and cached
#[derive(Clone)]
pub struct LazyExact(Arc<Node>);

struct Node {
    kind: Kind,
    approx: OnceCell<CgarF64>,
    exact: OnceCell<CgarRational>,
}

#[derive(Clone)]
enum Kind {
    LeafApprox(CgarF64),
    LeafExact(Arc<CgarRational>),
    Add(LazyExact, LazyExact),
    Sub(LazyExact, LazyExact),
    Mul(LazyExact, LazyExact),
    Div(LazyExact, LazyExact),
    Neg(LazyExact),
}

impl LazyExact {
    /* ========= Constructors ========= */

    pub fn from_f64(v: f64) -> Self {
        Self::from_cgar_f64(CgarF64(v))
    }

    pub fn from_i32(v: i32) -> Self {
        // represent exactly in the exact leaf; approx mirrors it
        let exact = CgarRational::from(v);
        let approx = CgarF64(v as f64);
        Self::from_leafs(exact, approx)
    }

    pub fn from_rug_rational(r: Rational) -> Self {
        Self::from_cgar_rational(CgarRational(r))
    }

    pub fn from_cgar_f64(v: CgarF64) -> Self {
        LazyExact(Arc::new(Node {
            kind: Kind::LeafApprox(v.clone()),
            approx: {
                let cell = OnceCell::new();
                let _ = cell.set(v);
                cell
            },
            exact: OnceCell::new(),
        }))
    }

    pub fn from_cgar_rational(v: CgarRational) -> Self {
        let approx = CgarF64(v.to_f64().unwrap_or(0.0));
        Self::from_leafs(v, approx)
    }

    fn from_leafs(exact: CgarRational, approx: CgarF64) -> Self {
        let exact_arc = Arc::new(exact);
        LazyExact(Arc::new(Node {
            kind: Kind::LeafExact(exact_arc.clone()),
            approx: {
                let cell = OnceCell::new();
                let _ = cell.set(approx);
                cell
            },
            exact: {
                let cell = OnceCell::new();
                // leaf exact is known eagerly; store an owned clone
                let _ = cell.set((*exact_arc).clone());
                cell
            },
        }))
    }

    /* ========= Basic queries ========= */

    /// Cheap approximate value (always available, memoized).
    pub fn approx(&self) -> CgarF64 {
        self.0
            .approx
            .get_or_init(|| match &self.0.kind {
                Kind::LeafApprox(a) => a.clone(),
                Kind::LeafExact(e) => CgarF64(e.to_f64().unwrap_or(0.0)),
                Kind::Add(a, b) => a.approx() + b.approx(),
                Kind::Sub(a, b) => a.approx() - b.approx(),
                Kind::Mul(a, b) => a.approx() * b.approx(),
                Kind::Div(a, b) => a.approx() / b.approx(),
                Kind::Neg(x) => -x.approx(),
            })
            .clone()
    }

    /// Exact value; computed lazily and cached.
    pub fn exact(&self) -> CgarRational {
        if let Some(v) = self.0.exact.get() {
            return v.clone();
        }
        let v = match &self.0.kind {
            Kind::LeafApprox(a) => CgarRational::from(a.0),
            Kind::LeafExact(e_arc) => CgarRational::clone(Arc::as_ref(e_arc)),
            Kind::Add(a, b) => &a.exact() + &b.exact(),
            Kind::Sub(a, b) => &a.exact() - &b.exact(),
            Kind::Mul(a, b) => &a.exact() * &b.exact(),
            Kind::Div(a, b) => {
                let denom = b.exact();
                assert!(!denom.is_zero(), "LazyExact: division by zero in exact()");
                &a.exact() / &denom
            }
            Kind::Neg(x) => -&x.exact(),
        };
        let _ = self.0.exact.set(v.clone());
        v
    }

    /// Sign with lazy exact fallback:
    /// - If |approx| > query_tolerance: use sign(approx)
    /// - else compute exact and use sign(exact)
    /// Returns -1, 0, or +1.
    pub fn sign(&self) -> i8 {
        let a = self.approx();
        let tol = CgarF64::query_tolerance();
        if a.abs() > tol {
            return if a.0 > 0.0 { 1 } else { -1 };
        }
        // Straddles zero → evaluate exactly.
        let e = self.exact();
        if e.is_zero() {
            0
        } else if e.is_positive() {
            1
        } else {
            -1
        }
    }

    pub fn is_zero(&self) -> bool {
        self.sign() == 0
    }

    /// Compare against zero without exposing Ordering yet (keeps it simple).
    pub fn cmp_zero(&self) -> std::cmp::Ordering {
        match self.sign() {
            -1 => std::cmp::Ordering::Less,
            0 => std::cmp::Ordering::Equal,
            _ => std::cmp::Ordering::Greater,
        }
    }

    /* ========= Internal helpers ========= */

    #[inline]
    fn new(kind: Kind) -> Self {
        // Seed approx cheaply to maximize early pruning in predicates
        let approx = match &kind {
            Kind::LeafApprox(a) => a.clone(),
            Kind::LeafExact(e) => CgarF64(e.to_f64().unwrap_or(0.0)),
            Kind::Add(a, b) => a.approx() + b.approx(),
            Kind::Sub(a, b) => a.approx() - b.approx(),
            Kind::Mul(a, b) => a.approx() * b.approx(),
            Kind::Div(a, b) => a.approx() / b.approx(),
            Kind::Neg(x) => -x.approx(),
        };
        LazyExact(Arc::new(Node {
            kind,
            approx: {
                let cell = OnceCell::new();
                let _ = cell.set(approx);
                cell
            },
            exact: OnceCell::new(),
        }))
    }
}

/* ========= Operator overloads (build expression DAGs) ========= */

impl Add for LazyExact {
    type Output = LazyExact;
    fn add(self, rhs: LazyExact) -> LazyExact {
        LazyExact::new(Kind::Add(self, rhs))
    }
}

impl<'a, 'b> Add<&'b LazyExact> for &'a LazyExact {
    type Output = LazyExact;
    fn add(self, rhs: &'b LazyExact) -> LazyExact {
        LazyExact::new(Kind::Add(self.clone(), rhs.clone()))
    }
}

impl Sub for LazyExact {
    type Output = LazyExact;
    fn sub(self, rhs: LazyExact) -> LazyExact {
        LazyExact::new(Kind::Sub(self, rhs))
    }
}

impl<'a, 'b> Sub<&'b LazyExact> for &'a LazyExact {
    type Output = LazyExact;
    fn sub(self, rhs: &'b LazyExact) -> LazyExact {
        LazyExact::new(Kind::Sub(self.clone(), rhs.clone()))
    }
}

impl Mul for LazyExact {
    type Output = LazyExact;
    fn mul(self, rhs: LazyExact) -> LazyExact {
        LazyExact::new(Kind::Mul(self, rhs))
    }
}

impl<'a, 'b> Mul<&'b LazyExact> for &'a LazyExact {
    type Output = LazyExact;
    fn mul(self, rhs: &'b LazyExact) -> LazyExact {
        LazyExact::new(Kind::Mul(self.clone(), rhs.clone()))
    }
}

impl Div for LazyExact {
    type Output = LazyExact;
    fn div(self, rhs: LazyExact) -> LazyExact {
        LazyExact::new(Kind::Div(self, rhs))
    }
}

impl<'a, 'b> Div<&'b LazyExact> for &'a LazyExact {
    type Output = LazyExact;
    fn div(self, rhs: &'b LazyExact) -> LazyExact {
        LazyExact::new(Kind::Div(self.clone(), rhs.clone()))
    }
}

impl Neg for LazyExact {
    type Output = LazyExact;
    fn neg(self) -> LazyExact {
        LazyExact::new(Kind::Neg(self))
    }
}

impl<'a> Neg for &'a LazyExact {
    type Output = LazyExact;
    fn neg(self) -> LazyExact {
        LazyExact::new(Kind::Neg(self.clone()))
    }
}

/* ========= Conversions ========= */

impl From<f64> for LazyExact {
    fn from(v: f64) -> Self {
        Self::from_f64(v)
    }
}
impl From<i32> for LazyExact {
    fn from(v: i32) -> Self {
        Self::from_i32(v)
    }
}
impl From<Rational> for LazyExact {
    fn from(v: Rational) -> Self {
        Self::from_rug_rational(v)
    }
}
impl From<CgarF64> for LazyExact {
    fn from(v: CgarF64) -> Self {
        Self::from_cgar_f64(v)
    }
}
impl From<CgarRational> for LazyExact {
    fn from(v: CgarRational) -> Self {
        Self::from_cgar_rational(v)
    }
}

/* ========= Debug ========= */

impl fmt::Debug for LazyExact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show approx eagerly; exact only if already realized
        let a = self.approx().0;
        if let Some(e) = self.0.exact.get() {
            write!(f, "LazyExact(approx={:.6}, exact={:?})", a, e)
        } else {
            write!(f, "LazyExact(approx={:.6}, exact=<lazy>)", a)
        }
    }
}

impl From<&LazyExact> for CgarF64 {
    fn from(x: &LazyExact) -> Self {
        x.approx()
    }
}
impl From<&LazyExact> for CgarRational {
    fn from(x: &LazyExact) -> Self {
        x.exact()
    }
}

impl LazyExact {
    pub fn force_exact(&self) {
        let _ = self.exact();
    }
    pub fn has_exact(&self) -> bool {
        self.0.exact.get().is_some()
    }
}
