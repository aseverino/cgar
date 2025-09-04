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

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::sync::Arc;

use num_traits::ToPrimitive;
use once_cell::sync::OnceCell;
use rug::Rational;

use crate::geometry::util::{f64_next_down, f64_next_up};
use crate::numeric::ball::Ball;
use crate::numeric::scalar::{FromRef, RefInto, Scalar};
use crate::numeric::{cgar_f64::CgarF64, cgar_rational::CgarRational};
use crate::operations::{Abs, One, Zero};

use std::sync::atomic::AtomicBool;

pub static ENABLE_PANIC_ON_EXACT: AtomicBool = AtomicBool::new(false);

#[derive(Clone)]
pub struct LazyExact(Arc<Node>);

impl Default for LazyExact {
    fn default() -> Self {
        LazyExact::from_i32(0)
    }
}

impl Scalar for LazyExact {
    fn min(self, other: Self) -> Self {
        if (&self - &other).sign() <= 0 {
            self
        } else {
            other
        }
    }
    fn max(self, other: Self) -> Self {
        if (&self - &other).sign() >= 0 {
            self
        } else {
            other
        }
    }

    fn from_num_den(num: i32, den: i32) -> Self {
        LazyExact::from_cgar_rational(CgarRational::from_num_den(num, den))
    }

    fn tolerance() -> Self {
        LazyExact::from_i32(0)
    }

    fn tolerance_squared() -> Self {
        let t = Self::tolerance();
        t.clone() * t
    }

    fn point_merge_threshold() -> Self {
        LazyExact::from_num_den(1, 1_000_000)
    }

    fn edge_degeneracy_threshold() -> Self {
        LazyExact::from_num_den(1, 100_000)
    }

    fn area_degeneracy_threshold() -> Self {
        LazyExact::from_num_den(1, 10_000_000)
    }

    fn query_tolerance() -> Self {
        LazyExact::from_num_den(1, 100_000)
    }

    fn query_tolerance_squared() -> Self {
        let t = Self::query_tolerance();
        t.clone() * t
    }

    fn point_merge_threshold_squared() -> Self {
        let t = Self::point_merge_threshold();
        t.clone() * t
    }

    /// Sign with lazy exact fallback:
    /// - If |approx| > query_tolerance: use sign(approx)
    /// - else compute exact and use sign(exact)
    /// Returns -1, 0, or +1.
    fn sign(&self) -> i8 {
        let b = self.ball_ref();
        if let Some(s) = b.sign_if_certain() {
            return s; // decided in double with a sound, expression-specific bound
        }
        // Uncertain -> exact
        let e = self.exact();
        if e.is_zero() {
            0
        } else if e.is_positive() {
            1
        } else {
            -1
        }
    }

    // approximate equality:
    // - fast path: |approx(self - other)| <= query_tolerance
    // - else: exact equality
    fn approx_eq(&self, other: &Self) -> bool {
        let diff = self - other;
        let b = diff.ball_ref();
        // User “query tolerance” becomes a *margin* on top of proven radius
        let tol = CgarF64::query_tolerance().0;
        if b.m.abs() <= b.r + tol {
            return true;
        }
        diff.exact().is_zero()
    }

    #[inline(always)]
    fn cmp_ref(a: &Self, b: &Self) -> Ordering {
        use core::cmp::Ordering::*;

        // 0) Cheap pointer equality
        if std::sync::Arc::ptr_eq(&a.0, &b.0) {
            return Equal;
        }

        // 1) Fast interval filter (if you have it)
        if let (Some((alo, ahi)), Some((blo, bhi))) = (a.double_interval(), b.double_interval()) {
            if ahi < blo {
                return Less;
            }
            if bhi < alo {
                return Greater;
            }
        }

        // 2) If both are exact doubles (or both reduced to machine floats)
        if let (Some(da), Some(db)) = (a.as_f64_if_exact(), b.as_f64_if_exact()) {
            return da.total_cmp(&db);
        }

        // 3) Exact comparator WITHOUT building (a-b) node.
        // Implement this inside LazyExact to walk/evaluate just enough to
        // decide sign(a - b) and return {-1,0,1}.
        match LazyExact::cmp_exact(a, b) {
            -1 => Less,
            0 => Equal,
            1 => Greater,
            _ => unreachable!(),
        }
    }

    /// Try to get f64 value without forcing exact computation
    fn as_f64_fast(&self) -> Option<f64> {
        // Fast path: check if we already have exact computed
        if let Some(exact) = self.0.exact.get() {
            return exact.to_f64();
        }

        // Use ball approximation directly if tight enough
        let ball = self.ball_ref();
        if ball.r <= 1e-14 * ball.m.abs() {
            return Some(ball.m);
        }

        None
    }

    #[inline(always)]
    fn double_interval(&self) -> Option<(f64, f64)> {
        let b = self.ball_ref();
        // Ball is already outward-safe; just return center +/- radius.
        Some((f64_next_down(b.m - b.r), f64_next_up(b.m + b.r)))
    }
}

impl PartialEq for LazyExact {
    fn eq(&self, other: &Self) -> bool {
        self.exact() == other.exact()
    }
}
impl Eq for LazyExact {}

impl PartialOrd for LazyExact {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // try interval fast path
        let diff = self - other;
        if let Some(s) = diff.ball_ref().sign_if_certain() {
            return Some(match s {
                -1 => Ordering::Less,
                0 => Ordering::Equal,
                _ => Ordering::Greater,
            });
        }
        // fallback exact
        self.exact().partial_cmp(&other.exact())
    }
}

impl Hash for LazyExact {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(h) = self.0.hashed_exact.get() {
            state.write_u64(*h);
            return;
        }
        let mut hasher = ahash::AHasher::default();
        self.exact().hash(&mut hasher);
        let h = hasher.finish();
        let _ = self.0.hashed_exact.set(h);
        state.write_u64(h);
    }
}

impl ToPrimitive for LazyExact {
    fn to_i64(&self) -> Option<i64> {
        self.exact().to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.exact().to_u64()
    }
    fn to_f32(&self) -> Option<f32> {
        self.exact().to_f32()
    }
    fn to_f64(&self) -> Option<f64> {
        self.exact().to_f64()
    }
}

impl FromRef<CgarF64> for LazyExact {
    fn from_ref(value: &CgarF64) -> Self {
        LazyExact(Arc::new(Node {
            kind: Kind::LeafApprox(value.clone()),
            approx: {
                let cell = OnceCell::new();
                let _ = cell.set(value.clone());
                cell
            },
            exact: OnceCell::new(),
            ball: OnceCell::new(),
            hashed_exact: OnceCell::new(),
            depth: 1,
        }))
    }
}

impl FromRef<CgarRational> for LazyExact {
    fn from_ref(value: &CgarRational) -> Self {
        let approx = CgarF64(value.to_f64().unwrap_or(0.0));
        LazyExact::from_leafs(value.clone(), approx)
    }
}

impl FromRef<LazyExact> for LazyExact {
    fn from_ref(value: &LazyExact) -> Self {
        value.clone()
    }
}

impl RefInto<CgarF64> for LazyExact {
    fn ref_into(&self) -> CgarF64 {
        self.approx()
    }
}

impl RefInto<CgarRational> for LazyExact {
    fn ref_into(&self) -> CgarRational {
        self.exact()
    }
}

impl RefInto<LazyExact> for LazyExact {
    fn ref_into(&self) -> LazyExact {
        self.clone()
    }
}

struct Node {
    kind: Kind,
    approx: OnceCell<CgarF64>,
    exact: OnceCell<CgarRational>,
    ball: OnceCell<Ball>,
    hashed_exact: OnceCell<u64>,
    depth: u32,
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
    const MAX_DEPTH: u32 = 1024;

    /// Fast comparison that avoids exact computation when possible
    pub fn cmp_fast(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Try ball-based comparison first
        let diff = self - other;
        if let Some(sign) = diff.ball_ref().sign_if_certain() {
            return Some(match sign {
                -1 => std::cmp::Ordering::Less,
                0 => std::cmp::Ordering::Equal,
                _ => std::cmp::Ordering::Greater,
            });
        }
        None // Need exact computation
    }

    /// Return the f64 exactly represented by this node (only for true f64 leaves).
    #[inline(always)]
    pub fn as_f64_if_exact(&self) -> Option<f64> {
        match &self.0.kind {
            // A true f64 leaf
            Kind::LeafApprox(a) => {
                let v = a.0;
                if v.is_finite() { Some(v) } else { None }
            }
            // An exact rational leaf that happens to fit exactly in f64
            Kind::LeafExact(e) => {
                if let Some(v) = e.to_f64() {
                    // accept only if round-trip is exact (dyadic)
                    if CgarRational::from(v) == **e {
                        Some(v)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    #[inline(always)]
    pub fn cmp_exact(a: &Self, b: &Self) -> i8 {
        use core::cmp::Ordering::*;
        let ae = a.exact();
        let be = b.exact();
        match ae.partial_cmp(&be).unwrap() {
            Less => -1,
            Equal => 0,
            Greater => 1,
        }
    }

    #[inline]
    fn compute_depth(kind: &Kind) -> u32 {
        use Kind::*;
        match kind {
            LeafApprox(_) | LeafExact(_) => 1,
            Add(a, b) | Sub(a, b) | Mul(a, b) | Div(a, b) => 1 + a.0.depth.max(b.0.depth),
            Neg(x) => 1 + x.0.depth,
        }
    }

    #[inline]
    fn exact_ready(&self) -> Option<&CgarRational> {
        self.0.exact.get()
    }

    #[inline]
    fn fold_if_both_exact(kind: Kind) -> Option<LazyExact> {
        use Kind::*;
        match &kind {
            Add(a, b) => {
                if let (Some(la), Some(lb)) = (a.exact_ready(), b.exact_ready()) {
                    let e = la + lb;
                    let a_ = a.approx();
                    let b_ = b.approx();
                    let approx = a_ + b_;
                    return Some(LazyExact::from_leafs(e, approx));
                }
            }
            Sub(a, b) => {
                if let (Some(la), Some(lb)) = (a.exact_ready(), b.exact_ready()) {
                    let e = la - lb;
                    let approx = a.approx() - b.approx();
                    return Some(LazyExact::from_leafs(e, approx));
                }
            }
            Mul(a, b) => {
                if let (Some(la), Some(lb)) = (a.exact_ready(), b.exact_ready()) {
                    let e = la * lb;
                    let approx = a.approx() * b.approx();
                    return Some(LazyExact::from_leafs(e, approx));
                }
            }
            Div(a, b) => {
                if let (Some(la), Some(lb)) = (a.exact_ready(), b.exact_ready()) {
                    assert!(!lb.is_zero(), "LazyExact: division by zero");
                    let e = la / lb;
                    let approx = a.approx() / b.approx();
                    return Some(LazyExact::from_leafs(e, approx));
                }
            }
            Neg(x) => {
                if let Some(le) = x.exact_ready() {
                    let e = -le;
                    let approx = -x.approx();
                    return Some(LazyExact::from_leafs(e, approx));
                }
            }
            _ => {}
        }
        None
    }

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
            ball: OnceCell::new(),
            hashed_exact: OnceCell::new(),
            depth: 1,
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
            ball: OnceCell::new(),
            hashed_exact: OnceCell::new(),
            depth: 1,
        }))
    }

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
        // if ENABLE_PANIC_ON_EXACT.load(std::sync::atomic::Ordering::Relaxed) {
        //     panic!("test");
        // }
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

    #[inline]
    fn is_exact_zero(&self) -> bool {
        // cheap check first:
        if let Some(e) = self.0.exact.get() {
            return e.is_zero();
        }
        false
    }
    #[inline]
    fn is_exact_one(&self) -> bool {
        if let Some(e) = self.0.exact.get() {
            return e == &CgarRational::from(1);
        }
        false
    }

    #[inline]
    fn simplify_identities(kind: Kind) -> Kind {
        use Kind::*;
        match &kind {
            Add(a, b) => {
                if a.is_exact_zero() {
                    return b.clone().0.kind.clone();
                }
                if b.is_exact_zero() {
                    return a.clone().0.kind.clone();
                }
            }
            Sub(a, b) => {
                if b.is_exact_zero() {
                    return a.clone().0.kind.clone();
                }
            }
            Mul(a, b) => {
                if a.is_exact_zero() || b.is_exact_zero() {
                    return Kind::LeafExact(Arc::new(CgarRational::from(0)));
                }
                if a.is_exact_one() {
                    return b.clone().0.kind.clone();
                }
                if b.is_exact_one() {
                    return a.clone().0.kind.clone();
                }
            }
            Div(a, b) => {
                if a.is_exact_zero() {
                    return Kind::LeafExact(Arc::new(CgarRational::from(0)));
                }
                if b.is_exact_one() {
                    return a.clone().0.kind.clone();
                }
            }
            Neg(x) => {
                if x.is_exact_zero() {
                    return Kind::LeafExact(Arc::new(CgarRational::from(0)));
                }
            }
            _ => {}
        }
        kind
    }

    #[inline]
    fn new(mut kind: Kind) -> Self {
        kind = LazyExact::simplify_identities(kind);

        if let Some(folded) = LazyExact::fold_if_both_exact(kind.clone()) {
            return folded;
        }

        let depth = LazyExact::compute_depth(&kind);
        if depth > Self::MAX_DEPTH {
            // force a leaf
            let approx = match &kind {
                Kind::LeafApprox(a) => a.clone(),
                Kind::LeafExact(e) => CgarF64(e.to_f64().unwrap_or(0.0)),
                Kind::Add(a, b) => a.approx() + b.approx(),
                Kind::Sub(a, b) => a.approx() - b.approx(),
                Kind::Mul(a, b) => a.approx() * b.approx(),
                Kind::Div(a, b) => a.approx() / b.approx(),
                Kind::Neg(x) => -x.approx(),
            };
            let exact = match &kind {
                Kind::LeafApprox(a) => CgarRational::from(a.0),
                Kind::LeafExact(e) => e.as_ref().clone(),
                Kind::Add(a, b) => &a.exact() + &b.exact(),
                Kind::Sub(a, b) => &a.exact() - &b.exact(),
                Kind::Mul(a, b) => &a.exact() * &b.exact(),
                Kind::Div(a, b) => {
                    let d = b.exact();
                    assert!(!d.is_zero());
                    &a.exact() / &d
                }
                Kind::Neg(x) => -&x.exact(),
            };
            return LazyExact::from_leafs(exact, approx);
        }

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
                let c = OnceCell::new();
                let _ = c.set(approx);
                c
            },
            exact: OnceCell::new(),
            ball: OnceCell::new(),
            hashed_exact: OnceCell::new(),
            depth,
        }))
    }
}

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

    #[inline]
    fn ball(&self) -> Ball {
        *self
            .0
            .ball
            .get_or_init(|| Self::eval_ball_kind(&self.0.kind))
    }

    #[inline]
    fn ball_ref(&self) -> &Ball {
        self.0
            .ball
            .get_or_init(|| Self::eval_ball_kind(&self.0.kind))
    }

    #[inline]
    fn eval_ball_kind(kind: &Kind) -> Ball {
        use Kind::*;
        match kind {
            LeafApprox(a) => Ball::from_f64(a.0),
            LeafExact(e) => Ball::from_f64(e.to_f64().unwrap_or(0.0)),
            Add(a, b) => a.ball_ref().add(b.ball()),
            Sub(a, b) => a.ball_ref().sub(b.ball()),
            Mul(a, b) => a.ball_ref().mul(b.ball()),
            Div(a, b) => a.ball_ref().div(b.ball()),
            Neg(x) => a_neg(x),
        }
    }
}

impl Zero for LazyExact {
    fn zero() -> Self {
        LazyExact::from_i32(0)
    }

    fn is_zero(&self) -> bool {
        self.sign() == 0
    }

    fn is_positive(&self) -> bool {
        self.sign() > 0
    }

    fn is_negative(&self) -> bool {
        self.sign() < 0
    }

    fn is_positive_or_zero(&self) -> bool {
        self.sign() >= 0
    }

    fn is_negative_or_zero(&self) -> bool {
        self.sign() <= 0
    }
}

impl One for LazyExact {
    fn one() -> Self {
        LazyExact::from_i32(1)
    }
}

impl Abs for LazyExact {
    fn abs(&self) -> Self {
        if self.is_negative() {
            -self.clone()
        } else {
            self.clone()
        }
    }
}

impl AddAssign<&LazyExact> for LazyExact {
    fn add_assign(&mut self, rhs: &LazyExact) {
        if let (Some(ae), Some(be)) = (self.0.exact.get(), rhs.0.exact.get()) {
            let e = ae + be;
            let approx = self.approx() + rhs.approx();
            *self = LazyExact::from_leafs(e, approx);
        } else {
            *self = &*self + rhs;
        }
    }
}
impl SubAssign<&LazyExact> for LazyExact {
    fn sub_assign(&mut self, rhs: &LazyExact) {
        if let (Some(ae), Some(be)) = (self.0.exact.get(), rhs.0.exact.get()) {
            let e = ae - be;
            let approx = self.approx() - rhs.approx();
            *self = LazyExact::from_leafs(e, approx);
        } else {
            *self = &*self - &rhs;
        }
    }
}

#[inline]
fn a_neg(x: &LazyExact) -> Ball {
    x.ball_ref().neg()
}
