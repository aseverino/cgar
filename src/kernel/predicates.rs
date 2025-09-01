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

use crate::geometry::Point2;
use crate::geometry::point::PointOps;
use crate::geometry::segment::Segment;
use crate::geometry::spatial_element::SpatialElement;
use crate::geometry::vector::VectorOps;
use crate::geometry::{point::Point, vector::Vector};
use crate::numeric::cgar_f64::CgarF64;
use crate::numeric::scalar::{RefInto, Scalar};
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TrianglePoint {
    Off,
    OnEdge,
    OnVertex,
    In,
}

pub fn are_equal<T: Scalar, const N: usize>(p1: &Point<T, N>, p2: &Point<T, N>) -> bool
where
    for<'a> &'a T: Sub<&'a T, Output = T> + Mul<&'a T, Output = T>,
{
    for i in 0..N {
        if !(&p1.coords[i] - &p2.coords[i]).abs().is_zero() {
            return false;
        }
    }

    return true;
}

pub fn are_collinear<T, const N: usize>(a: &Point<T, N>, b: &Point<T, N>, c: &Point<T, N>) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    for i in 0..N {
        let ui = &b.coords[i] - &a.coords[i];
        let vi = &c.coords[i] - &a.coords[i];

        if ui.abs().is_positive() {
            // first non-zero component gives the candidate scale factor
            let r = &vi / &ui;

            // every remaining coordinate must satisfy vj = r * uj
            for j in (i + 1)..N {
                let uj = &b.coords[j] - &a.coords[j];
                let vj = &c.coords[j] - &a.coords[j];
                if (&vj - &(&uj * &r)).abs().is_positive() {
                    return false; // breaks proportionality
                }
            }
            return true; // all coordinates match
        } else if vi.abs().is_positive() {
            return false; // ui ≈ 0 but vi isn’t ⇒ not collinear
        }
    }
    // all ui ≈ 0  ⇒  A and B coincide; collinear iff C coincides too
    true
}

pub fn triangle_is_degenerate<T: Scalar, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> bool
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Coincidence or collinearity → zero area
    are_equal(a, b) || are_equal(a, c) || are_equal(b, c) || are_collinear(a, b, c)
}

pub fn is_point_on_segment<T, const N: usize>(p: &Point<T, N>, seg: &Segment<T, N>) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
    T: PartialOrd, // needed for comparisons inside the loop
{
    // 1.  If P, A, B are not collinear, P cannot lie on AB
    if !are_collinear(p, &seg.a, &seg.b) {
        return false;
    }

    // 2.  For every coordinate axis, P must lie between A and B
    for i in 0..N {
        let ai = &seg.a.coords[i];
        let bi = &seg.b.coords[i];

        // min_i, max_i bounds
        let (min_i, max_i) = if ai <= bi { (ai, bi) } else { (bi, ai) };

        let pi = &p.coords[i];
        if pi < min_i || pi > max_i {
            return false; // outside on some axis ⇒ not on segment
        }
    }

    true
}

pub fn point_u_on_segment<T: Scalar + PartialOrd, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    p: &Point<T, N>,
) -> Option<T>
where
    Point<T, N>: PointOps<T, N, Vector = crate::geometry::vector::Vector<T, N>>,
    crate::geometry::vector::Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Direction and offset
    let ab = (b - a).as_vector();
    let ap = (p - a).as_vector();

    // Degenerate segment?
    let ab2 = ab.dot(&ab);
    if ab2.is_zero() {
        return if are_equal(a, p) {
            Some(T::zero())
        } else if are_equal(b, p) {
            Some(T::one())
        } else {
            None
        };
    }

    // Must be collinear with AB
    if !are_collinear(a, b, p) {
        return None;
    }

    // Parametric coordinate along AB
    let u = ap.dot(&ab) / ab2;
    if u < T::zero() || u > T::one() {
        None
    } else {
        Some(u)
    }
}

pub fn point_in_or_on_triangle_old<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> TrianglePoint
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Barycentric setup
    let v0 = (c - a).as_vector();
    let v1 = (b - a).as_vector();
    let v2 = (p - a).as_vector();

    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot11 = v1.dot(&v1);
    let dot12 = v1.dot(&v2);

    let denom = &dot00 * &dot11 - &dot01 * &dot01;

    // Degenerate triangle (zero/near-zero area)
    if denom.abs() < T::tolerance() {
        return TrianglePoint::Off;
    }

    let inv = T::one() / denom;
    let u = &(&dot11 * &dot02 - &dot01 * &dot12) * &inv;
    let v = &(&dot00 * &dot12 - &dot01 * &dot02) * &inv;
    let sum_uv = &u + &v;
    let w = &T::one() - &sum_uv;

    // Classification with tolerance
    let e = T::tolerance();
    let neg_e = e.clone().neg();

    // Outside if any barycentric is below -eps or u+v exceeds 1+eps
    if u < neg_e || v < neg_e || sum_uv > &T::one() + &e {
        return TrianglePoint::Off;
    }

    // On if any barycentric is within eps of the boundary
    // Check if on vertex (distance to vertex is zero)
    if are_equal(p, a) || are_equal(p, b) || are_equal(p, c) {
        return TrianglePoint::OnVertex;
    }

    // Check if on edge (one barycentric coordinate is zero)
    if u.is_zero() || v.is_zero() || w.is_zero() {
        return TrianglePoint::OnEdge;
    }

    TrianglePoint::In
}

#[inline(always)]
pub fn point_in_or_on_triangle<T: Scalar + PartialOrd, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> TrianglePoint
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    #[inline(always)]
    fn near_edge_approx<TS: Scalar>(e0: &TS, e1: &TS, e2: &TS) -> bool {
        let eps = RefInto::<CgarF64>::ref_into(&TS::query_tolerance()).0;
        let e0f = RefInto::<CgarF64>::ref_into(e0).0;
        let e1f = RefInto::<CgarF64>::ref_into(e1).0;
        let e2f = RefInto::<CgarF64>::ref_into(e2).0;
        e0f.abs() <= eps || e1f.abs() <= eps || e2f.abs() <= eps
    }

    #[inline(always)]
    fn classify_from_edges<TS: Scalar + PartialOrd>(e0: &TS, e1: &TS, e2: &TS) -> TrianglePoint {
        let zero = TS::zero();

        // exact zero flags
        let z0 = e0.is_zero();
        let z1 = e1.is_zero();
        let z2 = e2.is_zero();
        let zc = (z0 as u8) + (z1 as u8) + (z2 as u8);

        // mixed signs => strictly outside
        let has_neg = (!z0 && *e0 < zero) || (!z1 && *e1 < zero) || (!z2 && *e2 < zero);
        let has_pos = (!z0 && *e0 > zero) || (!z1 && *e1 > zero) || (!z2 && *e2 > zero);
        if has_neg && has_pos {
            return TrianglePoint::Off;
        }

        // boundary cases
        if zc >= 2 {
            return TrianglePoint::OnVertex; // exactly at a vertex
        }
        if zc == 1 {
            return TrianglePoint::OnEdge; // exactly on a unique edge
        }

        // inside (all nonzero and same sign)
        TrianglePoint::In
    }

    if N == 2 {
        // degenerate?
        let abx = &b[0] - &a[0];
        let aby = &b[1] - &a[1];
        let acx = &c[0] - &a[0];
        let acy = &c[1] - &a[1];
        let area2 = &(&abx * &acy) - &(&aby * &acx);
        if area2.is_zero() {
            return TrianglePoint::Off;
        }

        // edge functions
        let bcx = &c[0] - &b[0];
        let bcy = &c[1] - &b[1];
        let cax = &a[0] - &c[0];
        let cay = &a[1] - &c[1];

        let apx = &p[0] - &a[0];
        let apy = &p[1] - &a[1];
        let bpx = &p[0] - &b[0];
        let bpy = &p[1] - &b[1];
        let cpx = &p[0] - &c[0];
        let cpy = &p[1] - &c[1];

        let e0 = &(&abx * &apy) - &(&aby * &apx);
        let e1 = &(&bcx * &bpy) - &(&bcy * &bpx);
        let e2 = &(&cax * &cpy) - &(&cay * &cpx);

        // near-edge fast path -> confirm exact boundary kind
        if near_edge_approx::<T>(&e0, &e1, &e2) && (e0.is_zero() || e1.is_zero() || e2.is_zero()) {
            // figure out vertex vs edge by zero-count
            let zc = (e0.is_zero() as u8) + (e1.is_zero() as u8) + (e2.is_zero() as u8);
            return if zc >= 2 {
                TrianglePoint::OnVertex
            } else {
                TrianglePoint::OnEdge
            };
        }

        return classify_from_edges(&e0, &e1, &e2);
    } else if N == 3 {
        // normal and degeneracy
        let ab = (b - a).as_vector_3();
        let ac = (c - a).as_vector_3();
        let n = ab.cross(&ac);
        if n[0].is_zero() && n[1].is_zero() && n[2].is_zero() {
            return TrianglePoint::Off;
        }

        // edge functions projected on n
        let ap = (p - a).as_vector_3();
        let bp = (p - b).as_vector_3();
        let cp = (p - c).as_vector_3();
        let bc = (c - b).as_vector_3();
        let ca = (a - c).as_vector_3();

        let e0 = ab.cross(&ap).dot(&n);
        let e1 = bc.cross(&bp).dot(&n);
        let e2 = ca.cross(&cp).dot(&n);

        if near_edge_approx::<T>(&e0, &e1, &e2) && (e0.is_zero() || e1.is_zero() || e2.is_zero()) {
            let zc = (e0.is_zero() as u8) + (e1.is_zero() as u8) + (e2.is_zero() as u8);
            return if zc >= 2 {
                TrianglePoint::OnVertex
            } else {
                TrianglePoint::OnEdge
            };
        }

        return classify_from_edges(&e0, &e1, &e2);
    } else {
        panic!("point_in_or_on_triangle only supports N = 2 or N = 3");
    }
}

#[inline]
pub fn orient2d<T: Scalar>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>) -> T
where
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    // det | b-a, c-a |
    let det = (&b[0] - &a[0]) * (&c[1] - &a[1]) - (&b[1] - &a[1]) * (&c[0] - &a[0]);
    // light guard
    let eps = (b[0].abs() + b[1].abs() + c[0].abs() + c[1].abs() + a[0].abs() + a[1].abs())
        * T::from(1e-15);
    if det.abs() <= eps { T::zero() } else { det }
}

#[inline]
pub fn incircle<T: Scalar>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>, d: &Point2<T>) -> T
where
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    // Compute with standard determinant expansion; apply light scaling guard.
    // (Assumes abc CCW. If not, caller should flip sign or reorder.)
    let ax = &a[0] - &d[0];
    let ay = &a[1] - &d[1];
    let a2 = &ax * &ax + &ay * &ay;
    let bx = &b[0] - &d[0];
    let by = &b[1] - &d[1];
    let b2 = &bx * &bx + &by * &by;
    let cx = &c[0] - &d[0];
    let cy = &c[1] - &d[1];
    let c2 = &cx * &cx + &cy * &cy;

    let det = &ax * &(&(&by * &c2) - &(&b2 * &cy)) - &ay * &(&(&bx * &c2) - &(&b2 * &cx))
        + &a2 * &(&(&bx * &cy) - &(&by * &cx));

    let scale =
        (&ax.abs() + &ay.abs() + bx.abs() + by.abs() + cx.abs() + cy.abs()).max(T::from(1.0));
    let eps = &scale * &scale * T::from(1e-14);
    if det.abs() <= eps { T::zero() } else { det }
}

#[inline]
pub fn bbox<T: Scalar>(pts: &[Point2<T>]) -> (T, T, T, T)
where
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let mut minx = &pts[0][0];
    let mut miny = &pts[0][1];
    let mut maxx = &pts[0][0];
    let mut maxy = &pts[0][1];
    for p in &pts[1..] {
        if (&p[0] - &minx).is_negative() {
            minx = &p[0];
        }
        if (&p[1] - &miny).is_negative() {
            miny = &p[1];
        }
        if (&p[0] - &maxx).is_positive() {
            maxx = &p[0];
        }
        if (&p[1] - maxy).is_positive() {
            maxy = &p[1];
        }
    }
    (minx.clone(), miny.clone(), maxx.clone(), maxy.clone())
}

#[inline]
pub fn centroid2<T: Scalar>(a: &Point2<T>, b: &Point2<T>) -> Point2<T>
where
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    Point::<T, 2>::from_vals([
        T::from_num_den(1, 2) * (&a[0] + &b[0]),
        T::from_num_den(1, 2) * (&a[1] + &b[1]),
    ])
}
