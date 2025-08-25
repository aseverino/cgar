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

use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
    time::Instant,
};

use smallvec::SmallVec;

use crate::{
    geometry::{
        Point2,
        point::{Point, PointOps},
        segment::Segment,
        spatial_element::SpatialElement,
        util::*,
        vector::{Vector, VectorOps},
    },
    numeric::scalar::Scalar,
    operations::Zero,
};

#[derive(Clone, Debug)]
pub enum TriTriIntersectionResult<T: Scalar, const N: usize> {
    Proper(Segment<T, N>),
    Coplanar(Segment<T, N>),
    CoplanarPolygon(Vec<Segment<T, N>>),
    None,
}

/// Return true if 2D point `p` lies inside (or on) the triangle `tri` = [(x0,y0),(x1,y1),(x2,y2)].
/// Uses a barycentric‐coordinate test.
fn point_in_tri_2d<T>(p: &Point2<T>, tri: &[Point2<T>; 3]) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let (x, y) = (&p.coords[0], &p.coords[1]);
    let (x0, y0) = (&tri[0].coords[0], &tri[0].coords[1]);
    let (x1, y1) = (&tri[1].coords[0], &tri[1].coords[1]);
    let (x2, y2) = (&tri[2].coords[0], &tri[2].coords[1]);
    // Compute barycentric coords
    let denom = &(&(y1 - y2) * &(x0 - x2)) + &(&(x2 - x1) * &(y0 - y2));
    if denom.is_zero() {
        // degenerate triangle
        return false;
    }
    let u = &(&(&(y1 - y2) * &(x - &x2)) + &(&(x2 - x1) * &(y - &y2))) / &denom;
    let v = &(&(&(y2 - y0) * &(x - &x2)) + &(&(x0 - x2) * &(y - &y2))) / &denom;
    u.is_positive_or_zero()
        && v.is_positive_or_zero()
        && ((&u + &v) - T::one()).is_negative_or_zero()
}

/// If segments [a→b] and [c→d] intersect in 2D, return the intersection point.
/// Otherwise return None.  (Colinear or parallel ⇒ None.)
pub fn segment_intersect_2d<T: Scalar>(
    a0: &Point<T, 2>,
    a1: &Point<T, 2>,
    b0: &Point<T, 2>,
    b1: &Point<T, 2>,
) -> Option<Segment<T, 2>>
where
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Mul<&'a T, Output = T>,
{
    // 2D segment intersection including overlap
    // direction vectors
    let da = [&a1[0] - &a0[0], &a1[1] - &a0[1]];
    let db = [&b1[0] - &b0[0], &b1[1] - &b0[1]];
    // cross of directions
    let denom = &da[0] * &db[1] - &da[1] * &db[0];
    if denom.is_zero() {
        // parallel or colinear
        // check colinearity via cross of (b0 - a0) and da
        let diff = [&b0[0] - &a0[0], &b0[1] - &a0[1]];
        let cross = &diff[0] * &da[1] - &diff[1] * &da[0];
        if cross.is_zero() {
            // colinear: project onto A's parameter t
            let len2 = &da[0] * &da[0] + &da[1] * &da[1];
            if !len2.is_positive() {
                return None;
            }
            let t0 = &(&diff[0] * &da[0] + &diff[1] * &da[1]) / &len2;
            let t1 = &(&(&b1[0] - &a0[0]) * &da[0] + &(&b1[1] - &a0[1]) * &da[1]) / &len2;
            let swap = (&t0 - &t1).is_positive();
            let (tmin, tmax) = if swap { (t1, t0) } else { (t0, t1) };
            let zero = T::zero();
            let one = T::one();
            let start = if tmin.is_negative() {
                zero.clone()
            } else {
                tmin.clone()
            };
            let end = if (&tmax - &one).is_positive() {
                one.clone()
            } else {
                tmax.clone()
            };
            if (&end - &start).is_positive_or_zero() {
                let p_start = Point::<T, 2>::from_vals([
                    &a0[0] + &(&da[0] * &start),
                    &a0[1] + &(&da[1] * &start),
                ]);
                let p_end = Point::<T, 2>::from_vals([
                    &a0[0] + &(&da[0] * &end),
                    &a0[1] + &(&da[1] * &end),
                ]);
                return Some(Segment::new(&p_start, &p_end));
            }
        }
        return None;
    }
    // lines intersect at single point, solve via cross ratios
    let diff = [&b0[0] - &a0[0], &b0[1] - &a0[1]];
    let s = &(&diff[0] * &db[1] - &diff[1] * &db[0]) / &denom;
    let u = &(&diff[0] * &da[1] - &diff[1] * &da[0]) / &denom;

    let eps = T::tolerance();
    let one = T::one();
    if (&s + &eps).is_positive_or_zero()
        && (&(&one + &eps) - &s).is_positive_or_zero()
        && (&u + &eps).is_positive_or_zero()
        && (&(&one + &eps) - &u).is_positive_or_zero()
    {
        let ix = &a0[0] + &(&s * &da[0]);
        let iy = &a0[1] + &(&s * &da[1]);
        return Some(Segment::new(
            &Point::<T, 2>::from_vals([ix.clone(), iy.clone()]),
            &Point::<T, 2>::from_vals([ix, iy]),
        ));
    }
    None
}

/// Project a 3D triangle onto `axis`, returning (min,max).
fn project_3d_triangle<T: Scalar, const N: usize>(
    axis: &Vector<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> (T, T)
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let p = |p: &Point<T, N>| {
        let v: Vector<T, N> = p.as_vector();
        v.dot(axis)
    };
    let p0 = p(a);
    let p1 = p(b);
    let p2 = p(c);

    let mut min = p0.clone();
    let mut max = p0;
    if (&p1 - &min).is_negative() {
        min = p1.clone();
    }
    if (&p1 - &max).is_positive() {
        max = p1.clone();
    }
    if (&p2 - &min).is_negative() {
        min = p2.clone();
    }
    if (&p2 - &max).is_positive() {
        max = p2.clone();
    }
    (min, max)
}

fn coplanar_tri_tri_intersection<T: Scalar, const N: usize>(
    p: &[&Point<T, N>; 3],
    q: &[&Point<T, N>; 3],
    n: &Vector<T, N>,
) -> TriTriIntersectionResult<T, N>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let (i0, i1, drop) = coplanar_axes(n);

    // 2) build 2D points
    let to2d = |p: &Point<T, N>| project_to_2d(p, i0, i1);
    let t1 = [to2d(&p[2]), to2d(&p[0]), to2d(&p[1])];
    let t2 = [to2d(&q[2]), to2d(&q[0]), to2d(&q[1])];

    // 3) collect vertices of one triangle inside the other
    let mut pts: Vec<Point<T, N>> = Vec::new();
    for point in &t1 {
        if point_in_tri_2d(point, &t2) {
            pts.push(back_project_to_3d(point, i0, i1, drop, p[2]));
        }
    }
    for point in &t2 {
        if point_in_tri_2d(point, &t1) {
            pts.push(back_project_to_3d(point, i0, i1, drop, q[2]));
        }
    }

    // 4) edge-edge intersections in 2D
    let edges1 = [(&t1[0], &t1[1]), (&t1[1], &t1[2]), (&t1[2], &t1[0])];
    let edges2 = [(&t2[0], &t2[1]), (&t2[1], &t2[2]), (&t2[2], &t2[0])];
    for (a, b) in &edges1 {
        for (c, d) in &edges2 {
            if let Some(s) = segment_intersect_2d(a, b, c, d) {
                // Here we use p0 as the reference for the dropped axis, but you could interpolate if desired
                pts.push(back_project_to_3d(&s.a, i0, i1, drop, &p[2]));
            }
        }
    }

    // 5) dedupe by geometric distance using point-merge threshold (approx-first)
    let merge = T::point_merge_threshold();
    let merge2 = &merge * &merge;
    let mut uniq: Vec<Point<T, N>> = Vec::new();
    'dedupe_coplanar: for p3 in pts {
        for q3 in &uniq {
            let d = (&p3 - q3).as_vector();
            let d2 = d.dot(&d);
            if (&d2 - &merge2).is_negative_or_zero() {
                continue 'dedupe_coplanar;
            }
        }
        uniq.push(p3);
    }

    match uniq.len() {
        2 => TriTriIntersectionResult::Coplanar(Segment::new(&uniq[0], &uniq[1])),
        m if m >= 3 => {
            // 1) project all uniq back to 2D
            let to2d = |p: &Point<T, N>| project_to_2d(p, i0, i1);
            let mut pts2d: Vec<Point2<T>> = uniq.iter().map(to2d).collect();
            // 2) compute 2D convex hull indices (e.g. monotone‐chain)
            let hull_idx = convex_hull_2d_indices(&mut pts2d);
            // 3) turn hull edges back into 3D segments
            let mut segs = Vec::with_capacity(hull_idx.len());
            for w in 0..hull_idx.len() {
                let i = hull_idx[w];
                let j = hull_idx[(w + 1) % hull_idx.len()];
                let a3 = back_project_to_3d(&pts2d[i], i0, i1, drop, p[2]);
                let b3 = back_project_to_3d(&pts2d[j], i0, i1, drop, p[2]);
                segs.push(Segment::new(&a3, &b3));
            }
            TriTriIntersectionResult::CoplanarPolygon(segs)
        }
        _ => TriTriIntersectionResult::None,
    }
}

pub fn convex_hull_2d_indices<T>(pts: &Vec<Point2<T>>) -> Vec<usize>
where
    T: Scalar,
    Point2<T>: PointOps<T, 2, Vector = Vector<T, 2>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let n = pts.len();
    if n < 3 {
        return (0..n).collect();
    }

    // 1) sort indices by (x, then y)
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.sort_by(|&i, &j| {
        let dx = &pts[i].coords[0] - &pts[j].coords[0];
        if dx.is_negative() {
            Ordering::Less
        } else if dx.is_positive() {
            Ordering::Greater
        } else {
            let dy = &pts[i].coords[1] - &pts[j].coords[1];
            if dy.is_negative() {
                Ordering::Less
            } else if dy.is_positive() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    });

    // 2) build lower hull
    let mut lower = Vec::new();
    for &i in &idxs {
        while lower.len() >= 2 {
            let j = lower[lower.len() - 2];
            let k = lower[lower.len() - 1];
            let pts_k: &Point<T, 2> = &pts[k];
            let pts_j: &Point<T, 2> = &pts[j];
            // cross((pts[k] - pts[j]), (pts[i] - pts[j]))
            let cross = {
                let x1 = &pts_k.coords[0] - &pts_j.coords[0];
                let y1 = &pts_k.coords[1] - &pts_j.coords[1];
                let x2 = &pts[i].coords[0] - &pts_j.coords[0];
                let y2 = &pts[i].coords[1] - &pts_j.coords[1];
                &x1 * &y2 - &y1 * &x2
            };
            if cross.is_negative_or_zero() {
                lower.pop();
            } else {
                break;
            }
        }
        lower.push(i);
    }

    // 3) build upper hull
    let mut upper = Vec::new();
    for &i in idxs.iter().rev() {
        while upper.len() >= 2 {
            let j = upper[upper.len() - 2];
            let k = upper[upper.len() - 1];
            let pts_k: &Point<T, 2> = &pts[k];
            let pts_j: &Point<T, 2> = &pts[j];
            let cross = {
                let x1 = &pts_k.coords[0] - &pts_j.coords[0];
                let y1 = &pts_k.coords[1] - &pts_j.coords[1];
                let x2 = &pts[i].coords[0] - &pts_j.coords[0];
                let y2 = &pts[i].coords[1] - &pts_j.coords[1];
                &x1 * &y2 - &y1 * &x2
            };
            if cross.is_negative_or_zero() {
                upper.pop();
            } else {
                break;
            }
        }
        upper.push(i);
    }

    // 4) concatenate, dropping the duplicate endpoints
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Computes the segment where triangles T1=(p0,p1,p2) and T2=(q0,q1,q2) intersect.
/// Returns `None` if they don’t intersect, or `Some((a,b))` where `a` and `b` are
/// the two endpoints of the intersection segment (possibly `a==b` if they touch at a point).
pub fn tri_tri_intersection<T: Scalar, const N: usize>(
    p: &[&Point<T, N>; 3],
    q: &[&Point<T, N>; 3],
) -> TriTriIntersectionResult<T, N>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Plane of Q: n2·x + d2 = 0
    let v01 = (q[0] - q[2]).as_vector();
    let v02 = (q[1] - q[2]).as_vector();
    let n2 = v01.cross(&v02);
    if n2.is_zero() {
        // Degenerate Q
        return TriTriIntersectionResult::None;
    }
    let d2 = -n2.dot(&q[2].as_vector());

    // Signed distances of P to plane(Q)
    let d_p0 = &n2.dot(&p[0].as_vector()) + &d2;
    let d_p1 = &n2.dot(&p[1].as_vector()) + &d2;
    let d_p2 = &n2.dot(&p[2].as_vector()) + &d2;

    let zero_p = [d_p0.is_zero(), d_p1.is_zero(), d_p2.is_zero()];
    let zc_p = zero_p.iter().filter(|z| **z).count();

    // Two vertices of P exactly on plane(Q) ⇒ coplanar segment (rare)
    if zc_p == 2 {
        let (i, j) = match zero_p {
            [true, true, false] => (0, 1),
            [true, false, true] => (0, 2),
            [false, true, true] => (1, 2),
            _ => unreachable!(),
        };
        if let Some((a3, b3)) = clip_segment_to_triangle_in_plane(p[i], p[j], q, &n2) {
            return TriTriIntersectionResult::Coplanar(Segment::new(&a3, &b3));
        }
    }

    // Strictly on one side of plane(Q) ⇒ no intersection
    if (d_p0.is_positive() && d_p1.is_positive() && d_p2.is_positive())
        || (d_p0.is_negative() && d_p1.is_negative() && d_p2.is_negative())
    {
        return TriTriIntersectionResult::None;
    }

    // Strictly coplanar case (rare)
    if d_p0.is_zero() && d_p1.is_zero() && d_p2.is_zero() {
        return coplanar_tri_tri_intersection(p, q, &n2);
    }

    // Precompute 2D test for triangle Q (edge functions; no divisions)
    let (qi0, qi1, qdrop) = coplanar_axes(&n2);
    let q2d = [
        project_to_2d(q[0], qi0, qi1),
        project_to_2d(q[1], qi0, qi1),
        project_to_2d(q[2], qi0, qi1),
    ];
    let area_q = {
        let x0 = &q2d[1][0] - &q2d[0][0];
        let y0 = &q2d[1][1] - &q2d[0][1];
        let x1 = &q2d[2][0] - &q2d[0][0];
        let y1 = &q2d[2][1] - &q2d[0][1];
        &x0 * &y1 - &y0 * &x1
    };
    let q_ccw = area_q.is_positive();
    let inside_q_2d = |r3: &Point<T, N>| {
        let r = project_to_2d(r3, qi0, qi1);
        let e0 = {
            let ex = &q2d[1][0] - &q2d[0][0];
            let ey = &q2d[1][1] - &q2d[0][1];
            let rx = &r[0] - &q2d[0][0];
            let ry = &r[1] - &q2d[0][1];
            &ex * &ry - &ey * &rx
        };
        let e1 = {
            let ex = &q2d[2][0] - &q2d[1][0];
            let ey = &q2d[2][1] - &q2d[1][1];
            let rx = &r[0] - &q2d[1][0];
            let ry = &r[1] - &q2d[1][1];
            &ex * &ry - &ey * &rx
        };
        let e2 = {
            let ex = &q2d[0][0] - &q2d[2][0];
            let ey = &q2d[0][1] - &q2d[2][1];
            let rx = &r[0] - &q2d[2][0];
            let ry = &r[1] - &q2d[2][1];
            &ex * &ry - &ey * &rx
        };
        if q_ccw {
            e0.is_positive_or_zero() && e1.is_positive_or_zero() && e2.is_positive_or_zero()
        } else {
            e0.is_negative_or_zero() && e1.is_negative_or_zero() && e2.is_negative_or_zero()
        }
    };

    // On-the-fly dedupe
    let merge = T::point_merge_threshold();
    let merge2 = &merge * &merge;
    let mut uniq = SmallVec::<[Point<T, N>; 8]>::new();
    let mut push_uniq = |pt: Point<T, N>| {
        for q3 in &uniq {
            let d = (&pt - q3).as_vector();
            if (&d.dot(&d) - &merge2).is_negative_or_zero() {
                return 0;
            }
        }
        uniq.push(pt);
        return uniq.len();
    };

    // Intersections from P-edges against plane(Q) using precomputed distances
    for (ia, ib, da, db) in [
        (0usize, 1usize, &d_p0, &d_p1),
        (1usize, 2usize, &d_p1, &d_p2),
        (2usize, 0usize, &d_p2, &d_p0),
    ] {
        let prod = da * db;
        if prod.is_negative_or_zero() && !(da.is_zero() && db.is_zero()) {
            let denom = da - db; // da - db
            if denom.is_zero() {
                continue;
            }
            let t = da / &denom; // t = da / (da - db)
            let dir = (p[ib] - p[ia]).as_vector();
            let ip = p[ia].add_vector(&dir.scale(&t));

            if inside_q_2d(&ip) && push_uniq(ip) == 2 {
                // Fast path: both endpoints obtained from P edges; done.
                return TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[1]));
            }
        }
    }

    // Plane of P: n1·x + d1 = 0
    let u01 = (p[0] - p[2]).as_vector();
    let u02 = (p[1] - p[2]).as_vector();
    let n1 = u01.cross(&u02);
    if n1.is_zero() {
        // Degenerate P
        return if uniq.len() == 2 {
            TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[1]))
        } else if uniq.len() == 1 {
            TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[0]))
        } else {
            TriTriIntersectionResult::None
        };
    }
    let d1 = -n1.dot(&p[2].as_vector());

    // Signed distances of Q to plane(P)
    let d_q0 = &n1.dot(&q[0].as_vector()) + &d1;
    let d_q1 = &n1.dot(&q[1].as_vector()) + &d1;
    let d_q2 = &n1.dot(&q[2].as_vector()) + &d1;

    let zero_q = [d_q0.is_zero(), d_q1.is_zero(), d_q2.is_zero()];
    let zc_q = zero_q.iter().filter(|z| **z).count();

    if zc_q == 2 {
        let (i, j) = match zero_q {
            [true, true, false] => (0, 1),
            [true, false, true] => (0, 2),
            [false, true, true] => (1, 2),
            _ => unreachable!(),
        };
        if let Some((a3, b3)) = clip_segment_to_triangle_in_plane(q[i], q[j], p, &n1) {
            return TriTriIntersectionResult::Coplanar(Segment::new(&a3, &b3));
        }
    }

    if (d_q0.is_positive() && d_q1.is_positive() && d_q2.is_positive())
        || (d_q0.is_negative() && d_q1.is_negative() && d_q2.is_negative())
    {
        // Nothing more to gather
    } else {
        // Precompute 2D test for triangle P
        let (pi0, pi1, _pdrop) = coplanar_axes(&n1);
        let p2d = [
            project_to_2d(p[0], pi0, pi1),
            project_to_2d(p[1], pi0, pi1),
            project_to_2d(p[2], pi0, pi1),
        ];
        let area_p = {
            let x0 = &p2d[1][0] - &p2d[0][0];
            let y0 = &p2d[1][1] - &p2d[0][1];
            let x1 = &p2d[2][0] - &p2d[0][0];
            let y1 = &p2d[2][1] - &p2d[0][1];
            &x0 * &y1 - &y0 * &x1
        };
        let p_ccw = area_p.is_positive();
        let inside_p_2d = |r3: &Point<T, N>| {
            let r = project_to_2d(r3, pi0, pi1);
            let e0 = {
                let ex = &p2d[1][0] - &p2d[0][0];
                let ey = &p2d[1][1] - &p2d[0][1];
                let rx = &r[0] - &p2d[0][0];
                let ry = &r[1] - &p2d[0][1];
                &ex * &ry - &ey * &rx
            };
            let e1 = {
                let ex = &p2d[2][0] - &p2d[1][0];
                let ey = &p2d[2][1] - &p2d[1][1];
                let rx = &r[0] - &p2d[1][0];
                let ry = &r[1] - &p2d[1][1];
                &ex * &ry - &ey * &rx
            };
            let e2 = {
                let ex = &p2d[0][0] - &p2d[2][0];
                let ey = &p2d[0][1] - &p2d[2][1];
                let rx = &r[0] - &p2d[2][0];
                let ry = &r[1] - &p2d[2][1];
                &ex * &ry - &ey * &rx
            };
            if p_ccw {
                e0.is_positive_or_zero() && e1.is_positive_or_zero() && e2.is_positive_or_zero()
            } else {
                e0.is_negative_or_zero() && e1.is_negative_or_zero() && e2.is_negative_or_zero()
            }
        };

        // Intersections from Q-edges against plane(P)
        for (ia, ib, da, db) in [
            (0usize, 1usize, &d_q0, &d_q1),
            (1usize, 2usize, &d_q1, &d_q2),
            (2usize, 0usize, &d_q2, &d_q0),
        ] {
            let prod = da * db;
            if prod.is_negative_or_zero() && !(da.is_zero() && db.is_zero()) {
                let denom = da - db;
                if denom.is_zero() {
                    continue;
                }
                let t = da / &denom;
                let dir = (q[ib] - q[ia]).as_vector();
                let iq = q[ia].add_vector(&dir.scale(&t));
                if inside_p_2d(&iq) && push_uniq(iq) == 2 {
                    return TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[1]));
                }
            }
        }
    }

    // Finalize
    match uniq.len() {
        0 => TriTriIntersectionResult::None,
        1 => TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[0])),
        2 => TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[1])),
        _ => {
            // Pick farthest pair
            let mut max_d2 = T::zero();
            let mut best = (0usize, 1usize);
            for i in 0..uniq.len() {
                for j in (i + 1)..uniq.len() {
                    let v = (&uniq[j] - &uniq[i]).as_vector();
                    let d2 = v.dot(&v);
                    if (&d2 - &max_d2).is_positive() {
                        max_d2 = d2;
                        best = (i, j);
                    }
                }
            }
            TriTriIntersectionResult::Proper(Segment::new(&uniq[best.0], &uniq[best.1]))
        }
    }
}

/// Intersect the segment [a,b] against plane (n·x + d = 0).
/// Returns `Some(Point3)` if it crosses or touches, else `None`.
fn intersect_edge_plane<T: Scalar, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    n: &Point<T, N>,
    d: &T,
) -> Option<Point<T, N>>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // signed distances:
    let da = &n.as_vector().dot(&a.as_vector()) + &d;
    let db = &n.as_vector().dot(&b.as_vector()) + &d;

    // if both on same side (and nonzero), no cross
    if (&da * &db).is_positive() {
        return None;
    }

    // Check for division by zero
    let denominator = &da - &db;
    if denominator.is_zero() {
        // Edge parallel to plane. If it's on the plane, no point should be returned.
        return None;
    }

    // compute interpolation parameter t = da / (da - db)
    let t = &da / &(&da - &db);

    // point = a + t*(b - a)
    let dir = (b - a).as_vector();
    let offset = dir.scale(&t);
    Some(a.add_vector(&offset))
}

/// Returns true if `p` lies inside triangle `(a,b,c)` on the plane with normal `n`.
/// We use barycentric coordinates in 3D.
fn point_in_tri<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // compute vectors
    let v0 = (c - a).as_vector();
    let v1 = (b - a).as_vector();
    let v2 = (p - a).as_vector();

    // dot products
    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot11 = v1.dot(&v1);
    let dot12 = v1.dot(&v2);

    // barycentric coords
    let inv_denom = &T::one() / &(&(&dot00 * &dot11) - &(&dot01 * &dot01));
    let u = &(&(&dot11 * &dot02) - &(&dot01 * &dot12)) * &inv_denom;
    let v = &(&(&dot00 * &dot12) - &(&dot01 * &dot02)) * &inv_denom;

    // inside if u>=0, v>=0, u+v<=1
    u.is_positive_or_zero()
        && v.is_positive_or_zero()
        && ((&u + &v) - T::one()).is_negative_or_zero()
}

/// Given a normal, return the indices of the two axes to keep (largest dropped).
fn coplanar_axes<T: Scalar, const N: usize>(n: &Vector<T, N>) -> (usize, usize, usize)
where
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    let na = [n[0].abs(), n[1].abs(), n[2].abs()];
    let (i0, i1, drop) = if (&na[0] - &na[1]).is_positive() && (&na[0] - &na[2]).is_positive() {
        (1, 2, 0)
    } else if (&na[1] - &na[2]).is_positive() {
        (0, 2, 1)
    } else {
        (0, 1, 2)
    };
    (i0, i1, drop)
}

/// Project a 3D point onto a 2D plane using the provided axes.
fn project_to_2d<T: Scalar, const N: usize>(p: &Point<T, N>, i0: usize, i1: usize) -> Point2<T> {
    Point2::<T>::from_vals([p[i0].clone(), p[i1].clone()])
}

/// Back-project a 2D point into 3D, using a reference 3D point for the dropped axis.
pub fn back_project_to_3d<T: Scalar, const N: usize>(
    p: &Point2<T>,
    i0: usize,
    i1: usize,
    drop: usize,
    reference: &Point<T, N>,
) -> Point<T, N>
where
    Point<T, N>: SpatialElement<T, N>,
{
    let mut coords = Point::<T, N>::zero();
    coords[i0] = p.coords[0].clone();
    coords[i1] = p.coords[1].clone();
    coords[drop] = reference[drop].clone();
    coords
}

fn clip_segment_to_triangle_in_plane<T: Scalar, const N: usize>(
    a3: &Point<T, N>,
    b3: &Point<T, N>,
    tri: &[&Point<T, N>; 3],
    n: &Vector<T, N>,
) -> Option<(Point<T, N>, Point<T, N>)>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let (i0, i1, drop) = coplanar_axes(n);
    let a = project_to_2d(a3, i0, i1);
    let b = project_to_2d(b3, i0, i1);
    let t = [
        project_to_2d(tri[0], i0, i1),
        project_to_2d(tri[1], i0, i1),
        project_to_2d(tri[2], i0, i1),
    ];

    // ensure CCW triangle (left side is inside)
    let area = {
        let x0 = &t[1][0] - &t[0][0];
        let y0 = &t[1][1] - &t[0][1];
        let x1 = &t[2][0] - &t[0][0];
        let y1 = &t[2][1] - &t[0][1];
        &x0 * &y1 - &y0 * &x1
    };
    let tri_ccw = area.is_positive();
    let tri_edges = if tri_ccw {
        [(0, 1), (1, 2), (2, 0)]
    } else {
        [(0, 2), (2, 1), (1, 0)]
    };

    // parametric segment s(t)=a + t*(b-a)
    let dx = &b[0] - &a[0];
    let dy = &b[1] - &a[1];
    let mut t0 = T::zero();
    let mut t1 = T::one();

    let is_left = |vi: &Point2<T>, vj: &Point2<T>, x: &Point2<T>| {
        // cross( vj-vi, x-vi )
        let ex = &vj[0] - &vi[0];
        let ey = &vj[1] - &vi[1];
        let rx = &x[0] - &vi[0];
        let ry = &x[1] - &vi[1];
        &ex * &ry - &ey * &rx
    };

    for (i, j) in tri_edges {
        // signed “inside-ness” values at endpoints
        let f0 = is_left(&t[i], &t[j], &a);
        let f1 = is_left(&t[i], &t[j], &b);

        let f0_neg = f0.is_negative();
        let f1_neg = f1.is_negative();

        if f0_neg && f1_neg {
            return None; // fully outside
        }
        if f0_neg || f1_neg {
            // Find t where f(t)=0 along the segment.
            // f(t) = is_left(vi,vj, a + t*(b-a)) is linear in t:
            // f(t) = f0 + t * ((vj-vi) x (b-a))
            let denom = {
                let ex = &t[j][0] - &t[i][0];
                let ey = &t[j][1] - &t[i][1];
                &ex * &dy - &ey * &dx
            };
            if denom.is_zero() {
                // parallel edge: skip (shouldn’t happen for proper triangles)
                continue;
            }
            let t_hit = &(&T::zero() - &f0) / &denom; // t where it enters/leaves
            if f0_neg {
                // entering
                if (&t_hit - &t0).is_positive() {
                    t0 = t_hit;
                }
            } else {
                // leaving
                if (&t_hit - &t1).is_negative() {
                    t1 = t_hit;
                }
            }
            if (&t0 - &t1).is_positive_or_zero() {
                return None;
            }
        }
    }

    if t0.is_negative() && t1.is_negative() {
        return None;
    }
    if t0.is_positive() && (&t0 - &t1).is_positive_or_zero() {
        return None;
    }

    let lerp3 = |t: &T| {
        let ox = &dx * t;
        let oy = &dy * t;
        let p2 = Point2::<T>::from_vals([&a[0] + &ox, &a[1] + &oy]);
        back_project_to_3d::<T, N>(&p2, i0, i1, drop, a3) // drop axis from a3
    };
    let pa = lerp3(&t0);
    let pb = lerp3(&t1);
    if (&pb - &pa)
        .as_vector()
        .dot(&(&pb - &pa).as_vector())
        .is_positive()
    {
        Some((pa, pb))
    } else {
        None
    }
}

/// Reusable precomputation for a triangle:
/// - plane: n·x + d = 0
/// - coplanar projection axes (i0,i1,drop)
/// - projected 2D triangle and orientation for edge-function inside tests
#[derive(Clone)]
pub struct TriPrecomp<T: Scalar, const N: usize> {
    pub n: Vector<T, N>,
    pub d: T,
    pub axes: (usize, usize, usize),
    pub tri2d: [Point2<T>; 3],
    pub ccw: bool,
    pub degenerate: bool,
}

impl<T: Scalar, const N: usize> TriPrecomp<T, N>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    Point2<T>: PointOps<T, 2, Vector = Vector<T, 2>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    #[inline(always)]
    pub fn new(p: &[&Point<T, N>; 3]) -> Self {
        // Plane
        let e01 = (p[0] - p[2]).as_vector();
        let e02 = (p[1] - p[2]).as_vector();
        let n = e01.cross(&e02);
        if n.is_zero() {
            return Self {
                n,
                d: T::zero(),
                axes: (0, 1, 2),
                tri2d: [
                    Point2::<T>::from_vals([T::zero(), T::zero()]),
                    Point2::<T>::from_vals([T::zero(), T::zero()]),
                    Point2::<T>::from_vals([T::zero(), T::zero()]),
                ],
                ccw: false,
                degenerate: true,
            };
        }
        let d = -n.dot(&p[2].as_vector());

        // Axes and 2D triangle
        let (i0, i1, drop) = coplanar_axes(&n);
        let tri2d = [
            project_to_2d(p[0], i0, i1),
            project_to_2d(p[1], i0, i1),
            project_to_2d(p[2], i0, i1),
        ];

        // Orientation
        let area = {
            let x0 = &tri2d[1][0] - &tri2d[0][0];
            let y0 = &tri2d[1][1] - &tri2d[0][1];
            let x1 = &tri2d[2][0] - &tri2d[0][0];
            let y1 = &tri2d[2][1] - &tri2d[0][1];
            &x0 * &y1 - &y0 * &x1
        };
        let ccw = area.is_positive();

        Self {
            n,
            d,
            axes: (i0, i1, drop),
            tri2d,
            ccw,
            degenerate: false,
        }
    }

    /// 2D edge-function point-in-triangle test using precomputed 2D triangle and orientation.
    #[inline(always)]
    pub fn inside_2d(&self, r3: &Point<T, N>) -> bool {
        let (i0, i1, _) = self.axes;
        let r = project_to_2d(r3, i0, i1);

        let e0 = {
            let ex = &self.tri2d[1][0] - &self.tri2d[0][0];
            let ey = &self.tri2d[1][1] - &self.tri2d[0][1];
            let rx = &r[0] - &self.tri2d[0][0];
            let ry = &r[1] - &self.tri2d[0][1];
            &ex * &ry - &ey * &rx
        };
        let e1 = {
            let ex = &self.tri2d[2][0] - &self.tri2d[1][0];
            let ey = &self.tri2d[2][1] - &self.tri2d[1][1];
            let rx = &r[0] - &self.tri2d[1][0];
            let ry = &r[1] - &self.tri2d[1][1];
            &ex * &ry - &ey * &rx
        };
        let e2 = {
            let ex = &self.tri2d[0][0] - &self.tri2d[2][0];
            let ey = &self.tri2d[0][1] - &self.tri2d[2][1];
            let rx = &r[0] - &self.tri2d[2][0];
            let ry = &r[1] - &self.tri2d[2][1];
            &ex * &ry - &ey * &rx
        };

        if self.ccw {
            e0.is_positive_or_zero() && e1.is_positive_or_zero() && e2.is_positive_or_zero()
        } else {
            e0.is_negative_or_zero() && e1.is_negative_or_zero() && e2.is_negative_or_zero()
        }
    }
}

/// Optimized intersection using precomputed data for both triangles.
/// This avoids recomputing planes, axes, and projected triangles per pair.
pub fn tri_tri_intersection_with_precomp<T: Scalar, const N: usize>(
    p: &[&Point<T, N>; 3],
    q: &[&Point<T, N>; 3],
    pre_p: &TriPrecomp<T, N>,
    pre_q: &TriPrecomp<T, N>,
) -> TriTriIntersectionResult<T, N>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if pre_p.degenerate || pre_q.degenerate {
        return TriTriIntersectionResult::None;
    }

    // Signed distances of P to plane(Q): n2·pi + d2
    let d_p0 = &pre_q.n.dot(&p[0].as_vector()) + &pre_q.d;
    let d_p1 = &pre_q.n.dot(&p[1].as_vector()) + &pre_q.d;
    let d_p2 = &pre_q.n.dot(&p[2].as_vector()) + &pre_q.d;

    let zero_p = [d_p0.is_zero(), d_p1.is_zero(), d_p2.is_zero()];
    let zc_p = zero_p.iter().filter(|z| **z).count();

    if zc_p == 2 {
        // Coplanar segment of P against triangle Q
        let (i, j) = match zero_p {
            [true, true, false] => (0, 1),
            [true, false, true] => (0, 2),
            [false, true, true] => (1, 2),
            _ => unreachable!(),
        };
        if let Some((a3, b3)) = clip_segment_to_triangle_in_plane(p[i], p[j], q, &pre_q.n) {
            return TriTriIntersectionResult::Coplanar(Segment::new(&a3, &b3));
        }
    }

    // Strict separation by plane(Q)
    if (d_p0.is_positive() && d_p1.is_positive() && d_p2.is_positive())
        || (d_p0.is_negative() && d_p1.is_negative() && d_p2.is_negative())
    {
        return TriTriIntersectionResult::None;
    }

    // All three distances zero ⇒ coplanar
    if d_p0.is_zero() && d_p1.is_zero() && d_p2.is_zero() {
        return coplanar_tri_tri_intersection(p, q, &pre_q.n);
    }

    // Dedupe buffer for intersection points
    let merge = T::point_merge_threshold();
    let merge2 = &merge * &merge;
    let mut uniq = SmallVec::<[Point<T, N>; 4]>::new();
    let mut push_uniq = |pt: Point<T, N>| {
        for q3 in &uniq {
            let d = (&pt - q3).as_vector();
            if (&d.dot(&d) - &merge2).is_negative_or_zero() {
                return 0;
            }
        }
        uniq.push(pt);
        uniq.len()
    };

    // Collect intersections from P-edges against plane(Q): use cached distances
    for (ia, ib, da, db) in [
        (0usize, 1usize, &d_p0, &d_p1),
        (1usize, 2usize, &d_p1, &d_p2),
        (2usize, 0usize, &d_p2, &d_p0),
    ] {
        let prod = da * db;
        if prod.is_negative_or_zero() && !(da.is_zero() && db.is_zero()) {
            let denom = da - db;
            if denom.is_zero() {
                continue;
            }
            // t = da / (da - db)
            let t = da / &denom;
            let dir = (p[ib] - p[ia]).as_vector();
            let ip = p[ia].add_vector(&dir.scale(&t));
            if pre_q.inside_2d(&ip) && push_uniq(ip) == 2 {
                return TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[1]));
            }
        }
    }

    // Signed distances of Q to plane(P)
    let d_q0 = &pre_p.n.dot(&q[0].as_vector()) + &pre_p.d;
    let d_q1 = &pre_p.n.dot(&q[1].as_vector()) + &pre_p.d;
    let d_q2 = &pre_p.n.dot(&q[2].as_vector()) + &pre_p.d;

    let zero_q = [d_q0.is_zero(), d_q1.is_zero(), d_q2.is_zero()];
    let zc_q = zero_q.iter().filter(|z| **z).count();

    if zc_q == 2 {
        let (i, j) = match zero_q {
            [true, true, false] => (0, 1),
            [true, false, true] => (0, 2),
            [false, true, true] => (1, 2),
            _ => unreachable!(),
        };
        if let Some((a3, b3)) = clip_segment_to_triangle_in_plane(q[i], q[j], p, &pre_p.n) {
            return TriTriIntersectionResult::Coplanar(Segment::new(&a3, &b3));
        }
    }

    if (d_q0.is_positive() && d_q1.is_positive() && d_q2.is_positive())
        || (d_q0.is_negative() && d_q1.is_negative() && d_q2.is_negative())
    {
        // nothing more
    } else {
        // Q-edges against plane(P)
        for (ia, ib, da, db) in [
            (0usize, 1usize, &d_q0, &d_q1),
            (1usize, 2usize, &d_q1, &d_q2),
            (2usize, 0usize, &d_q2, &d_q0),
        ] {
            let prod = da * db;
            if prod.is_negative_or_zero() && !(da.is_zero() && db.is_zero()) {
                let denom = da - db;
                if denom.is_zero() {
                    continue;
                }
                let t = da / &denom;
                let dir = (q[ib] - q[ia]).as_vector();
                let iq = q[ia].add_vector(&dir.scale(&t));
                if pre_p.inside_2d(&iq) && push_uniq(iq) == 2 {
                    return TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[1]));
                }
            }
        }
    }

    // Finalize
    match uniq.len() {
        0 => TriTriIntersectionResult::None,
        1 => TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[0])),
        2 => TriTriIntersectionResult::Proper(Segment::new(&uniq[0], &uniq[1])),
        _ => {
            // farthest pair
            let mut max_d2 = T::zero();
            let mut best = (0usize, 1usize);
            for i in 0..uniq.len() {
                for j in (i + 1)..uniq.len() {
                    let v = (&uniq[j] - &uniq[i]).as_vector();
                    let d2 = v.dot(&v);
                    if (&d2 - &max_d2).is_positive() {
                        max_d2 = d2;
                        best = (i, j);
                    }
                }
            }
            TriTriIntersectionResult::Proper(Segment::new(&uniq[best.0], &uniq[best.1]))
        }
    }
}

/// Convenience overload: reuse precomputed data for `p`, compute once for `q`.
#[inline(always)]
pub fn tri_tri_intersection_with_precomp_p<T: Scalar, const N: usize>(
    p: &[&Point<T, N>; 3],
    q: &[&Point<T, N>; 3],
    pre_p: &TriPrecomp<T, N>,
) -> TriTriIntersectionResult<T, N>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let pre_q = TriPrecomp::new(q);
    if pre_q.degenerate {
        return TriTriIntersectionResult::None;
    }
    tri_tri_intersection_with_precomp(p, q, pre_p, &pre_q)
}

// New code for early triangle calculations
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichTri {
    P,
    Q,
}

#[derive(Clone, Debug)]
pub enum ContactOnTri<T: Scalar> {
    Vertex(usize),                    // exact vertex index 0..2 on that triangle
    Edge { e: (usize, usize), u: T }, // parameter along edge (i->j), u in [0,1]
    Interior { bary: (T, T, T) },     // barycentric coords (λ0, λ1, λ2) w.r.t. that triangle
}

#[derive(Clone, Debug)]
pub struct EndpointInfo<T: Scalar, const N: usize> {
    pub point: Point<T, N>,
    pub on_p: ContactOnTri<T>,
    pub on_q: ContactOnTri<T>,
}

#[derive(Clone, Debug)]
pub enum TriTriIntersectionDetailed<T: Scalar, const N: usize> {
    Proper {
        ends: [EndpointInfo<T, N>; 2],
    },
    Coplanar {
        segs: Vec<(EndpointInfo<T, N>, EndpointInfo<T, N>)>,
    },
    None,
}

#[inline(always)]
fn edge_param_along<T: Scalar, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    p: &Point<T, N>,
) -> T
where
    for<'a> &'a T: Sub<&'a T, Output = T> + Div<&'a T, Output = T> + Add<&'a T, Output = T>,
{
    // Choose the axis with the largest |b[i] - a[i]| to compute t robustly
    let mut k = 0usize;
    let mut best = (&b[0] - &a[0]).abs();
    for i in 1..N {
        let e = (&b[i] - &a[i]).abs();
        if (&e - &best).is_positive() {
            best = e;
            k = i;
        }
    }
    if best.is_zero() {
        // Degenerate edge; return 0 to avoid div-by-zero (caller will ignore t)
        return T::zero();
    }
    &(&p[k] - &a[k]) / &(&b[k] - &a[k])
}

#[inline(always)]
fn classify_on_tri<T: Scalar, const N: usize>(
    r3: &Point<T, N>,
    tri3: &[&Point<T, N>; 3],
    pre: &TriPrecomp<T, N>,
) -> ContactOnTri<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Point2<T>: PointOps<T, 2, Vector = Vector<T, 2>>,
    Vector<T, 2>: VectorOps<T, 2>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Project point with the same axes used to build pre.tri2d
    let (i0, i1, _) = pre.axes;
    let r = project_to_2d(r3, i0, i1);

    // Edge functions in the projected plane (consistent with pre.ccw)
    let e0 = {
        let ex = &pre.tri2d[1][0] - &pre.tri2d[0][0];
        let ey = &pre.tri2d[1][1] - &pre.tri2d[0][1];
        let rx = &r[0] - &pre.tri2d[0][0];
        let ry = &r[1] - &pre.tri2d[0][1];
        &ex * &ry - &ey * &rx
    };
    let e1 = {
        let ex = &pre.tri2d[2][0] - &pre.tri2d[1][0];
        let ey = &pre.tri2d[2][1] - &pre.tri2d[1][1];
        let rx = &r[0] - &pre.tri2d[1][0];
        let ry = &r[1] - &pre.tri2d[1][1];
        &ex * &ry - &ey * &rx
    };
    let e2 = {
        let ex = &pre.tri2d[0][0] - &pre.tri2d[2][0];
        let ey = &pre.tri2d[0][1] - &pre.tri2d[2][1];
        let rx = &r[0] - &pre.tri2d[2][0];
        let ry = &r[1] - &pre.tri2d[2][1];
        &ex * &ry - &ey * &rx
    };

    let z0 = e0.is_zero();
    let z1 = e1.is_zero();
    let z2 = e2.is_zero();

    // Vertex cases: two zero edge-functions
    if z0 && z2 {
        return ContactOnTri::Vertex(0);
    } else if z0 && z1 {
        return ContactOnTri::Vertex(1);
    } else if z1 && z2 {
        return ContactOnTri::Vertex(2);
    }

    // Edge cases: exactly one edge-function is zero
    if z0 {
        // Edge (0,1)
        let u = edge_param_along::<T, N>(tri3[0], tri3[1], r3);
        return ContactOnTri::Edge { e: (0, 1), u };
    }
    if z1 {
        // Edge (1,2)
        let u = edge_param_along::<T, N>(tri3[1], tri3[2], r3);
        return ContactOnTri::Edge { e: (1, 2), u };
    }
    if z2 {
        // Edge (2,0)
        let u = edge_param_along::<T, N>(tri3[2], tri3[0], r3);
        return ContactOnTri::Edge { e: (2, 0), u };
    }

    // Interior: compute barycentric coordinates in the same 2D projection
    // area = cross(t1 - t0, t2 - t0)
    let area = {
        let x0 = &pre.tri2d[1][0] - &pre.tri2d[0][0];
        let y0 = &pre.tri2d[1][1] - &pre.tri2d[0][1];
        let x1 = &pre.tri2d[2][0] - &pre.tri2d[0][0];
        let y1 = &pre.tri2d[2][1] - &pre.tri2d[0][1];
        &x0 * &y1 - &y0 * &x1
    };
    // λ0 = cross(b - r, c - r) / area
    // λ1 = cross(c - r, a - r) / area
    // λ2 = cross(a - r, b - r) / area
    let bx = &pre.tri2d[1][0] - &r[0];
    let by = &pre.tri2d[1][1] - &r[1];
    let cx = &pre.tri2d[2][0] - &r[0];
    let cy = &pre.tri2d[2][1] - &r[1];
    let ax = &pre.tri2d[0][0] - &r[0];
    let ay = &pre.tri2d[0][1] - &r[1];

    let num0 = &(&bx * &cy) - &(&by * &cx);
    let num1 = &(&cx * &ay) - &(&cy * &ax);
    let num2 = &(&ax * &by) - &(&ay * &bx);

    // area must be non-zero for non-degenerate triangles (guaranteed by pre.degenerate)
    let inv_area = &T::one() / &area;
    let l0 = &num0 * &inv_area;
    let l1 = &num1 * &inv_area;
    let l2 = &num2 * &inv_area;

    ContactOnTri::Interior { bary: (l0, l1, l2) }
}

#[inline(always)]
fn merge_contact<T: Scalar>(a: &mut ContactOnTri<T>, b: ContactOnTri<T>) {
    // Prefer more specific classifications: Vertex > Edge > Interior
    match (&*a, b) {
        (ContactOnTri::Vertex(_), _) => {}
        (ContactOnTri::Edge { .. }, ContactOnTri::Vertex(v)) => {
            *a = ContactOnTri::Vertex(v);
        }
        (ContactOnTri::Interior { .. }, ContactOnTri::Edge { e, u }) => {
            *a = ContactOnTri::Edge { e, u };
        }
        (ContactOnTri::Interior { .. }, ContactOnTri::Vertex(v)) => {
            *a = ContactOnTri::Vertex(v);
        }
        _ => {}
    }
}

#[inline(always)]
fn push_uniq_info<T: Scalar, const N: usize>(
    uniq: &mut Vec<EndpointInfo<T, N>>,
    merge2: &T,
    mut info: EndpointInfo<T, N>,
) where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    for u in uniq.iter_mut() {
        let d = (&info.point - &u.point).as_vector();
        let d2 = d.dot(&d);
        if (&d2 - merge2).is_negative_or_zero() {
            // Merge classifications (prefer most specific)
            merge_contact(&mut u.on_p, info.on_p);
            merge_contact(&mut u.on_q, info.on_q);
            return;
        }
    }
    uniq.push(info);
}

#[inline(always)]
fn snap_exterior_to_edge<T: Scalar, const N: usize>(
    point: &Point<T, N>,
    tri: &[&Point<T, N>; 3],
    contact: ContactOnTri<T>,
) -> ContactOnTri<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if let ContactOnTri::Interior { bary: (l0, l1, l2) } = contact.clone() {
        // Check if any barycentric coordinate is negative (exterior point)
        let l0_neg = l0.is_negative();
        let l1_neg = l1.is_negative();
        let l2_neg = l2.is_negative();

        if l0_neg || l1_neg || l2_neg {
            // Find closest edge by computing distance to each edge
            let mut min_dist2 = T::from(1e300);
            let mut closest_edge = (0usize, 1usize);
            let mut closest_u = T::zero();

            // Check all three edges
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let edge_vec = (tri[j] - tri[i]).as_vector();
                let to_point = (point - tri[i]).as_vector();

                // Project point onto edge
                let edge_len2 = edge_vec.dot(&edge_vec);
                if edge_len2.is_zero() {
                    continue; // Degenerate edge
                }

                let t = &to_point.dot(&edge_vec) / &edge_len2;
                let clamped_t = if t.is_negative() {
                    T::zero()
                } else if (&t - &T::one()).is_positive() {
                    T::one()
                } else {
                    t
                };

                // Point on edge
                let edge_point = tri[i].add_vector(&edge_vec.scale(&clamped_t));
                let dist_vec = (point - &edge_point).as_vector();
                let dist2 = dist_vec.dot(&dist_vec);

                if (&dist2 - &min_dist2).is_negative() {
                    min_dist2 = dist2;
                    closest_edge = (i, j);
                    closest_u = clamped_t;
                }
            }

            return ContactOnTri::Edge {
                e: closest_edge,
                u: closest_u,
            };
        }
    }

    contact
}

pub fn tri_tri_intersection_with_precomp_detailed<T: Scalar, const N: usize>(
    p: &[&Point<T, N>; 3],
    q: &[&Point<T, N>; 3],
    pre_p: &TriPrecomp<T, N>,
    pre_q: &TriPrecomp<T, N>,
) -> TriTriIntersectionDetailed<T, N>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Point2<T>: PointOps<T, 2, Vector = Vector<T, 2>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    Vector<T, 2>: VectorOps<T, 2>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if pre_p.degenerate || pre_q.degenerate {
        return TriTriIntersectionDetailed::None;
    }

    // Signed distances of P to plane(Q)
    let d_p0 = &pre_q.n.dot(&p[0].as_vector()) + &pre_q.d;
    let d_p1 = &pre_q.n.dot(&p[1].as_vector()) + &pre_q.d;
    let d_p2 = &pre_q.n.dot(&p[2].as_vector()) + &pre_q.d;

    let zero_p = [d_p0.is_zero(), d_p1.is_zero(), d_p2.is_zero()];
    let zc_p = zero_p.iter().filter(|z| **z).count();

    if zc_p == 2 {
        // Coplanar segment of P on triangle Q
        let (i, j) = match zero_p {
            [true, true, false] => (0, 1),
            [true, false, true] => (0, 2),
            [false, true, true] => (1, 2),
            _ => unreachable!(),
        };
        if let Some((a3, b3)) = clip_segment_to_triangle_in_plane(p[i], p[j], q, &pre_q.n) {
            // Classify endpoints on both triangles
            let on_q_a = classify_on_tri::<T, N>(&a3, q, &pre_q);
            let on_q_b = classify_on_tri::<T, N>(&b3, q, &pre_q);
            let on_p_a = classify_on_tri::<T, N>(&a3, p, &pre_p);
            let on_p_b = classify_on_tri::<T, N>(&b3, p, &pre_p);
            return TriTriIntersectionDetailed::Proper {
                ends: [
                    EndpointInfo {
                        point: a3,
                        on_p: on_p_a,
                        on_q: on_q_a,
                    },
                    EndpointInfo {
                        point: b3,
                        on_p: on_p_b,
                        on_q: on_q_b,
                    },
                ],
            };
        }
    }

    // Strict separation by plane(Q)
    if (d_p0.is_positive() && d_p1.is_positive() && d_p2.is_positive())
        || (d_p0.is_negative() && d_p1.is_negative() && d_p2.is_negative())
    {
        return TriTriIntersectionDetailed::None;
    }

    // Completely coplanar
    if d_p0.is_zero() && d_p1.is_zero() && d_p2.is_zero() {
        // Defer to coplanar handler: build polygon edges, then classify endpoints
        match coplanar_tri_tri_intersection(p, q, &pre_q.n) {
            TriTriIntersectionResult::Proper(seg) | TriTriIntersectionResult::Coplanar(seg) => {
                let a3 = seg.a.clone();
                let b3 = seg.b.clone();
                let on_q_a = classify_on_tri::<T, N>(&a3, q, &pre_q);
                let on_q_b = classify_on_tri::<T, N>(&b3, q, &pre_q);
                let on_p_a = classify_on_tri::<T, N>(&a3, p, &pre_p);
                let on_p_b = classify_on_tri::<T, N>(&b3, p, &pre_p);
                return TriTriIntersectionDetailed::Proper {
                    ends: [
                        EndpointInfo {
                            point: a3,
                            on_p: on_p_a,
                            on_q: on_q_a,
                        },
                        EndpointInfo {
                            point: b3,
                            on_p: on_p_b,
                            on_q: on_q_b,
                        },
                    ],
                };
            }
            TriTriIntersectionResult::CoplanarPolygon(segs) => {
                let mut out = Vec::with_capacity(segs.len());
                for s in segs {
                    let a3 = s.a.clone();
                    let b3 = s.b.clone();
                    let on_q_a = classify_on_tri::<T, N>(&a3, q, &pre_q);
                    let on_q_b = classify_on_tri::<T, N>(&b3, q, &pre_q);
                    let on_p_a = classify_on_tri::<T, N>(&a3, p, &pre_p);
                    let on_p_b = classify_on_tri::<T, N>(&b3, p, &pre_p);
                    out.push((
                        EndpointInfo {
                            point: a3,
                            on_p: on_p_a,
                            on_q: on_q_a,
                        },
                        EndpointInfo {
                            point: b3,
                            on_p: on_p_b,
                            on_q: on_q_b,
                        },
                    ));
                }
                return TriTriIntersectionDetailed::Coplanar { segs: out };
            }
            TriTriIntersectionResult::None => return TriTriIntersectionDetailed::None,
        }
    }

    // Collect unique endpoints, merging duplicates
    let merge = T::point_merge_threshold();
    let merge2 = &merge * &merge;
    let mut uniq: Vec<EndpointInfo<T, N>> = Vec::with_capacity(2);

    // Intersections from P-edges against plane(Q) (reuse cached distances)
    for (ia, ib, da, db) in [
        (0usize, 1usize, &d_p0, &d_p1),
        (1usize, 2usize, &d_p1, &d_p2),
        (2usize, 0usize, &d_p2, &d_p0),
    ] {
        let prod = da * db;
        if prod.is_negative_or_zero() && !(da.is_zero() && db.is_zero()) {
            let denom = da - db;
            if denom.is_zero() {
                continue;
            }

            let t = da / &denom;
            let dir = (p[ib] - p[ia]).as_vector();
            let ip = p[ia].add_vector(&dir.scale(&t));

            if !pre_q.inside_2d(&ip) {
                continue;
            }

            let on_p = if t.is_zero() {
                ContactOnTri::Vertex(ia)
            } else if (&(&t - &T::one())).is_zero() {
                ContactOnTri::Vertex(ib)
            } else {
                ContactOnTri::Edge {
                    e: (ia, ib),
                    u: t.clone(),
                }
            };

            // Classify vs Q and snap if exterior
            let on_q = classify_on_tri::<T, N>(&ip, q, &pre_q);
            let on_q = snap_exterior_to_edge(&ip, q, on_q);

            push_uniq_info(
                &mut uniq,
                &merge2,
                EndpointInfo {
                    point: ip,
                    on_p,
                    on_q,
                },
            );
            if uniq.len() == 2 {
                return TriTriIntersectionDetailed::Proper {
                    ends: [uniq.swap_remove(0), uniq.swap_remove(0)],
                };
            }
        }
    }

    // Now distances of Q to plane(P), and Q-edges intersections
    let d_q0 = &pre_p.n.dot(&q[0].as_vector()) + &pre_p.d;
    let d_q1 = &pre_p.n.dot(&q[1].as_vector()) + &pre_p.d;
    let d_q2 = &pre_p.n.dot(&q[2].as_vector()) + &pre_p.d;

    let zero_q = [d_q0.is_zero(), d_q1.is_zero(), d_q2.is_zero()];
    let zc_q = zero_q.iter().filter(|z| **z).count();
    if zc_q == 2 {
        let (i, j) = match zero_q {
            [true, true, false] => (0, 1),
            [true, false, true] => (0, 2),
            [false, true, true] => (1, 2),
            _ => unreachable!(),
        };
        if let Some((a3, b3)) = clip_segment_to_triangle_in_plane(q[i], q[j], p, &pre_p.n) {
            // Classify endpoints on both triangles and return
            let on_p_a = classify_on_tri::<T, N>(&a3, p, &pre_p);
            let on_p_b = classify_on_tri::<T, N>(&b3, p, &pre_p);
            let on_q_a = classify_on_tri::<T, N>(&a3, q, &pre_q);
            let on_q_b = classify_on_tri::<T, N>(&b3, q, &pre_q);
            return TriTriIntersectionDetailed::Proper {
                ends: [
                    EndpointInfo {
                        point: a3,
                        on_p: on_p_a,
                        on_q: on_q_a,
                    },
                    EndpointInfo {
                        point: b3,
                        on_p: on_p_b,
                        on_q: on_q_b,
                    },
                ],
            };
        }
    }

    if (d_q0.is_positive() && d_q1.is_positive() && d_q2.is_positive())
        || (d_q0.is_negative() && d_q1.is_negative() && d_q2.is_negative())
    {
        // nothing more
    } else {
        // Q-edges against plane(P)
        for (ia, ib, da, db) in [
            (0usize, 1usize, &d_q0, &d_q1),
            (1usize, 2usize, &d_q1, &d_q2),
            (2usize, 0usize, &d_q2, &d_q0),
        ] {
            let prod = da * db;
            if prod.is_negative_or_zero() && !(da.is_zero() && db.is_zero()) {
                let denom = da - db;
                if denom.is_zero() {
                    continue;
                }
                let u = da / &denom;
                let dir = (q[ib] - q[ia]).as_vector();
                let iq = q[ia].add_vector(&dir.scale(&u));

                if !pre_p.inside_2d(&iq) {
                    continue;
                }

                let on_q = if u.is_zero() {
                    ContactOnTri::Vertex(ia)
                } else if (&(&u - &T::one())).is_zero() {
                    ContactOnTri::Vertex(ib)
                } else {
                    ContactOnTri::Edge {
                        e: (ia, ib),
                        u: u.clone(),
                    }
                };

                // Classify vs P and snap if exterior
                let on_p = classify_on_tri::<T, N>(&iq, p, &pre_p);
                let on_p = snap_exterior_to_edge(&iq, p, on_p);

                push_uniq_info(
                    &mut uniq,
                    &merge2,
                    EndpointInfo {
                        point: iq,
                        on_p,
                        on_q,
                    },
                );
                if uniq.len() == 2 {
                    return TriTriIntersectionDetailed::Proper {
                        ends: [uniq.swap_remove(0), uniq.swap_remove(0)],
                    };
                }
            }
        }
    }

    match uniq.len() {
        0 => TriTriIntersectionDetailed::None,
        1 => TriTriIntersectionDetailed::Proper {
            ends: [uniq[0].clone(), uniq[0].clone()],
        },
        2 => TriTriIntersectionDetailed::Proper {
            ends: [uniq.swap_remove(0), uniq.swap_remove(0)],
        },
        _ => {
            // Farthest pair (keep associated classifications)
            let mut max_d2 = T::zero();
            let mut best = (0usize, 1usize);
            for i in 0..uniq.len() {
                for j in (i + 1)..uniq.len() {
                    let v = (&uniq[j].point - &uniq[i].point).as_vector();
                    let d2 = v.dot(&v);
                    if (&d2 - &max_d2).is_positive() {
                        max_d2 = d2;
                        best = (i, j);
                    }
                }
            }
            TriTriIntersectionDetailed::Proper {
                ends: [uniq[best.0].clone(), uniq[best.1].clone()],
            }
        }
    }
}

#[inline(always)]
pub fn tri_tri_intersection_detailed<T: Scalar, const N: usize>(
    p: &[&Point<T, N>; 3],
    q: &[&Point<T, N>; 3],
) -> TriTriIntersectionDetailed<T, N>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    Point2<T>: PointOps<T, 2, Vector = Vector<T, 2>>,
    Vector<T, 2>: VectorOps<T, 2>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let pre_p = TriPrecomp::new(p);
    let pre_q = TriPrecomp::new(q);
    tri_tri_intersection_with_precomp_detailed(p, q, &pre_p, &pre_q)
}
