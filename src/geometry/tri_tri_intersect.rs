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
    collections::HashSet,
    ops::{Add, Div, Mul, Sub},
    time::Instant,
};

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
            let (tmin, tmax) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
            let start = tmin.max(0.0.into());
            let end = tmax.min(1.0.into());
            if start <= end {
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

    if s >= (-EPS).into()
        && s <= (1.0 + EPS).into()
        && u >= (-EPS).into()
        && u <= (1.0 + EPS).into()
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
    if &p1 < &min {
        min = p1.clone();
    }
    if &p1 > &max {
        max = p1.clone();
    }
    if &p2 < &min {
        min = p2.clone();
    }
    if &p2 > &max {
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

    // 5) dedupe
    let mut set = HashSet::new();
    let mut uniq: Vec<Point<T, N>> = Vec::new();
    println!("adding an intersection:");
    for p in pts {
        if set.insert(p.clone()) {
            uniq.push(p);
        }
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
        let xi = &pts[i].coords[0];
        let xj = &pts[j].coords[0];
        xi.partial_cmp(xj)
            .unwrap()
            .then_with(|| (&pts[i].coords[1]).partial_cmp(&pts[j].coords[1]).unwrap())
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
            if cross <= T::zero() {
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
            if cross <= T::zero() {
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
    // 1) Build plane of T2: n2·x + d2 = 0
    let v01 = (q[0] - q[2]).as_vector();
    let v02 = (q[1] - q[2]).as_vector();
    let n2 = v01.cross(&v02);
    let d2 = -n2.dot(&q[2].as_vector());

    // signed distances of p-verts to T2's plane
    let d_p0 = &n2.dot(&p[0].as_vector()) + &d2;
    let d_p1 = &n2.dot(&p[1].as_vector()) + &d2;
    let d_p2 = &n2.dot(&p[2].as_vector()) + &d2;

    let zero_p = [d_p0.is_zero(), d_p1.is_zero(), d_p2.is_zero()];
    let zc_p = zero_p.iter().filter(|z| **z).count();

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

    // Fall back for strictly co-planar
    if d_p0.is_zero() && d_p1.is_zero() && d_p2.is_zero() {
        return coplanar_tri_tri_intersection(p, q, &n2);
    }

    // 2) Now do the regular non-coplanar plane‐edge clipping:
    let mut pts = Vec::new();
    for (a, b) in [(&p[0], &p[1]), (&p[1], &p[2]), (&p[2], &p[0])] {
        if let Some(ip) = intersect_edge_plane(&a, &b, &n2.0, &d2) {
            if point_in_tri(&ip, &q[0], &q[1], &q[2]) {
                pts.push(ip);
            }
        }
    }

    // 3) Build plane of T1:
    let u01 = (p[0] - p[2]).as_vector();
    let u02 = (p[1] - p[2]).as_vector();
    let n1 = u01.cross(&u02);
    let d1 = -n1.dot(&p[2].as_vector());

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

    // 4) Clip edges of T2 against T1’s plane:
    for (a, b) in [(&q[0], &q[1]), (&q[1], &q[2]), (&q[2], &q[0])] {
        if let Some(ip) = intersect_edge_plane(&a, &b, &n1.0, &d1) {
            if point_in_tri(&ip, &p[0], &p[1], &p[2]) {
                pts.push(ip);
            }
        }
    }

    // 5) Deduplicate exactly via HashSet
    let mut set = HashSet::new();
    let mut uniq = Vec::new();
    for p in pts {
        if set.insert(p.clone()) {
            uniq.push(p);
        }
    }

    // 6) Make sure both points are on T1
    let on_a: Vec<_> = uniq
        .iter()
        //.filter(|this_p| point_in_tri(this_p, &p[0], &p[1], &p[2]))
        .cloned()
        .collect();

    // 6) Return based on unique points count
    match on_a.len() {
        0 => TriTriIntersectionResult::None,
        1 => TriTriIntersectionResult::Proper(Segment::new(&on_a[0], &on_a[0])),
        2 => TriTriIntersectionResult::Proper(Segment::new(&on_a[0], &on_a[1])),
        _ => {
            // More than 2 points - select the two most distant points
            let mut max_dist_sq = T::zero();
            let mut best_pair = (0, 1);

            for i in 0..on_a.len() {
                for j in (i + 1)..on_a.len() {
                    let diff = (&on_a[j] - &on_a[i]).as_vector();
                    let dist_sq = diff.dot(&diff);
                    if dist_sq > max_dist_sq {
                        max_dist_sq = dist_sq;
                        best_pair = (i, j);
                    }
                }
            }

            let (i, j) = best_pair;
            TriTriIntersectionResult::Proper(Segment::new(&on_a[i], &on_a[j]))
            // TriTriIntersectionResult::Proper(Segment::new(&uniq[best_pair.0], &uniq[best_pair.1]))
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
fn coplanar_axes<T: Scalar, const N: usize>(n: &Vector<T, N>) -> (usize, usize, usize) {
    let na = [n[0].abs(), n[1].abs(), n[2].abs()];
    let (i0, i1, drop) = if na[0] > na[1] && na[0] > na[2] {
        (1, 2, 0)
    } else if na[1] > na[2] {
        (0, 2, 1)
    } else {
        (0, 1, 2)
    };
    (i0, i1, drop)
}

/// Project a 3D point onto a 2D plane using the provided axes.
fn project_to_2d<T: Scalar, const N: usize>(p: &Point<T, N>, i0: usize, i1: usize) -> Point2<T> {
    Point2::from_vals([p[i0].clone(), p[i1].clone()])
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
                if t_hit > t0 {
                    t0 = t_hit;
                }
            } else {
                // leaving
                if t_hit < t1 {
                    t1 = t_hit;
                }
            }
            if !(t0 < t1) && !(&t0 - &t1).is_negative() {
                return None;
            }
        }
    }

    if t0.is_negative() && t1.is_negative() {
        return None;
    }
    if t0.is_positive() && t0 >= t1 {
        return None;
    }

    let lerp3 = |t: &T| {
        let ox = &dx * t;
        let oy = &dy * t;
        let p2 = Point2::from_vals([&a[0] + &ox, &a[1] + &oy]);
        back_project_to_3d(&p2, i0, i1, drop, a3) // drop axis from a3
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
