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
};

use crate::{
    geometry::{
        Point2, Point3, Segment3, Vector3,
        point::{Point, PointOps},
        segment::Segment,
        spatial_element::SpatialElement,
        util::*,
        vector::{Vector, VectorOps},
    },
    numeric::scalar::Scalar,
    operations::Zero,
};

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
    if denom == T::zero() {
        // degenerate triangle
        return false;
    }
    let u = &(&(&(y1 - y2) * &(x - &x2)) + &(&(x2 - x1) * &(y - &y2))) / &denom;
    let v = &(&(&(y2 - y0) * &(x - &x2)) + &(&(x0 - x2) * &(y - &y2))) / &denom;
    u >= T::zero() && v >= T::zero() && (&u + &v) <= T::one()
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
fn project_3d_triangle<T>(axis: &Vector3<T>, a: &Point3<T>, b: &Point3<T>, c: &Point3<T>) -> (T, T)
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let p = |p: &Point3<T>| {
        let v: Vector3<T> = p.as_vector().into();
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

fn coplanar_tri_tri_intersection<T>(
    p: &[Point3<T>; 3],
    q: &[Point3<T>; 3],
    n: &Vector3<T>,
) -> Option<Segment3<T>>
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let (i0, i1, drop) = coplanar_axes(n);

    // 2) build 2D points
    let to2d = |p: &Point3<T>| project_to_2d(p, i0, i1);
    let t1 = [to2d(&p[2]), to2d(&p[0]), to2d(&p[1])];
    let t2 = [to2d(&q[2]), to2d(&q[0]), to2d(&q[1])];

    // 3) collect vertices of one triangle inside the other
    let mut pts: Vec<Point3<T>> = Vec::new();
    for point in &t1 {
        if point_in_tri_2d(point, &t2) {
            pts.push(back_project_to_3d(point, i0, i1, drop, &p[2]));
        }
    }
    for point in &t2 {
        if point_in_tri_2d(point, &t1) {
            pts.push(back_project_to_3d(point, i0, i1, drop, &q[2]));
        }
    }

    // 4) edge-edge intersections in 2D
    let edges1 = [
        (t1[0].clone(), t1[1].clone()),
        (t1[1].clone(), t1[2].clone()),
        (t1[2].clone(), t1[0].clone()),
    ];
    let edges2 = [
        (t2[0].clone(), t2[1].clone()),
        (t2[1].clone(), t2[2].clone()),
        (t2[2].clone(), t2[0].clone()),
    ];
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
    let mut uniq: Vec<Point3<T>> = Vec::new();
    for p in pts {
        if set.insert(p.clone()) {
            uniq.push(p)
        }
    }
    if uniq.len() == 2 {
        Some(Segment::new(&uniq[0], &uniq[1]))
    } else if uniq.len() == 1 {
        Some(Segment::new(&uniq[0], &uniq[0]))
    } else {
        None
    }
}

/// Computes the segment where triangles T1=(p0,p1,p2) and T2=(q0,q1,q2) intersect.
/// Returns `None` if they don’t intersect, or `Some((a,b))` where `a` and `b` are
/// the two endpoints of the intersection segment (possibly `a==b` if they touch at a point).
pub fn tri_tri_intersection<T>(p: &[Point3<T>; 3], q: &[Point3<T>; 3]) -> Option<Segment3<T>>
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) Build plane of T2: n2·x + d2 = 0
    let v01 = (&q[0] - &q[2]).as_vector();
    let v02 = (&q[1] - &q[2]).as_vector();
    let n2 = v01.cross(&v02);
    let d2 = -n2.dot(&q[2].as_vector().into());

    // signed distances of p-verts to T2’s plane
    let d_p0 = &n2.dot(&p[2].as_vector().into()) + &d2;
    let d_p1 = &n2.dot(&p[0].as_vector().into()) + &d2;
    let d_p2 = &n2.dot(&p[1].as_vector().into()) + &d2;

    //  → Fall back for strictly co-planar
    if d_p0 == T::zero() && d_p1 == T::zero() && d_p2 == T::zero() {
        // Must handle co-planar intersection in 2D
        return coplanar_tri_tri_intersection(p, q, &n2);
    }

    // 2) Now do the regular non-coplanar plane‐edge clipping:
    let mut pts = Vec::new();
    for (a, b) in [(&p[2], &p[0]), (&p[0], &p[1]), (&p[1], &p[2])] {
        if let Some(ip) = intersect_edge_plane(&a, &b, &n2.clone().into(), &d2) {
            if point_in_tri(&ip, &q[2], &q[0], &q[1]) {
                pts.push(ip);
            }
        }
    }

    // 3) Build plane of T1:
    let u01 = (&p[0] - &p[2]).as_vector();
    let u02 = (&p[1] - &p[2]).as_vector();
    let n1 = u01.cross(&u02);
    let d1 = -n1.dot(&p[2].as_vector().into());

    // 4) Clip edges of T2 against T1’s plane:
    for (a, b) in [(&q[2], &q[0]), (&q[0], &q[1]), (&q[1], &q[2])] {
        if let Some(ip) = intersect_edge_plane(&a, &b, &n1.clone().into(), &d1) {
            if point_in_tri(&ip, &p[2], &p[0], &p[1]) {
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
        .filter(|this_p| point_in_tri(this_p, &p[2], &p[0], &p[1]))
        .cloned()
        .collect();

    // 7) Return
    match on_a.len() {
        2 => Some(Segment::new(&on_a[0], &on_a[1])),
        1 => Some(Segment::new(&on_a[0], &on_a[0])),
        _ => None,
    }
}

/// Intersect the segment [a,b] against plane (n·x + d = 0).
/// Returns `Some(Point3)` if it crosses or touches, else `None`.
fn intersect_edge_plane<T>(a: &Point3<T>, b: &Point3<T>, n: &Point3<T>, d: &T) -> Option<Point3<T>>
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // signed distances:
    let da = &n.as_vector().dot(&a.as_vector()) + &d;
    let db = &n.as_vector().dot(&b.as_vector()) + &d;

    // if both on same side (and nonzero), no cross
    if &da * &db > T::zero() {
        return None;
    }

    // Check for division by zero
    let denominator = &da - &db;
    if denominator == T::zero() {
        // Edge is parallel to plane, check if it lies on the plane
        if da == T::zero() {
            return Some(a.clone()); // Both points are on the plane
        }
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
fn point_in_tri<T>(p: &Point3<T>, a: &Point3<T>, b: &Point3<T>, c: &Point3<T>) -> bool
where
    T: Scalar,
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
    u >= T::zero() && v >= T::zero() && (&u + &v) <= T::one()
}

/// Given a normal, return the indices of the two axes to keep (largest dropped).
fn coplanar_axes<T: Scalar>(n: &Vector3<T>) -> (usize, usize, usize) {
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
fn project_to_2d<T: Scalar>(p: &Point3<T>, i0: usize, i1: usize) -> Point2<T> {
    Point2::from_vals([p[i0].clone(), p[i1].clone()])
}

/// Back-project a 2D point into 3D, using a reference 3D point for the dropped axis.
fn back_project_to_3d<T: Scalar>(
    p: &Point2<T>,
    i0: usize,
    i1: usize,
    drop: usize,
    reference: &Point3<T>,
) -> Point3<T>
where
    Point3<T>: SpatialElement<T, 3>,
{
    let mut coords = Point3::zero();
    coords[i0] = p.coords[0].clone();
    coords[i1] = p.coords[1].clone();
    coords[drop] = reference[drop].clone();
    coords
}
