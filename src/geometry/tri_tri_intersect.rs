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
        Point3, Vector3, point::PointOps, spatial_element::SpatialElement, vector::VectorOps,
    },
    numeric::scalar::Scalar,
    operations::Zero,
};

/// Fast 3D triangle–triangle overlap test (Möller 1997).
/// Returns true if T1=(p0,p1,p2) and T2=(q0,q1,q2) intersect.
pub fn tri_tri_overlap<T>(
    p0: &Point3<T>,
    p1: &Point3<T>,
    p2: &Point3<T>,
    q0: &Point3<T>,
    q1: &Point3<T>,
    q2: &Point3<T>,
) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) plane test for triangle T1
    let e1 = (p1 - p0).as_vector();
    let e2 = (p2 - p0).as_vector();
    let n1 = e1.cross(&e2);

    let d0 = n1.dot(&(q0 - p0).as_vector().into());
    let d1 = n1.dot(&(q1 - p0).as_vector().into());
    let d2 = n1.dot(&(q2 - p0).as_vector().into());

    if (d0 > T::zero() && d1 > T::zero() && d2 > T::zero())
        || (d0 < T::zero() && d1 < T::zero() && d2 < T::zero())
    {
        return false;
    }
    // coplanar?
    if d0 == T::zero() && d1 == T::zero() && d2 == T::zero() {
        return coplanar_tri_tri(p0, p1, p2, q0, q1, q2, &n1);
    }

    // 2) plane test for triangle T2
    let f1 = (q1 - q0).as_vector();
    let f2 = (q2 - q0).as_vector();
    let n2 = f1.cross(&f2);

    let e0 = n2.dot(&(p0 - q0).as_vector().into());
    let e1_ = n2.dot(&(p1 - q0).as_vector().into());
    let e2_ = n2.dot(&(p2 - q0).as_vector().into());

    if (e0 > T::zero() && e1_ > T::zero() && e2_ > T::zero())
        || (e0 < T::zero() && e1_ < T::zero() && e2_ < T::zero())
    {
        return false;
    }

    // 3) 9 cross‐edge SAT tests
    let tri_axes = [
        e1.cross(&f1),
        e1.cross(&f2),
        e1.cross(&(q0 - q2).as_vector()),
        e2.cross(&f1),
        e2.cross(&f2),
        e2.cross(&(q0 - q2).as_vector()),
        (p0 - p2).as_vector().cross(&f1),
        (p0 - p2).as_vector().cross(&f2),
        (p0 - p2).as_vector().cross(&(q0 - q2).as_vector()),
    ];

    for axis in &tri_axes {
        if Vector3::<T>::zero() == *axis {
            continue; // parallel edges
        }
        let (min1, max1) = project_3d_triangle(axis, p0, p1, p2);
        let (min2, max2) = project_3d_triangle(axis, q0, q1, q2);
        if max1 < min2 || max2 < min1 {
            return false;
        }
    }

    true
}

/// Return true if 2D point `p` lies inside (or on) the triangle `tri` = [(x0,y0),(x1,y1),(x2,y2)].
/// Uses a barycentric‐coordinate test.
fn point_in_tri_2d<T>(p: (T, T), tri: &[(T, T); 3]) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let (x, y) = p;
    let (x0, y0) = &tri[0];
    let (x1, y1) = &tri[1];
    let (x2, y2) = &tri[2];
    // Compute barycentric coords
    let denom = &(&(y1 - y2) * &(x0 - x2)) + &(&(x2 - x1) * &(y0 - y2));
    if denom == T::zero() {
        // degenerate triangle
        return false;
    }
    let u = &(&(&(y1 - y2) * &(&x - &x2)) + &(&(x2 - x1) * &(&y - &y2))) / &denom;
    let v = &(&(&(y2 - y0) * &(&x - &x2)) + &(&(x0 - x2) * &(&y - &y2))) / &denom;
    u >= T::zero() && v >= T::zero() && (&u + &v) <= T::one()
}

/// If segments [a→b] and [c→d] intersect in 2D, return the intersection point.
/// Otherwise return None.  (Colinear or parallel ⇒ None.)
fn segment_intersect_2d<T>(a: (T, T), b: (T, T), c: (T, T), d: (T, T)) -> Option<(T, T)>
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let (x1, y1) = a;
    let (x2, y2) = b;
    let (x3, y3) = c;
    let (x4, y4) = d;
    // denominator = cross(dir_ab, dir_cd)
    let denom = &(&(&y4 - &y3) * &(&x2 - &x1)) - &(&(&x4 - &x3) * &(&y2 - &y1));
    if denom == T::zero() {
        return None; // parallel or colinear
    }
    // compute parameters t,u
    let t = &(&(&(&x4 - &x3) * &(&y1 - &y3)) - &(&(&y4 - &y3) * &(&x1 - &x3))) / &denom;
    let u = &(&(&(&x2 - &x1) * &(&y1 - &y3)) - &(&(&y2 - &y1) * &(&x1 - &x3))) / &denom;
    // check if intersection lies on both segments
    if t < T::zero() || t > T::one() || u < T::zero() || u > T::one() {
        return None;
    }
    // intersection = a + t*(b-a)
    let ix = &x1 + &(&t * &(&x2 - &x1));
    let iy = &y1 + &(&t * &(&y2 - &y1));
    Some((ix, iy))
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

/// Handle the coplanar case by projecting both triangles into 2D
/// and performing a 2D SAT on the plane.
fn coplanar_tri_tri<T>(
    p0: &Point3<T>,
    p1: &Point3<T>,
    p2: &Point3<T>,
    q0: &Point3<T>,
    q1: &Point3<T>,
    q2: &Point3<T>,
    n: &Vector3<T>,
) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) Choose the projection plane by dropping the largest normal component
    let (i0, i1) = {
        let na = [n[0].abs(), n[1].abs(), n[2].abs()];
        if na[0] > na[1] && na[0] > na[2] {
            (1, 2)
        } else if na[1] > na[2] {
            (0, 2)
        } else {
            (0, 1)
        }
    };

    // 2) Build 2D tuples (x,y) for each triangle
    let to2d = |p: &Point3<T>| (p[i0].clone(), p[i1].clone());
    let t1: [(T, T); 3] = [to2d(p0), to2d(p1), to2d(p2)];
    let t2: [(T, T); 3] = [to2d(q0), to2d(q1), to2d(q2)];

    // 3) A small helper to project a 2D triangle onto an axis
    let project_2d = |axis: (T, T), tri: &[(T, T); 3]| -> (T, T) {
        let (ax, ay) = axis;
        let mut min = &(&tri[0].0 * &ax) + &(&tri[0].1 * &ay);
        let mut max = min.clone();
        for &(ref x, ref y) in &tri[1..] {
            let proj = &(&(x * &ax) + &(y * &ay));
            if proj < &min {
                min = proj.clone();
            }
            if proj > &max {
                max = proj.clone();
            }
        }
        (min, max)
    };

    // 4) SAT on edges of the first triangle
    for &(i, j) in &[(0, 1), (1, 2), (2, 0)] {
        let dx = &t1[j].0 - &t1[i].0;
        let dy = &t1[j].1 - &t1[i].1;
        let axis = (-dy, dx);
        let (min1, max1) = project_2d(axis.clone(), &t1);
        let (min2, max2) = project_2d(axis.clone(), &t2);
        if max1 < min2 || max2 < min1 {
            return false;
        }
    }

    // 5) SAT on edges of the second triangle
    for &(i, j) in &[(0, 1), (1, 2), (2, 0)] {
        let dx = &(t2[j].0) - &(t2[i].0);
        let dy = &(t2[j].1) - &(t2[i].1);
        let axis = (-dy, dx);
        let (min1, max1) = project_2d(axis.clone(), &t1);
        let (min2, max2) = project_2d(axis.clone(), &t2);
        if max1 < min2 || max2 < min1 {
            return false;
        }
    }

    // No separating axis found in 2D ⇒ triangles overlap
    true
}

fn coplanar_tri_tri_intersection<T>(
    p0: &Point3<T>,
    p1: &Point3<T>,
    p2: &Point3<T>,
    q0: &Point3<T>,
    q1: &Point3<T>,
    q2: &Point3<T>,
    n: &Vector3<T>,
) -> Option<(Point3<T>, Point3<T>)>
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
    let t1 = [to2d(p0), to2d(p1), to2d(p2)];
    let t2 = [to2d(q0), to2d(q1), to2d(q2)];

    // 3) collect vertices of one triangle inside the other
    let mut pts: Vec<Point3<T>> = Vec::new();
    for (i, (x, y)) in t1.iter().enumerate() {
        if point_in_tri_2d((x.clone(), y.clone()), &t2) {
            pts.push(back_project_to_3d(&x, &y, i0, i1, drop, &p0));
        }
    }
    for (i, (x, y)) in t2.iter().enumerate() {
        if point_in_tri_2d((x.clone(), y.clone()), &t1) {
            pts.push(back_project_to_3d(&x, &y, i0, i1, drop, &q0));
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
            if let Some((ix, iy)) = segment_intersect_2d(a.clone(), b.clone(), c.clone(), d.clone())
            {
                // Here we use p0 as the reference for the dropped axis, but you could interpolate if desired
                pts.push(back_project_to_3d(&ix, &iy, i0, i1, drop, p0));
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
        Some((uniq[0].clone(), uniq[1].clone()))
    } else if uniq.len() == 1 {
        Some((uniq[0].clone(), uniq[0].clone()))
    } else {
        None
    }
}

/// Computes the segment where triangles T1=(p0,p1,p2) and T2=(q0,q1,q2) intersect.
/// Returns `None` if they don’t intersect, or `Some((a,b))` where `a` and `b` are
/// the two endpoints of the intersection segment (possibly `a==b` if they touch at a point).
pub fn tri_tri_intersection<T>(
    p0: &Point3<T>,
    p1: &Point3<T>,
    p2: &Point3<T>,
    q0: &Point3<T>,
    q1: &Point3<T>,
    q2: &Point3<T>,
) -> Option<(Point3<T>, Point3<T>)>
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) Build plane of T2: n2·x + d2 = 0
    let v01 = (q1 - q0).as_vector();
    let v02 = (q2 - q0).as_vector();
    let n2 = v01.cross(&v02);
    let d2 = -n2.dot(&q0.as_vector().into());

    // signed distances of p-verts to T2’s plane
    let d_p0 = &n2.dot(&p0.as_vector().into()) + &d2;
    let d_p1 = &n2.dot(&p1.as_vector().into()) + &d2;
    let d_p2 = &n2.dot(&p2.as_vector().into()) + &d2;

    //  → Fall back for strictly co-planar
    if d_p0 == T::zero() && d_p1 == T::zero() && d_p2 == T::zero() {
        // Must handle co-planar intersection in 2D
        return coplanar_tri_tri_intersection(p0, p1, p2, q0, q1, q2, &n2);
    }

    // 2) Now do the regular non-coplanar plane‐edge clipping:
    let mut pts = Vec::new();
    for &(a, b) in &[(p0, p1), (p1, p2), (p2, p0)] {
        if let Some(ip) = intersect_edge_plane(a, b, &n2.clone().into(), &d2) {
            if point_in_tri(&ip, q0, q1, q2) {
                pts.push(ip);
            }
        }
    }

    // 3) Build plane of T1:
    let u01 = (p1 - p0).as_vector();
    let u02 = (p2 - p0).as_vector();
    let n1 = u01.cross(&u02);
    let d1 = -n1.dot(&p0.as_vector().into());

    // 4) Clip edges of T2 against T1’s plane:
    for &(a, b) in &[(q0, q1), (q1, q2), (q2, q0)] {
        if let Some(ip) = intersect_edge_plane(a, b, &n1.clone().into(), &d1) {
            if point_in_tri(&ip, p0, p1, p2) {
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

    // 6) Return as before
    match uniq.len() {
        2 => Some((uniq[0].clone(), uniq[1].clone())),
        1 => Some((uniq[0].clone(), uniq[0].clone())),
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
fn project_to_2d<T: Scalar>(p: &Point3<T>, i0: usize, i1: usize) -> (T, T) {
    (p[i0].clone(), p[i1].clone())
}

/// Back-project a 2D point into 3D, using a reference 3D point for the dropped axis.
fn back_project_to_3d<T: Scalar>(
    x: &T,
    y: &T,
    i0: usize,
    i1: usize,
    drop: usize,
    reference: &Point3<T>,
) -> Point3<T>
where
    Point3<T>: SpatialElement<T, 3>,
{
    let mut coords = core::array::from_fn(|_| T::zero());
    coords[i0] = x.clone();
    coords[i1] = y.clone();
    coords[drop] = reference[drop].clone();
    Point3::from_vals(coords)
}
