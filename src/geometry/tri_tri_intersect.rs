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

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{
    geometry::{
        point::{Point3, PointOps},
        vector::{Vector3, VectorOps},
    },
    mesh::point_trait::PointTrait,
    operations::{Abs, Pow, Sqrt, Zero},
};
use num_traits::Float;

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
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero + Neg<Output = T>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) plane test for triangle T1
    let e1 = p1.sub(p0).as_vector();
    let e2 = p2.sub(p0).as_vector();
    let n1 = e1.cross(&e2);

    let d0 = n1.dot(&q0.sub(p0).as_vector());
    let d1 = n1.dot(&q1.sub(p0).as_vector());
    let d2 = n1.dot(&q2.sub(p0).as_vector());

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
    let f1 = q1.sub(q0).as_vector();
    let f2 = q2.sub(q0).as_vector();
    let n2 = f1.cross(&f2);

    let e0 = n2.dot(&p0.sub(q0).as_vector());
    let e1_ = n2.dot(&p1.sub(q0).as_vector());
    let e2_ = n2.dot(&p2.sub(q0).as_vector());

    if (e0 > T::zero() && e1_ > T::zero() && e2_ > T::zero())
        || (e0 < T::zero() && e1_ < T::zero() && e2_ < T::zero())
    {
        return false;
    }

    // 3) 9 cross‐edge SAT tests
    let tri_axes = [
        e1.cross(&f1),
        e1.cross(&f2),
        e1.cross(&q0.sub(q2).as_vector()),
        e2.cross(&f1),
        e2.cross(&f2),
        e2.cross(&q0.sub(q2).as_vector()),
        p0.sub(p2).as_vector().cross(&f1),
        p0.sub(p2).as_vector().cross(&f2),
        p0.sub(p2).as_vector().cross(&q0.sub(q2).as_vector()),
    ];

    for axis in &tri_axes {
        if *axis == Vector3::<T>::zero() {
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

/// Project a 3D triangle onto `axis`, returning (min,max).
fn project_3d_triangle<T>(axis: &Vector3<T>, a: &Point3<T>, b: &Point3<T>, c: &Point3<T>) -> (T, T)
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero + Neg<Output = T>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let p = |p: &Point3<T>| {
        let v = p.as_vector();
        v.dot(axis)
    };
    let p0 = p(a);
    let p1 = p(b);
    let p2 = p(c);

    let mut min = p0.clone();
    let mut max = p0.clone();
    if p1 < min {
        min = p1.clone()
    }
    if p1 > max {
        max = p1.clone()
    }
    if p2 < min {
        min = p2.clone()
    }
    if p2 > max {
        max = p2.clone()
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
    T: Clone + PartialOrd + Abs + Pow + Sqrt + Zero + Neg<Output = T>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) Choose the projection plane by dropping the largest normal component
    let (i0, i1) = {
        let na = [n.x.clone().abs(), n.y.clone().abs(), n.z.clone().abs()];
        if na[0] > na[1] && na[0] > na[2] {
            (1, 2)
        } else if na[1] > na[2] {
            (0, 2)
        } else {
            (0, 1)
        }
    };

    // 2) Build 2D tuples (x,y) for each triangle
    let to2d = |p: &Point3<T>| (p.coord(i0).clone(), p.coord(i1).clone());
    let t1: [(T, T); 3] = [to2d(p0), to2d(p1), to2d(p2)];
    let t2: [(T, T); 3] = [to2d(q0), to2d(q1), to2d(q2)];

    // 3) A small helper to project a 2D triangle onto an axis
    let project_2d = |axis: (T, T), tri: &[(T, T); 3]| -> (T, T) {
        let (ax, ay) = axis;
        let mut min = &(&tri[0].0 * &ax) + &(&tri[0].1 * &ay);
        let mut max = min.clone();
        for &(ref x, ref y) in &tri[1..] {
            let proj = &(x * &ax) + &(y * &ay);
            if proj < min {
                min = proj.clone()
            }
            if proj > max {
                max = proj.clone()
            }
        }
        (min, max)
    };

    // 4) SAT on edges of the first triangle
    for &(i, j) in &[(0, 1), (1, 2), (2, 0)] {
        let dx = &t1[j].0 - &t1[i].0;
        let dy = &t1[j].1 - &t1[i].1;
        let axis = (-dy.clone(), dx.clone());
        let (min1, max1) = project_2d(axis.clone(), &t1);
        let (min2, max2) = project_2d(axis, &t2);
        if max1 < min2 || max2 < min1 {
            return false;
        }
    }

    // 5) SAT on edges of the second triangle
    for &(i, j) in &[(0, 1), (1, 2), (2, 0)] {
        let dx = &(t2[j].0) - &(t2[i].0);
        let dy = &(t2[j].1) - &(t2[i].1);
        let axis = (-dy.clone(), dx.clone());
        let (min1, max1) = project_2d(axis.clone(), &t1);
        let (min2, max2) = project_2d(axis, &t2);
        if max1 < min2 || max2 < min1 {
            return false;
        }
    }

    // No separating axis found in 2D ⇒ triangles overlap
    true
}
