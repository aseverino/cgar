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

use cgar::{
    geometry::{Point3, spatial_element::SpatialElement, tri_tri_intersect::tri_tri_intersection},
    numeric::cgar_f64::CgarF64,
};

fn _sort_pair<T: PartialOrd + Clone>(mut a: T, mut b: T) -> (T, T) {
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    (a, b)
}

#[test]
fn test_coplanar_overlap()
//where
//    Point3<f64>: From<Vector3<f64>>,
{
    // Two right triangles in the z=0 plane sharing the diagonal from (0,1) to (1,0)
    let t1 = [
        Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]),
        Point3::from_vals([1.0, 0.0, 0.0]),
        Point3::from_vals([0.0, 1.0, 0.0]),
    ];
    let t2 = [
        Point3::from_vals([0.0, 1.0, 0.0]),
        Point3::from_vals([1.0, 0.0, 0.0]),
        Point3::from_vals([1.0, 1.0, 0.0]),
    ];

    let seg = tri_tri_intersection(&t1, &t2).expect("triangles should intersect");

    // We expect the endpoints (0,1,0) and (1,0,0), in either order.
    let (a, b) = (
        (seg.a[0].clone(), seg.a[1].clone(), seg.a[2].clone()),
        (seg.b[0].clone(), seg.b[1].clone(), seg.b[2].clone()),
    );
    let wanted = [
        (CgarF64(0.0), CgarF64(1.0), CgarF64(0.0)),
        (1.0.into(), 0.0.into(), 0.0.into()),
    ];
    let mut actual = [a, b];
    actual.sort_by(|p, q| p.partial_cmp(q).unwrap_or(Ordering::Equal));
    let mut expected = wanted;
    expected.sort_by(|p, q| p.partial_cmp(q).unwrap_or(Ordering::Equal));
    assert_eq!(actual, expected);
}

#[test]
fn test_coplanar_disjoint() {
    // Two triangles in the z=0 plane that do not touch
    let t1 = [
        Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]),
        Point3::from_vals([1.0, 0.0, 0.0]),
        Point3::from_vals([0.0, 1.0, 0.0]),
    ];
    let t2 = [
        Point3::from_vals([2.0, 2.0, 0.0]),
        Point3::from_vals([3.0, 2.0, 0.0]),
        Point3::from_vals([2.0, 3.0, 0.0]),
    ];

    let seg = tri_tri_intersection(&t1, &t2);
    assert!(
        seg.is_none(),
        "disjoint coplanar triangles should not intersect"
    );
}

#[test]
fn test_non_coplanar_slice_f64() {
    // A horizontal triangle in z=0
    let t1 = [
        Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]),
        Point3::from_vals([1.0, 0.0, 0.0]),
        Point3::from_vals([0.0, 1.0, 0.0]),
    ];
    // A vertical triangle that slices through t1, intersecting along a segment
    let t2 = [
        Point3::from_vals([0.2, 0.2, -1.0]),
        Point3::from_vals([0.2, 0.2, 1.0]),
        Point3::from_vals([0.8, 0.8, 1.0]),
    ];

    let seg = tri_tri_intersection(&t1, &t2).expect("should slice and intersect");

    // The two intersection points should lie on z=0 at (0.2,0.2) and (0.5,0.5)
    let (a2, b2) = (
        (seg.a[0].clone(), seg.a[1].clone()),
        (seg.b[0].clone(), seg.b[1].clone()),
    );
    let pts = [a2, b2];
    println!("Intersection points: {:?}", pts);
    assert!(pts.contains(&(0.2.into(), 0.2.into())));
    assert!(pts.contains(&(0.5.into(), 0.5.into())));
}
