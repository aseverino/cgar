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

use cgar::{
    geometry::{
        Point2, Point3, Segment2, Vector2, Vector3,
        point::PointOps,
        segment::{Segment3, SegmentOps},
        spatial_element::SpatialElement,
        vector::{Cross2, Cross3, VectorOps},
    },
    numeric::cgar_f64::CgarF64,
};

// #[test]
// fn test_distance_2() {
//     let p1 = Point2::<CgarF64>::from_vals([0.0, 0.0]);
//     let p2 = Point2::from_vals([3.0, 4.0]);
//     assert_eq!(p1.distance_to(&p2), 5.0.into());
// }

#[test]
fn test_vector_cross_2() {
    let v1 = Vector2::<CgarF64>::from_vals([1.0, 0.0]);
    let v2 = Vector2::from_vals([0.0, 1.0]);
    assert_eq!(v1.0.as_vector_2().cross(&v2.0.as_vector_2()), 1.0.into());
}

// #[test]
// fn test_segment_length_2() {
//     let s = Segment2::new(
//         &Point2::<CgarF64>::from_vals([0.0, 0.0]),
//         &Point2::from_vals([0.0, 5.0]),
//     );
//     assert_eq!(s.length(), 5.0.into());
// }

// #[test]
// fn test_distance_3() {
//     let p1 = Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]);
//     let p2 = Point3::<CgarF64>::from_vals([1.0, 2.0, 2.0]);
//     // √(1² + 2² + 2²) = √9 = 3
//     assert_eq!(p1.distance_to(&p2), 3.0.into());
// }

#[test]
fn test_vector_cross_3()
where
    Vector3<CgarF64>: Cross3<CgarF64>,
{
    let v1 = Vector3::<CgarF64>::from_vals([1.0, 0.0, 0.0]);
    let v2 = Vector3::<CgarF64>::from_vals([0.0, 1.0, 0.0]);
    // i × j = k
    assert_eq!(
        v1.0.as_vector_3().cross(&v2.0.as_vector_3()),
        Vector3::from_vals([0.0, 0.0, 1.0])
    );
}

// #[test]
// fn test_segment_length_3() {
//     let a = Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]);
//     let b = Point3::<CgarF64>::from_vals([0.0, 3.0, 4.0]);
//     let s = Segment3::<CgarF64>::new(&a, &b);
//     // length is √(0² + 3² + 4²) = 5
//     assert_eq!(s.length(), 5.0.into());
// }
