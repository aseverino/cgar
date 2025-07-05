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

use cgar::geometry::segment::Segment3;
use cgar::geometry::{
    Point2, Point3, Segment2, SegmentIntersection3, segment_segment_intersection_3,
};
use cgar::geometry::{SegmentIntersection2, segment_segment_intersection_2};
use cgar::numeric::cgar_f64::CgarF64;

#[test]
fn test_simple_intersection_2() {
    let s1 = Segment2::<CgarF64>::new(&Point2::new(0.0, 0.0), &Point2::new(2.0, 2.0));
    let s2 = Segment2::new(&Point2::new(0.0, 2.0), &Point2::new(2.0, 0.0));
    let res = segment_segment_intersection_2(&s1, &s2, 1e-9.into());
    assert_eq!(res, SegmentIntersection2::Point(Point2::new(1.0, 1.0)));
}

#[test]
fn test_no_intersection_2() {
    let s1 = Segment2::<CgarF64>::new(&Point2::new(0.0, 0.0), &Point2::new(1.0, 0.0));
    let s2 = Segment2::new(&Point2::new(0.0, 1.0), &Point2::new(1.0, 1.0));
    let res = segment_segment_intersection_2(&s1, &s2, 1e-9.into());
    assert_eq!(res, SegmentIntersection2::None);
}

#[test]
fn test_collinear_overlap_2() {
    let s1 = Segment2::<CgarF64>::new(&Point2::new(0.0, 0.0), &Point2::new(2.0, 0.0));
    let s2 = Segment2::new(&Point2::new(1.0, 0.0), &Point2::new(3.0, 0.0));
    let res = segment_segment_intersection_2(&s1, &s2, 1e-9.into());
    assert_eq!(
        res,
        SegmentIntersection2::Overlapping(Segment2::new(
            &Point2::new(1.0, 0.0),
            &Point2::new(2.0, 0.0),
        ))
    );
}

#[test]
fn test_simple_intersection_3() {
    let s1 = Segment3::<CgarF64>::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(2.0, 2.0, 0.0));
    let s2 = Segment3::new(&Point3::new(0.0, 2.0, 0.0), &Point3::new(2.0, 0.0, 0.0));
    let res = segment_segment_intersection_3(&s1, &s2, 1e-9.into());
    assert_eq!(res, SegmentIntersection3::Point(Point3::new(1.0, 1.0, 0.0)));
}

#[test]
fn test_no_intersection_3() {
    let s1 = Segment3::<CgarF64>::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(1.0, 0.0, 0.0));
    let s2 = Segment3::new(&Point3::new(0.0, 1.0, 1.0), &Point3::new(1.0, 1.0, 1.0));
    let res = segment_segment_intersection_3(&s1, &s2, 1e-9.into());
    assert_eq!(res, SegmentIntersection3::None);
}

#[test]
fn test_collinear_overlap_3() {
    let s1 = Segment3::<CgarF64>::new(&Point3::new(0.0, 0.0, 0.0), &Point3::new(2.0, 0.0, 0.0));
    let s2 = Segment3::new(&Point3::new(1.0, 0.0, 0.0), &Point3::new(3.0, 0.0, 0.0));
    let res = segment_segment_intersection_3(&s1, &s2, 1e-9.into());
    assert_eq!(
        res,
        SegmentIntersection3::Overlapping(Segment3::new(
            &Point3::new(1.0, 0.0, 0.0),
            &Point3::new(2.0, 0.0, 0.0),
        ))
    );
}
