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

use cgar::geometry::spatial_element::SpatialElement;
use cgar::geometry::{Point2, Segment2};
use cgar::kernel::{are_collinear, are_equal, is_point_on_segment};
use cgar::numeric::cgar_f64::CgarF64;

#[test]
fn test_are_equal() {
    let p1 = Point2::<CgarF64>::from_vals([1.00000000001, 2.0]);
    let p2 = Point2::from_vals([1.00000000002, 2.0]);
    assert!(are_equal(&p1, &p2));
}

#[test]
fn test_are_collinear() {
    let a = Point2::<CgarF64>::from_vals([0.0, 0.0]);
    let b = Point2::from_vals([1.0, 1.0]);
    let c = Point2::from_vals([2.0, 2.0]);
    assert!(are_collinear(&a, &b, &c));
}

#[test]
fn test_point_on_segment() {
    let seg = Segment2::<CgarF64>::new(
        &Point2::from_vals([0.0, 0.0]),
        &Point2::from_vals([2.0, 2.0]),
    );
    let p_on = Point2::from_vals([1.0, 1.0]);
    let p_off = Point2::from_vals([3.0, 3.0]);

    assert!(is_point_on_segment(&p_on, &seg));
    assert!(!is_point_on_segment(&p_off, &seg));
}
