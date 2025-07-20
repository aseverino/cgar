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
use cgar::geometry::spatial_element::SpatialElement;
use cgar::geometry::util::EPS;
use cgar::geometry::{Point2, Point3, Segment2};
use cgar::kernel::{are_collinear, are_equal, is_point_on_segment, orient};
use cgar::numeric::cgar_f64::CgarF64;
use cgar::numeric::cgar_rational::CgarRational;
use rug::Rational;

#[test]
fn test_orient2d() {
    let a = Point2::<CgarF64>::from_vals([0.0, 0.0]);
    let b = Point2::<CgarF64>::from_vals([1.0, 0.0]);
    let c = Point2::<CgarF64>::from_vals([0.0, 1.0]);

    let res = orient(&[a, b, c]);
    assert!(res > CgarF64(0.0));
}

#[test]
fn test_are_equal_2() {
    let a = Point2::<CgarF64>::from_vals([1.0, 2.0]);
    let b = Point2::<CgarF64>::from_vals([1.0 + EPS / 2.0, 2.0]);
    let c = Point2::<CgarF64>::from_vals([1.0 + EPS * 10.0, 2.0]);

    assert!(are_equal(&a, &b));
    assert!(!are_equal(&a, &c));
}

#[test]
fn test_are_collinear_2() {
    let a = Point2::<CgarF64>::from_vals([0.0, 0.0]);
    let b = Point2::<CgarF64>::from_vals([1.0, 1.0]);
    let c = Point2::<CgarF64>::from_vals([2.0, 2.0]);

    assert!(are_collinear(&a, &b, &c));

    let d = Point2::from_vals([2.0, 2.000001]);
    assert!(!are_collinear(&a, &b, &d));
}

#[test]
fn test_is_point_on_segment_2() {
    let seg = Segment2::new(
        &Point2::<CgarF64>::from_vals([0.0, 0.0]),
        &Point2::from_vals([2.0, 2.0]),
    );
    let on = Point2::from_vals([1.0, 1.0]);
    let off = Point2::from_vals([3.0, 3.0]);

    assert!(is_point_on_segment(&on, &seg));
    assert!(!is_point_on_segment(&off, &seg));
}

fn _eps() -> CgarRational {
    CgarRational(Rational::from((1, 1_000_000_000))) // 1e-9
}

#[test]
fn test_bigrational_orient2d() {
    let a = Point2::<CgarRational>::from_vals([0, 0]);
    let b = Point2::<CgarRational>::from_vals([1, 0]);
    let c = Point2::<CgarRational>::from_vals([0, 1]);

    let res = orient(&[a, b, c]);
    assert!(res.0 > 0);
}

#[test]
fn test_orient3d() {
    let a = Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]);
    let b = Point3::from_vals([1.0, 0.0, 0.0]);
    let c = Point3::from_vals([0.0, 1.0, 0.0]);
    let d = Point3::from_vals([0.0, 0.0, 1.0]);

    let res = orient(&[a, b, c, d]);
    assert!(res.0 > 0.0);
}

#[test]
fn test_are_equal_3() {
    let p1 = Point3::<CgarF64>::from_vals([1.0, 2.0, 3.0]);
    let p2 = Point3::<CgarF64>::from_vals([1.0 + EPS / 2.0, 2.0, 3.0]);
    let p3 = Point3::<CgarF64>::from_vals([1.0 + EPS * 10.0, 2.0, 3.0]);

    assert!(are_equal(&p1, &p2));
    assert!(!are_equal(&p1, &p3));
}

#[test]
fn test_are_collinear_3() {
    let a = Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]);
    let b = Point3::from_vals([1.0, 1.0, 1.0]);
    let c = Point3::from_vals([2.0, 2.0, 2.0]);
    assert!(are_collinear(&a, &b, &c));

    let d = Point3::from_vals([2.0, 2.0, 2.000001]);
    assert!(!are_collinear(&a, &b, &d));
}

#[test]
fn test_is_point_on_segment_3() {
    let seg = Segment3::<CgarF64>::new(
        &Point3::from_vals([0.0, 0.0, 0.0]),
        &Point3::from_vals([2.0, 2.0, 2.0]),
    );
    let on = Point3::from_vals([1.0, 1.0, 1.0]);
    let off = Point3::from_vals([3.0, 3.0, 3.0]);

    assert!(is_point_on_segment(&on, &seg));
    assert!(!is_point_on_segment(&off, &seg));
}

fn _eps_rational() -> CgarRational {
    CgarRational(Rational::from((1, 1_000_000_000))) // 1e-9
}

#[test]
fn test_bigrational_orient3d() {
    let a = Point3::<CgarRational>::from_vals([0, 0, 0]);
    let b = Point3::from_vals([1, 0, 0]);
    let c = Point3::from_vals([0, 1, 0]);
    let d = Point3::from_vals([0, 0, 1]);

    let res: CgarRational = orient(&[a, b, c, d]);
    // expect positive rational
    assert!(res.0 > Rational::from((0, 1)));
}
