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

use cgar::geometry::{Point2, Segment2};
use cgar::kernel::{BigRationalKernel, F64Kernel, Kernel};
use cgar::numeric::cgar_rational::CgarRational;
use rug::Rational;

const EPS: f64 = 1e-9;

#[test]
fn test_orient2d() {
    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(1.0, 0.0);
    let c = Point2::new(0.0, 1.0);

    let res = F64Kernel::orient2d(&a, &b, &c);
    assert!(res > 0.0);
}

#[test]
fn test_are_equal() {
    let a = Point2::new(1.0, 2.0);
    let b = Point2::new(1.0 + EPS / 2.0, 2.0);
    let c = Point2::new(1.0 + EPS * 10.0, 2.0);

    assert!(F64Kernel::are_equal(&a, &b, EPS));
    assert!(!F64Kernel::are_equal(&a, &c, EPS));
}

#[test]
fn test_are_collinear() {
    let a = Point2::new(0.0, 0.0);
    let b = Point2::new(1.0, 1.0);
    let c = Point2::new(2.0, 2.0);

    assert!(F64Kernel::are_collinear(&a, &b, &c, EPS));

    let d = Point2::new(2.0, 2.000001);
    assert!(!F64Kernel::are_collinear(&a, &b, &d, EPS));
}

#[test]
fn test_is_point_on_segment() {
    let seg = Segment2::new(&Point2::new(0.0, 0.0), &Point2::new(2.0, 2.0));
    let on = Point2::new(1.0, 1.0);
    let off = Point2::new(3.0, 3.0);

    assert!(F64Kernel::is_point_on_segment(&on, &seg, EPS));
    assert!(!F64Kernel::is_point_on_segment(&off, &seg, EPS));
}

fn eps() -> CgarRational {
    CgarRational(Rational::from((1, 1_000_000_000))) // 1e-9
}

#[test]
fn test_bigrational_orient2d() {
    let a = Point2::new(
        CgarRational(Rational::from(0)),
        CgarRational(Rational::from(0)),
    );
    let b = Point2::new(
        CgarRational(Rational::from(1)),
        CgarRational(Rational::from(0)),
    );
    let c = Point2::new(
        CgarRational(Rational::from(0)),
        CgarRational(Rational::from(1)),
    );

    let res = BigRationalKernel::orient2d(&a, &b, &c);
    assert!(res.0 > 0);
}

// #[test]
// fn test_bigrational_equal_and_collinear() {
//     let a = Point2::new(1.0, 1.0);
//     let b = Point2::new(1.0000000001, 1.0);
//     let eps = eps();

//     assert!(BigRationalKernel::are_equal(&a, &b, eps.clone()));
//     assert!(BigRationalKernel::are_collinear(
//         &Point2::new(0.0, 0.0),
//         &Point2::new(1.0, 1.0),
//         &Point2::new(2.0, 2.0),
//         eps.clone()
//     ));
// }

// #[test]
// fn test_bigrational_on_segment() {
//     let seg = Segment2::new(&Point2::new(0.0, 0.0), &Point2::new(2.0, 2.0));
//     let p_on = Point2::new(1.0, 1.0);
//     let p_off = Point2::new(3.0, 3.0);

//     assert!(BigRationalKernel::is_point_on_segment(&p_on, &seg, eps()));
//     assert!(!BigRationalKernel::is_point_on_segment(&p_off, &seg, eps()));
// }
