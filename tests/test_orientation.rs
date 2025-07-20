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

use cgar::geometry::Point2;
use cgar::geometry::Point3;
use cgar::geometry::spatial_element::SpatialElement;
use cgar::geometry::util::EPS;
use cgar::kernel::orient;
use cgar::numeric::cgar_f64::CgarF64;
use cgar::operations::Abs;

#[test]
fn ccw_test() {
    let a = Point2::<CgarF64>::from_vals([0.0, 0.0]);
    let b = Point2::from_vals([1.0, 0.0]);
    let c = Point2::from_vals([0.0, 1.0]);

    assert!(orient(&[a, b, c]) > 0.0.into()); // Counter-clockwise
}

#[test]
fn orientation_3d_positive_volume() {
    let a = Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]);
    let b = Point3::from_vals([1.0, 0.0, 0.0]);
    let c = Point3::from_vals([0.0, 1.0, 0.0]);
    let d = Point3::from_vals([0.0, 0.0, 1.0]); // above the abc plane

    let vol = orient(&[a, b, c, d]);
    assert!(vol > 0.0.into());
}

#[test]
fn orientation_3d_negative_volume() {
    let a = Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]);
    let b = Point3::from_vals([1.0, 0.0, 0.0]);
    let c = Point3::from_vals([0.0, 1.0, 0.0]);
    let d = Point3::from_vals([0.0, 0.0, -1.0]); // below the abc plane

    let vol = orient(&[a, b, c, d]);
    assert!(vol < 0.0.into());
}

#[test]
fn orientation_3d_coplanar() {
    let a = Point3::<CgarF64>::from_vals([0.0, 0.0, 0.0]);
    let b = Point3::from_vals([1.0, 0.0, 0.0]);
    let c = Point3::from_vals([0.0, 1.0, 0.0]);
    let d = Point3::from_vals([1.0, 1.0, 0.0]); // lies in the same z=0 plane

    let vol = orient(&[a, b, c, d]);
    assert!(vol.abs() < EPS.into()); // small epsilon to account for floating point
}
