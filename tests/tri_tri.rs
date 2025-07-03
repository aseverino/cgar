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

use cgar::geometry::point::Point3;
use cgar::geometry::tri_tri_intersect::tri_tri_overlap;

#[test]
fn test_triangles_overlap() {
    let t1 = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
    ];
    let t2 = [
        Point3::new(0.1, 0.1, 0.0),
        Point3::new(0.9, 0.1, 0.0),
        Point3::new(0.1, 0.9, 0.0),
    ];
    assert!(tri_tri_overlap(
        &t1[0], &t1[1], &t1[2], &t2[0], &t2[1], &t2[2]
    ));

    let t3 = [
        Point3::new(2.0, 2.0, 0.0),
        Point3::new(3.0, 2.0, 0.0),
        Point3::new(2.0, 3.0, 0.0),
    ];
    assert!(!tri_tri_overlap(
        &t1[0], &t1[1], &t1[2], &t3[0], &t3[1], &t3[2]
    ));
}
