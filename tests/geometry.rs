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

use cgar::geometry::{Point2, Segment2, Vector2};

#[test]
fn test_distance() {
    let p1 = Point2::new(0.0, 0.0);
    let p2 = Point2::new(3.0, 4.0);
    assert_eq!(p1.distance_to(&p2), 5.0);
}

#[test]
fn test_vector_cross() {
    let v1 = Vector2::new(1.0, 0.0);
    let v2 = Vector2::new(0.0, 1.0);
    assert_eq!(v1.cross(v2), 1.0);
}

#[test]
fn test_segment_length() {
    let s = Segment2::new(&Point2::new(0.0, 0.0), &Point2::new(0.0, 5.0));
    assert_eq!(s.length(), 5.0);
}
