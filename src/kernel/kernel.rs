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

// ...existing code...
use crate::{
    geometry::{
        point::Point,
        segment::Segment,
        vector::{Vector, VectorOps},
    },
    numeric::scalar::Scalar,
    operations::{Abs, Pow, Sqrt},
};

pub trait Kernel2 {
    type FT: Scalar;
    type Point2;
    type Segment2;
    type Vector2;

    // Sign of oriented area (CCW>0, CW<0, 0 if collinear)
    fn orient2d(a: &Self::Point2, b: &Self::Point2, c: &Self::Point2) -> Self::FT;

    // Convenience predicates
    fn are_equal2(a: &Self::Point2, b: &Self::Point2) -> bool;
    fn are_collinear2(a: &Self::Point2, b: &Self::Point2, c: &Self::Point2) -> bool;
    fn is_point_on_segment2(p: &Self::Point2, s: &Self::Segment2) -> bool;
    fn point_u_on_segment2(
        a: &Self::Point2,
        b: &Self::Point2,
        p: &Self::Point2,
    ) -> Option<Self::FT>;
}

pub trait Kernel3 {
    type FT: Scalar;
    type Point3;
    type Segment3;
    type Vector3;

    // Signed volume of tetra (a,b,c,d): ((b-a) x (c-a)) · (d-a)
    fn orient3d(a: &Self::Point3, b: &Self::Point3, c: &Self::Point3, d: &Self::Point3)
    -> Self::FT;

    // Convenience predicates
    fn are_equal3(a: &Self::Point3, b: &Self::Point3) -> bool;
    fn are_collinear3(a: &Self::Point3, b: &Self::Point3, c: &Self::Point3) -> bool;
    fn is_point_on_segment3(p: &Self::Point3, s: &Self::Segment3) -> bool;
    fn point_u_on_segment3(
        a: &Self::Point3,
        b: &Self::Point3,
        p: &Self::Point3,
    ) -> Option<Self::FT>;
}
