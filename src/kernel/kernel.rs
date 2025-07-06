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

use crate::operations::{Abs, Pow, Sqrt};

pub trait Kernel2 {
    type FT: Clone + PartialOrd + Abs + Pow + Sqrt;
    type Point2;
    type Segment2;
    type Vector2;

    // fn orient2d(a: &Self::Point2, b: &Self::Point2, c: &Self::Point2) -> Self::FT;
    // fn are_equal(a: &Self::Point2, b: &Self::Point2, eps: Self::FT) -> bool;
    // fn are_collinear(a: &Self::Point2, b: &Self::Point2, c: &Self::Point2, eps: Self::FT) -> bool;
    // fn is_point_on_segment(p: &Self::Point2, s: &Self::Segment2, eps: Self::FT) -> bool;
}

pub trait Kernel3 {
    type FT: Clone + PartialOrd + Abs + Pow + Sqrt;
    type Point3;
    type Segment3;
    type Vector3;

    // fn orient3d(a: &Self::Point3, b: &Self::Point3, c: &Self::Point3) -> Self::FT;
    // fn are_equal(a: &Self::Point3, b: &Self::Point3, eps: Self::FT) -> bool;
    // fn are_collinear(a: &Self::Point3, b: &Self::Point3, c: &Self::Point3, eps: Self::FT) -> bool;
    // fn is_point_on_segment(p: &Self::Point3, s: &Self::Segment3, eps: Self::FT) -> bool;
}
