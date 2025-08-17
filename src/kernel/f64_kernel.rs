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

use crate::{
    geometry::{
        Point3,
        point::{Point, PointOps},
        segment::Segment,
        vector::{Vector, VectorOps},
    },
    kernel::{
        kernel::{Kernel2, Kernel3},
        orientation::orient,
        predicates::{are_collinear, are_equal, is_point_on_segment, point_u_on_segment},
    },
    numeric::cgar_f64::CgarF64,
};

pub struct F64Kernel;

impl Kernel2 for F64Kernel {
    type FT = CgarF64;
    type Point2 = Point<CgarF64, 2>;
    type Segment2 = Segment<CgarF64, 2>;
    type Vector2 = Vector<CgarF64, 2>;

    #[inline]
    fn orient2d(a: &Self::Point2, b: &Self::Point2, c: &Self::Point2) -> Self::FT {
        orient::<CgarF64, 2>(&[a.clone(), b.clone(), c.clone()])
    }

    #[inline]
    fn are_equal2(a: &Self::Point2, b: &Self::Point2) -> bool {
        are_equal(a, b)
    }

    #[inline]
    fn are_collinear2(a: &Self::Point2, b: &Self::Point2, c: &Self::Point2) -> bool {
        are_collinear(a, b, c)
    }

    #[inline]
    fn is_point_on_segment2(p: &Self::Point2, s: &Self::Segment2) -> bool {
        is_point_on_segment(p, s)
    }

    #[inline]
    fn point_u_on_segment2(
        a: &Self::Point2,
        b: &Self::Point2,
        p: &Self::Point2,
    ) -> Option<Self::FT> {
        point_u_on_segment(a, b, p)
    }
}

impl Kernel3 for F64Kernel
where
    Point3<CgarF64>: PointOps<CgarF64, 3>,
{
    type FT = CgarF64;
    type Point3 = Point<CgarF64, 3>;
    type Segment3 = Segment<CgarF64, 3>;
    type Vector3 = Vector<CgarF64, 3>;

    #[inline]
    fn orient3d(
        a: &Self::Point3,
        b: &Self::Point3,
        c: &Self::Point3,
        d: &Self::Point3,
    ) -> Self::FT {
        let ab = (&b.clone() - a).as_vector();
        let ac = (&c.clone() - a).as_vector();
        let ad = (&d.clone() - a).as_vector();
        ab.cross(&ac).dot(&ad)
    }

    #[inline]
    fn are_equal3(a: &Self::Point3, b: &Self::Point3) -> bool {
        are_equal(a, b)
    }

    #[inline]
    fn are_collinear3(a: &Self::Point3, b: &Self::Point3, c: &Self::Point3) -> bool {
        are_collinear(a, b, c)
    }

    #[inline]
    fn is_point_on_segment3(p: &Self::Point3, s: &Self::Segment3) -> bool {
        is_point_on_segment(p, s)
    }

    #[inline]
    fn point_u_on_segment3(
        a: &Self::Point3,
        b: &Self::Point3,
        p: &Self::Point3,
    ) -> Option<Self::FT> {
        point_u_on_segment(a, b, p)
    }
}
