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

use std::ops::{Index, IndexMut};

use smallvec::SmallVec;

use crate::{geometry::segment::Segment, numeric::scalar::Scalar};

#[derive(Debug, Clone)]
pub struct IntersectionEndPoint<T: Scalar, const N: usize> {
    pub vertex_hint: Option<[usize; 2]>,
    pub half_edge_hint: Option<usize>,
    pub half_edge_u_hint: Option<T>,
    pub faces_hint: Option<[usize; 2]>,
    pub barycentric_hint: Option<(T, T, T)>,
    pub resulting_vertex: Option<usize>,
}

impl<T: Scalar, const N: usize> IntersectionEndPoint<T, N> {
    pub fn new(
        // point: Point<T, N>,
        vertex_hint: Option<[usize; 2]>,
        half_edge_hint: Option<usize>,
        half_edge_u_hint: Option<T>,
        faces_hint: Option<[usize; 2]>,
        barycentric_hint: Option<(T, T, T)>,
        resulting_vertex: Option<usize>,
    ) -> Self {
        Self {
            // point,
            vertex_hint,
            half_edge_hint,
            half_edge_u_hint,
            faces_hint: faces_hint,
            barycentric_hint,
            resulting_vertex,
        }
    }
    pub fn new_default() -> Self {
        Self {
            // point: Point::default(),
            vertex_hint: None,
            half_edge_hint: None,
            half_edge_u_hint: None,
            faces_hint: None,
            barycentric_hint: None,
            resulting_vertex: None,
        }
    }
}

impl<T: Scalar, const N: usize> Default for IntersectionEndPoint<T, N> {
    fn default() -> Self {
        Self::new_default()
    }
}

#[derive(Debug, Clone)]
pub struct IntersectionSegment<T: Scalar, const N: usize> {
    pub a: IntersectionEndPoint<T, N>,
    pub b: IntersectionEndPoint<T, N>,
    pub segment: Segment<T, N>,
    pub initial_face_reference: usize,
    pub links: SmallVec<[usize; 2]>,
    pub coplanar: bool,
    pub invalidated: bool,
    pub split: bool,
}

impl<T: Scalar, const N: usize> Index<usize> for IntersectionSegment<T, N> {
    type Output = IntersectionEndPoint<T, N>;
    fn index(&self, i: usize) -> &Self::Output {
        if i == 0 {
            &self.a
        } else if i == 1 {
            &self.b
        } else {
            panic!("Index out of bounds for Segment: {}", i);
        }
    }
}
impl<T: Scalar, const N: usize> IndexMut<usize> for IntersectionSegment<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        if i == 0 {
            &mut self.a
        } else if i == 1 {
            &mut self.b
        } else {
            panic!("Index out of bounds for Segment: {}", i);
        }
    }
}

impl<T: Scalar, const N: usize> IntersectionSegment<T, N> {
    pub fn new(
        a: IntersectionEndPoint<T, N>,
        b: IntersectionEndPoint<T, N>,
        segment: &Segment<T, N>,
        initial_face_reference: usize,
        coplanar: bool,
    ) -> Self {
        Self {
            a: a,
            b: b,
            segment: segment.clone(),
            initial_face_reference,
            links: SmallVec::new(),
            coplanar,
            invalidated: false,
            split: true,
        }
    }
    pub fn new_default(
        a: IntersectionEndPoint<T, N>,
        b: IntersectionEndPoint<T, N>,
        segment: &Segment<T, N>,
        initial_face_reference: usize,
    ) -> Self {
        Self::new(a, b, segment, initial_face_reference, false)
    }
}

impl<T: Scalar, const N: usize> Default for IntersectionSegment<T, N> {
    fn default() -> Self {
        Self::new_default(
            IntersectionEndPoint::default(),
            IntersectionEndPoint::default(),
            &Segment::default(),
            usize::MAX,
        )
    }
}
