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

use super::{face::Face, half_edge::HalfEdge, point_trait::PointTrait, vertex::Vertex};

#[derive(Debug)]
pub struct Mesh<T, P: PointTrait<T>> {
    pub vertices: Vec<Vertex<T, P>>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<Face>,
}

impl<T, P: PointTrait<T>> Mesh<T, P> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            half_edges: Vec::new(),
            faces: Vec::new(),
        }
    }

    pub fn add_vertex(&mut self, position: P) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(Vertex::new(position));
        idx
    }

    /// Adds a triangle face given three vertex indices (in CCW order).
    /// Note: this is a naive non-twin-connected insertion for now.
    pub fn add_triangle(&mut self, v0: usize, v1: usize, v2: usize) -> usize {
        let he0 = self.half_edges.len();
        let he1 = he0 + 1;
        let he2 = he0 + 2;
        let face_idx = self.faces.len();

        self.half_edges.push(HalfEdge {
            vertex: v1,
            face: Some(face_idx),
            next: he1,
            prev: he2,
            twin: usize::MAX,
        });

        self.half_edges.push(HalfEdge {
            vertex: v2,
            face: Some(face_idx),
            next: he2,
            prev: he0,
            twin: usize::MAX,
        });

        self.half_edges.push(HalfEdge {
            vertex: v0,
            face: Some(face_idx),
            next: he0,
            prev: he1,
            twin: usize::MAX,
        });

        self.vertices[v0].half_edge.get_or_insert(he0);
        self.vertices[v1].half_edge.get_or_insert(he1);
        self.vertices[v2].half_edge.get_or_insert(he2);

        self.faces.push(Face::new(he0));

        face_idx
    }
}
