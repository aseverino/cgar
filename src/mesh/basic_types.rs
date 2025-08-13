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

use std::collections::HashMap;

use smallvec::SmallVec;

use crate::{
    mesh::{face::Face, half_edge::HalfEdge, vertex::Vertex},
    numeric::scalar::Scalar,
};

#[derive(Debug, Clone, Default)]
pub struct Triangle {
    pub face_idx: usize,
    pub vertices: [usize; 3],
}

#[derive(Debug, Clone, Default)]
pub struct FaceSplitMap {
    pub face: usize,
    pub new_faces: SmallVec<[Triangle; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SplitResultKind {
    NoSplit,
    SplitFace,
    SplitEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointInMeshResult {
    Outside,
    OnSurface,
    Inside,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RayCastResult {
    Outside,
    OnSurface,
    Inside,
}

#[derive(Debug, Clone)]
pub struct SplitResult {
    pub kind: SplitResultKind,
    pub vertex: usize,
    pub new_faces: [usize; 4], // up to 4 new faces can be created
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexSource {
    A,
    B,
}

#[derive(Debug, Clone)]
pub struct Mesh<T: Scalar, const N: usize> {
    pub vertices: Vec<Vertex<T, N>>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<Face>,

    pub edge_map: HashMap<(usize, usize), usize>,
    pub(crate) vertex_spatial_hash: HashMap<(i64, i64, i64), Vec<usize>>,
    pub(crate) face_split_map: HashMap<usize, FaceSplitMap>,
    pub half_edge_split_map: HashMap<usize, (usize, usize)>,
}
