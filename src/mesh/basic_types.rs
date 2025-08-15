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

/// CCW one-ring snapshot around a vertex.
#[derive(Debug, Clone)]
pub struct VertexRing {
    pub center: usize,
    pub halfedges_ccw: Vec<usize>, // outgoing half-edges from `center`
    pub neighbors_ccw: Vec<usize>, // target(vertex) of each half-edge
    pub faces_ccw: Vec<Option<usize>>, // incident face for each wedge
    pub is_border: bool,
}

/// CCW one-ring snapshots around the two endpoints of an edge (v0,v1),
/// plus convenient derived info for edge-collapse checks.
#[derive(Debug, Clone)]
pub struct PairRing {
    pub v0: usize,
    pub v1: usize,
    pub ring0: VertexRing, // CCW ring around v0
    pub ring1: VertexRing, // CCW ring around v1

    /// index in ring0.neighbors_ccw where neighbor == v1 (if edge exists)
    pub idx_v1_in_ring0: Option<usize>,
    /// index in ring1.neighbors_ccw where neighbor == v0 (if edge exists)
    pub idx_v0_in_ring1: Option<usize>,

    /// The two “opposite” vertices across the edge (v0,v1):
    /// a = third vertex of face(v0->v1), b = third vertex of face(v1->v0)
    /// None for borders/removed faces.
    pub opposite_a: Option<usize>,
    pub opposite_b: Option<usize>,

    /// Neighbor sets (excluding {v0,v1})
    pub common_neighbors: std::collections::HashSet<usize>,
    pub union_neighbors: std::collections::HashSet<usize>,

    /// True if the edge is on the border (i.e., one of the incident faces is None/removed).
    pub is_border_edge: bool,
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
