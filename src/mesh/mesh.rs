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
        Aabb, AabbTree,
        plane::Plane,
        point::{Point, PointOps},
        segment::{Segment, SegmentOps},
        spatial_element::SpatialElement,
        vector::{Vector, VectorOps},
    },
    mesh::face,
    numeric::scalar::Scalar,
    operations::Zero,
};

use super::{face::Face, half_edge::HalfEdge, vertex::Vertex};
use core::panic;
use smallvec::*;
use std::{
    array::from_fn,
    collections::{HashMap, HashSet, VecDeque},
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
    time::Instant,
};
use std::{convert::TryInto, sync::Arc};

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

#[derive(Debug, Clone)]
pub struct IntersectionEndPoint<T: Scalar, const N: usize> {
    // pub point: Point<T, N>,
    pub vertex_hint: Option<[usize; 2]>,
    pub half_edge_hint: Option<usize>,
    pub half_edge_u_hint: Option<T>,
    pub face_hint: Option<usize>, // only if interior
    pub barycentric_hint: Option<(T, T, T)>,
}

impl<T: Scalar, const N: usize> IntersectionEndPoint<T, N> {
    pub fn new(
        // point: Point<T, N>,
        vertex_hint: Option<[usize; 2]>,
        half_edge_hint: Option<usize>,
        half_edge_u_hint: Option<T>,
        face_hint: Option<usize>,
        barycentric_hint: Option<(T, T, T)>,
    ) -> Self {
        Self {
            // point,
            vertex_hint,
            half_edge_hint,
            half_edge_u_hint,
            face_hint: face_hint,
            barycentric_hint,
        }
    }
    pub fn new_default() -> Self {
        Self {
            // point: Point::default(),
            vertex_hint: None,
            half_edge_hint: None,
            half_edge_u_hint: None,
            face_hint: None,
            barycentric_hint: None,
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
    pub resulting_vertices_pair: [usize; 2],
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
        resulting_vertices_pair: [usize; 2],
        coplanar: bool,
    ) -> Self {
        Self {
            a: a,
            b: b,
            segment: segment.clone(),
            initial_face_reference,
            resulting_vertices_pair,
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
        Self::new(
            a,
            b,
            segment,
            initial_face_reference,
            [usize::MAX, usize::MAX],
            false,
        )
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

#[derive(Debug, Clone)]
pub struct Mesh<T: Scalar, const N: usize> {
    pub vertices: Vec<Vertex<T, N>>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<Face>,

    pub edge_map: HashMap<(usize, usize), usize>,
    vertex_spatial_hash: HashMap<(i64, i64, i64), Vec<usize>>,
    face_split_map: HashMap<usize, FaceSplitMap>,
    pub half_edge_split_map: HashMap<usize, (usize, usize)>,
}

impl<T: Scalar, const N: usize> Mesh<T, N> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            half_edges: Vec::new(),
            faces: Vec::new(),
            edge_map: HashMap::new(),
            vertex_spatial_hash: HashMap::new(),
            face_split_map: HashMap::new(),
            half_edge_split_map: HashMap::new(),
        }
    }

    pub fn face_normal(&self, face_idx: usize) -> Vector<T, N>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face_vertices = self
            .face_vertices(face_idx)
            .map(|v| &self.vertices[v].position);
        let edge1 = (face_vertices[1] - face_vertices[0]).as_vector();
        let edge2 = (face_vertices[2] - face_vertices[0]).as_vector();
        edge1.cross(&edge2).normalized()
    }

    pub fn face_from_vertices(&self, v0: usize, v1: usize, v2: usize) -> usize {
        // find a face with the given vertex indices
        // It's more efficient to get faces around each vertex and check for a match
        let fs0 = self.faces_around_vertex(v0);
        let fs1 = self.faces_around_vertex(v1);
        let fs2 = self.faces_around_vertex(v2);

        for &face_id in fs0.iter() {
            if fs1.contains(&face_id) && fs2.contains(&face_id) {
                return face_id;
            }
        }
        usize::MAX
    }

    pub fn plane_from_face(&self, face_idx: usize) -> Plane<T, N>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let verts = self.face_vertices(face_idx); // [usize; 3]
        let v0 = &self.vertices[verts[0]].position;
        let v1 = &self.vertices[verts[1]].position;
        let v2 = &self.vertices[verts[2]].position;

        Plane::from_points(v0, v1, v2)
    }

    pub fn build_boundary_map(
        &self,
        intersection_segments: &[IntersectionSegment<T, N>],
    ) -> HashSet<(usize, usize)>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        Segment<T, N>: SegmentOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>
            + Neg<Output = T>,
    {
        fn ordered(a: usize, b: usize) -> (usize, usize) {
            if a < b { (a, b) } else { (b, a) }
        }

        let mut boundary_edges = HashSet::new();

        for (seg_idx, seg) in intersection_segments.iter().enumerate() {
            let direction = intersection_segments[seg_idx].segment.direction();
            let he = self
                .edge_map
                .get(&(
                    seg.resulting_vertices_pair[0],
                    seg.resulting_vertices_pair[1],
                ))
                .expect("Edge map must contain the segment vertices pair");

            let face0 = self.half_edges[*he]
                .face
                .expect("Half-edge must have a face");
            let twin = self.find_valid_half_edge(self.half_edges[*he].twin, &seg.segment.a);
            let face1 = self.half_edges[twin]
                .face
                .expect("Half-edge must have a face");

            boundary_edges.insert(ordered(face0, face1));

            // for &linked_idx in &seg.links {
            //     if linked_idx == seg_idx || linked_idx >= intersection_segments.len() {
            //         continue;
            //     }

            //     let linked = &intersection_segments[linked_idx];

            //     let faces = [
            //         (
            //             seg.resulting_faces[0],
            //             linked.resulting_faces[0],
            //             &seg.segment.a,
            //             &linked.segment.a,
            //         ),
            //         (
            //             seg.resulting_faces[1],
            //             linked.resulting_faces[1],
            //             &seg.segment.b,
            //             &linked.segment.b,
            //         ),
            //     ];

            //     for (f0, f1, pa, pb) in faces {
            //         let fa = self.find_exact_valid_face(f0, pa, None).unwrap_or(f0);
            //         let fb = self.find_exact_valid_face(f1, pb, None).unwrap_or(f1);
            //         boundary_edges.insert(ordered(fa, fb));
            //     }
            // }
        }

        boundary_edges
    }

    pub fn build_face_adjacency_graph(&self) -> HashMap<usize, Vec<usize>> {
        let mut adjacency_graph = HashMap::new();

        for face_idx in 0..self.faces.len() {
            let mut adjacent_faces = Vec::new();

            // Use iterative approach with safety limits
            if let Some(half_edges) = get_face_half_edges_iterative(self, face_idx) {
                for he_idx in half_edges {
                    if he_idx < self.half_edges.len() {
                        let twin_idx = self.half_edges[he_idx].twin;

                        if twin_idx != usize::MAX && twin_idx < self.half_edges.len() {
                            if let Some(twin_face) = self.half_edges[twin_idx].face {
                                if twin_face != face_idx
                                    && twin_face < self.faces.len()
                                    && self.faces[twin_face].half_edge != usize::MAX
                                {
                                    adjacent_faces.push(twin_face);
                                }
                            }
                        }
                    }
                }
            }

            adjacency_graph.insert(face_idx, adjacent_faces);
        }

        adjacency_graph
    }

    pub fn remove_invalidated_faces(&mut self) {
        let original_count = self.faces.len();

        // Filter out invalidated faces
        let mut new_faces = Vec::new();
        let mut face_mapping = vec![None; self.faces.len()];

        for (old_idx, face) in self.faces.iter().enumerate() {
            if !face.removed {
                let new_idx = new_faces.len();
                face_mapping[old_idx] = Some(new_idx);
                new_faces.push(face.clone());
            }
        }

        // Update half-edge face references
        for he in &mut self.half_edges {
            if he.removed {
                continue; // Skip removed half-edges
            }
            if let Some(ref mut face_ref) = he.face {
                if let Some(new_face_idx) = face_mapping[*face_ref] {
                    *face_ref = new_face_idx;
                } else {
                    he.face = None; // Face was invalidated
                }
            }
        }

        // Replace faces
        self.faces = new_faces;
    }

    /// Safe version of face_vertices that validates indices
    pub fn face_vertices_safe(&self, f: usize) -> Option<Vec<usize>> {
        get_face_vertices_safe(self, f)
    }

    /// Remove duplicate vertices and update all references
    /// Returns the number of vertices removed
    pub fn remove_duplicate_vertices(&mut self) -> usize
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        if self.vertices.is_empty() {
            return 0;
        }

        let tolerance = T::point_merge_threshold();
        let initial_count = self.vertices.len();

        // Build spatial hash for efficient duplicate detection
        let mut spatial_groups: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();

        for (vertex_idx, vertex) in self.vertices.iter().enumerate() {
            let hash_key = self.position_to_hash_key(&vertex.position);
            spatial_groups.entry(hash_key).or_default().push(vertex_idx);
        }

        // Find duplicates within each spatial group
        let mut vertex_mapping = (0..self.vertices.len()).collect::<Vec<_>>();
        let mut duplicates = HashSet::new();

        for group in spatial_groups.values() {
            if group.len() < 2 {
                continue;
            }

            // Check all pairs within the group
            for i in 0..group.len() {
                if duplicates.contains(&group[i]) {
                    continue;
                }

                for j in (i + 1)..group.len() {
                    if duplicates.contains(&group[j]) {
                        continue;
                    }

                    let pos_i = &self.vertices[group[i]].position;
                    let pos_j = &self.vertices[group[j]].position;

                    if pos_i.distance_to(pos_j) <= tolerance {
                        // Mark j as duplicate of i
                        vertex_mapping[group[j]] = group[i];
                        duplicates.insert(group[j]);
                    }
                }
            }
        }

        if duplicates.is_empty() {
            return 0;
        }

        // Create compacted vertex list and final mapping
        let mut new_vertices = Vec::new();
        let mut old_to_new = vec![usize::MAX; self.vertices.len()];

        for (old_idx, vertex) in self.vertices.iter().enumerate() {
            if !duplicates.contains(&old_idx) {
                let new_idx = new_vertices.len();
                old_to_new[old_idx] = new_idx;
                new_vertices.push(vertex.clone());
            }
        }

        // Update mapping for duplicates to point to their canonical vertex's new index
        for &duplicate_idx in &duplicates {
            let canonical_idx = vertex_mapping[duplicate_idx];
            old_to_new[duplicate_idx] = old_to_new[canonical_idx];
        }

        // Update all half-edge vertex references
        for half_edge in &mut self.half_edges {
            if half_edge.removed {
                continue;
            }

            let old_vertex = half_edge.vertex;
            if old_vertex < old_to_new.len() {
                if let Some(&new_vertex) = old_to_new.get(old_vertex) {
                    if new_vertex != usize::MAX {
                        half_edge.vertex = new_vertex;
                    }
                }
            }
        }

        // Update vertex half-edge references
        for vertex in &mut new_vertices {
            if let Some(he_idx) = vertex.half_edge {
                // Verify the half-edge still exists and is valid
                if he_idx >= self.half_edges.len() || self.half_edges[he_idx].removed {
                    vertex.half_edge = None;
                }
            }
        }

        // Rebuild edge map with new vertex indices
        let mut new_edge_map = HashMap::new();
        for (&(old_v1, old_v2), &he_idx) in &self.edge_map {
            if old_v1 < old_to_new.len() && old_v2 < old_to_new.len() {
                let new_v1 = old_to_new[old_v1];
                let new_v2 = old_to_new[old_v2];

                if new_v1 != usize::MAX && new_v2 != usize::MAX && new_v1 != new_v2 {
                    // Only keep edges that don't become self-loops
                    new_edge_map.insert((new_v1, new_v2), he_idx);
                }
            }
        }

        // Rebuild spatial hash with new vertex indices
        let mut new_spatial_hash: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (vertex_idx, vertex) in new_vertices.iter().enumerate() {
            let hash_key = self.position_to_hash_key(&vertex.position);
            new_spatial_hash
                .entry(hash_key)
                .or_default()
                .push(vertex_idx);
        }

        // Update mesh data structures
        self.vertices = new_vertices;
        self.edge_map = new_edge_map;
        self.vertex_spatial_hash = new_spatial_hash;

        // Remove any degenerate faces that may have been created
        // self.remove_degenerate_faces();

        initial_count - self.vertices.len()
    }

    pub fn remove_unused_vertices(&mut self) {
        // 1. Find used vertices
        let mut used = vec![false; self.vertices.len()];
        for face_idx in 0..self.faces.len() {
            // Skip null or removed faces
            if self.faces[face_idx].null || self.faces[face_idx].removed {
                continue;
            }
            for &v_idx in &self.face_vertices(face_idx) {
                used[v_idx] = true;
            }
        }

        // 2. Build mapping from old to new indices
        let mut old_to_new = vec![None; self.vertices.len()];
        let mut new_vertices = Vec::new();
        for (old_idx, vertex) in self.vertices.iter().enumerate() {
            if used[old_idx] {
                let new_idx = new_vertices.len();
                new_vertices.push(vertex);
                old_to_new[old_idx] = Some(new_idx);
            }
        }

        // 3. Remap faces (skip invalidated ones)
        let mut new_mesh = Mesh::new();
        for v in &new_vertices {
            new_mesh.add_vertex(v.position.clone());
        }
        for face_idx in 0..self.faces.len() {
            // Skip null or removed faces
            if self.faces[face_idx].null || self.faces[face_idx].removed {
                continue;
            }
            let vs = self.face_vertices(face_idx);
            let mapped: Vec<usize> = vs.iter().map(|&vi| old_to_new[vi].unwrap()).collect();
            if mapped[0] != mapped[1] && mapped[1] != mapped[2] && mapped[2] != mapped[0] {
                new_mesh.add_triangle(mapped[0], mapped[1], mapped[2]);
            }
        }
        *self = new_mesh;
    }

    pub fn build_face_tree(&self) -> AabbTree<T, N, Point<T, N>, usize>
    where
        T: Scalar,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut face_aabbs = Vec::with_capacity(self.faces.len());

        let _start = Instant::now();
        for (face_idx, face) in self.faces.iter().enumerate() {
            if face.removed {
                continue;
            }

            let hes = self.face_half_edges(face_idx);

            if hes[1] >= self.half_edges.len()
                || hes[2] >= self.half_edges.len()
                || self.half_edges[hes[2]].next != hes[0]
            {
                continue;
            }

            let v0_idx = self.half_edges[hes[0]].vertex;
            let v1_idx = self.half_edges[hes[1]].vertex;
            let v2_idx = self.half_edges[hes[2]].vertex;

            if v0_idx >= self.vertices.len()
                || v1_idx >= self.vertices.len()
                || v2_idx >= self.vertices.len()
            {
                continue;
            }

            // **GENERIC N-DIMENSIONAL COORDINATE PROCESSING**
            let p0 = &self.vertices[v0_idx].position;
            let p1 = &self.vertices[v1_idx].position;
            let p2 = &self.vertices[v2_idx].position;

            // Find min/max for each coordinate dimension
            let mut min_coords = from_fn(|_| T::from(0));
            let mut max_coords = from_fn(|_| T::from(0));

            for i in 0..N {
                // Direct comparison without intermediate storage
                let coord0 = &p0[i];
                let coord1 = &p1[i];
                let coord2 = &p2[i];

                let min_val = if coord0 <= coord1 && coord0 <= coord2 {
                    coord0
                } else if coord1 <= coord2 {
                    coord1
                } else {
                    coord2
                };

                let max_val = if coord0 >= coord1 && coord0 >= coord2 {
                    coord0
                } else if coord1 >= coord2 {
                    coord1
                } else {
                    coord2
                };

                min_coords[i] = min_val.clone();
                max_coords[i] = max_val.clone();
            }

            let aabb =
                Aabb::from_points(&Point::from_vals(min_coords), &Point::from_vals(max_coords));

            face_aabbs.push((aabb, face_idx));
        }

        let tree = AabbTree::build(face_aabbs);

        tree
    }

    /// Compute the AABB of face `f`.
    pub fn face_aabb(&self, f: usize) -> Aabb<T, N, Point<T, N>>
    where
        T: Scalar,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face = &self.faces[f];
        if face.removed || face.half_edge == usize::MAX || self.half_edges[face.half_edge].removed {
            // Return degenerate AABB for invalid faces
            let origin = Point::<T, N>::from_vals(from_fn(|_| T::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        let hes = self.face_half_edges(f);

        // Safety checks
        if hes[1] >= self.half_edges.len() || hes[2] >= self.half_edges.len() {
            let origin = Point::<T, N>::from_vals(from_fn(|_| T::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        let v0_idx = self.half_edges[hes[0]].vertex;
        let v1_idx = self.half_edges[hes[1]].vertex;
        let v2_idx = self.half_edges[hes[2]].vertex;

        // Safety checks for vertex indices
        if v0_idx >= self.vertices.len()
            || v1_idx >= self.vertices.len()
            || v2_idx >= self.vertices.len()
        {
            let origin = Point::<T, N>::from_vals(from_fn(|_| T::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        let p0 = &self.vertices[v0_idx].position;
        let p1 = &self.vertices[v1_idx].position;
        let p2 = &self.vertices[v2_idx].position;

        // Compute AABB from the three vertices directly
        compute_triangle_aabb(p0, p1, p2)
    }

    pub fn add_vertex(&mut self, position: Point<T, N>) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(Vertex::new(position));
        idx
    }

    /// Find existing vertex near position using spatial hash
    fn find_nearby_vertex(&self, pos: &Point<T, N>, tolerance: T) -> Option<usize>
    where
        Point<T, N>: PointOps<T, N>,
    {
        let center_key = self.position_to_hash_key(pos);

        // Check center cell and 26 neighboring cells (3x3x3 grid)
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (center_key.0 + dx, center_key.1 + dy, center_key.2 + dz);

                    if let Some(candidates) = self.vertex_spatial_hash.get(&key) {
                        for &vi in candidates {
                            if vi < self.vertices.len()
                                && self.vertices[vi].position.distance_to(pos) < tolerance
                            {
                                return Some(vi);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Convert position to hash key (quantized to grid)
    fn position_to_hash_key(&self, pos: &Point<T, N>) -> (i64, i64, i64) {
        let grid_size = T::point_merge_threshold().to_f64().unwrap_or(1e-5);
        (
            (pos[0].to_f64().unwrap() / grid_size) as i64,
            (pos[1].to_f64().unwrap() / grid_size) as i64,
            (pos[2].to_f64().unwrap() / grid_size) as i64,
        )
    }

    pub fn faces_containing_point_aabb(
        &self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        p: &Point<T, N>,
    ) -> Vec<usize>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>
            + Neg<Output = T>,
    {
        let tolerance = T::query_tolerance();

        // **OPTIMIZED: Tighter query AABB**
        let query_aabb = Aabb::from_points(
            &Point::<T, N>::from_vals(from_fn(|i| &p[i] - &tolerance)),
            &Point::<T, N>::from_vals(from_fn(|i| &p[i] + &tolerance)),
        );

        let mut candidates = Vec::new();
        aabb_tree.query(&query_aabb, &mut candidates);

        // **OPTIMIZED: Early exit with tree rebuild**
        if candidates.is_empty() {
            // Try with much larger search radius before rebuilding
            let large_tolerance = T::tolerance() * T::from(100.0);
            let large_query_aabb = Aabb::from_points(
                &Point::<T, N>::from_vals(from_fn(|i| &p[i] - &large_tolerance)),
                &Point::<T, N>::from_vals(from_fn(|i| &p[i] + &large_tolerance)),
            );

            aabb_tree.query(&large_query_aabb, &mut candidates);

            // Only rebuild if absolutely no faces found in large area
            if candidates.is_empty() {
                drop(candidates);
                *aabb_tree = self.build_face_tree();
                candidates = Vec::new();
                aabb_tree.query(&query_aabb, &mut candidates);

                if candidates.is_empty() {
                    return Vec::new();
                }
            }
        }

        let mut result = Vec::with_capacity(candidates.len().min(8)); // Pre-allocate reasonable size

        for &face_idx in &candidates {
            let he0 = self.faces[*face_idx].half_edge;

            if he0 >= self.half_edges.len() {
                continue;
            }

            let he1 = self.half_edges[he0].next;
            let he2 = self.half_edges[he1].next;

            if he1 >= self.half_edges.len()
                || he2 >= self.half_edges.len()
                || self.half_edges[he2].next != he0
            {
                continue;
            }

            let v0_idx = self.half_edges[he0].vertex;
            let v1_idx = self.half_edges[he1].vertex;
            let v2_idx = self.half_edges[he2].vertex;

            if self.edge_map.get(&(v0_idx, v1_idx)).is_none()
                || self.edge_map.get(&(v1_idx, v2_idx)).is_none()
                || self.edge_map.get(&(v2_idx, v0_idx)).is_none()
            {
                panic!(
                    "Invalid face edges in mesh: {}, {}, {}",
                    v0_idx, v1_idx, v2_idx
                );
            }

            if v0_idx >= self.vertices.len()
                || v1_idx >= self.vertices.len()
                || v2_idx >= self.vertices.len()
            {
                continue;
            }

            let a = &self.vertices[v0_idx].position;
            let b = &self.vertices[v1_idx].position;
            let c = &self.vertices[v2_idx].position;

            let ab = (b - a).as_vector();
            let ac = (c - a).as_vector();
            let ap = (p - a).as_vector();

            let n = ab.cross(&ac);
            let dist = n.dot(&ap).abs();

            // **FIXED: Use consistent tolerance**
            if dist > tolerance {
                continue;
            }

            if point_in_triangle(p, a, b, c) {
                result.push(*face_idx);
            }
        }

        // **OPTIMIZED: Tree rebuild detection**
        if result.is_empty() && !candidates.is_empty() {
            // Candidates existed but none contained point - possible stale tree
            *aabb_tree = self.build_face_tree();
            return self.faces_containing_point_aabb(aabb_tree, p);
        }

        result
    }

    /// Adds a triangle face given three vertex indices (in CCW order).
    /// If a given bordering face is inexistent, a "null" face is added.
    /// Returns the index of the newly created face.
    pub fn add_triangle(&mut self, v0: usize, v1: usize, v2: usize) -> usize {
        // CCW triangle directed edges
        let edge_vertices = [(v0, v1), (v1, v2), (v2, v0)];

        // Pre-scan: find reusable border half-edges (owned by null faces)
        let mut reuse_he: [Option<usize>; 3] = [None, None, None];
        let mut reuse_face_idx: Option<usize> = None;
        let mut extra_promoted_nulls: Vec<usize> = Vec::new();

        for (i, &(from, to)) in edge_vertices.iter().enumerate() {
            if let Some(&he_idx) = self.edge_map.get(&(from, to)) {
                if let Some(fidx) = self.half_edges[he_idx].face {
                    if self.faces[fidx].null {
                        reuse_he[i] = Some(he_idx);
                        if let Some(chosen) = reuse_face_idx {
                            if chosen != fidx {
                                // This is another (different) null face; we’ll retire it later.
                                extra_promoted_nulls.push(fidx);
                            }
                        } else {
                            reuse_face_idx = Some(fidx);
                        }
                    } else {
                        // Directed edge already belongs to a real face -> non-manifold if we add another.
                        debug_assert!(
                            false,
                            "add_triangle: directed edge ({},{}) already used by a real face",
                            from, to
                        );
                    }
                }
            }
        }

        // Decide which face index to use for the new triangle
        let face_idx = if let Some(fidx) = reuse_face_idx {
            // We’ll flip this null face to real below.
            fidx
        } else {
            // Reserve a new real face slot NOW so the index stays stable
            let idx = self.faces.len();
            // Temporary placeholder; will set .half_edge and null flag after edges are wired
            self.faces.push(Face::new(0));
            idx
        };

        // Build/reuse the 3 half-edges that bound this new face
        let mut edge_indices = [usize::MAX; 3];

        for (i, &(from, to)) in edge_vertices.iter().enumerate() {
            if let Some(he_idx) = reuse_he[i] {
                // Reuse a border half-edge: promote its face to this real face index
                self.half_edges[he_idx].face = Some(face_idx);
                edge_indices[i] = he_idx;
            } else if let Some(&existing_idx) = self.edge_map.get(&(from, to)) {
                // Exists but not null (already used by a real face) -> non-manifold; keep assert.
                debug_assert!(
                    false,
                    "add_triangle: reusing non-null directed edge ({},{})",
                    from, to
                );
                edge_indices[i] = existing_idx; // keep mesh consistent in release builds
            } else {
                // Create a brand-new half-edge for this triangle
                let he_idx = self.half_edges.len();
            let mut he = HalfEdge::new(to);
            he.face = Some(face_idx);
                self.half_edges.push(he);
                self.edge_map.insert((from, to), he_idx);
                edge_indices[i] = he_idx;

                // Try to hook twin; otherwise create a border twin + a null face
                if let Some(&rev_idx) = self.edge_map.get(&(to, from)) {
                    self.half_edges[he_idx].twin = rev_idx;
                    self.half_edges[rev_idx].twin = he_idx;
                } else {
                    // Create border half-edge (to -> from) owned by a new null face
                    let border_idx = self.half_edges.len();
                    let mut bhe = HalfEdge::new(from);
                    bhe.twin = he_idx;
                    // temporary self-loop; an outer ring builder can rewire later
                    bhe.next = border_idx;
                    bhe.prev = border_idx;
                    self.half_edges.push(bhe);
                    self.edge_map.insert((to, from), border_idx);

                    self.half_edges[he_idx].twin = border_idx;

                    let nf = self.faces.len();
                    self.faces.push(Face::new_null(border_idx));
                    self.half_edges[border_idx].face = Some(nf);
                }
            }
        }

        // Link the triangle ring
        let e0 = edge_indices[0];
        let e1 = edge_indices[1];
        let e2 = edge_indices[2];
        self.half_edges[e0].next = e1;
        self.half_edges[e0].prev = e2;
        self.half_edges[e1].next = e2;
        self.half_edges[e1].prev = e0;
        self.half_edges[e2].next = e0;
        self.half_edges[e2].prev = e1;

        // Attach representative half-edges to vertices
        self.vertices[v0].half_edge.get_or_insert(e0);
        self.vertices[v1].half_edge.get_or_insert(e1);
        self.vertices[v2].half_edge.get_or_insert(e2);

        // Finalize the face record
        if let Some(_reused) = reuse_face_idx {
            // Flip null → real, and point it at one of the boundary half-edges
            self.faces[face_idx].null = false; // or self.faces[face_idx].null = false;
            self.faces[face_idx].half_edge = e0;
        } else {
            // We reserved a new face earlier; fill its boundary pointer now
            self.faces[face_idx].half_edge = e0;
            // ensure it’s not null
            self.faces[face_idx].null = false;
        }

        // Any *other* null faces that had their sole half-edge promoted are now empty → retire them
        for f in extra_promoted_nulls {
            if f != face_idx && self.faces[f].null {
                self.faces[f].removed = true;
                // Optional: self.faces[f].half_edge = usize::MAX;
            }
        }

        face_idx
    }

    /// Return the centroid of face `f` as a Vec<f64> of length = dimensions().
    /// Currently works for any dimension, but returns a flat Vec.
    pub fn face_centroid(&self, f: usize) -> Vector<T, N>
    where
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face_vertices = self.face_vertices(f);
        let n = T::from(face_vertices.len() as f64);

        let mut centroid: Vector<T, N> = Vector::zero();

        for &v_idx in &face_vertices {
            let coords = self.vertices[v_idx].position.coords();
            for i in 0..3.min(N) {
                centroid[i] += &coords[i];
            }
        }

        for coord in centroid.coords_mut().iter_mut() {
            *coord = &*coord / &n;
        }

        centroid
    }

    pub fn face_area(&self, f: usize) -> T
    where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        match N {
            2 => self.face_area_2d(f),
            3 => self.face_area_3d(f),
            _ => panic!("face_area only supports 2D and 3D"),
        }
    }

    fn face_area_2d(&self, f: usize) -> T
    where
        T: Scalar,
        Point<T, 2>: PointOps<T, 2, Vector = Vector<T, 2>>,
        Vector<T, 2>: VectorOps<T, 2, Cross = T>, // Cross product is scalar in 2D
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face_vertices = self.face_vertices(f);
        if face_vertices.len() != 3 {
            panic!("face_area only works for triangular faces");
        }

        let a = &self.vertices[face_vertices[0]].position;
        let b = &self.vertices[face_vertices[1]].position;
        let c = &self.vertices[face_vertices[2]].position;

        let ab = b - a;
        let ac = c - a;

        let ab_2d = Vector::<T, 2>::from_vals([ab[0].clone(), ab[1].clone()]);
        let ac_2d = Vector::<T, 2>::from_vals([ac[0].clone(), ac[1].clone()]);

        let cross_product = ab_2d.cross(&ac_2d);
        cross_product.abs() / T::from_num_den(2, 1)
    }

    fn face_area_3d(&self, f: usize) -> T
    where
        T: Scalar,
        Point<T, 3>: PointOps<T, 3, Vector = Vector<T, 3>>,
        Vector<T, 3>: VectorOps<T, 3, Cross = Vector<T, 3>>, // Cross product is vector in 3D
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face_vertices = self.face_vertices(f);
        if face_vertices.len() != 3 {
            panic!("face_area only works for triangular faces");
        }

        let a = &self.vertices[face_vertices[0]].position;
        let b = &self.vertices[face_vertices[1]].position;
        let c = &self.vertices[face_vertices[2]].position;

        let ab = b - a;
        let ac = c - a;

        let ab_3d = Vector::<T, 3>::from_vals([ab[0].clone(), ab[1].clone(), ab[2].clone()]);
        let ac_3d = Vector::<T, 3>::from_vals([ac[0].clone(), ac[1].clone(), ac[2].clone()]);

        let cross_product = ab_3d.cross(&ac_3d);
        cross_product.norm() / T::from_num_den(2, 1)
    }

    /// Enumerate all outgoing half-edges from `v` exactly once,
    /// in CCW order.  Works even on meshes with open boundaries,
    /// *provided* you’ve first called `build_boundary_loops()`.
    pub fn outgoing_half_edges(&self, v: usize) -> Vec<usize> {
        let start = self.vertices[v]
            .half_edge
            .expect("vertex has no incident edges");
        let mut result = Vec::new();
        let mut h = start;
        loop {
            result.push(h);
            let t = self.half_edges[h].twin;
            // Now that every edge has a twin (real or ghost), we never hit usize::MAX
            h = self.half_edges[t].next;
            if h == start {
                break;
            }
        }
        result
    }

    /// Returns true if vertex `v` has any outgoing ghost edge (face == None).
    pub fn is_boundary_vertex(&self, v: usize) -> bool {
        self.outgoing_half_edges(v)
            .into_iter()
            .any(|he| self.half_edges[he].face.is_none())
    }

    // /// Returns all vertex indices that lie on at least one boundary loop.
    pub fn boundary_vertices(&self) -> Vec<usize> {
        (0..self.vertices.len())
            .filter(|&v| self.is_boundary_vertex(v))
            .collect()
    }

    /// Returns the one-ring neighbors of vertex `v`.
    pub fn one_ring_neighbors(&self, v: usize) -> Vec<usize> {
        self.outgoing_half_edges(v)
            .iter()
            .map(|&he_idx| self.half_edges[he_idx].vertex)
            .collect()
    }

    /// Builds boundary loops for the mesh.
    pub fn build_boundary_loops(&mut self) {
        let mut seen = HashSet::new();
        let original_count = self.half_edges.len();

        for start in 0..original_count {
            if self.half_edges[start].twin != usize::MAX || seen.contains(&start) {
                continue;
            }

            // 1) Gather the full hole cycle (may include interior edges)
            let mut hole_cycle = Vec::new();
            let mut he = start;
            let mut steps = 0;
            loop {
                // ← add a safety guard
                if steps > original_count {
                    panic!(
                        "build_boundary_loops: boundary cycle failed to close (start={} current_he={})",
                        start, he
                    );
                }
                steps += 1;
                seen.insert(he);
                hole_cycle.push(he);
                let prev = self.half_edges[he].prev;
                let twin = self.half_edges[prev].twin;
                he = if twin != usize::MAX { twin } else { prev };
                if he == start {
                    break;
                }
            }

            // 2) Filter to *just* the boundary edges
            let boundary_cycle: Vec<usize> = hole_cycle
                .into_iter()
                .filter(|&bhe| bhe < original_count && self.half_edges[bhe].twin == usize::MAX)
                .collect();

            // 3) Spawn one ghost per boundary half-edge
            let mut ghosts = Vec::with_capacity(boundary_cycle.len());
            for &bhe in &boundary_cycle {
                let origin = {
                    let prev = self.half_edges[bhe].prev;
                    self.half_edges[prev].vertex
                };
                let mut ghost = HalfEdge::new(origin);
                ghost.face = None;
                ghost.twin = bhe;
                let g_idx = self.half_edges.len();
                self.half_edges[bhe].twin = g_idx;
                self.half_edges.push(ghost);
                ghosts.push(g_idx);
            }

            // 4) Link the *ghosts* in cycle order
            let n = ghosts.len();
            for i in 0..n {
                let g = ghosts[i];
                let g_next = ghosts[(i + 1) % n];
                let g_prev = ghosts[(i + n - 1) % n];
                self.half_edges[g].next = g_next;
                self.half_edges[g].prev = g_prev;
            }
        }
    }

    /// Returns the indices of the half-edges bounding face `f`,
    /// in CCW order.
    pub fn face_half_edges(&self, f: usize) -> SmallVec<[usize; 3]> {
        if self.faces[f].removed {
            panic!("face_half_edges called on removed face {}", f);
        }
        let mut result = SmallVec::new();
        let start = self.faces[f].half_edge;
        let mut h = start;
        loop {
            result.push(h);
            h = self.half_edges[h].next;

            if h == start {
                break;
            }
        }
        result
    }

    /// Returns the vertex indices around face `f`,
    /// in CCW order.
    pub fn face_vertices(&self, f: usize) -> [usize; 3] {
        let vertices: Vec<usize> = self
            .face_half_edges(f)
            .into_iter()
            .map(|he| self.half_edges[he].vertex)
            .collect();

        [vertices[0], vertices[1], vertices[2]]
    }

    pub fn point_on_half_edge(&self, he: usize, p: &Point<T, N>) -> Option<T>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let start = &self.vertices[self.half_edges[self.half_edges[he].prev].vertex].position;
        let end = &self.vertices[self.half_edges[he].vertex].position;

        point_on_segment(start, end, p)
    }

    pub fn segment_from_half_edge(&self, he: usize) -> Segment<T, N> {
        let target = self.half_edges[he].vertex;
        let source = self.half_edges[self.half_edges[he].prev].vertex;
        Segment::new(
            &self.vertices[source].position,
            &self.vertices[target].position,
        )
    }

    pub fn are_faces_adjacent(&self, f1: usize, f2: usize) -> bool {
        // Check if any half-edge of f1 is a twin of any half-edge of f2
        for h1 in self.face_half_edges(f1) {
            for h2 in self.face_half_edges(f2) {
                if self.half_edges[h1].twin == h2 {
                    return true;
                }
            }
        }
        false
    }

    /// Returns each boundary loop as a Vec of vertex indices, CCW around the hole.
    pub fn boundary_loops(&self) -> Vec<Vec<usize>> {
        let mut loops = Vec::new();
        let mut seen = HashSet::new();

        for (i, he) in self.half_edges.iter().enumerate() {
            // only process ghost edges (face == None) once
            if he.face.is_none() && !seen.contains(&i) {
                let mut loop_vs = Vec::new();
                let mut curr = i;
                loop {
                    seen.insert(curr);
                    // each ghost.he.vertex is the “to”-vertex on the boundary
                    loop_vs.push(self.half_edges[curr].vertex);
                    curr = self.half_edges[curr].next;
                    if curr == i {
                        break;
                    }
                }
                loops.push(loop_vs);
            }
        }
        loops
    }

    pub fn point_in_mesh(
        &self,
        tree: &AabbTree<T, 3, Point<T, 3>, usize>,
        p: &Point<T, 3>,
    ) -> PointInMeshResult
    where
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut inside_count = 0;
        let mut total_rays = 0;
        let mut on_surface = false;

        let rays = vec![
            Vector::from_vals([T::one(), T::zero(), T::zero()]),
            Vector::from_vals([T::zero(), T::one(), T::zero()]),
            Vector::from_vals([T::zero(), T::zero(), T::one()]),
            Vector::from_vals([T::from(-1.0), T::zero(), T::zero()]),
            Vector::from_vals([T::zero(), T::from(-1.0), T::zero()]),
            Vector::from_vals([T::zero(), T::zero(), T::from(-1.0)]),
        ];

        for r in rays {
            match self.cast_ray(p, &r, tree) {
                Some(RayCastResult::Inside) => {
                    inside_count += 1;
                    total_rays += 1;
                }
                Some(RayCastResult::OnSurface) => {
                    on_surface = true;
                    total_rays += 1;
                }
                Some(RayCastResult::Outside) => {
                    total_rays += 1;
                }
                None => {}
            }
        }

        if on_surface {
            PointInMeshResult::OnSurface
        } else if total_rays > 0 && inside_count > total_rays / 2 {
            PointInMeshResult::Inside
        } else {
            PointInMeshResult::Outside
        }
    }

    fn cast_ray(
        &self,
        p: &Point<T, 3>,
        dir: &Vector<T, 3>,
        tree: &AabbTree<T, 3, Point<T, 3>, usize>,
    ) -> Option<RayCastResult>
    where
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut hits: Vec<T> = Vec::new();
        let mut touches_surface = false;

        // Create ray AABB for tree query
        let far_multiplier = T::from(1000.0);
        let far_point = Point::<T, 3>::from_vals([
            &p[0] + &(&dir[0] * &far_multiplier),
            &p[1] + &(&dir[1] * &far_multiplier),
            &p[2] + &(&dir[2] * &far_multiplier),
        ]);
        let ray_aabb = Aabb::from_points(p, &far_point);

        // Query tree for faces that intersect ray
        let mut candidate_faces = Vec::new();
        tree.query(&ray_aabb, &mut candidate_faces);

        // Test only candidate faces (not all faces!)
        for &fi in &candidate_faces {
            let vs_idxs = self.face_vertices(*fi);

            // Use references to existing vertex positions
            let v0 = &self.vertices[vs_idxs[0]].position;
            let v1 = &self.vertices[vs_idxs[1]].position;
            let v2 = &self.vertices[vs_idxs[2]].position;

            if let Some(t) = self.ray_triangle_intersection(&p.coords(), dir, [v0, v1, v2]) {
                if t.abs() <= T::tolerance() {
                    touches_surface = true;
                } else if t > T::tolerance() {
                    hits.push(t);
                }
            }
        }

        if hits.is_empty() {
            return None;
        }

        // Remove duplicates and count
        hits.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if touches_surface {
            Some(RayCastResult::OnSurface)
        } else if !hits.is_empty() {
            // Deduplicate and count intersections (same logic as before)
            hits.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut filtered_hits = 0;
            let mut last_t = None;

            for t in hits {
                if last_t
                    .as_ref()
                    .map_or(true, |lt: &T| (&t - &lt).abs() > T::tolerance())
                {
                    filtered_hits += 1;
                    last_t = Some(t);
                }
            }

            Some(if filtered_hits % 2 == 1 {
                RayCastResult::Inside
            } else {
                RayCastResult::Outside
            })
        } else {
            None
        }
    }

    /// Robust ray-triangle intersection using Möller-Trumbore algorithm
    fn ray_triangle_intersection(
        &self,
        ray_origin: &[T; 3],
        ray_dir: &Vector<T, 3>,
        triangle: [&Point<T, N>; 3],
    ) -> Option<T>
    where
        T: Scalar,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        // Triangle vertices
        let v0 = triangle[0];
        let v1 = triangle[1];
        let v2 = triangle[2];

        // Triangle edges
        let edge1 = Vector::<T, 3>::from_vals([&v1[0] - &v0[0], &v1[1] - &v0[1], &v1[2] - &v0[2]]);

        let edge2 = Vector::<T, 3>::from_vals([&v2[0] - &v0[0], &v2[1] - &v0[1], &v2[2] - &v0[2]]);

        // Cross product: ray_dir x edge2
        let h = Vector::<T, 3>::from_vals([
            &ray_dir[1] * &edge2[2] - &ray_dir[2] * &edge2[1],
            &ray_dir[2] * &edge2[0] - &ray_dir[0] * &edge2[2],
            &ray_dir[0] * &edge2[1] - &ray_dir[1] * &edge2[0],
        ]);

        // Determinant
        let a = edge1.dot(&h);

        if a.abs().is_zero() {
            return None; // Ray is parallel to triangle
        }

        let f = T::one() / a;

        // Vector from v0 to ray origin
        let s = Vector::<T, 3>::from_vals([
            &ray_origin[0] - &v0[0],
            &ray_origin[1] - &v0[1],
            &ray_origin[2] - &v0[2],
        ]);

        // Calculate u parameter
        let u = &f * &s.dot(&h);

        if u < T::zero() || u > T::one() {
            return None; // Intersection outside triangle
        }

        // Cross product: s x edge1
        let q = Vector::<T, 3>::from_vals([
            &s[1] * &edge1[2] - &s[2] * &edge1[1],
            &s[2] * &edge1[0] - &s[0] * &edge1[2],
            &s[0] * &edge1[1] - &s[1] * &edge1[0],
        ]);

        // Calculate v parameter
        let v = &f * &ray_dir.dot(&q);

        if v < T::zero() || &u + &v > T::one() {
            return None; // Intersection outside triangle
        }

        // Calculate t parameter (distance along ray)
        let t = &f * &edge2.dot(&q);
        Some(t)

        // if t.is_positive() {
        //     Some(t) // Valid intersection
        // } else {
        //     None // Intersection behind ray origin or too close
        // }
    }

    /// Flip an interior edge given one of its half‐edges `he`.
    /// Returns Err if `he` is on the boundary (i.e. twin or face is None).
    pub fn flip_edge(&mut self, he_a: usize) -> Result<(), &'static str> {
        // --- 1) validity checks ---
        let he_d = self.half_edges[he_a].twin;
        if he_d == usize::MAX {
            return Err("cannot flip a boundary edge");
        }
        let f0 = self.half_edges[he_a].face.ok_or("no face on he")?;
        let f1 = self.half_edges[he_d].face.ok_or("no face on twin")?;

        // --- 2) collect the six half‐edges around that edge ---
        let he_b = self.half_edges[he_a].next;
        let he_c = self.half_edges[he_a].prev;
        let he_e = self.half_edges[he_d].next;
        let he_f = self.half_edges[he_d].prev;

        // --- 3) pull off the four corner vertices ---
        let _u = self.half_edges[he_c].vertex; // c->u
        let _v = self.half_edges[he_a].vertex; // u->v
        let c = self.half_edges[he_b].vertex; // v->c
        let d = self.half_edges[he_e].vertex; // u->d

        // --- 4) reassign the two halves of the diagonal to c->d and d->c ---
        self.half_edges[he_a].vertex = d; // now u->d
        self.half_edges[he_d].vertex = c; // now v->c

        // --- 5) stitch up face f0 to be the triangle (c, d, u) ---
        // We pick the cycle [he_c, he_a, he_b] so that dests are [u, d, c]:
        self.half_edges[he_c].next = he_a;
        self.half_edges[he_a].next = he_b;
        self.half_edges[he_b].next = he_c;

        self.half_edges[he_a].prev = he_c;
        self.half_edges[he_b].prev = he_a;
        self.half_edges[he_c].prev = he_b;

        self.faces[f1].half_edge = he_c; // start anywhere in that cycle
        // println!("setting face {} half-edge {}", f1, he_c);

        // --- 6) stitch up face f1 to be the triangle (d, c, v) ---
        // We pick the cycle [he_e, he_d, he_f] so that dests are [d, c, v]:
        self.half_edges[he_e].next = he_d;
        self.half_edges[he_d].next = he_f;
        self.half_edges[he_f].next = he_e;

        self.half_edges[he_d].prev = he_e;
        self.half_edges[he_f].prev = he_d;
        self.half_edges[he_e].prev = he_f;

        self.faces[f0].half_edge = he_e;
        // println!("setting face {} half-edge {}", f0, he_e);

        Ok(())
    }

    /// Collapse the interior edge `he` by merging its dest‐vertex into its origin‐vertex,
    /// removing the two incident faces and any degenerate triangles that produce.
    ///
    /// This rebuilds the mesh from scratch, so all indices and edge_map are reconstructed.
    pub fn collapse_edge_rebuild(&mut self, he: usize) -> Result<(), &'static str> {
        // 1) Preconditions (same as before) …
        let he_d = self.half_edges[he].twin;
        if he_d == usize::MAX {
            return Err("cannot collapse a boundary edge");
        }

        let f0 = self.half_edges[he].face.ok_or("he has no face")?;
        let f1 = self.half_edges[he_d].face.ok_or("twin has no face")?;

        // 2) Identify u->v and record the three hole corners c, u, d
        let he_b = self.half_edges[he].next; // v -> c
        let he_c = self.half_edges[he].prev; // c -> u
        let he_e = self.half_edges[he_d].next; // u -> d

        let u = self.half_edges[he_c].vertex; // origin u
        let c = self.half_edges[he_b].vertex; // one corner c
        let d = self.half_edges[he_e].vertex; // the other corner d

        // 3) Build old_to_new map & vertex list (same as before) …
        let remove_v = self.half_edges[he].vertex; // v
        let mut old_to_new = vec![None; self.vertices.len()];
        let mut new_positions = Vec::with_capacity(self.vertices.len() - 1);
        for (i, vert) in self.vertices.iter().enumerate() {
            if i == remove_v {
                continue;
            }
            let ni = new_positions.len();
            old_to_new[i] = Some(ni);
            new_positions.push(vert.position.clone());
        }
        // redirect the removed v -> the kept u
        old_to_new[remove_v] = old_to_new[u];

        // 4) Collect surviving faces
        let mut new_faces = Vec::new();
        for (fi, _) in self.faces.iter().enumerate() {
            if fi == f0 || fi == f1 {
                continue;
            }
            let vs = self.face_vertices(fi);
            let mapped: [usize; 3] = [
                old_to_new[vs[0]].unwrap(),
                old_to_new[vs[1]].unwrap(),
                old_to_new[vs[2]].unwrap(),
            ];
            if mapped[0] != mapped[1] && mapped[1] != mapped[2] && mapped[2] != mapped[0] {
                new_faces.push(mapped);
            }
        }

        // 5) **If no faces survived**, triangulate the hole [c,u,d]
        if new_faces.is_empty() {
            let mc = old_to_new[c].unwrap();
            let mu = old_to_new[u].unwrap();
            let md = old_to_new[d].unwrap();
            // One triangle filling the hole:
            new_faces.push([mc, mu, md]);
        }

        // 6) Rebuild the mesh
        let mut new_mesh = Mesh::new();
        for pos in new_positions {
            new_mesh.add_vertex(pos);
        }
        for tri in new_faces {
            new_mesh.add_triangle(tri[0], tri[1], tri[2]);
        }
        new_mesh.build_boundary_loops();

        *self = new_mesh;
        Ok(())
    }

    // pub fn split_edge_by_u(
    //     &mut self,
    //     aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
    //     he: usize,
    //     u: &T,
    // ) -> Result<SplitResult, &'static str>
    // where
    //     T: Scalar,
    //     Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    //     Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    //     for<'a> &'a T: Sub<&'a T, Output = T>
    //         + Mul<&'a T, Output = T>
    //         + Add<&'a T, Output = T>
    //         + Div<&'a T, Output = T>,
    // {
    //     let he = self.find_valid_half_edge(he, &u);
    //     let source = self.half_edges[self.half_edges[he].prev].vertex;
    //     let segment = self.segment_from_half_edge(he);
    //     let segment_direction = segment.direction();
    //     let new_point = &self.vertices[source].position + &segment_direction.scale(&u).0;
    //     self.split_edge(aabb_tree, he, &new_point)
    // }

    pub fn split_edge(
        &mut self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        he: usize,
        pos: &Point<T, N>,
    ) -> Result<SplitResult, &'static str>
    where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let he_ca = self.find_valid_half_edge(he, pos);
        let he_ab = self.half_edges[he_ca].next;
        let he_bc = self.half_edges[he_ab].next;

        let he_ac = self.half_edges[he_ca].twin;
        let he_cd = self.half_edges[he_ac].next;
        let he_da = self.half_edges[he_cd].next;

        let ex_he_ba = self.half_edges[he_ab].twin;
        let ex_he_cb = self.half_edges[he_bc].twin;
        let ex_he_dc = self.half_edges[he_cd].twin;
        let ex_he_ad = self.half_edges[he_da].twin;

        let a = self.half_edges[he_ca].vertex;
        let b = self.half_edges[he_ab].vertex;
        let c = self.half_edges[he_bc].vertex;
        let d = self.half_edges[he_cd].vertex;

        // Create new vertex at split position
        let w = self.vertices.len();
        self.vertex_spatial_hash
            .entry(self.position_to_hash_key(pos))
            .or_default()
            .push(w);
        //self.vertices.push(Vertex::new(pos.clone()));
        self.add_vertex(pos.clone());

        // let mut new_face_results = Vec::new();
        let original_face_1 = self.half_edges[he_ca].face.unwrap();
        let original_face_2 = self.half_edges[he_ac].face.unwrap();

        // 1. Build the new faces (4 in total) and create their half-edges (no twins for now)
        let evs = [
            (w, b), // face wbc
            (c, w),
            (w, a), // face wab
            (b, w),
            (w, c), // face wcd
            (d, w),
            (w, d), // face wda
            (a, w),
        ];

        let existing_evs = [he_bc, he_ab, he_cd, he_da];

        let base_face_idx = self.faces.len();

        let base_half_edge_idx = self.half_edges.len();

        // let he_wb = base_half_edge_idx;
        let he_cw = base_half_edge_idx + 1;
        let he_wa = base_half_edge_idx + 2;
        // let he_bw = base_half_edge_idx + 3;
        let he_wc = base_half_edge_idx + 4;
        // let he_dw = base_half_edge_idx + 5;
        // let he_wd = base_half_edge_idx + 6;
        let he_aw = base_half_edge_idx + 7;

        let mut i = 0;
        for (from, to) in evs {
            let mut he = HalfEdge::new(to);
            he.face = Some(base_face_idx + i / 2);
            he.vertex = to;
            self.half_edges.push(he);
            self.edge_map.insert((from, to), base_half_edge_idx + i);
            i += 1;
        }

        for i in 0..4 {
            let base_he_idx = base_half_edge_idx + i * 2;
            let edge_indices = [base_he_idx, existing_evs[i], base_he_idx + 1];
            self.half_edges[edge_indices[0]].next = edge_indices[1];
            self.half_edges[edge_indices[0]].prev = edge_indices[2];
            self.half_edges[edge_indices[1]].next = edge_indices[2];
            self.half_edges[edge_indices[1]].prev = edge_indices[0];
            self.half_edges[edge_indices[2]].next = edge_indices[0];
            self.half_edges[edge_indices[2]].prev = edge_indices[1];

            self.faces.push(Face::new(edge_indices[0]));
            self.faces[base_face_idx + i].half_edge = edge_indices[0];
        }

        let twins = [
            (0, 3), // w -> b | b -> w
            (1, 4), // c -> w | w -> c
            (2, 7), // w -> a | a -> w
            (5, 6), // d -> w | w -> d
        ];

        for i in 0..4 {
            self.half_edges[base_half_edge_idx + twins[i].0].twin = base_half_edge_idx + twins[i].1;
            self.half_edges[base_half_edge_idx + twins[i].1].twin = base_half_edge_idx + twins[i].0;
        }

        // Update vertices half-edges
        self.vertices[w].half_edge = Some(base_half_edge_idx); // w -> b
        self.vertices[a].half_edge = Some(he_ab); // a -> b
        self.vertices[b].half_edge = Some(he_bc); // b -> c
        self.vertices[c].half_edge = Some(he_cd); // c -> d
        self.vertices[d].half_edge = Some(he_da); // d -> a

        // Remove old edges from edge map
        self.edge_map.remove(&(c, a));
        self.edge_map.remove(&(a, c));

        // External twins (unchanged neighboring faces)
        self.half_edges[ex_he_ba].twin = he_ab; // external b->a | a->b
        self.half_edges[he_ab].twin = ex_he_ba;

        self.half_edges[ex_he_cb].twin = he_bc; // external c->b | b->c
        self.half_edges[he_bc].twin = ex_he_cb;

        self.half_edges[ex_he_dc].twin = he_cd; // external d->c | c->d
        self.half_edges[he_cd].twin = ex_he_dc;

        self.half_edges[ex_he_ad].twin = he_da; // external a->d | d->a
        self.half_edges[he_da].twin = ex_he_ad;

        let triangle_wbc = Triangle {
            face_idx: base_face_idx,
            vertices: [w, b, c],
        };
        let triangle_wab = Triangle {
            face_idx: base_face_idx + 1,
            vertices: [w, a, b],
        };
        let triangle_wcd = Triangle {
            face_idx: base_face_idx + 2,
            vertices: [w, c, d],
        };
        let triangle_wda = Triangle {
            face_idx: base_face_idx + 3,
            vertices: [w, d, a],
        };

        // Mark old faces as removed
        self.faces[original_face_1].removed = true;
        self.faces[original_face_2].removed = true;

        // Mark old half-edges as removed
        self.half_edges[he_ca].removed = true;
        self.half_edges[he_ac].removed = true;

        self.half_edges[he_bc].face = Some(base_face_idx);
        self.half_edges[he_ab].face = Some(base_face_idx + 1);
        self.half_edges[he_cd].face = Some(base_face_idx + 2);
        self.half_edges[he_da].face = Some(base_face_idx + 3);

        // Connect the new half-edges to their respective vertices
        self.half_edges[he_bc].vertex = c; // b -> c
        self.half_edges[he_ab].vertex = b; // a -> b
        self.half_edges[he_cd].vertex = d; // c -> d
        self.half_edges[he_da].vertex = a; // d -> a

        self.half_edge_split_map.insert(he_ca, (he_cw, he_wa));
        self.half_edge_split_map.insert(he_ac, (he_aw, he_wc));

        let face_split = FaceSplitMap {
            face: original_face_1,
            new_faces: smallvec![triangle_wbc, triangle_wab],
        };
        self.face_split_map.insert(original_face_1, face_split);

        let face_split = FaceSplitMap {
            face: original_face_2,
            new_faces: smallvec![triangle_wcd, triangle_wda],
        };
        self.face_split_map.insert(original_face_2, face_split);

        let split_result = SplitResult {
            kind: SplitResultKind::SplitEdge,
            vertex: w,
            new_faces: [
                base_face_idx,
                base_face_idx + 1,
                base_face_idx + 2,
                base_face_idx + 3,
            ],
        };

        for &new_face_idx in &split_result.new_faces {
            let face_aabb = self.face_aabb(new_face_idx);
            aabb_tree.insert(face_aabb, new_face_idx);
        }

        aabb_tree.invalidate(&original_face_1);
        aabb_tree.invalidate(&original_face_2);

        Ok(split_result)
    }

    pub fn validate_connectivity(&self) {
        // For every half-edge, check next/prev/twin consistency:
        for (i, he) in self.half_edges.iter().enumerate() {
            if he.removed {
                continue; // skip removed half-edges
            }
            assert_eq!(
                self.half_edges[he.next].prev, i,
                "he {} next -> prev mismatch",
                i
            );
            assert_eq!(
                self.half_edges[he.prev].next, i,
                "he {} prev -> next mismatch",
                i
            );
            assert_eq!(
                self.half_edges[he.twin].twin, i,
                "he {} twin -> twin mismatch",
                i
            );

            // face must match the one it's stored on:
            if let Some(f) = he.face {
                // check that f really contains i somewhere in its cycle…
                let mut cur = self.faces[f].half_edge;
                let mut found = false;
                loop {
                    if cur == i {
                        found = true;
                        break;
                    }
                    cur = self.half_edges[cur].next;
                    assert!(cur != self.faces[f].half_edge, "he {} not in face {}", i, f);
                }
            }
        }

        let mut edge_set = HashSet::new();
        for (i, he) in self.half_edges.iter().enumerate() {
            if he.removed {
                continue;
            }
            let src = self.half_edges[he.prev].vertex;
            let dst = he.vertex;
            assert!(
                edge_set.insert((src, dst)),
                "duplicate half-edge ({},{})",
                src,
                dst
            );
        }

        // Check every face's entry half_edge still belongs to that face:
        for (fi, face) in self.faces.iter().enumerate() {
            if face.removed {
                continue;
            }
            let start = face.half_edge;
            let mut cur = start;
            loop {
                assert_eq!(
                    self.half_edges[cur].face,
                    Some(fi),
                    "face {} half-edge {} points at wrong face",
                    fi,
                    cur
                );
                cur = self.half_edges[cur].next;
                if cur == start {
                    break;
                }
            }
        }

        // Check every vertex's half_edge really points at a target half-edge:
        for (vi, v) in self.vertices.iter().enumerate() {
            if let Some(he0) = v.half_edge {
                let prev = self.half_edges[he0].prev;
                let prev_he = &self.half_edges[prev];
                assert_eq!(
                    prev_he.vertex, // the 'to' of the previous edge is 'from' of this one
                    vi,             // should match the current vertex index
                    "vertex {}: half_edge {} is not outgoing from this vertex (prev edge points to {})",
                    vi,
                    he0,
                    prev_he.vertex
                );
            }
        }
    }

    /// Splits the interior edge `he` by inserting a new vertex at `pos`.
    /// The two adjacent triangles are each subdivided into two, yielding
    /// four faces in place of the original two.  Returns the new vertex index.
    pub fn split_edge_rebuild(
        &mut self,
        he: usize,
        pos: Point<T, N>,
    ) -> Result<usize, &'static str> {
        // 1) Pre-flight checks
        let he_twin = self.half_edges[he].twin;
        if he_twin == usize::MAX {
            return Err("cannot split a boundary edge");
        }
        let f0 = self.half_edges[he].face.ok_or("he has no face")?;
        let f1 = self.half_edges[he_twin].face.ok_or("twin has no face")?;

        // 2) Gather old vertex positions
        let mut old_positions: Vec<Point<T, N>> =
            self.vertices.iter().map(|v| v.position.clone()).collect();
        // the new vertex gets the next index in that list
        let new_old_idx = old_positions.len();
        old_positions.push(pos.clone());

        // 3) Identify u, v (edge endpoints)
        let u = {
            // the half-edge before `he` ends at u
            let prev = self.half_edges[he].prev;
            self.half_edges[prev].vertex
        };
        let v = self.half_edges[he].vertex;

        // 4) Build the new face list
        let mut new_face_tris: Vec<[usize; 3]> = Vec::with_capacity(self.faces.len() + 2);

        for fid in 0..self.faces.len() {
            if fid == f0 || fid == f1 {
                // subdivide this face
                let vs = self.face_vertices(fid); // CCW triple
                // find whether the edge appears as u->v or v->u
                let mut handled = false;
                for i in 0..3 {
                    let a = vs[i];
                    let b = vs[(i + 1) % 3];
                    let c = vs[(i + 2) % 3];
                    if a == u && b == v {
                        // orientation u->v->c
                        new_face_tris.push([u, new_old_idx, c]);
                        new_face_tris.push([new_old_idx, v, c]);
                        handled = true;
                        break;
                    }
                    if a == v && b == u {
                        // orientation v->u->c
                        new_face_tris.push([v, new_old_idx, c]);
                        new_face_tris.push([new_old_idx, u, c]);
                        handled = true;
                        break;
                    }
                }
                if !handled {
                    return Err("split edge not found in one of its faces");
                }
            } else {
                // keep an untouched face
                let vs = self.face_vertices(fid);
                new_face_tris.push([vs[0], vs[1], vs[2]]);
            }
        }

        // 5) Rebuild the mesh from scratch
        let mut new_mesh = Mesh::new();
        // re-add all vertices
        for p in old_positions {
            new_mesh.add_vertex(p);
        }
        // re-add all faces
        for tri in new_face_tris {
            new_mesh.add_triangle(tri[0], tri[1], tri[2]);
        }
        // re-generate boundary & twin links
        new_mesh.build_boundary_loops();

        // 6) Replace self and return the new-vertex index
        *self = new_mesh;
        Ok(new_old_idx)
    }

    pub fn faces_around_face(&self, face: usize) -> [usize; 3] {
        let mut result = [usize::MAX; 3];
        let mut current_he_idx = self.faces[face].half_edge;
        if current_he_idx == usize::MAX {
            panic!("Face has no half-edge.");
        }
        let starting_he_idx = current_he_idx;

        let mut i = 0;
        loop {
            let current_he = &self.half_edges[current_he_idx];
            if let Some(face_idx) = current_he.face
                && !self.faces[face_idx].removed
            {
                result[i] = face_idx; // valid half_edges should always point to valid faces
                i += 1;
                if i == 3 {
                    break; // we only need 3 faces
                }
            }

            let twin_idx = current_he.twin;
            let next_idx = self.half_edges[twin_idx].next;

            if next_idx == starting_he_idx {
                break;
            }

            current_he_idx = next_idx;
        }

        result
    }

    pub fn faces_around_vertex(&self, vertex: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut current_he_idx = self.vertices[vertex]
            .half_edge
            .expect("Vertex has no half-edge.");
        let starting_he_idx = current_he_idx;

        loop {
            let current_he = &self.half_edges[current_he_idx];
            if let Some(face_idx) = current_he.face {
                if !self.faces[face_idx].null && !self.faces[face_idx].removed {
                    result.push(face_idx); // valid half_edges should always point to valid faces
                }
            }

            let twin_idx = current_he.twin;
            let next_idx = self.half_edges[twin_idx].next;

            if next_idx == starting_he_idx {
                break;
            }

            current_he_idx = next_idx;
        }

        result
    }

    pub fn get_first_half_edge_intersection_on_face(
        &self,
        face: usize,
        from: &Point<T, N>,
        direction: &Vector<T, N>,
    ) -> Option<(usize, T, T)>
    where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>
            + Neg<Output = T>,
    {
        if self.faces[face].removed {
            panic!("Cannot find intersection on a removed face");
        }
        let mut closest_he = None;
        let mut closest_t = None;
        let mut closest_u = None;
        let plane = self.plane_from_face(face);
        let (u, v) = plane.basis(); // u, v lie in the plane
        let origin = plane.origin(); // any point on the face

        let origin_n =
            Point::<T, N>::from_vals(from_fn(
                |i| {
                    if i < N { origin[i].clone() } else { T::zero() }
                },
            ));

        let from_2d = {
            let d = from.as_vector() - origin_n.as_vector();
            Point::<T, 2>::new([d.dot(&u), d.dot(&v)])
        };

        let dir_2d = {
            let d = &direction;
            Vector::<T, 2>::new([d.dot(&u), d.dot(&v)])
        };

        // Get triangle half-edges
        let hes = self.face_half_edges(face);

        if self.half_edges[hes[0]].removed
            || self.half_edges[hes[1]].removed
            || self.half_edges[hes[2]].removed
        {
            panic!("Cannot find intersection on a face with removed half-edges");
        }

        let pts: [Point<T, 2>; 3] = hes
            .iter()
            .map(|&he_idx| {
                let p3d = &self.vertices[self.half_edges[he_idx].vertex].position;
                let d = (p3d - &origin_n).as_vector();
                Point::new([d.dot(&u), d.dot(&v)])
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        for i in 0..3 {
            let he_idx = hes[i];
            let a = &pts[(i + 2) % 3];
            let b = &pts[i];
            if let Some((t, u)) = ray_segment_intersection_2d(&from_2d, &dir_2d, a, b) {
                if t.is_positive() && (closest_t.is_none() || &t < &closest_t.clone().unwrap()) {
                    closest_he = Some(he_idx);
                    closest_t = Some(t);
                    closest_u = Some(u);
                }
            }
        }

        if let Some(he) = closest_he {
            return Some((he, closest_t.unwrap(), closest_u.unwrap()));
        }
        None
    }

    // Given a ray that will be created from `from` and `direction`, this function finds the first intersection
    // with a half-edge that is in the opposite side of `start_vertex` on one of its connecting faces.
    pub fn get_first_half_edge_intersection(
        &mut self,
        start_vertex: usize,
        from: &Point<T, N>,
        direction: &Vector<T, N>,
    ) -> (usize, T, T)
    where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>
            + Neg<Output = T>,
    {
        let mut closest_he = None;
        let mut closest_t = None;
        let mut closest_u = None;

        // First, let's find all faces that contain the start_vertex
        let faces = self.faces_around_vertex(start_vertex);
        for f in faces {
            if let Some((he, t, u)) =
                self.get_first_half_edge_intersection_on_face(f, from, direction)
            {
                if t.is_positive() && (closest_t.is_none() || &t < &closest_t.clone().unwrap()) {
                    closest_he = Some(he);
                    closest_t = Some(t);
                    closest_u = Some(u);
                }
            }
        }

        // let closest_he_q = self.vertices[self.half_edges[closest_he.unwrap()].vertex]
        //     .position
        //     .clone();
        // let closest_he_p = self.vertices[self.half_edges
        //     [self.find_valid_half_edge(self.half_edges[closest_he.unwrap()].twin)]
        // .vertex]
        //     .position
        //     .clone();

        // let t = closest_t.unwrap();

        // let hit = from + &direction.scale(&t).0;
        // let hit_edge_vector = (&closest_he_q - &closest_he_p).as_vector();
        // let relative_vector = &hit.as_vector() - &closest_he_p.as_vector();

        // let u = relative_vector.dot(&hit_edge_vector) / hit_edge_vector.dot(&hit_edge_vector);

        match (closest_he, closest_t, closest_u) {
            (Some(he), Some(t), Some(u)) => (he, t, u),
            _ => panic!("Ray did not intersect any edge from start vertex"),
        }
    }

    /// Method 1: Direct 3D parametric intersection
    fn parametric_line_intersection_3d(
        &self,
        p: &Point<T, N>,
        q: &Point<T, N>,
        a: &Point<T, N>,
        b: &Point<T, N>,
    ) -> Option<(Point<T, N>, T)>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let d1 = (q - p).as_vector();
        let d2 = (b - a).as_vector();
        let r = (a - p).as_vector();

        // Solve: p + t*d1 = a + s*d2
        // This gives us: t*d1 - s*d2 = r

        let d1_dot_d1 = d1.dot(&d1);
        let d1_dot_d2 = d1.dot(&d2);
        let d2_dot_d2 = d2.dot(&d2);
        let d1_dot_r = d1.dot(&r);
        let d2_dot_r = d2.dot(&r);

        let denom = &d1_dot_d1 * &d2_dot_d2 - &d1_dot_d2 * &d1_dot_d2;
        let eps = T::tolerance();

        if &denom.abs() < &eps {
            return None; // Lines are parallel
        }

        let t = (&d2_dot_d2 * &d1_dot_r - &d1_dot_d2 * &d2_dot_r) / denom.clone();
        let s = (&d1_dot_d1 * &d2_dot_r - &d1_dot_d2 * &d1_dot_r) / denom;

        // Check if intersection is within both line segments
        let eps_bounds = T::tolerance();
        if t >= -eps_bounds.clone()
            && t <= T::one() + eps_bounds.clone()
            && s >= -eps_bounds.clone()
            && s <= T::one() + eps_bounds
        {
            let intersection = p + &d1.scale(&t).0;
            return Some((intersection, t));
        }

        None
    }

    /// Method 2: Projection-based intersection for a specific axis
    fn projected_intersection_2d(
        &self,
        p: &Point<T, N>,
        q: &Point<T, N>,
        a: &Point<T, N>,
        b: &Point<T, N>,
        drop_axis: usize,
    ) -> Option<(Point<T, N>, T)>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        // Choose two axes to keep (drop the specified axis)
        let (i0, i1) = match drop_axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };

        // Project to 2D
        let p2 = Point::<T, 2>::from_vals([p.coords()[i0].clone(), p.coords()[i1].clone()]);
        let q2 = Point::<T, 2>::from_vals([q.coords()[i0].clone(), q.coords()[i1].clone()]);
        let a2 = Point::<T, 2>::from_vals([a.coords()[i0].clone(), a.coords()[i1].clone()]);
        let b2 = Point::<T, 2>::from_vals([b.coords()[i0].clone(), b.coords()[i1].clone()]);

        // **CHECK FOR DEGENERATE PROJECTIONS FIRST**
        let segment_2d_len_sq =
            (&q2[0] - &p2[0]) * (&q2[0] - &p2[0]) + (&q2[1] - &p2[1]) * (&q2[1] - &p2[1]);
        let edge_2d_len_sq =
            (&b2[0] - &a2[0]) * (&b2[0] - &a2[0]) + (&b2[1] - &a2[1]) * (&b2[1] - &a2[1]);

        if segment_2d_len_sq < T::tolerance() || edge_2d_len_sq < T::tolerance() {
            return None; // Degenerate projection
        }

        // **USE PARAMETRIC 2D INTERSECTION (NOT segment_intersect_2d)**
        let dir_segment = [&q2[0] - &p2[0], &q2[1] - &p2[1]];
        let dir_edge = [&b2[0] - &a2[0], &b2[1] - &a2[1]];

        // Solve: p2 + t*dir_segment = a2 + s*dir_edge
        let denom = &dir_segment[0] * &dir_edge[1] - &dir_segment[1] * &dir_edge[0];

        if denom.abs() <= T::tolerance() {
            return None; // Parallel in 2D
        }

        let diff = [&a2[0] - &p2[0], &a2[1] - &p2[1]];
        let t = (&diff[0] * &dir_edge[1] - &diff[1] * &dir_edge[0]) / denom.clone();
        let s = (&diff[0] * &dir_segment[1] - &diff[1] * &dir_segment[0]) / denom;

        // Check bounds
        let eps = T::tolerance();
        if t >= -eps.clone()
            && t <= T::one() + eps.clone()
            && s >= -eps.clone()
            && s <= T::one() + eps
        {
            // Lift back to 3D by interpolating along the original segment
            let dir_3d = (q - p).as_vector();
            let intersection_3d = p + &dir_3d.scale(&t).0;
            return Some((intersection_3d, t));
        }

        None
    }

    /// Method 3: Closest approach between line segments
    fn closest_approach_intersection(
        &self,
        p: &Point<T, N>,
        q: &Point<T, N>,
        a: &Point<T, N>,
        b: &Point<T, N>,
    ) -> Option<(Point<T, N>, T)>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let d1 = (q - p).as_vector();
        let d2 = (b - a).as_vector();
        let r = (p - a).as_vector();

        let a_val = d1.dot(&d1);
        let b_val = d1.dot(&d2);
        let c_val = d2.dot(&d2);
        let d_val = d1.dot(&r);
        let e_val = d2.dot(&r);

        let denom = &a_val * &c_val - &b_val * &b_val;
        let eps = T::tolerance();

        if denom.abs() < eps {
            return None;
        }

        let t = (&b_val * &e_val - &c_val * &d_val) / denom.clone();
        let s = (&a_val * &e_val - &b_val * &d_val) / denom;

        // Check if closest approach is within reasonable bounds and distance
        if t >= T::from(-0.1) && t <= T::from(1.1) && s >= T::from(-0.1) && s <= T::from(1.1) {
            let point1 = p + &d1.scale(&t).0;
            let point2 = a + &d2.scale(&s).0;
            let distance = point1.distance_to(&point2);

            if distance < T::tolerance() {
                // Lines are close enough to be considered intersecting
                let intersection = p + &d1.scale(&t.clone().max(T::zero()).min(T::one())).0;
                let clamped_t = &t.max(T::zero()).min(T::one());
                return Some((intersection, clamped_t.clone()));
            }
        }

        None
    }

    pub fn split_face(
        &mut self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        face: usize,
        p: &Point<T, N>,
    ) -> Option<SplitResult>
    where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face = self.find_valid_face(face, p);
        if self.faces[face].removed {
            panic!("Cannot find or insert vertex on a removed face");
        }

        let he_ab = self.find_valid_half_edge(self.faces[face].half_edge, &p);
        let he_bc = self.half_edges[he_ab].next;
        let he_ca = self.half_edges[he_bc].next;

        let w = self.add_vertex(p.clone());

        let a = self.half_edges[he_ca].vertex;
        let b = self.half_edges[he_ab].vertex;
        let c = self.half_edges[he_bc].vertex;

        // if self.half_edges[self.vertices[vs[0]].half_edge.unwrap()]
        //     .face
        //     .unwrap()
        //     != face
        // {
        //     panic!("Face half-edge does not match the face");
        // }
        // if self.half_edges[self.vertices[vs[1]].half_edge.unwrap()]
        //     .face
        //     .unwrap()
        //     != face
        // {
        //     panic!("Face half-edge does not match the face");
        // }
        // if self.half_edges[self.vertices[vs[2]].half_edge.unwrap()]
        //     .face
        //     .unwrap()
        //     != face
        // {
        //     panic!("Face half-edge does not match the face");
        // }

        let subface_1_verts = [a, w, c];
        let subface_2_verts = [c, w, b];
        let subface_3_verts = [b, w, a];

        let subface_1_idx = self.faces.len();
        let subface_2_idx = self.faces.len() + 1;
        let subface_3_idx = self.faces.len() + 2;

        // Create subface 1
        let base_he_idx_1 = self.half_edges.len();
        let edge_vertices_1 = [
            (subface_1_verts[0], subface_1_verts[1]), // a -> w
            (subface_1_verts[1], subface_1_verts[2]), // w -> c
                                                      // c -> a
        ];

        // Create half-edges for face 1
        for (_i, &(_from, to)) in edge_vertices_1.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(subface_1_idx);
            self.half_edges.push(he);
        }

        // Link face 1 half-edges
        let edge_indices_1 = [base_he_idx_1, base_he_idx_1 + 1, he_ca];
        self.half_edges[edge_indices_1[0]].next = edge_indices_1[1];
        self.half_edges[edge_indices_1[0]].prev = edge_indices_1[2];
        self.half_edges[edge_indices_1[1]].next = edge_indices_1[2];
        self.half_edges[edge_indices_1[1]].prev = edge_indices_1[0];
        self.half_edges[edge_indices_1[2]].next = edge_indices_1[0];
        self.half_edges[edge_indices_1[2]].prev = edge_indices_1[1];

        // Create face 2
        let base_he_idx_2 = self.half_edges.len();
        let edge_vertices_2 = [
            (subface_2_verts[0], subface_2_verts[1]), // c -> w
            (subface_2_verts[1], subface_2_verts[2]), // w -> b
                                                      // b -> c
        ];

        // Create half-edges for face 2
        for (_i, &(_from, to)) in edge_vertices_2.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(subface_2_idx);
            self.half_edges.push(he);
        }

        // Link face 2 half-edges
        let edge_indices_2 = [base_he_idx_2, base_he_idx_2 + 1, he_bc];
        self.half_edges[edge_indices_2[0]].next = edge_indices_2[1];
        self.half_edges[edge_indices_2[0]].prev = edge_indices_2[2];
        self.half_edges[edge_indices_2[1]].next = edge_indices_2[2];
        self.half_edges[edge_indices_2[1]].prev = edge_indices_2[0];
        self.half_edges[edge_indices_2[2]].next = edge_indices_2[0];
        self.half_edges[edge_indices_2[2]].prev = edge_indices_2[1];

        // Create face 3
        let base_he_idx_3 = self.half_edges.len();
        let edge_vertices_3 = [
            (subface_3_verts[0], subface_3_verts[1]), // b -> w
            (subface_3_verts[1], subface_3_verts[2]), // w -> a
                                                      // a -> b
        ];

        // Create half-edges for face 3
        for (_i, &(_from, to)) in edge_vertices_3.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(subface_3_idx);
            self.half_edges.push(he);
        }

        // Link face 3 half-edges
        let edge_indices_3 = [base_he_idx_3, base_he_idx_3 + 1, he_ab];
        self.half_edges[edge_indices_3[0]].next = edge_indices_3[1];
        self.half_edges[edge_indices_3[0]].prev = edge_indices_3[2];
        self.half_edges[edge_indices_3[1]].next = edge_indices_3[2];
        self.half_edges[edge_indices_3[1]].prev = edge_indices_3[0];
        self.half_edges[edge_indices_3[2]].next = edge_indices_3[0];
        self.half_edges[edge_indices_3[2]].prev = edge_indices_3[1];

        self.faces.push(Face::new(edge_indices_1[0]));
        self.faces.push(Face::new(edge_indices_2[0]));
        self.faces.push(Face::new(edge_indices_3[0]));

        self.faces[face].removed = true;

        // self.half_edges[base_he_idx_1].face = Some(subface_1_idx);
        // self.half_edges[base_he_idx_1 + 1].face = Some(subface_1_idx);
        self.half_edges[he_ca].face = Some(subface_1_idx);

        // self.half_edges[base_he_idx_2].face = Some(subface_2_idx);
        // self.half_edges[base_he_idx_2 + 1].face = Some(subface_2_idx);
        self.half_edges[he_bc].face = Some(subface_2_idx);

        // self.half_edges[base_he_idx_3].face = Some(subface_3_idx);
        // self.half_edges[base_he_idx_3 + 1].face = Some(subface_3_idx);
        self.half_edges[he_ab].face = Some(subface_3_idx);

        self.vertices[w].half_edge = Some(base_he_idx_2 + 1); // w -> b
        self.vertices[a].half_edge = Some(he_ab);
        self.vertices[b].half_edge = Some(he_bc);
        self.vertices[c].half_edge = Some(he_ca);

        self.half_edges[base_he_idx_1].vertex = w; // a -> w
        self.half_edges[base_he_idx_1 + 1].vertex = c; // w -> c

        self.half_edges[base_he_idx_2].vertex = w; // c -> w
        self.half_edges[base_he_idx_2 + 1].vertex = b; // w -> b

        self.half_edges[base_he_idx_3].vertex = w; // b -> w
        self.half_edges[base_he_idx_3 + 1].vertex = a; // w -> a

        self.faces[subface_1_idx].half_edge = base_he_idx_1;
        self.faces[subface_2_idx].half_edge = base_he_idx_2;
        self.faces[subface_3_idx].half_edge = base_he_idx_3;

        // internal twins
        // w -> a and a -> w, respectively
        self.half_edges[base_he_idx_3 + 1].twin = base_he_idx_1;
        self.half_edges[base_he_idx_1].twin = base_he_idx_3 + 1;

        // w -> c and c -> w, respectively
        self.half_edges[base_he_idx_1 + 1].twin = base_he_idx_2;
        self.half_edges[base_he_idx_2].twin = base_he_idx_1 + 1;

        // w -> b and b -> w, respectively
        self.half_edges[base_he_idx_2 + 1].twin = base_he_idx_3;
        self.half_edges[base_he_idx_3].twin = base_he_idx_2 + 1;

        self.edge_map.insert((a, w), base_he_idx_1);
        self.edge_map.insert((w, c), base_he_idx_1 + 1);

        self.edge_map.insert((c, w), base_he_idx_2);
        self.edge_map.insert((w, b), base_he_idx_2 + 1);

        self.edge_map.insert((b, w), base_he_idx_3);
        self.edge_map.insert((w, a), base_he_idx_3 + 1);

        let triangle_awc = Triangle {
            face_idx: subface_1_idx,
            vertices: [a, w, c],
        };
        let triangle_cwb = Triangle {
            face_idx: subface_2_idx,
            vertices: [c, w, b],
        };
        let triangle_bwa = Triangle {
            face_idx: subface_3_idx,
            vertices: [b, w, a],
        };

        let face_split = FaceSplitMap {
            face: face,
            new_faces: smallvec![triangle_awc, triangle_cwb, triangle_bwa],
        };
        self.face_split_map.insert(face, face_split);

        let split_result = SplitResult {
            kind: SplitResultKind::SplitFace,
            vertex: w,
            new_faces: [subface_1_idx, subface_2_idx, subface_3_idx, usize::MAX],
        };

        for i in 0..3 {
            let face_aabb = self.face_aabb(split_result.new_faces[i]);
            aabb_tree.insert(face_aabb, split_result.new_faces[i]);
        }

        aabb_tree.invalidate(&face);

        Some(split_result)
    }

    pub fn half_edge_between(&self, vi0: usize, vi1: usize) -> Option<usize> {
        // Check edge_map for direct connection
        if let Some(&he_idx) = self.edge_map.get(&(vi0, vi1)) {
            return Some(he_idx);
        }
        // Also check the reverse direction (for undirected edge)
        if let Some(&he_idx) = self.edge_map.get(&(vi1, vi0)) {
            return Some(he_idx);
        }
        None
    }

    pub fn point_is_on_some_half_edge(&self, face: usize, point: &Point<T, N>) -> Option<(usize, T)>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        if self.faces[face].removed {
            return None;
        }
        // Check if point is on any edge of the face
        for &he in &self.face_half_edges(face) {
            let src = self.half_edges[self.half_edges[he].prev].vertex;
            let dst = self.half_edges[he].vertex;
            let ps = &self.vertices[src].position;
            let pd = &self.vertices[dst].position;
            // Is point on segment [ps, pd]?
            if let Some(u) = point_position_on_segment(ps, pd, point) {
                return Some((he, u));
            }
        }
        None
    }

    pub fn find_valid_face(&self, face_idx: usize, point: &Point<T, N>) -> usize
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        // base case: if this face is still around, we’re done
        if !self.faces[face_idx].removed {
            return face_idx;
        }

        // if it was split, try each child
        if let Some(mapping) = self.face_split_map.get(&face_idx) {
            for tri in &mapping.new_faces {
                let [i0, i1, i2] = tri.vertices;
                if point_in_or_on_triangle(
                    point,
                    &self.vertices[i0].position,
                    &self.vertices[i1].position,
                    &self.vertices[i2].position,
                ) {
                    // recurse into that sub-face
                    return self.find_valid_face(tri.face_idx, point);
                }
            }
        }

        panic!(
            "find_valid_face: no child triangle contains point {:?}",
            point
        );
    }

    pub fn find_valid_half_edge(&self, mut he_idx: usize, point_hint: &Point<T, N>) -> usize
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        if !self.half_edges[he_idx].removed {
            return he_idx;
        }
        loop {
            // see if this half-edge was split
            if let Some(&(child_a, child_b)) = self.half_edge_split_map.get(&he_idx) {
                // println!(
                //     "find_valid_half_edge: Half-edge {} was split into {} and {}",
                //     he_idx, child_a, child_b
                // );
                // test child A
                if child_a != usize::MAX {
                    let target = &self.vertices[self.half_edges[child_a].vertex].position;
                    let twin_he = self.find_valid_half_edge(self.half_edges[child_a].twin, target);
                    let origin = &self.vertices[self.half_edges[twin_he].vertex].position;
                    if point_position_on_segment(origin, point_hint, target).is_some() {
                        // println!("A find_valid_half_edge: Found valid half-edge {}", child_a);
                        he_idx = child_a;
                        continue;
                    }
                }
                // test child B
                if child_b != usize::MAX {
                    let target = &self.vertices[self.half_edges[child_b].vertex].position;
                    let twin_he = self.find_valid_half_edge(self.half_edges[child_b].twin, target);
                    let origin = &self.vertices[self.half_edges[twin_he].vertex].position;
                    if point_position_on_segment(origin, point_hint, target).is_some() {
                        // println!("B find_valid_half_edge: Found valid half-edge {}", child_b);
                        he_idx = child_b;
                        continue;
                    }
                }
            }
            // no split (or neither child matched), so this is our leaf
            break;
        }

        if self.half_edges[he_idx].removed {
            panic!("find_valid_half_edge: Half-edge is removed");
        }

        he_idx
    }

    fn point_in_face(&self, face_idx: usize, point: &Point<T, N>) -> bool
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face_vertices = self.face_vertices(face_idx);
        if face_vertices.len() != 3 {
            return false;
        }

        let v0 = &self.vertices[face_vertices[0]].position;
        let v1 = &self.vertices[face_vertices[1]].position;
        let v2 = &self.vertices[face_vertices[2]].position;

        point_in_or_on_triangle(point, v0, v1, v2)
    }

    pub fn barycentric_coords_on_face(
        &self,
        face_idx: usize,
        point: &Point<T, N>,
    ) -> Option<(T, T, T)>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face_vertices = self.face_vertices(face_idx);
        if face_vertices.len() != 3 {
            return None;
        }

        let v0 = &self.vertices[face_vertices[0]].position;
        let v1 = &self.vertices[face_vertices[1]].position;
        let v2 = &self.vertices[face_vertices[2]].position;

        if point_in_or_on_triangle(point, v0, v1, v2) {
            barycentric_coords(point, v0, v1, v2)
        } else {
            None
        }
    }

    pub fn filter_degenerate_segments(segments: &mut Vec<IntersectionSegment<T, N>>)
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let tolerance = T::tolerance();

        // Only filter truly degenerate segments
        segments.retain(|seg| {
            let length = seg.segment.length();
            if length < tolerance { false } else { true }
        });
    }

    /// Checks if two vertices have an edge between them in the mesh.
    pub fn are_vertices_connected(&self, vertex_a: usize, vertex_b: usize) -> bool {
        return self.edge_map.contains_key(&(vertex_a, vertex_b))
            || self.edge_map.contains_key(&(vertex_b, vertex_a));
    }

    /// Checks if two vertices have an edge between them and return the half-edge pointing to B if it exists.
    pub fn vertices_connection(&self, vertex_a: usize, vertex_b: usize) -> usize {
        if vertex_a >= self.vertices.len() || vertex_b >= self.vertices.len() {
            return usize::MAX; // Invalid vertex indices
        }
        self.edge_map
            .get(&(vertex_a, vertex_b))
            .copied()
            .unwrap_or(usize::MAX)
    }
}

fn point_in_or_on_triangle<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // barycentric coordinates
    let v0 = (c - a).as_vector();
    let v1 = (b - a).as_vector();
    let v2 = (p - a).as_vector();
    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot11 = v1.dot(&v1);
    let dot12 = v1.dot(&v2);

    let inv = &T::one() / &(&dot00 * &dot11 - &dot01 * &dot01);
    let u = &(&dot11 * &dot02 - &dot01 * &dot12) * &inv;
    let v = &(&dot00 * &dot12 - &dot01 * &dot02) * &inv;

    // allow on‐edge within small epsilon
    let e = T::tolerance();
    let neg_e = e.clone().neg();
    u >= neg_e && v >= neg_e && u + v <= T::one() + e
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexSource {
    A,
    B,
}

/// Standard squared‐distance from a point to a triangle in 3D
/// (see Christer Ericson, *Real-Time Collision Detection*)
pub fn distance_point_triangle_squared<T: Scalar>(
    p: &Point<T, 3>,
    a: &Point<T, 3>,
    b: &Point<T, 3>,
    c: &Point<T, 3>,
) -> T
where
    Point<T, 3>: PointOps<T, 3, Vector = Vector<T, 3>>,
    Vector<T, 3>: VectorOps<T, 3, Cross = Vector<T, 3>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) Compute vectors
    let ab = Vector::<T, 3>::from_vals([&b[0] - &a[0], &b[1] - &a[1], &b[2] - &a[2]]);
    let ac = Vector::<T, 3>::from_vals([&c[0] - &a[0], &c[1] - &a[1], &c[2] - &a[2]]);
    let ap = Vector::<T, 3>::from_vals([&p[0] - &a[0], &p[1] - &a[1], &p[2] - &a[2]]);

    // Face normal squared‐length
    let n = ab.cross(&ac);
    let nn2 = n.dot(&n);

    // Degenerate triangle?  (zero area)
    if nn2.is_zero() {
        return distance_point_segment_squared(p, a, b)
            .min(distance_point_segment_squared(p, b, c))
            .min(distance_point_segment_squared(p, c, a));
    }

    // 2) Compute barycentric coords to find the closest point on the *infinite plane*
    let d1 = &ab[0] * &ap[0] + &ab[1] * &ap[1] + &ab[2] * &ap[2];
    let d2 = &ac[0] * &ap[0] + &ac[1] * &ap[1] + &ac[2] * &ap[2];

    if d1.is_negative_or_zero() && d2.is_negative_or_zero() {
        return &ap[0] * &ap[0] + &ap[1] * &ap[1] + &ap[2] * &ap[2];
    }

    // 3) Check “vertex region B”
    let bp = Point::<T, 3>::from_vals([&p[0] - &b[0], &p[1] - &b[1], &p[2] - &b[2]]);
    let d3 = &ab[0] * &bp[0] + &ab[1] * &bp[1] + &ab[2] * &bp[2];
    let d4 = &ac[0] * &bp[0] + &ac[1] * &bp[1] + &ac[2] * &bp[2];
    if d3.is_positive_or_zero() && d4 <= d3 {
        return &bp[0] * &bp[0] + &bp[1] * &bp[1] + &bp[2] * &bp[2];
    }

    // 4) Edge AB?
    let vc = &d1 * &d4 - &d3 * &d2;
    if vc.is_negative_or_zero() && d1.is_positive_or_zero() && d3.is_negative_or_zero() {
        let v = &d1 / &(&d1 - &d3);
        let proj = Point::<T, 3>::from_vals([
            &a[0] + &(&v * &ab[0]),
            &a[1] + &(&v * &ab[1]),
            &a[2] + &(&v * &ab[2]),
        ]);
        let diff = Point::<T, 3>::from_vals([&p[0] - &proj[0], &p[1] - &proj[1], &p[2] - &proj[2]]);
        return &diff[0] * &diff[0] + &diff[1] * &diff[1] + &diff[2] * &diff[2];
    }

    // 5) Edge AC?
    let cp = Point::<T, 3>::from_vals([&p[0] - &c[0], &p[1] - &c[1], &p[2] - &c[2]]);
    let d5 = &ab[0] * &cp[0] + &ab[1] * &cp[1] + &ab[2] * &cp[2];
    let d6 = &ac[0] * &cp[0] + &ac[1] * &cp[1] + &ac[2] * &cp[2];
    if d6.is_positive_or_zero() && d5 <= d6 {
        let w = &d6 / &(&d6 - &d2);
        let proj = Point::<T, 3>::from_vals([
            &a[0] + &(&w * &ac[0]),
            &a[1] + &(&w * &ac[1]),
            &a[2] + &(&w * &ac[2]),
        ]);
        let diff = Point::<T, 3>::from_vals([&p[0] - &proj[0], &p[1] - &proj[1], &p[2] - &proj[2]]);
        return &diff[0] * &diff[0] + &diff[1] * &diff[1] + &diff[2] * &diff[2];
    }

    // 6) Edge BC?
    let vb = &d5 * &d2 - &d1 * &d6;
    if vb.is_negative_or_zero()
        && (&d4 - &d3).is_positive_or_zero()
        && (&d5 - &d6).is_positive_or_zero()
    {
        // parameter t along BC
        let t = (&d4 - &d3) / ((&d4 - &d3) + (&d5 - &d6));
        // B + t*(C–B)
        let proj = Point::from_vals([
            &b[0] + &(&(&c[0] - &b[0]) * &t),
            &b[1] + &(&(&c[1] - &b[1]) * &t),
            &b[2] + &(&(&c[2] - &b[2]) * &t),
        ]);
        let diff = Point::from_vals([&p[0] - &proj[0], &p[1] - &proj[1], &p[2] - &proj[2]]);
        return &diff[0] * &diff[0] + &diff[1] * &diff[1] + &diff[2] * &diff[2];
    }

    // project p onto the plane
    let t_plane = &ap.dot(&n) / &nn2;
    let proj = Point::from(&p.clone().as_vector() - &n.scale(&t_plane));

    // compute barycentrics of 'proj' in triangle {a,b,c}
    let v0 = ac;
    let v1 = ab;
    let v2 = &proj - &a;
    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot11 = v1.dot(&v1);
    let dot02 = v0.dot(&v2.as_vector());
    let dot12 = v1.dot(&v2.as_vector());
    let inv_denom = T::one() / (&dot00 * &dot11 - &dot01 * &dot01);
    let u = &(&dot11 * &dot02 - &dot01 * &dot12) * &inv_denom;
    let v = &(&dot00 * &dot12 - &dot01 * &dot02) * &inv_denom;

    if u >= T::zero() && v >= T::zero() && u + v <= T::one() {
        let d_plane = ap.dot(&n);
        return &d_plane * &d_plane / nn2;
    }

    // if we get here, that means numerical jitter kicked us out of face region
    // but we’ve already tested all three edges above, so this *shouldn’t* happen.
    // As a safe fallback, return the minimum of the three edge distances:
    distance_point_segment_squared(p, a, b)
        .min(distance_point_segment_squared(p, b, c))
        .min(distance_point_segment_squared(p, c, a))
}

pub fn point_position_on_segment<T: Scalar, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    p: &Point<T, N>,
) -> Option<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let ab = (b - a).as_vector();
    let ap = (p - a).as_vector();

    let ab_len_squared = ab.norm_squared();
    if ab_len_squared.is_zero() {
        return None; // segment is degenerate
    }

    let u = ap.dot(&ab) / ab_len_squared;

    // Reconstruct point on segment line
    let proj = ab.scale(&u);
    let rejection = &ap - &proj;

    // If rejection is non-zero, p is not on the line
    if rejection.norm_squared().is_zero()
        && u.is_positive_or_zero()
        && (&u - &T::one()).is_negative_or_zero()
    {
        Some(u)
    } else {
        None
    }
}

fn point_on_segment<T: Scalar, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    p: &Point<T, N>,
) -> Option<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Use direct point-to-point distances (more reliable)
    let edge_length = a.distance_to(b);
    let split_distance = a.distance_to(p);
    let distance_from_end = b.distance_to(p);

    // Early exit for degenerate edge
    if edge_length <= T::edge_degeneracy_threshold() {
        return if a.distance_to(p).is_zero() {
            Some(T::zero())
        } else if b.distance_to(p).is_zero() {
            Some(T::one())
        } else {
            None
        };
    }

    // Parametric check
    let ab = b - a;
    let ap = p - a;
    let ab_dot_ab = ab.as_vector().dot(&ab.as_vector());
    let ab_dot_ap = ab.as_vector().dot(&ap.as_vector());

    if ab_dot_ab == T::zero() {
        return if a.distance_to(p).is_zero() {
            Some(T::zero())
        } else {
            None
        };
    }

    let t = ab_dot_ap / ab_dot_ab;
    if t.is_negative() || t > T::one() {
        return None;
    }

    let point_threshold = T::point_merge_threshold();

    // Check if point is at vertices
    if split_distance <= point_threshold {
        return Some(T::zero()); // Point is at vertex a
    }
    if distance_from_end <= point_threshold {
        return Some(T::one()); // Point is at vertex b
    }

    // Check if edge split would be valid
    if !should_split_edge(&edge_length, &split_distance) {
        return None;
    }

    // Verify point is actually on the line segment
    let closest = a + &ab.as_vector().scale(&t).0;
    if closest.distance_to(p) <= T::point_merge_threshold() {
        Some(t)
    } else {
        None
    }
}

fn distance_point_segment_squared<T: Scalar>(p: &Point<T, 3>, a: &Point<T, 3>, b: &Point<T, 3>) -> T
where
    Point<T, 3>: PointOps<T, 3, Vector = Vector<T, 3>>,
    Vector<T, 3>: VectorOps<T, 3>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let ab = b - a;
    let mut t = (p - a).as_vector().dot(&ab.as_vector()) / ab.as_vector().dot(&ab.as_vector());
    //.clamp(T::zero(), T::one());

    if t.is_negative() {
        t = T::zero();
    } else if t > T::one() {
        t = T::one();
    }

    let ab_by_t = ab.as_vector().scale(&t);

    let proj = Point::from(&a.as_vector() + &ab_by_t);
    (p - &proj).as_vector().dot(&(p - &proj).as_vector())
}

fn is_closed_loop<T: Scalar, const N: usize>(
    loop_segments: &Vec<usize>,
    segments: &Vec<IntersectionSegment<T, N>>,
) -> bool
where
    Point<T, N>: PointOps<T, N>,
{
    if loop_segments.len() < 3 {
        return false;
    }

    let tolerance = T::tolerance();
    let first_seg = &segments[loop_segments[0]];
    let last_seg = &segments[*loop_segments.last().unwrap()];

    // Check if the loop forms a closed cycle
    let first_start = &first_seg.segment.a;
    let first_end = &first_seg.segment.b;
    let last_start = &last_seg.segment.a;
    let last_end = &last_seg.segment.b;

    // Check various connection patterns
    (first_start.distance_to(last_end) < tolerance)
        || (first_start.distance_to(last_start) < tolerance)
        || (first_end.distance_to(last_start) < tolerance)
        || (first_end.distance_to(last_end) < tolerance)
}

/// Helper function with improved error handling
fn get_face_vertices_safe<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_idx: usize,
) -> Option<Vec<usize>> {
    // Use existing safe method
    mesh.face_vertices_safe(face_idx)
}

/// Find shared face using iterative approach (no recursion)
fn find_shared_face_iterative<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    vertex_a: usize,
    vertex_b: usize,
) -> Option<usize> {
    for (face_idx, _face) in mesh.faces.iter().enumerate() {
        // Use safe vertex extraction with timeout
        if let Some(face_vertices) = get_face_vertices_with_timeout(mesh, face_idx, 100) {
            if face_vertices.contains(&vertex_a) && face_vertices.contains(&vertex_b) {
                return Some(face_idx);
            }
        }
    }
    None
}

/// Safe face vertex extraction with iteration limit
fn get_face_vertices_with_timeout<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_idx: usize,
    max_iterations: usize,
) -> Option<Vec<usize>> {
    if face_idx >= mesh.faces.len() {
        return None;
    }

    let face = &mesh.faces[face_idx];
    if face.removed || face.half_edge >= mesh.half_edges.len() {
        return None;
    }

    let mut result = Vec::new();
    let start_he = face.half_edge;
    let mut current_he = start_he;
    let mut iterations = 0;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            // println!("WARNING: Face traversal timeout for face {}", face_idx);
            return None;
        }

        if current_he >= mesh.half_edges.len() || mesh.half_edges[current_he].removed {
            return None;
        }

        result.push(mesh.half_edges[current_he].vertex);

        current_he = mesh.half_edges[current_he].next;
        if current_he >= mesh.half_edges.len() || mesh.half_edges[current_he].removed {
            return None;
        }

        if current_he == start_he {
            break;
        }
    }

    Some(result)
}

/// Find connection through adjacent faces (non-recursive)
fn find_adjacent_face_connection<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    start_vertex: usize,
    target_vertex: usize,
) -> Option<AdjacentFaceConnection>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Find all faces containing start_vertex
    let start_faces = find_faces_containing_vertex_safe(mesh, start_vertex);
    let target_faces = find_faces_containing_vertex_safe(mesh, target_vertex);

    // Look for faces that share an edge
    for &start_face in &start_faces {
        for &target_face in &target_faces {
            if start_face == target_face {
                continue; // Same face already handled
            }

            if mesh.faces[start_face].removed || mesh.faces[target_face].removed {
                panic!("removed faces!");
            }

            // Check if faces are adjacent
            if let Some(shared_edge) = find_shared_edge_between_faces(mesh, start_face, target_face)
            {
                return Some(AdjacentFaceConnection {
                    start_face,
                    target_face,
                    intermediate_vertex: shared_edge.0, // One vertex of shared edge
                });
            }
        }
    }

    None
}

#[derive(Debug)]
struct AdjacentFaceConnection {
    start_face: usize,
    target_face: usize,
    intermediate_vertex: usize,
}

/// Find faces containing vertex with safety limits
fn find_faces_containing_vertex_safe<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    vertex: usize,
) -> Vec<usize> {
    let mut faces = Vec::new();
    let max_faces = 1000; // Safety limit

    for (face_idx, face) in mesh.faces.iter().enumerate() {
        if faces.len() >= max_faces {
            // println!("WARNING: Too many faces contain vertex {}", vertex);
            break;
        }

        if face.removed {
            continue;
        }

        let face_vertices = mesh.face_vertices(face_idx);

        if face_vertices.contains(&vertex) {
            faces.push(face_idx);
        }
    }

    faces
}

/// Find shared edge between two faces
fn find_shared_edge_between_faces<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_a: usize,
    face_b: usize,
) -> Option<(usize, usize)> {
    let vertices_a = mesh.face_vertices(face_a);
    let vertices_b = mesh.face_vertices(face_b);

    // Check all edges of face_a against face_b
    for i in 0..vertices_a.len() {
        let v1 = vertices_a[i];
        let v2 = vertices_a[(i + 1) % vertices_a.len()];

        // Check if this edge exists in face_b
        for j in 0..vertices_b.len() {
            let u1 = vertices_b[j];
            let u2 = vertices_b[(j + 1) % vertices_b.len()];

            if (v1 == u1 && v2 == u2) || (v1 == u2 && v2 == u1) {
                return Some((v1, v2));
            }
        }
    }

    None
}

fn is_manifold_boundary_loop<T: Scalar, const N: usize>(
    loop_segments: &Vec<usize>,
    segments: &Vec<IntersectionSegment<T, N>>,
) -> bool
where
    Point<T, N>: PointOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if loop_segments.len() < 3 {
        return false;
    }

    // Check if each vertex has exactly 2 connections (manifold property)
    let mut vertex_degree = HashMap::new();

    for &seg_idx in loop_segments {
        if seg_idx >= segments.len() {
            return false;
        }

        let seg = &segments[seg_idx];

        // Use quantized coordinates for robust vertex identification
        let a_key = quantize_point(&seg.segment.a);
        let b_key = quantize_point(&seg.segment.b);

        *vertex_degree.entry(a_key).or_insert(0) += 1;
        *vertex_degree.entry(b_key).or_insert(0) += 1;
    }

    // All vertices should have degree 2 for a manifold loop
    vertex_degree.values().all(|&degree| degree == 2)
}

// In quantize_point function
fn quantize_point<T: Scalar, const N: usize>(point: &Point<T, N>) -> String
where
    Point<T, N>: PointOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let tolerance = T::point_merge_threshold(); // Use point-specific threshold

    if tolerance.is_zero() {
        // Handle exact arithmetic case
        let coords: Vec<String> = (0..N).map(|i| format!("{:?}", point[i])).collect();
        return coords.join(",");
    }

    let quantized_coords: Vec<i64> = (0..N)
        .map(|i| {
            let coord = &point[i];
            let ratio = coord / &tolerance;
            let ratio_f64 = ratio.to_f64().unwrap();
            ratio_f64.round() as i64
        })
        .collect();
    format!("{:?}", quantized_coords)
}

// Iterative face half-edge traversal to eliminate recursion
fn get_face_half_edges_iterative<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_idx: usize,
) -> Option<Vec<usize>> {
    if face_idx >= mesh.faces.len() {
        return None;
    }

    let face = &mesh.faces[face_idx];
    if face.half_edge == usize::MAX || mesh.half_edges[face.half_edge].removed {
        return None;
    }

    let start_he = face.half_edge;
    if start_he >= mesh.half_edges.len() {
        return None;
    }

    let mut result = Vec::new();
    let mut current_he = start_he;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 50; // Safety limit for triangular faces

    loop {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            // println!(
            //     "WARNING: Face {} half-edge traversal exceeded limit",
            //     face_idx
            // );
            return None;
        }

        if current_he >= mesh.half_edges.len() {
            return None;
        }

        result.push(current_he);

        current_he = mesh.half_edges[current_he].next;
        if current_he >= mesh.half_edges.len() {
            return None;
        }

        if current_he == start_he {
            break;
        }
    }

    Some(result)
}

fn should_split_edge<T: Scalar>(edge_length: &T, split_point_distance: &T) -> bool
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let min_edge_length = T::edge_degeneracy_threshold();
    let min_split_distance = T::point_merge_threshold();

    // Don't split if:
    // 1. Edge is too short
    // 2. Split point is too close to start
    // 3. Remaining segment would be too short
    edge_length > &min_edge_length
        && split_point_distance > &min_split_distance
        && (edge_length - &split_point_distance) > min_split_distance
}

fn compute_triangle_aabb<T: Scalar, const N: usize>(
    p0: &Point<T, N>,
    p1: &Point<T, N>,
    p2: &Point<T, N>,
) -> Aabb<T, N, Point<T, N>>
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let mut min_coords = p0.coords().clone();
    let mut max_coords = p0.coords().clone();

    // Compare with p1
    for i in 0..N {
        if p1[i] < min_coords[i] {
            min_coords[i] = p1[i].clone();
        }
        if p1[i] > max_coords[i] {
            max_coords[i] = p1[i].clone();
        }
    }

    // Compare with p2
    for i in 0..N {
        if p2[i] < min_coords[i] {
            min_coords[i] = p2[i].clone();
        }
        if p2[i] > max_coords[i] {
            max_coords[i] = p2[i].clone();
        }
    }

    Aabb::from_points(&Point::from_vals(min_coords), &Point::from_vals(max_coords))
}

fn point_in_triangle<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let v0 = (c - a).as_vector();
    let v1 = (b - a).as_vector();
    let v2 = (p - a).as_vector();

    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot11 = v1.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot12 = v1.dot(&v2);

    let denom = &dot00 * &dot11 - &dot01 * &dot01;
    if denom.abs().is_zero() {
        return false; // Degenerate triangle
    }

    let inv_denom = T::one() / denom;
    let u = &(&dot11 * &dot02 - &dot01 * &dot12) * &inv_denom;
    let v = &(&dot00 * &dot12 - &dot01 * &dot02) * &inv_denom;

    let eps = T::tolerance();
    u >= -eps.clone() && v >= -eps.clone() && (&u + &v) <= (T::one() + eps)
}

fn barycentric_coords<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> Option<(T, T, T)>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let v0 = (b - a).as_vector();
    let v1 = (c - a).as_vector();
    let v2 = (p - a).as_vector();

    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);

    let denom = &d00 * &d11 - &d01 * &d01;
    if denom.abs() < T::tolerance() {
        return None; // Degenerate triangle
    }

    let v = (&d11 * &d20 - &d01 * &d21) / denom.clone();
    let w = (&d00 * &d21 - &d01 * &d20) / denom;
    let u = &T::one() - &(&v - &w);

    // println!("Barycentric coords: u={:?}, v={:?}, w={:?}", u, v, w);

    Some((u, v, w))
}

fn point_in_triangle_barycentric<T: Scalar, const N: usize>(bary: &(T, T, T)) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let (u, v, w) = bary;
    let zero = T::zero();
    let one = T::one();
    let eps = T::tolerance();

    // Allow small negative values due to floating-point error
    u >= &(&zero - &eps)
        && v >= &(&zero - &eps)
        && w >= &(&zero - &eps)
        && u <= &(&one + &eps)
        && v <= &(&one + &eps)
        && w <= &(&one + &eps)
}

fn ray_segment_intersection_2d<T: Scalar>(
    ray_origin: &Point<T, 2>,
    ray_dir: &Vector<T, 2>,
    seg_a: &Point<T, 2>,
    seg_b: &Point<T, 2>,
) -> Option<(T, T)>
where
    Point<T, 2>: PointOps<T, 2, Vector = Vector<T, 2>>,
    Vector<T, 2>: VectorOps<T, 2, Cross = T>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let v1 = (ray_origin - seg_a).as_vector();
    let v2 = (seg_b - seg_a).as_vector();
    let v3 = Vector::new([-ray_dir[1].clone(), ray_dir[0].clone()]);

    let denom = v2.dot(&v3);
    if denom.abs().is_zero() {
        return None; // parallel
    }

    let t = &v2.cross(&v1) / &denom;
    let u = &v1.dot(&v3) / &denom;

    if t.is_positive() && u.is_positive() && (&u - &T::one()).is_negative_or_zero() {
        Some((t, u))
    } else {
        None
    }
}

fn ray_stays_within_face<T: Scalar, const N: usize>(
    face_vertices: &[&Point<T, N>], // original 3D vertex positions of the face
    point_3d: &Point<T, N>,         // point lying on the edge
    dir_3d: &Vector<T, N>,          // direction to test
    epsilon: &T,                    // small step value, e.g., 1e-6
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Step 1: Compute basis (u, v) for face plane
    let edge1 = (face_vertices[1] - face_vertices[0]).as_vector();
    let edge2 = (face_vertices[2] - face_vertices[0]).as_vector();
    let normal = edge1.cross(&edge2);
    let u = edge1.normalized(); // assume non-degenerate
    let v = normal.cross(&u).normalized();

    // Step 2: Project point and direction into 2D
    let origin = &face_vertices[0];
    let to_point = point_3d - origin;
    let point_2d = Point::<T, 2>::new([to_point.as_vector().dot(&u), to_point.as_vector().dot(&v)]);
    let dir_2d = Vector::<T, 2>::new([dir_3d.dot(&u), dir_3d.dot(&v)]);

    let offset_point = &point_2d + &(dir_2d.scale(epsilon)).0;

    // Step 3: Project face vertices into 2D
    let face_2d: Vec<Point<T, 2>> = face_vertices
        .iter()
        .map(|v| {
            let delta = (*v - *origin).as_vector();
            Point::new([delta.dot(&u), delta.dot(&v.as_vector())])
        })
        .collect();

    // Step 4: Run point-in-polygon test for offset_point
    point_in_polygon_2d(&face_2d, &offset_point)
}

fn point_in_polygon_2d<T: Scalar, const N: usize>(polygon: &[Point<T, 2>], p: &Point<T, 2>) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let mut inside = false;
    let n = polygon.len();

    for i in 0..n {
        let a = &polygon[i];
        let b = &polygon[(i + 1) % n];

        let (x, y) = (&p[0], &p[1]);
        let (x0, y0) = (&a[0], &a[1]);
        let (x1, y1) = (&b[0], &b[1]);

        let cond1 = (y0 <= y && y < y1) || (y1 <= y && y < y0);
        if cond1 {
            let cross = (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0);
            if cross.is_negative() {
                inside = !inside;
            }
        }
    }

    inside
}

fn find_ray_face_boundary_intersection<T: Scalar, const N: usize>(
    ray_origin: &Point<T, N>,
    ray_direction: &Vector<T, N>,
    face_vertices: &[&Point<T, N>; 3],
    tolerance: &T,
) -> Option<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let mut min_t = None;

    // Check intersection with each edge of the triangle
    for i in 0..3 {
        let edge_start = face_vertices[i];
        let edge_end = face_vertices[(i + 1) % 3];

        if let Some(t) =
            ray_line_intersection_2d(ray_origin, ray_direction, edge_start, edge_end, tolerance)
        {
            if t > *tolerance {
                // Ignore trivial hits at starting point
                match min_t {
                    None => min_t = Some(t),
                    Some(current_min) if t < current_min => min_t = Some(t),
                    _ => {}
                }
            }
        }
    }

    min_t
}

fn ray_line_intersection_2d<T: Scalar, const N: usize>(
    ray_origin: &Point<T, N>,
    ray_direction: &Vector<T, N>,
    line_start: &Point<T, N>,
    line_end: &Point<T, N>,
    tolerance: &T,
) -> Option<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Handle degenerate line segment
    let line_dir = (line_end - line_start).as_vector();
    let line_length_sq = line_dir.norm_squared();

    if line_length_sq <= tolerance * tolerance {
        return None; // Degenerate line segment
    }

    // Project to dominant 2D plane to avoid numerical issues
    let abs_dir = [
        ray_direction[0].abs(),
        ray_direction[1].abs(),
        ray_direction[2].abs(),
    ];

    // Choose projection plane by dropping the axis with largest ray direction component
    let (coord0, coord1) = if abs_dir[0] >= abs_dir[1] && abs_dir[0] >= abs_dir[2] {
        (1, 2) // Drop X, use YZ plane
    } else if abs_dir[1] >= abs_dir[2] {
        (0, 2) // Drop Y, use XZ plane  
    } else {
        (0, 1) // Drop Z, use XY plane
    };

    // Extract 2D coordinates
    let ray_origin_2d =
        Point::<T, 2>::new([ray_origin[coord0].clone(), ray_origin[coord1].clone()]);

    let ray_dir_2d =
        Vector::<T, 2>::new([ray_direction[coord0].clone(), ray_direction[coord1].clone()]);

    let line_start_2d =
        Point::<T, 2>::new([line_start[coord0].clone(), line_start[coord1].clone()]);

    let line_end_2d = Point::<T, 2>::new([line_end[coord0].clone(), line_end[coord1].clone()]);

    // Check if 2D projection is degenerate
    let ray_2d_length_sq = ray_dir_2d.norm_squared();
    let line_2d_length_sq = (&line_end_2d - &line_start_2d).as_vector().norm_squared();

    if ray_2d_length_sq <= tolerance * tolerance || line_2d_length_sq <= tolerance * tolerance {
        return None; // Degenerate projection
    }

    // Parametric intersection: ray_origin + t * ray_dir = line_start + s * line_dir
    let line_dir_2d = (&line_end_2d - &line_start_2d).as_vector();
    let origin_diff = (&line_start_2d - &ray_origin_2d).as_vector();

    // Solve 2x2 linear system using Cramer's rule
    let det = ray_dir_2d[0].clone() * line_dir_2d[1].clone()
        - ray_dir_2d[1].clone() * line_dir_2d[0].clone();

    if det.abs() <= *tolerance {
        return None; // Parallel lines
    }

    let t = (origin_diff[0].clone() * line_dir_2d[1].clone()
        - origin_diff[1].clone() * line_dir_2d[0].clone())
        / det.clone();
    let s = (origin_diff[0].clone() * ray_dir_2d[1].clone()
        - origin_diff[1].clone() * ray_dir_2d[0].clone())
        / det;

    // Check if intersection is within line segment bounds
    if s >= T::zero() && s <= T::one() && t > *tolerance {
        Some(t)
    } else {
        None
    }
}
