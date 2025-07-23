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
        Aabb, AabbTree, Point3,
        point::{Point, PointOps},
        segment::{Segment, SegmentOps},
        spatial_element::SpatialElement,
        tri_tri_intersect::{self, tri_tri_intersection},
        util::EPS,
        vector::{Vector, VectorOps},
    },
    io::obj::write_obj,
    numeric::{cgar_f64::CgarF64, scalar::Scalar},
    operations::{Abs, Pow, Sqrt, Zero},
};
use num_traits::Float;
use rand::prelude::*;

use super::{face::Face, half_edge::HalfEdge, vertex::Vertex};
use core::panic;
use std::{
    array::from_fn,
    collections::{HashMap, HashSet, VecDeque},
    ops::{Add, Div, Mul, Sub},
    time::Instant,
};
use std::{convert::TryInto, f64::consts::PI};

#[derive(Debug, PartialEq, Eq)]
pub enum BooleanOp {
    Union,
    Intersection,
    Difference,
}

#[derive(Debug, Clone, Default)]
pub struct FaceSplitMap {
    pub face: usize,
    pub new_faces: Vec<usize>,
}

pub enum SplitResultKind {
    NoSplit,
    SplitFace,
    SplitEdge,
}

pub struct SplitResult {
    kind: SplitResultKind,
    new_vertex: usize,
    new_faces: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct IntersectionSegment<T: Scalar, const N: usize> {
    pub segment: Segment<T, N>,
    pub faces_a: [usize; 2],
    pub faces_b: [usize; 2],
    pub links: Vec<usize>, // indices of intersection segments that this segment link to
}

impl<T: Scalar, const N: usize> IntersectionSegment<T, N> {
    pub fn new(segment: Segment<T, N>, faces_a: [usize; 2], faces_b: [usize; 2]) -> Self {
        Self {
            segment,
            faces_a,
            faces_b,
            links: Vec::new(),
        }
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
        }
    }

    pub fn face_half_edges_safe(&self, f: usize) -> Option<Vec<usize>> {
        if f >= self.faces.len() {
            return None;
        }

        let face = &self.faces[f];
        if face.half_edge == usize::MAX {
            return None;
        }

        get_face_half_edges_safe(self, f)
    }

    /// Safe version of face_vertices that validates indices
    pub fn face_vertices_safe(&self, f: usize) -> Option<Vec<usize>> {
        get_face_vertices_safe(self, f)
    }

    /// Returns all half-edges that originate from vertex `v`
    /// This is essential for mesh traversal and topology operations
    pub fn vertex_half_edges(&self, v: usize) -> Vec<usize> {
        let mut result = Vec::new();

        // Get the starting half-edge for this vertex
        if let Some(start_he) = self.vertices[v].half_edge {
            let mut current_he = start_he;
            let mut iterations = 0;
            let max_iterations = self.half_edges.len(); // Safety limit

            loop {
                // Safety check to prevent infinite loops
                if iterations > max_iterations {
                    break;
                }

                // Add current half-edge if it originates from vertex v
                let prev_he = self.half_edges[current_he].prev;
                if self.half_edges[prev_he].vertex == v {
                    result.push(current_he);
                }

                // Move to twin and then next to get the next outgoing edge
                let twin_he = self.half_edges[current_he].twin;
                if twin_he == usize::MAX {
                    // Boundary case - we've hit a boundary, stop here
                    break;
                }

                current_he = self.half_edges[twin_he].next;
                iterations += 1;

                // Check if we've completed the loop
                if current_he == start_he {
                    break;
                }
            }
        }

        result
    }

    /// Computes the face normal vector for face `f`
    /// Uses cross product of two edge vectors
    pub fn face_normal(&self, f: usize) -> Vector<T, N>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let face_vertices = self.face_vertices(f);

        if face_vertices.len() < 3 {
            return Vector::zero(); // Degenerate face
        }

        // Get three vertices of the face
        let v0 = &self.vertices[face_vertices[0]].position;
        let v1 = &self.vertices[face_vertices[1]].position;
        let v2 = &self.vertices[face_vertices[2]].position;

        // Compute two edge vectors
        let edge1 = (v1 - v0).as_vector();
        let edge2 = (v2 - v0).as_vector();

        // Cross product gives face normal
        let normal = edge1.cross(&edge2);

        // Return normalized normal (handle zero-length case)
        let norm = normal.norm();
        if norm.is_zero() {
            Vector::zero() // Degenerate triangle
        } else {
            normal.normalized()
        }
    }

    fn average_face_area(&self) -> f64
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut total_area = 0.0;
        let mut count = 0;
        for f in 0..self.faces.len() {
            let area = self.face_area(f).to_f64().unwrap();
            total_area += area;
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            total_area / (count as f64)
        }
    }

    /// Remove a vertex and update all references.
    /// Removes any faces that use this vertex and updates indices in remaining faces.
    /// Returns true if the vertex was successfully removed.
    pub fn remove_vertex(&mut self, vertex_idx: usize) -> bool {
        if vertex_idx >= self.vertices.len() {
            return false; // Invalid index
        }

        // 1) Remove faces that use this vertex
        let mut faces_to_remove = Vec::new();
        for (face_idx, _) in self.faces.iter().enumerate() {
            let face_verts = self.face_vertices(face_idx);
            if face_verts.contains(&vertex_idx) {
                faces_to_remove.push(face_idx);
            }
        }

        // Remove faces in reverse order to maintain indices
        for &face_idx in faces_to_remove.iter().rev() {
            self.remove_face(face_idx);
        }

        // 2) Remove the vertex
        self.vertices.remove(vertex_idx);

        // 3) Update all vertex indices > vertex_idx (decrement by 1)
        self.update_vertex_indices_after_removal(vertex_idx);

        true
    }

    /// Remove a face and clean up associated half-edges
    pub fn remove_face(&mut self, face_idx: usize) {
        if face_idx >= self.faces.len() {
            return; // Invalid index
        }

        // Get half-edges for this face before removal
        let face_half_edges = self.face_half_edges(face_idx);

        // Remove half-edges associated with this face
        // Note: This is complex due to twin relationships, so we'll mark them for cleanup
        for &he_idx in &face_half_edges {
            // Clear face reference
            if he_idx < self.half_edges.len() {
                self.half_edges[he_idx].face = None;

                // Update twin relationships
                let twin_idx = self.half_edges[he_idx].twin;
                if twin_idx != usize::MAX && twin_idx < self.half_edges.len() {
                    self.half_edges[twin_idx].twin = usize::MAX;
                }
            }
        }

        // Remove from edge_map
        for &he_idx in &face_half_edges {
            if he_idx < self.half_edges.len() {
                let vertex = self.half_edges[he_idx].vertex;
                let prev_idx = self.half_edges[he_idx].prev;
                if prev_idx < self.half_edges.len() {
                    let prev_vertex = self.half_edges[prev_idx].vertex;
                    self.edge_map.remove(&(prev_vertex, vertex));
                }
            }
        }

        // Remove the face
        self.faces.remove(face_idx);

        // Update face indices in half-edges
        for he in &mut self.half_edges {
            if let Some(ref mut face_ref) = he.face {
                if *face_ref > face_idx {
                    *face_ref -= 1;
                }
            }
        }
    }

    /// Update all vertex indices after a vertex removal
    fn update_vertex_indices_after_removal(&mut self, removed_idx: usize) {
        // Update half-edges
        for he in &mut self.half_edges {
            if he.vertex > removed_idx {
                he.vertex -= 1;
            }
        }

        // Update vertex half-edge references
        for (v_idx, vertex) in self.vertices.iter_mut().enumerate() {
            if let Some(ref mut he_idx) = vertex.half_edge {
                // Find a valid half-edge for this vertex (since indices may have shifted)
                let mut found_valid = false;
                for (he_idx_search, he) in self.half_edges.iter().enumerate() {
                    if he.vertex == v_idx {
                        *he_idx = he_idx_search;
                        found_valid = true;
                        break;
                    }
                }
                if !found_valid {
                    vertex.half_edge = None;
                }
            }
        }

        // Update edge_map
        let old_edge_map = std::mem::take(&mut self.edge_map);
        for ((from, to), he_idx) in old_edge_map {
            let new_from = if from > removed_idx { from - 1 } else { from };
            let new_to = if to > removed_idx { to - 1 } else { to };

            // Only keep edges where both vertices still exist
            if new_from != removed_idx && new_to != removed_idx {
                self.edge_map.insert((new_from, new_to), he_idx);
            }
        }
    }

    pub fn remove_unused_vertices(&mut self) {
        // 1. Find used vertices
        let mut used = vec![false; self.vertices.len()];
        for face_idx in 0..self.faces.len() {
            // Skip invalidated faces
            if self.faces[face_idx].half_edge == usize::MAX {
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
                new_vertices.push(vertex.clone());
                old_to_new[old_idx] = Some(new_idx);
            }
        }

        // 3. Remap faces (skip invalidated ones)
        let mut new_mesh = Mesh::new();
        for v in &new_vertices {
            new_mesh.add_vertex(v.position.clone());
        }
        for face_idx in 0..self.faces.len() {
            // Skip invalidated faces
            if self.faces[face_idx].half_edge == usize::MAX {
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

    /// Finds an existing vertex at the given position on the specified face,
    /// or splits an edge/face to insert a new vertex at that position.
    /// Returns the vertex index.
    pub fn split_or_find_vertex_on_face(
        &mut self,
        face: usize,
        pos: &Point<T, N>,
    ) -> Option<SplitResult>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        // // println!("split_or_find_vertex_on_face");
        let tolerance = T::tolerance();

        // 1. Check if position matches an existing vertex on this face (still fast)
        let face_vertices = self.face_vertices(face);
        for &vi in &face_vertices {
            if self.vertices[vi].position.distance_to(pos) < tolerance {
                // println!("Found existing vertex {} at position {:?}", vi, pos);
                return Some(SplitResult {
                    kind: SplitResultKind::NoSplit,
                    new_vertex: vi,
                    new_faces: Vec::new(),
                });
            }
        }

        // 2. Use spatial hash to find nearby vertices (much faster than full scan)
        if let Some(existing_vi) = self.find_nearby_vertex(pos, tolerance.clone()) {
            // println!(
            //     "Found existing vertex {} near position {:?}",
            //     existing_vi, pos
            // );
            return Some(SplitResult {
                kind: SplitResultKind::NoSplit,
                new_vertex: existing_vi,
                new_faces: Vec::new(),
            });
        }

        // 3. Check if position lies on an edge of this face
        for &he in &self.face_half_edges(face) {
            let src = self.half_edges[self.half_edges[he].prev].vertex;
            let dst = self.half_edges[he].vertex;
            let edge_start = &self.vertices[src].position;
            let edge_end = &self.vertices[dst].position;

            if point_on_segment(edge_start, edge_end, pos) {
                // println!("ON SEGMENT");
                let edge_length = edge_start.distance_to(edge_end);
                let dist_to_start = edge_start.distance_to(pos);
                let dist_to_end = edge_end.distance_to(pos);

                if !should_split_edge(&edge_length, &dist_to_start) {
                    // Too close to start vertex
                    return Some(SplitResult {
                        kind: SplitResultKind::NoSplit,
                        new_vertex: src,
                        new_faces: Vec::new(),
                    });
                }

                if !should_split_edge(&edge_length, &dist_to_end) {
                    // Too close to end vertex
                    return Some(SplitResult {
                        kind: SplitResultKind::NoSplit,
                        new_vertex: dst,
                        new_faces: Vec::new(),
                    });
                }

                if let Ok(split_edge_result) = self.split_edge(he, &pos.clone()) {
                    // println!(
                    //     "Inserted new vertex {} on edge ({}, {}) at position {:?}",
                    //     split_edge_result.new_vertex, src, dst, pos
                    // );
                    return Some(split_edge_result);
                } else {
                    // println!(
                    //     "Failed to split edge ({}, {}) at position {:?}",
                    //     src, dst, pos
                    // );
                    return Some(SplitResult {
                        kind: SplitResultKind::SplitEdge,
                        new_vertex: self.add_vertex(pos.clone()),
                        new_faces: Vec::new(),
                    });
                }
            }
        }

        // 4. Position is inside the face - split the face into triangles
        if let Some(split_edge_result) = self.find_or_insert_vertex_on_face(face, pos) {
            // // println!("recursive!");
            return Some(split_edge_result);
        }

        for f in self.faces_containing_point(pos) {
            if let Some(split_edge_result) = self.find_or_insert_vertex_on_face(f, pos) {
                return Some(split_edge_result);
            }
        }

        // println!(
        //     "No suitable vertex found, adding new vertex at position {:?}",
        //     pos
        // );

        return None;
    }

    fn rebuild_spatial_hash(&mut self) {
        self.vertex_spatial_hash.clear();
        for (idx, vertex) in self.vertices.iter().enumerate() {
            let hash_key = self.position_to_hash_key(&vertex.position);
            self.vertex_spatial_hash
                .entry(hash_key)
                .or_insert_with(Vec::new)
                .push(idx);
        }
    }

    /// Adds an edge between two vertices if it doesn't already exist.
    /// Returns true if a new edge was added, false if it already existed.
    // pub fn add_edge_if_not_exists(&mut self, vi0: usize, vi1: usize) -> bool
    // where
    //     Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    //     Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    //     for<'a> &'a T: Sub<&'a T, Output = T>
    //         + Mul<&'a T, Output = T>
    //         + Add<&'a T, Output = T>
    //         + Div<&'a T, Output = T>,
    // {
    //     // Check if edge already exists
    //     if self.half_edge_between(vi0, vi1).is_some() {
    //         return false; // Edge already exists
    //     }

    //     // Find faces that contain both vertices
    //     let mut shared_faces = Vec::new();
    //     for face_idx in 0..self.faces.len() {
    //         if self.faces[face_idx].half_edge == usize::MAX {
    //             continue; // Skip invalidated faces
    //         }
    //         let face_verts = self.face_vertices(face_idx);
    //         if face_verts.contains(&vi0) && face_verts.contains(&vi1) {
    //             shared_faces.push(face_idx);
    //         }
    //     }

    //     match shared_faces.len() {
    //         0 => {
    //             self.carve_segment_across_faces(vi0, vi1);
    //             false
    //         }
    //         1 => {
    //             // Vertices are on the same face - split the face
    //             let face = shared_faces[0];
    //             self.split_segment_by_indices(face, vi0, vi1, false);
    //             true
    //         }
    //         _ => {
    //             // Multiple shared faces - edge already exists implicitly
    //             // This is typically an error case
    //             e// println!("Warning: Vertices share multiple faces");
    //             false
    //         }
    //     }
    // }

    /// Compute the AABB of face `f`.
    pub fn face_aabb(&self, f: usize) -> Aabb<T, N, Point<T, N>>
    where
        T: Scalar,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let _he = self.faces[f].half_edge;
        let vs = self.face_half_edges(f);
        let p0 = &self.vertices[self.half_edges[vs[0]].vertex].position;
        let p1 = &self.vertices[self.half_edges[vs[1]].vertex].position;
        let p2 = &self.vertices[self.half_edges[vs[2]].vertex].position;
        Aabb::from_points(p0, p1).union(&Aabb::from_points(p1, p2))
    }

    pub fn add_vertex(&mut self, position: Point<T, N>) -> usize {
        let idx = self.vertices.len();

        let hash_key = self.position_to_hash_key(&position);
        self.vertex_spatial_hash
            .entry(hash_key)
            .or_insert_with(Vec::new)
            .push(idx);

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

    pub fn faces_containing_point(&self, p: &Point<T, N>) -> Vec<usize>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut result = Vec::new();
        for f in 0..self.faces.len() {
            if self.faces[f].half_edge == usize::MAX {
                continue; // Skip invalidated faces
            }
            let vs = self.face_vertices(f);
            if vs.len() != 3 {
                continue;
            }
            let a = &self.vertices[vs[0]].position;
            let b = &self.vertices[vs[1]].position;
            let c = &self.vertices[vs[2]].position;

            // check coplanarity
            let n = (b - a).as_vector().cross(&(c - a).as_vector());
            if n.dot(&(p - a).as_vector()).abs().is_positive() {
                continue;
            }

            // check inside or on boundary
            if point_in_or_on_triangle(p, a, b, c) {
                result.push(f);
            }
        }
        result
    }

    /// Adds a triangle face given three vertex indices (in CCW order).
    /// Note: this is a naive non-twin-connected insertion for now.
    pub fn add_triangle(&mut self, v0: usize, v1: usize, v2: usize) -> usize {
        let face_idx = self.faces.len();
        let base_idx = self.half_edges.len();

        let edge_vertices = [(v0, v1), (v1, v2), (v2, v0)];

        let mut edge_indices = [0; 3];

        // Step 1: Create the 3 new half-edges
        for (i, &(from, to)) in edge_vertices.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(face_idx);
            let idx = base_idx + i;

            // Try to find twin edge (to → from)
            if let Some(&twin_idx) = self.edge_map.get(&(to, from)) {
                he.twin = twin_idx;
                self.half_edges[twin_idx].twin = idx;
            }

            self.edge_map.insert((from, to), idx);
            self.half_edges.push(he);
            edge_indices[i] = idx;
        }

        // Step 2: Link next/prev
        self.half_edges[edge_indices[0]].next = edge_indices[1];
        self.half_edges[edge_indices[0]].prev = edge_indices[2];
        self.half_edges[edge_indices[1]].next = edge_indices[2];
        self.half_edges[edge_indices[1]].prev = edge_indices[0];
        self.half_edges[edge_indices[2]].next = edge_indices[0];
        self.half_edges[edge_indices[2]].prev = edge_indices[1];

        // Step 3: Attach half-edge to vertices (first one only, for now)
        self.vertices[v0].half_edge.get_or_insert(edge_indices[0]);
        self.vertices[v1].half_edge.get_or_insert(edge_indices[1]);
        self.vertices[v2].half_edge.get_or_insert(edge_indices[2]);

        self.faces.push(Face::new(edge_indices[0]));
        face_idx
    }

    /// Return the centroid of face `f` as a Vec<f64> of length = dimensions().
    /// Currently works for any dimension, but returns a flat Vec.
    pub fn face_centroid(&self, f: usize) -> Vector<T, N>
    where
        Vector<T, N>: VectorOps<T, N>,
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
            *coord = coord.clone() / n.clone();
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
                he = if self.half_edges[prev].twin != usize::MAX {
                    self.half_edges[prev].twin
                } else {
                    prev
                };
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
    pub fn face_half_edges(&self, f: usize) -> Vec<usize> {
        let mut result = Vec::new();
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
    pub fn face_vertices(&self, f: usize) -> Vec<usize> {
        self.face_half_edges(f)
            .into_iter()
            .map(|he| self.half_edges[he].vertex)
            .collect()
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

    pub fn point_in_mesh(&self, tree: &AabbTree<T, 3, Point<T, 3>, usize>, p: &Point<T, 3>) -> bool
    where
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut inside_count = 0;
        let mut total_rays = 0;

        let rays = vec![
            Vector::from_vals([T::one(), T::zero(), T::zero()]),
            Vector::from_vals([T::zero(), T::one(), T::zero()]),
            Vector::from_vals([T::zero(), T::zero(), T::one()]),
            Vector::from_vals([T::from(-1.0), T::zero(), T::zero()]),
            Vector::from_vals([T::zero(), T::from(-1.0), T::zero()]),
            Vector::from_vals([T::zero(), T::zero(), T::from(-1.0)]),
        ];

        for r in rays {
            if let Some(is_inside) = self.cast_ray(p, &r, tree) {
                if is_inside {
                    inside_count += 1;
                }
                total_rays += 1;
            }
        }

        total_rays > 0 && inside_count > total_rays / 2
    }

    fn cast_ray(
        &self,
        p: &Point<T, 3>,
        dir: &Vector<T, 3>,
        tree: &AabbTree<T, 3, Point<T, 3>, usize>,
    ) -> Option<bool>
    where
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut hits: Vec<T> = Vec::new();

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
                if t > T::tolerance() {
                    hits.push(t);
                }
            }
        }

        if hits.is_empty() {
            return None;
        }

        // Remove duplicates and count
        hits.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut filtered_hits = Vec::new();
        let mut last_t = None;

        for t in hits {
            if last_t.as_ref().map_or(true, |lt: &T| {
                (t.clone() - lt.clone()).abs() > T::tolerance()
            }) {
                filtered_hits.push(t.clone());
                last_t = Some(t);
            }
        }

        Some(filtered_hits.len() % 2 == 1)
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

        // Cross product: ray_dir × edge2
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

        // Cross product: s × edge1
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

        if t.is_positive() {
            Some(t) // Valid intersection
        } else {
            None // Intersection behind ray origin or too close
        }
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
        let _u = self.half_edges[he_c].vertex; // c→u
        let _v = self.half_edges[he_a].vertex; // u→v
        let c = self.half_edges[he_b].vertex; // v→c
        let d = self.half_edges[he_e].vertex; // u→d

        // --- 4) reassign the two halves of the diagonal to c→d and d→c ---
        self.half_edges[he_a].vertex = d; // now u→d
        self.half_edges[he_d].vertex = c; // now v→c

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

        // 2) Identify u→v and record the three hole corners c, u, d
        let he_b = self.half_edges[he].next; // v → c
        let he_c = self.half_edges[he].prev; // c → u
        let he_e = self.half_edges[he_d].next; // u → d

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
        // redirect the removed v → the kept u
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

    pub fn split_edge(&mut self, he: usize, pos: &Point<T, N>) -> Result<SplitResult, &'static str>
    where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let prev = self.half_edges[he].prev;
        let u = self.half_edges[prev].vertex;
        let v = self.half_edges[he].vertex;

        let edge_start = &self.vertices[u].position;
        let edge_end = &self.vertices[v].position;
        let edge_length = edge_start.distance_to(edge_end);

        let split_point_distance_from_start = edge_start.distance_to(pos);
        let split_point_distance_from_end = edge_end.distance_to(pos);

        // println!(
        //     "Distance from start: {}, from end: {}",
        //     split_point_distance_from_start.to_f64().unwrap(),
        //     split_point_distance_from_end.to_f64().unwrap()
        // );

        // Check if we already have this position at existing vertices
        if pos == edge_start {
            return Ok(SplitResult {
                kind: SplitResultKind::NoSplit,
                new_vertex: u,
                new_faces: Vec::new(),
            });
        }
        if pos == edge_end {
            return Ok(SplitResult {
                kind: SplitResultKind::NoSplit,
                new_vertex: v,
                new_faces: Vec::new(),
            });
        }

        // Apply degeneracy checks to prevent creating tiny edges
        if !should_split_edge(&edge_length, &split_point_distance_from_start)
            || !should_split_edge(&edge_length, &split_point_distance_from_end)
        {
            // Return existing vertex instead of splitting
            if split_point_distance_from_start < split_point_distance_from_end {
                return Ok(SplitResult {
                    kind: SplitResultKind::NoSplit,
                    new_vertex: u,
                    new_faces: Vec::new(),
                });
            } else {
                return Ok(SplitResult {
                    kind: SplitResultKind::NoSplit,
                    new_vertex: v,
                    new_faces: Vec::new(),
                });
            }
        }

        // Create new vertex at split position
        let w = self.vertices.len();
        self.vertex_spatial_hash
            .entry(self.position_to_hash_key(pos))
            .or_default()
            .push(w);
        self.vertices.push(Vertex::new(pos.clone()));

        // Find all faces containing edge u-v (or v-u)
        let mut affected_faces = Vec::new();
        for face_idx in 0..self.faces.len() {
            if self.faces[face_idx].half_edge == usize::MAX {
                continue;
            }
            let vs = self.face_vertices(face_idx);
            for i in 0..vs.len() {
                let curr = vs[i];
                let next = vs[(i + 1) % vs.len()];
                if (curr == u && next == v) || (curr == v && next == u) {
                    affected_faces.push((face_idx, curr, next, vs[(i + 2) % vs.len()]));
                    break;
                }
            }
        }

        let mut new_face_results = Vec::new();

        for (face_idx, a, b, c) in affected_faces {
            // Invalidate original face
            self.faces[face_idx].half_edge = usize::MAX;
            // println!("invalidating face {}", face_idx);

            // Create two new triangles sharing the new vertex w
            let face_1_idx = self.faces.len();
            let face_2_idx = self.faces.len() + 1;

            // Triangle 1: (a, w, c) - contains original edge endpoint a
            // Triangle 2: (w, b, c) - contains original edge endpoint b
            let tri1_verts = [a, w, c];
            let tri2_verts = [w, b, c];

            // Create face 1
            let base_idx_1 = self.half_edges.len();
            let edge_vertices_1 = [
                (tri1_verts[0], tri1_verts[1]), // a → w
                (tri1_verts[1], tri1_verts[2]), // w → c
                (tri1_verts[2], tri1_verts[0]), // c → a
            ];

            for (i, &(from, to)) in edge_vertices_1.iter().enumerate() {
                let mut he = HalfEdge::new(to);
                he.face = Some(face_1_idx);
                let idx = base_idx_1 + i;

                if let Some(&twin_idx) = self.edge_map.get(&(to, from)) {
                    he.twin = twin_idx;
                    self.half_edges[twin_idx].twin = idx;
                } else {
                    he.twin = usize::MAX;
                }

                self.edge_map.insert((from, to), idx);
                self.half_edges.push(he);
            }

            // Link face 1 half-edges
            let edge_indices_1 = [base_idx_1, base_idx_1 + 1, base_idx_1 + 2];
            self.half_edges[edge_indices_1[0]].next = edge_indices_1[1];
            self.half_edges[edge_indices_1[0]].prev = edge_indices_1[2];
            self.half_edges[edge_indices_1[1]].next = edge_indices_1[2];
            self.half_edges[edge_indices_1[1]].prev = edge_indices_1[0];
            self.half_edges[edge_indices_1[2]].next = edge_indices_1[0];
            self.half_edges[edge_indices_1[2]].prev = edge_indices_1[1];

            // Create face 2
            let base_idx_2 = self.half_edges.len();
            let edge_vertices_2 = [
                (tri2_verts[0], tri2_verts[1]), // w → b
                (tri2_verts[1], tri2_verts[2]), // b → c
                (tri2_verts[2], tri2_verts[0]), // c → w
            ];

            for (i, &(from, to)) in edge_vertices_2.iter().enumerate() {
                let mut he = HalfEdge::new(to);
                he.face = Some(face_2_idx);
                let idx = base_idx_2 + i;

                if let Some(&twin_idx) = self.edge_map.get(&(to, from)) {
                    he.twin = twin_idx;
                    self.half_edges[twin_idx].twin = idx;
                } else {
                    he.twin = usize::MAX;
                }

                self.edge_map.insert((from, to), idx);
                self.half_edges.push(he);
            }

            // Link face 2 half-edges
            let edge_indices_2 = [base_idx_2, base_idx_2 + 1, base_idx_2 + 2];
            self.half_edges[edge_indices_2[0]].next = edge_indices_2[1];
            self.half_edges[edge_indices_2[0]].prev = edge_indices_2[2];
            self.half_edges[edge_indices_2[1]].next = edge_indices_2[2];
            self.half_edges[edge_indices_2[1]].prev = edge_indices_2[0];
            self.half_edges[edge_indices_2[2]].next = edge_indices_2[0];
            self.half_edges[edge_indices_2[2]].prev = edge_indices_2[1];

            // Update vertex half-edge references
            for &v_idx in &tri1_verts {
                self.vertices[v_idx]
                    .half_edge
                    .get_or_insert(edge_indices_1[0]);
            }
            for &v_idx in &tri2_verts {
                self.vertices[v_idx]
                    .half_edge
                    .get_or_insert(edge_indices_2[0]);
            }

            // Add faces to mesh
            self.faces.push(Face::new(edge_indices_1[0]));
            self.faces.push(Face::new(edge_indices_2[0]));

            new_face_results.extend_from_slice(&[face_1_idx, face_2_idx]);

            // Update face split map
            self.face_split_map
                .entry(face_idx)
                .or_default()
                .new_faces
                .extend_from_slice(&[face_1_idx, face_2_idx]);
        }

        // Remove old edge from edge_map
        self.edge_map.remove(&(u, v));
        self.edge_map.remove(&(v, u));

        Ok(SplitResult {
            kind: SplitResultKind::SplitEdge,
            new_vertex: w,
            new_faces: new_face_results,
        })
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
                // find whether the edge appears as u→v or v→u
                let mut handled = false;
                for i in 0..3 {
                    let a = vs[i];
                    let b = vs[(i + 1) % 3];
                    let c = vs[(i + 2) % 3];
                    if a == u && b == v {
                        // orientation u→v→c
                        new_face_tris.push([u, new_old_idx, c]);
                        new_face_tris.push([new_old_idx, v, c]);
                        handled = true;
                        break;
                    }
                    if a == v && b == u {
                        // orientation v→u→c
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

    /// Strategy 2: Robust edge intersection with multiple fallback methods
    fn find_edge_intersection(
        &mut self,
        face: usize,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
        segment_idx: usize,
        chain: &mut Vec<usize>,
        chain_idx: usize,
        curr_idx: usize,
    ) -> Option<(usize, usize)>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let p_pos = intersection_segments[segment_idx].segment.a.clone();
        let q_pos = intersection_segments[segment_idx].segment.b.clone();

        let mut best_intersection = None;
        let mut best_t = T::one();

        // Find best edge intersection
        for &he in &self.face_half_edges(face) {
            let src = self.half_edges[self.half_edges[he].prev].vertex;
            let dst = self.half_edges[he].vertex;

            if src == curr_idx || dst == curr_idx {
                continue;
            }

            let edge_start = &self.vertices[src].position;
            let edge_end = &self.vertices[dst].position;

            if let Some((intersection_point, t)) =
                self.compute_intersection(&p_pos, &q_pos, edge_start, edge_end)
            {
                let eps = T::tolerance();
                if t > eps && t < T::one() - eps && t < best_t {
                    best_t = t.clone();
                    best_intersection = Some((he, intersection_point));
                }
            }
        }

        if let Some((he, intersection_point)) = best_intersection {
            if let Ok(split_edge_result) = self.split_edge(he, &intersection_point) {
                let new_vertex_idx = split_edge_result.new_vertex;
                let new_vertex_pos = self.vertices[new_vertex_idx].position.clone();
                let mut containing_faces = self.faces_containing_point(&new_vertex_pos);
                if let Some(&new_face_b) = containing_faces.get(0) {
                    // 1) extract old endpoint & face_b
                    let (old_b_point, old_face_b) = {
                        let seg = &intersection_segments[segment_idx];
                        (seg.segment.b.clone(), seg.faces_b[0])
                    };

                    // 2) update the existing segment
                    {
                        let seg_mut = &mut intersection_segments[segment_idx];
                        seg_mut.segment.b = intersection_point.clone();
                        seg_mut.faces_b = [new_face_b, usize::MAX];
                        seg_mut.faces_a = [
                            split_edge_result.new_faces[0],
                            split_edge_result.new_faces[1],
                        ];
                    } // seg_mut goes out of scope here

                    // 3) build and push the new segment
                    let new_seg = IntersectionSegment::new(
                        Segment::new(&intersection_point, &old_b_point),
                        [new_face_b, usize::MAX],
                        [old_face_b, usize::MAX],
                    );
                    let new_idx = intersection_segments.len();
                    intersection_segments.push(new_seg);

                    // 4) update the chain
                    if chain_idx + 1 <= chain.len() {
                        chain.insert(chain_idx + 1, new_idx);
                    } else {
                        chain.push(new_idx);
                    }

                    // 5) finally link them up
                    let old_link = intersection_segments[segment_idx].links.pop();
                    intersection_segments[segment_idx].links.push(new_idx);
                    intersection_segments[new_idx].links.push(segment_idx);
                    intersection_segments[new_idx].links.push(old_link.unwrap());

                    // println!(
                    //     "SUCCESSFULLY SPLIT SEGMENT: {} -> [{}, {}]",
                    //     segment_idx, segment_idx, new_idx
                    // );

                    return Some((new_vertex_idx, new_face_b));
                }
            }
        }

        None
    }

    /// Robust intersection computation with multiple methods
    fn compute_intersection(
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
        // Method 1: 3D parametric line intersection
        if let Some(result) = self.parametric_line_intersection_3d(p, q, a, b) {
            return Some(result);
        }

        // Method 2: Try all projection planes
        for drop_axis in 0..3 {
            if let Some(result) = self.projected_intersection_2d(p, q, a, b, drop_axis) {
                return Some(result);
            }
        }

        // Method 3: Distance-based closest approach
        self.closest_approach_intersection(p, q, a, b)
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

        if denom.abs() < T::tolerance() {
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

    /// Helper: Check if two vectors are colinear with potential overlap
    fn are_colinear_with_overlap(&self, v1: &Vector<T, N>, v2: &Vector<T, N>) -> bool
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let cross = v1.cross(v2);
        cross.norm().is_zero()
    }

    pub fn split_segment_by_indices(
        &mut self,
        face: usize,
        vi0: usize,
        vi1: usize,
        update_loops: bool,
    ) where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        // 1) If there’s already a half‐edge vi0→vi1 on this face, split that edge
        if let Some(he) = self.find_half_edge_on_face(face, vi0, vi1) {
            let _ = self.split_edge(he, &self.vertices[vi1].position.clone());
        } else {
            // println!("VEIO AQUI NÃO QUERO--------------------------------------------------");
            // 2) No direct half‐edge: do a single two‐triangle split of this face
            let vs = self.face_vertices(face);
            let &third = vs.iter().find(|&&v| v != vi0 && v != vi1).unwrap();
            // collect all faces except `face`
            let mut new_tris = Vec::with_capacity(self.faces.len() + 1);
            for f in 0..self.faces.len() {
                if f != face && self.faces[f].half_edge != usize::MAX {
                    let fv = self.face_vertices(f);
                    new_tris.push([fv[0], fv[1], fv[2]]);
                }
            }
            // add the two halves of the split
            new_tris.push([vi0, vi1, third]);
            new_tris.push([vi1, vi0, third]);

            // rebuild entire mesh (vertices unchanged)
            let old_positions = self
                .vertices
                .iter()
                .map(|v| v.position.clone())
                .collect::<Vec<_>>();
            *self = Mesh::new();
            for pos in old_positions {
                self.add_vertex(pos);
            }
            for tri in new_tris {
                self.add_triangle(tri[0], tri[1], tri[2]);
            }
        }

        // 3) rebuild boundary loops if asked
        if update_loops {
            //self.build_boundary_loops();
        }
    }

    pub fn split_segment_by_indices_2(
        &mut self,
        face: usize,
        vi0: usize,
        vi1: usize,
        update_loops: bool,
    ) {
        // 1. Find half-edge from vi0 to vi1 or vi1 to vi0 on this face.
        let mut he_match = None;
        for &he in &self.face_half_edges(face) {
            let src = self.half_edges[self.half_edges[he].prev].vertex;
            let dst = self.half_edges[he].vertex;
            if (src == vi0 && dst == vi1) || (src == vi1 && dst == vi0) {
                he_match = Some((he, src, dst));
                break;
            }
        }

        match he_match {
            Some((he, src, dst)) => {
                // 2. Edge already exists as a half-edge in this face.
                // Optional: Split it if you want to insert a new vertex between, but for a segment split, we’re done.
                // No-op: edge already present as half-edge
                // Optionally, you could check if it’s a boundary edge and act accordingly.
                return;
            }
            None => {
                // 3. No half-edge directly between vi0 and vi1; need to split through intermediate vertices.
                // We'll find the path between vi0 and vi1 along the face boundary.
                // (Works for triangles and polygons.)
                let face_vs = self.face_vertices(face);
                // Find path vi0 -> ... -> vi1 along face boundary
                let mut path = Vec::new();
                let mut found = false;
                for i in 0..face_vs.len() {
                    if face_vs[i] == vi0 {
                        // Walk forward until we find vi1
                        let mut idx = i;
                        loop {
                            let next_idx = (idx + 1) % face_vs.len();
                            let v = face_vs[next_idx];
                            path.push(face_vs[idx]);
                            if v == vi1 {
                                path.push(v);
                                found = true;
                                break;
                            }
                            idx = next_idx;
                            if idx == i {
                                break; // Avoid infinite loops
                            }
                        }
                        if found {
                            break;
                        }
                        path.clear();
                    }
                }
                if !found {
                    panic!("Vertices not on the same face boundary");
                }
                // Now path = [vi0, ..., vi1]
                // For every consecutive pair, check if edge exists, else split at middle points.
                for pair in path.windows(2) {
                    let v_from = pair[0];
                    let v_to = pair[1];
                    let he = self.find_half_edge_on_face(face, v_from, v_to);
                    if he.is_none() {
                        // Should not happen: mesh connectivity error
                        panic!("No half-edge between consecutive face boundary vertices");
                    }
                    // For robust Booleans, only split at explicit new points.
                    // If we wanted to insert more points (not just along edges), do so here.
                }
                // After splitting, the edge should exist
            }
        }

        if update_loops {
            //self.build_boundary_loops();
        }
    }

    /// Utility: Finds the half-edge index on the given face from vi_from to vi_to.
    fn find_half_edge_on_face(&self, face: usize, vi_from: usize, vi_to: usize) -> Option<usize> {
        // Add safety check to prevent infinite loops
        let start_he = self.faces[face].half_edge;
        let mut current_he = start_he;
        let mut count = 0;
        let max_iterations = self.half_edges.len(); // Safety limit

        loop {
            // Safety check to prevent infinite loops
            if count > max_iterations {
                return None;
            }

            let src = self.half_edges[self.half_edges[current_he].prev].vertex;
            let dst = self.half_edges[current_he].vertex;

            if src == vi_from && dst == vi_to {
                return Some(current_he);
            }

            current_he = self.half_edges[current_he].next;
            count += 1;

            if current_he == start_he {
                break;
            }
        }

        None
    }

    /// Naïve but correct: returns the *closest* distance from `p` to ANY triangle in `self`
    /// You can optimize this later by using the tree to cull away distant faces.
    pub fn point_to_mesh_distance(
        &self,
        tree: &AabbTree<T, 3, Point<T, 3>, usize>,
        p: &Point3<T>,
    ) -> T
    where
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        // Use a large but reasonable search radius
        let search_radius = T::from(100.0); // Adjust based on your mesh scales

        let search_aabb = Aabb::from_points(
            &Point::<T, 3>::from_vals([
                &p[0] - &search_radius,
                &p[1] - &search_radius,
                &p[2] - &search_radius,
            ]),
            &Point::<T, 3>::from_vals([
                &p[0] + &search_radius,
                &p[1] + &search_radius,
                &p[2] + &search_radius,
            ]),
        );

        let mut candidate_faces = Vec::new();
        tree.query(&search_aabb, &mut candidate_faces);

        let mut min_distance_squared: Option<T> = None;

        for &fi in &candidate_faces {
            let vs = self.face_vertices(*fi);
            let v0 = Point::<T, 3>::from_vals([
                self.vertices[vs[0]].position[0].clone(),
                self.vertices[vs[0]].position[1].clone(),
                self.vertices[vs[0]].position[2].clone(),
            ]);
            let v1 = Point::<T, 3>::from_vals([
                self.vertices[vs[1]].position[0].clone(),
                self.vertices[vs[1]].position[1].clone(),
                self.vertices[vs[1]].position[2].clone(),
            ]);
            let v2 = Point::<T, 3>::from_vals([
                self.vertices[vs[2]].position[0].clone(),
                self.vertices[vs[2]].position[1].clone(),
                self.vertices[vs[2]].position[2].clone(),
            ]);

            let d2 = distance_point_triangle_squared(&p, &v0, &v1, &v2);

            if min_distance_squared.is_none() || d2 < *min_distance_squared.as_ref().unwrap() {
                min_distance_squared = Some(d2);
            }
        }

        min_distance_squared.unwrap_or(T::zero()).sqrt()
    }

    /// Given a face and a target point, finds an existing vertex (if close)
    /// or splits the appropriate edge and returns the new vertex index.
    pub fn find_or_insert_vertex_on_face(
        &mut self,
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
        // 1. Check if p coincides with a face vertex.
        let vs = self.face_vertices(face);
        let face_vs = self.face_vertices(face);
        for &vi in &face_vs {
            if self.vertices[vi].position.distance_to(p).is_zero() {
                return Some(SplitResult {
                    kind: SplitResultKind::NoSplit,
                    new_vertex: vi,
                    new_faces: Vec::new(),
                });
            }
        }

        // 2. Check if p is on a face edge, and split if so.
        for &he in &self.face_half_edges(face) {
            let src = self.half_edges[self.half_edges[he].prev].vertex;
            let dst = self.half_edges[he].vertex;
            let ps = &self.vertices[src].position;
            let pd = &self.vertices[dst].position;
            // Is p on segment [ps, pd]?
            if point_on_segment(ps, pd, p) {
                // Split edge and return new vertex index.
                let split_edge_result = self.split_edge(he, &p.clone()).unwrap();
                return Some(split_edge_result);
            }
        }

        // 3. If p is strictly inside this triangular face, split it into 3 (RARE case).
        let a = &self.vertices[vs[0]].position;
        let b = &self.vertices[vs[1]].position;
        let c = &self.vertices[vs[2]].position;
        if point_in_or_on_triangle(p, a, b, c) {
            // println!("Point is inside the triangle, splitting... RARE CASE *********************");
            // self.remove_unused_vertices();
            // self.build_boundary_loops();
            // remove_invalidated_faces(self);
            // let _ = write_obj(self, "/mnt/v/cgar_meshes/a_3.obj");
            // panic!("panic for now");
            // 3.a add new vertex
            let w = self.add_vertex(p.clone());

            // 3.b get the original face's half-edge before invalidation
            let original_he = self.faces[face].half_edge;

            // 3.c invalidate the original face
            self.faces[face].half_edge = usize::MAX;
            // println!("invalidating face {}", face);

            // 3.d create 3 new faces incrementally
            let new_face_1 = self.add_triangle(vs[0], vs[1], w); // (a,b,w)
            let new_face_2 = self.add_triangle(vs[1], vs[2], w); // (b,c,w)
            let new_face_3 = self.add_triangle(vs[2], vs[0], w); // (c,a,w)

            return Some(SplitResult {
                kind: SplitResultKind::SplitFace,
                new_vertex: w,
                new_faces: vec![new_face_1, new_face_2, new_face_3],
            });
        }

        None
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

    pub fn compute_mesh_bounds(&self) -> Aabb<T, 3, Point<T, 3>>
    where
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        if self.vertices.is_empty() {
            // Return a degenerate AABB if no vertices
            let origin = Point::<T, 3>::from_vals([T::zero(), T::zero(), T::zero()]);
            return Aabb::from_points(&origin, &origin);
        }

        let first_vertex = &self.vertices[0].position;
        let mut min_point = Point::<T, 3>::from_vals([
            first_vertex[0].clone(),
            first_vertex[1].clone(),
            first_vertex[2].clone(),
        ]);
        let mut max_point = Point::<T, 3>::from_vals([
            first_vertex[0].clone(),
            first_vertex[1].clone(),
            first_vertex[2].clone(),
        ]);

        for vertex in &self.vertices[1..] {
            let pos = &vertex.position;
            for i in 0..3 {
                if pos[i] < min_point[i] {
                    min_point[i] = pos[i].clone();
                }
                if pos[i] > max_point[i] {
                    max_point[i] = pos[i].clone();
                }
            }
        }

        Aabb::from_points(&min_point, &max_point)
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
    u >= -e.clone() && v >= -e.clone() && u + v <= T::one() + e
}

pub trait BooleanImpl<T>
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn boolean(&self, other: &Self, op: BooleanOp) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum VertexSource {
    A,
    B,
}

impl<T: Scalar> BooleanImpl<T> for Mesh<T, 3>
where
    T: Scalar,
    CgarF64: From<T>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn boolean(&self, other: &Mesh<T, 3>, op: BooleanOp) -> Mesh<T, 3> {
        let mut a = self.clone();
        let mut b = other.clone();
        a.build_boundary_loops();
        b.build_boundary_loops();

        // 1. Collect ALL intersection segments
        let mut intersection_segments = Vec::new();
        let tree_b = AabbTree::build((0..b.faces.len()).map(|i| (b.face_aabb(i), i)).collect());

        for fa in 0..a.faces.len() {
            if a.faces[fa].half_edge == usize::MAX {
                continue;
            }

            let mut candidates = Vec::new();
            tree_b.query(&a.face_aabb(fa), &mut candidates);

            let pa_idx = a.face_vertices(fa);
            let pa_vec: Vec<Point<T, 3>> = pa_idx
                .into_iter()
                .map(|vi| a.vertices[vi].position.clone())
                .collect();
            let pa: [Point<T, 3>; 3] = pa_vec.try_into().expect("Expected 3 vertices");

            for &fb in &candidates {
                let pb_idx = b.face_vertices(*fb);
                let pb_vec: Vec<Point<T, 3>> = pb_idx
                    .into_iter()
                    .map(|vi| b.vertices[vi].position.clone())
                    .collect();
                let pb: [Point<T, 3>; 3] = pb_vec.try_into().expect("Expected 3 vertices");

                if let Some(s) = tri_tri_intersection(&pa, &pb) {
                    if s.length().is_positive() {
                        intersection_segments.push(IntersectionSegment::new(
                            s,
                            [fa, usize::MAX],
                            [*fb, usize::MAX],
                        ));
                    }
                }
            }
        }

        filter_coplanar_intersections(&mut intersection_segments, &a, &b);

        // 2. Link intersection segments into chains/graphs
        let chain_roots = link_intersection_segments(&mut intersection_segments);

        // 3. Split faces along intersection curves
        let chains =
            split_mesh_along_intersection_curves(&mut a, &mut intersection_segments, &chain_roots);

        // a.faces.retain(|f| f.half_edge != usize::MAX);

        let mut test = a.clone();
        // 4. Rebuild and cleanup

        // let _ = write_obj(&test, "/mnt/v/cgar_meshes/a.obj");

        for i in &chains[0] {
            // let intersection = &intersection_segments[*i];
            // for f in intersection.faces_a {
            //     // println!("Face index: {}, half-edge {}", f, test.faces[f].half_edge);
            //     let f = find_valid_face(&test, f, &intersection.segment.a, true);
            //     if f.is_none() {
            //         // println!("Face not found in test mesh");
            //         continue;
            //     }
            //     test.faces[f.unwrap()].half_edge = usize::MAX; // Invalidate face
            // }

            let intersection = &intersection_segments[*i];
            // println!("{}: {:?}", i, intersection);
            // println!("------------------------------");
        }

        // test.faces[28].half_edge = usize::MAX;
        // test.faces[29].half_edge = usize::MAX;
        // test.faces[32].half_edge = usize::MAX;
        // test.faces[33].half_edge = usize::MAX; // Invalidate face 27 for testing
        // test.faces[34].half_edge = usize::MAX;
        // test.faces[35].half_edge = usize::MAX;

        test.remove_unused_vertices();
        test.build_boundary_loops();
        remove_invalidated_faces(&mut test);

        let _ = write_obj(&test, "/mnt/v/cgar_meshes/a_2.obj");

        // 5. Build trees for classification
        // let tree_a = AabbTree::build((0..a.faces.len()).map(|i| (a.face_aabb(i), i)).collect());
        // let tree_b = AabbTree::build((0..b.faces.len()).map(|i| (b.face_aabb(i), i)).collect());

        // 6. Create result mesh
        let mut result = Mesh::new();
        let mut vid_map = HashMap::new();

        // Add A vertices
        for (i, v) in a.vertices.iter().enumerate() {
            let ni = result.add_vertex(v.position.clone());
            vid_map.insert((VertexSource::A, i), ni);
        }

        // Classify A faces using topological method
        let a_classifications =
            classify_faces_inside_intersection_loops(&a, &b, &intersection_segments, &chains);

        // remove_invalidated_faces(&mut a);

        for (fa, inside) in a_classifications.iter().enumerate() {
            if a.faces[fa].half_edge == usize::MAX {
                continue; // Skip invalidated faces
            }

            if !*inside {
                let vs = a.face_vertices(fa);
                result.add_triangle(
                    vid_map[&(VertexSource::A, vs[0])],
                    vid_map[&(VertexSource::A, vs[1])],
                    vid_map[&(VertexSource::A, vs[2])],
                );
            }
        }

        // Handle B faces based on operation
        match op {
            BooleanOp::Union => {
                // Add B vertices and classify B faces
                for (i, v) in b.vertices.iter().enumerate() {
                    let ni = result.add_vertex(v.position.clone());
                    vid_map.insert((VertexSource::B, i), ni);
                }

                let b_classifications = classify_faces_inside_intersection_loops(
                    &b,
                    &a,
                    &intersection_segments,
                    &chains,
                );
                for (fb, keep) in b_classifications.iter().enumerate() {
                    if *keep {
                        let vs = b.face_vertices(fb);
                        result.add_triangle(
                            vid_map[&(VertexSource::B, vs[0])],
                            vid_map[&(VertexSource::B, vs[1])],
                            vid_map[&(VertexSource::B, vs[2])],
                        );
                    }
                }
            }
            BooleanOp::Intersection => {
                // Similar to Union but with intersection logic
                for (i, v) in b.vertices.iter().enumerate() {
                    let ni = result.add_vertex(v.position.clone());
                    vid_map.insert((VertexSource::B, i), ni);
                }

                let b_op = BooleanOp::Intersection; // Keep faces inside A
                let b_classifications = classify_faces_inside_intersection_loops(
                    &b,
                    &a,
                    &intersection_segments,
                    &chains,
                );
                for (fb, keep) in b_classifications.iter().enumerate() {
                    if *keep {
                        let vs = b.face_vertices(fb);
                        result.add_triangle(
                            vid_map[&(VertexSource::B, vs[0])],
                            vid_map[&(VertexSource::B, vs[1])],
                            vid_map[&(VertexSource::B, vs[2])],
                        );
                    }
                }
            }
            BooleanOp::Difference => {
                // Add B vertices and flip B faces that are inside A
                for (i, v) in b.vertices.iter().enumerate() {
                    let ni = result.add_vertex(v.position.clone());
                    vid_map.insert((VertexSource::B, i), ni);
                }

                let b_op = BooleanOp::Intersection; // Keep faces inside A, but flip them
                // let b_classifications =
                //     classify_faces_inside_intersection_loops(&b, &intersection_segments, &chains);
                // for (fb, keep) in b_classifications.iter().enumerate() {
                //     if *keep {
                //         let vs = b.face_vertices(fb);
                //         // Flip face orientation for caps
                //         result.add_triangle(
                //             vid_map[&(VertexSource::B, vs[2])],
                //             vid_map[&(VertexSource::B, vs[1])],
                //             vid_map[&(VertexSource::B, vs[0])],
                //         );
                //     }
                // }
            }
        }

        result.remove_unused_vertices();
        result.build_boundary_loops();
        result
    }
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
            &b[0] + &((&c[0] - &b[0]) * t.clone()),
            &b[1] + &((&c[1] - &b[1]) * t.clone()),
            &b[2] + &((&c[2] - &b[2]) * t),
        ]);
        let diff = Point::from_vals([&p[0] - &proj[0], &p[1] - &proj[1], &p[2] - &proj[2]]);
        return &diff[0] * &diff[0] + &diff[1] * &diff[1] + &diff[2] * &diff[2];
    }

    // project p onto the plane
    let t_plane = ap.dot(&n) / nn2.clone();
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
    let u = (&dot11 * &dot02 - &dot01 * &dot12) * inv_denom.clone();
    let v = (&dot00 * &dot12 - &dot01 * &dot02) * inv_denom;

    if u >= T::zero() && v >= T::zero() && u + v <= T::one() {
        let d_plane = ap.dot(&n);
        return d_plane.clone() * d_plane / nn2;
    }

    // if we get here, that means numerical jitter kicked us out of face region
    // but we’ve already tested all three edges above, so this *shouldn’t* happen.
    // As a safe fallback, return the minimum of the three edge distances:
    distance_point_segment_squared(p, a, b)
        .min(distance_point_segment_squared(p, b, c))
        .min(distance_point_segment_squared(p, c, a))
}

fn point_on_segment<T: Scalar, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    p: &Point<T, N>,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Use direct point-to-point distances (more reliable)
    let edge_length = a.distance_to(b); // ✅ Direct distance
    let split_distance = a.distance_to(p); // ✅ Direct distance  
    let distance_from_end = b.distance_to(p); // ✅ Direct distance

    // Early exit for degenerate edge
    if edge_length <= T::edge_degeneracy_threshold() {
        return a.distance_to(p).is_zero() || b.distance_to(p).is_zero();
    }

    // Parametric check (keep your existing logic)
    let ab = b - a;
    let ap = p - a;
    let ab_dot_ab = ab.as_vector().dot(&ab.as_vector());
    let ab_dot_ap = ab.as_vector().dot(&ap.as_vector());

    if ab_dot_ab == T::zero() {
        return a.distance_to(p).is_zero();
    }

    let t = ab_dot_ap / ab_dot_ab;
    if t.is_negative() || t > T::one() {
        return false;
    }

    // Use the corrected distance calculations
    let point_threshold = T::point_merge_threshold();

    // Check if point is at vertices
    if split_distance <= point_threshold || distance_from_end <= point_threshold {
        return false; // Point is at a vertex, not on edge
    }

    // Check if edge split would be valid
    if !should_split_edge(&edge_length, &split_distance) {
        return false;
    }

    // Verify point is actually on the line segment
    let closest = a + &ab.as_vector().scale(&t).0;
    closest.distance_to(p) <= T::point_merge_threshold()
}

fn filter_coplanar_intersections<T: Scalar, const N: usize>(
    segments: &mut Vec<IntersectionSegment<T, N>>,
    _mesh_a: &Mesh<T, N>,
    _mesh_b: &Mesh<T, N>,
) where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let tolerance = T::tolerance();
    let initial_count = segments.len();

    // Only filter truly degenerate segments
    segments.retain(|seg| {
        let length = seg.segment.length();
        if length < tolerance {
            // println!(
            //     "Filtered degenerate segment: {:?} -> {:?}",
            //     seg.segment.a, seg.segment.b
            // );
            false
        } else {
            true
        }
    });

    // println!(
    //     "Filtered intersections: {} -> {} segments",
    //     initial_count,
    //     segments.len()
    // );
}

fn traverse_chain<T: Scalar, const N: usize>(
    intersection_segments: &Vec<IntersectionSegment<T, N>>,
    root: usize,
) -> Vec<usize>
where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
{
    let mut result = Vec::new();
    let mut stack = vec![root];
    let mut visited = HashSet::new();

    while let Some(idx) = stack.pop() {
        if visited.insert(idx) {
            result.push(idx);

            // println!("links: {:?}", intersection_segments[idx].links);

            for &child in &intersection_segments[idx].links {
                if !visited.contains(&child) {
                    stack.push(child);
                }
            }
        }
    }
    result
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

fn deduplicate_segments_improved<T: Scalar, const N: usize>(
    segments: &mut Vec<IntersectionSegment<T, N>>,
) where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>,
{
    let tolerance = T::edge_degeneracy_threshold();

    // println!("=== IMPROVED DEDUPLICATION ===");
    let initial_count = segments.len();

    let mut unique_segments: Vec<IntersectionSegment<T, N>> = Vec::new();

    for (i, seg) in segments.drain(..).enumerate() {
        // Skip degenerate segments
        if seg.segment.length() <= tolerance {
            // println!("Removed degenerate segment {}", i);
            continue;
        }

        // Check for duplicates
        let mut is_duplicate = false;
        for existing in &unique_segments {
            if segments_are_equivalent(&seg.segment, &existing.segment) {
                // println!("Segment {} is duplicate", i);
                is_duplicate = true;
                break;
            }
        }

        if !is_duplicate {
            unique_segments.push(seg);
        }
    }

    *segments = unique_segments;
    // println!(
    //     "Deduplication: {} -> {} segments",
    //     initial_count,
    //     segments.len()
    // );
}

fn segments_are_equivalent<T: Scalar, const N: usize>(
    seg1: &Segment<T, N>,
    seg2: &Segment<T, N>,
) -> bool
where
    Point<T, N>: PointOps<T, N>,
{
    let tolerance = T::point_merge_threshold(); // Use point-specific threshold

    let forward_match = seg1.a.distance_to(&seg2.a) <= tolerance.clone()
        && seg1.b.distance_to(&seg2.b) <= tolerance.clone();

    let reverse_match = seg1.a.distance_to(&seg2.b) <= tolerance.clone()
        && seg1.b.distance_to(&seg2.a) <= tolerance;

    forward_match || reverse_match
}

fn get_or_create_point_index<T: Scalar, const N: usize>(
    point_map: &mut HashMap<String, usize>,
    counter: &mut usize,
    point: &Point<T, N>,
    tolerance: T,
) -> usize
where
    Point<T, N>: PointOps<T, N>,
{
    let key = if tolerance.is_zero() {
        // For exact arithmetic, use exact coordinate representation
        let coords: Vec<String> = (0..N).map(|i| format!("{:?}", point[i])).collect();
        coords.join(",")
    } else {
        // For floating-point, use quantized coordinates
        let quantized_coords = (0..N)
            .map(|i| {
                let coord = &point[i];
                let quantized =
                    (coord.to_f64().unwrap() / tolerance.to_f64().unwrap()).round() as i64;
                quantized
            })
            .collect::<Vec<_>>();
        format!("{:?}", quantized_coords)
    };

    if let Some(&existing_idx) = point_map.get(&key) {
        existing_idx
    } else {
        let new_idx = *counter;
        *counter += 1;
        point_map.insert(key, new_idx);
        new_idx
    }
}

fn find_point_index<T: Scalar, const N: usize>(
    point_map: &HashMap<String, usize>,
    point: &Point<T, N>,
    tolerance: T,
) -> Option<usize>
where
    Point<T, N>: PointOps<T, N>,
{
    let key = if tolerance.is_zero() {
        // For exact arithmetic, use exact coordinate representation
        let coords: Vec<String> = (0..N).map(|i| format!("{:?}", point[i])).collect();
        coords.join(",")
    } else {
        // For floating-point, use quantized coordinates
        let quantized_coords = (0..N)
            .map(|i| {
                let coord = &point[i];
                let quantized =
                    (coord.to_f64().unwrap() / tolerance.to_f64().unwrap()).round() as i64;
                quantized
            })
            .collect::<Vec<_>>();
        format!("{:?}", quantized_coords)
    };

    point_map.get(&key).copied()
}

fn select_best_boundary_loops<T: Scalar, const N: usize>(
    loops: Vec<Vec<usize>>,
    segments: &Vec<IntersectionSegment<T, N>>,
) -> Vec<Vec<usize>>
where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if loops.is_empty() {
        return Vec::new();
    }

    // Score loops by length, geometric properties, and topological significance
    let mut scored_loops: Vec<(f64, Vec<usize>)> = loops
        .into_iter()
        .map(|loop_segs| {
            let score = compute_boundary_loop_score(&loop_segs, segments);
            (score, loop_segs)
        })
        .collect();

    // Sort by score (higher is better)
    scored_loops.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // println!("Boundary loop scores:");
    // for (i, (score, loop_segs)) in scored_loops.iter().enumerate() {
    // println!(
    //     "  Loop {}: {} segments, score: {:.3}",
    //     i,
    //     loop_segs.len(),
    //     score
    // );
    // }

    // Select non-overlapping loops starting with highest score
    let mut selected_loops = Vec::new();
    let mut used_segments = HashSet::new();

    for (_, loop_segs) in scored_loops {
        let overlaps = loop_segs
            .iter()
            .any(|&seg_idx| used_segments.contains(&seg_idx));

        if !overlaps {
            for &seg_idx in &loop_segs {
                used_segments.insert(seg_idx);
            }
            selected_loops.push(loop_segs);
        }
    }

    selected_loops
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
fn link_intersection_segments<T: Scalar, const N: usize>(
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
) -> Vec<usize>
where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // println!("=== MANIFOLD BOUNDARY EXTRACTION ===");
    // println!("Input segments: {}", intersection_segments.len());

    if intersection_segments.is_empty() {
        return Vec::new();
    }

    // 1. Deduplicate segments
    deduplicate_segments_improved(intersection_segments);

    // 2. Extract boundary loops using manifold topology principles
    extract_manifold_boundary_loops(intersection_segments)
}

fn extract_manifold_boundary_loops<T: Scalar, const N: usize>(
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
) -> Vec<usize>
where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let tolerance = T::tolerance();

    // Build adjacency map using quantized endpoints
    let (adjacency_map, point_to_segments, point_map) =
        build_adjacency_map(intersection_segments, tolerance.clone());

    // Find all manifold loops using graph theory
    let loops = find_all_manifold_loops(
        &adjacency_map,
        &point_to_segments,
        &point_map,
        intersection_segments,
        tolerance.clone(),
    );

    // Select the best loops based on geometric criteria
    let selected_loops = select_best_boundary_loops(loops, intersection_segments);

    let mut loop_roots = Vec::new();
    for loop_segments in selected_loops {
        if loop_segments.len() < 3 {
            continue;
        }
        // 1) order segments so seg[i].b == seg[i+1].a
        let mut ordered = Vec::with_capacity(loop_segments.len());
        let mut rem = loop_segments.clone();
        ordered.push(rem.remove(0));

        while !rem.is_empty() {
            let last = *ordered.last().unwrap();
            let last_b = &intersection_segments[last].segment.b;
            let mut found = false;
            for i in 0..rem.len() {
                let idx = rem[i];
                let seg = &intersection_segments[idx].segment;
                if last_b.distance_to(&seg.a) <= tolerance.clone() {
                    ordered.push(idx);
                    rem.remove(i);
                    found = true;
                    break;
                }
                if last_b.distance_to(&seg.b) <= tolerance.clone() {
                    // flip this segment’s endpoints
                    let seg_mut = &mut intersection_segments[idx].segment;
                    std::mem::swap(&mut seg_mut.a, &mut seg_mut.b);
                    ordered.push(idx);
                    rem.remove(i);
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }
        // 2) link each in a closed cycle
        let n = ordered.len();

        // println!("Ordered loop segments len: {}", n);

        for (i, &idx) in ordered.iter().enumerate() {
            let prev = ordered[(i + n - 1) % n];
            let next = ordered[(i + 1) % n];
            intersection_segments[idx].links = vec![prev, next];
        }
        loop_roots.push(ordered[0]);
    }

    // println!("Selected {} manifold loops", loop_roots.len());
    loop_roots
}

fn build_adjacency_map<T: Scalar, const N: usize>(
    segments: &Vec<IntersectionSegment<T, N>>,
    tolerance: T,
) -> (
    HashMap<usize, Vec<(usize, usize)>>,
    HashMap<usize, Vec<usize>>,
    HashMap<String, usize>,
)
where
    Point<T, N>: PointOps<T, N>,
{
    let mut point_map = HashMap::new();
    let mut point_counter = 0;

    // Quantize all endpoints to create a point-to-index mapping
    for (seg_idx, seg) in segments.iter().enumerate() {
        let _a_idx = get_or_create_point_index(
            &mut point_map,
            &mut point_counter,
            &seg.segment.a,
            tolerance.clone(),
        );
        let _b_idx = get_or_create_point_index(
            &mut point_map,
            &mut point_counter,
            &seg.segment.b,
            tolerance.clone(),
        );
    }

    // Build segment adjacency map: segment_idx -> [(neighbor_seg_idx, shared_point_idx)]
    let mut adjacency_map: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    let mut point_to_segments: HashMap<usize, Vec<usize>> = HashMap::new();

    for (seg_idx, seg) in segments.iter().enumerate() {
        let a_idx = find_point_index(&point_map, &seg.segment.a, tolerance.clone()).unwrap();
        let b_idx = find_point_index(&point_map, &seg.segment.b, tolerance.clone()).unwrap();

        point_to_segments.entry(a_idx).or_default().push(seg_idx);
        point_to_segments.entry(b_idx).or_default().push(seg_idx);
    }

    // Build adjacency relationships
    for (point_idx, connected_segments) in &point_to_segments {
        for i in 0..connected_segments.len() {
            for j in (i + 1)..connected_segments.len() {
                let seg_i = connected_segments[i];
                let seg_j = connected_segments[j];

                adjacency_map
                    .entry(seg_i)
                    .or_default()
                    .push((seg_j, *point_idx));
                adjacency_map
                    .entry(seg_j)
                    .or_default()
                    .push((seg_i, *point_idx));
            }
        }
    }

    (adjacency_map, point_to_segments, point_map)
}

fn find_all_manifold_loops<T: Scalar, const N: usize>(
    adjacency_map: &HashMap<usize, Vec<(usize, usize)>>,
    point_to_segments: &HashMap<usize, Vec<usize>>,
    point_map: &HashMap<String, usize>,
    segments: &Vec<IntersectionSegment<T, N>>,
    tolerance: T,
) -> Vec<Vec<usize>>
where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
{
    let mut loops = Vec::new();
    let mut used_segments = HashSet::new();

    // Process each unused segment as a potential loop start
    for start_seg in 0..segments.len() {
        if used_segments.contains(&start_seg) {
            continue;
        }

        if let Some(loop_segments) = trace_manifold_loop(
            start_seg,
            adjacency_map,
            point_to_segments,
            point_map,
            segments,
            &used_segments,
            tolerance.clone(),
        ) {
            // Mark segments as used
            for &seg_idx in &loop_segments {
                used_segments.insert(seg_idx);
            }
            loops.push(loop_segments);
        }
    }

    // println!("Found {} potential manifold loops", loops.len());
    loops
}

fn trace_manifold_loop<T: Scalar, const N: usize>(
    start_seg: usize,
    adjacency_map: &HashMap<usize, Vec<(usize, usize)>>,
    _point_to_segments: &HashMap<usize, Vec<usize>>,
    point_map: &HashMap<String, usize>,
    segments: &Vec<IntersectionSegment<T, N>>,
    used_segments: &HashSet<usize>,
    tolerance: T,
) -> Option<Vec<usize>>
where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
{
    if used_segments.contains(&start_seg) {
        return None;
    }

    // Get the two endpoints of the starting segment
    let start_point_a =
        find_point_index(point_map, &segments[start_seg].segment.a, tolerance.clone())?;
    let start_point_b =
        find_point_index(point_map, &segments[start_seg].segment.b, tolerance.clone())?;

    // Try tracing from both endpoints to find the longest valid loop
    let loop_a = trace_from_endpoint(
        start_seg,
        start_point_b,
        start_point_a,
        adjacency_map,
        point_map,
        segments,
        used_segments,
        tolerance.clone(),
    );
    let loop_b = trace_from_endpoint(
        start_seg,
        start_point_a,
        start_point_b,
        adjacency_map,
        point_map,
        segments,
        used_segments,
        tolerance.clone(),
    );

    // Return the longer valid loop
    match (loop_a, loop_b) {
        (Some(a), Some(b)) => Some(if a.len() > b.len() { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn trace_from_endpoint<T: Scalar, const N: usize>(
    start_seg: usize,
    current_point: usize,
    target_point: usize,
    adjacency_map: &HashMap<usize, Vec<(usize, usize)>>,
    point_map: &HashMap<String, usize>,
    segments: &Vec<IntersectionSegment<T, N>>,
    used_segments: &HashSet<usize>,
    tolerance: T,
) -> Option<Vec<usize>>
where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
{
    let mut path = vec![start_seg];
    let mut visited_segments = HashSet::new();
    visited_segments.insert(start_seg);

    let mut current_seg = start_seg;
    let mut current_endpoint = current_point;

    let max_iterations = segments.len() * 2;
    let mut iterations = 0;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            break; // Safety break
        }

        // Find next segment connected to current endpoint
        let next_candidates = if let Some(neighbors) = adjacency_map.get(&current_seg) {
            neighbors
                .iter()
                .filter(|(seg_idx, point_idx)| {
                    *point_idx == current_endpoint
                        && !visited_segments.contains(seg_idx)
                        && !used_segments.contains(seg_idx)
                })
                .map(|(seg_idx, _)| *seg_idx)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        if next_candidates.is_empty() {
            break; // No more connections
        }

        // For manifold loops, prefer segments with exactly 2 connections (typical for loops)
        let next_seg = if next_candidates.len() == 1 {
            next_candidates[0]
        } else {
            // Choose the segment that forms the best geometric continuation
            *next_candidates.iter().min_by_key(|&&seg_idx| {
                // Prefer segments with fewer total connections (more likely to be on boundary)
                adjacency_map
                    .get(&seg_idx)
                    .map_or(0, |neighbors| neighbors.len())
            })?
        };

        visited_segments.insert(next_seg);
        path.push(next_seg);

        // Find the other endpoint of the next segment
        let next_seg_data = &segments[next_seg];
        let next_point_a =
            find_point_index(point_map, &next_seg_data.segment.a, tolerance.clone())?;
        let next_point_b =
            find_point_index(point_map, &next_seg_data.segment.b, tolerance.clone())?;

        current_endpoint = if next_point_a == current_endpoint {
            next_point_b
        } else {
            next_point_a
        };

        current_seg = next_seg;

        // Check for loop closure
        if current_endpoint == target_point && path.len() >= 3 {
            // println!("Found manifold loop with {} segments", path.len());
            return Some(path);
        }
    }

    // Return partial path if it's reasonably long
    if path.len() >= 3 {
        // println!("Found partial manifold path with {} segments", path.len());
        Some(path)
    } else {
        None
    }
}

fn get_face_half_edges_safe<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_idx: usize,
) -> Option<Vec<usize>> {
    if face_idx >= mesh.faces.len() {
        return None;
    }

    let face = &mesh.faces[face_idx];
    if face.half_edge == usize::MAX {
        return None;
    }

    let start_he = face.half_edge;
    if start_he >= mesh.half_edges.len() {
        return None;
    }

    let mut result = Vec::new();
    let mut current_he = start_he;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 100;

    loop {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
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

fn split_mesh_along_intersection_curves<T: Scalar>(
    mesh: &mut Mesh<T, 3>,
    intersection_segments: &mut Vec<IntersectionSegment<T, 3>>,
    chain_roots: &Vec<usize>,
) -> Vec<Vec<usize>>
where
    Point<T, 3>: PointOps<T, 3, Vector = Vector<T, 3>>,
    Vector<T, 3>: VectorOps<T, 3, Cross = Vector<T, 3>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // println!("=== SIMPLE FACE SPLITTING ===");

    let tolerance = T::tolerance();

    let mut chains = Vec::new();

    // Process each chain
    for (chain_idx, &root) in chain_roots.iter().enumerate() {
        // println!("Processing boundary chain {}", chain_idx);

        let mut chain = traverse_chain(intersection_segments, root);
        // println!("Initial chain length: {} segments", chain.len());

        let mut i = 0;
        while i < chain.len() {
            let seg_idx = chain[i];
            let seg = &intersection_segments[seg_idx];

            // println!(
            //     "Processing segment {}/{}: {:?} -> {:?}",
            //     i + 1,
            //     chain.len(),
            //     seg.segment.a,
            //     seg.segment.b
            // );

            process_segment_simple(
                mesh,
                intersection_segments,
                seg_idx,
                &mut chain,
                i,
                tolerance.clone(),
            );
            i += 1;
        }

        chains.push(chain);
    }

    // Remove all invalidated faces at the end
    // remove_invalidated_faces(mesh);
    // println!("Simple face splitting completed");

    chains
}

fn find_valid_face<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_idx: usize,
    point: &Point<T, N>,
    debug: bool,
) -> Option<usize>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if debug {
        // println!(
        //     "Searching for valid face containing point {:?} starting from face {}, he {}",
        //     point, face_idx, mesh.faces[face_idx].half_edge
        // );
    }
    // Recursively search for a valid face containing the point
    if face_idx < mesh.faces.len() && mesh.faces[face_idx].half_edge != usize::MAX {
        // println!("Face is valid.");
        return Some(face_idx);
    }

    if let Some(split_map) = mesh.face_split_map.get(&face_idx) {
        for &new_face_idx in &split_map.new_faces {
            if let Some(valid_idx) = find_valid_face(mesh, new_face_idx, point, debug) {
                if debug {
                    // println!(
                    //     "Found valid face {} for point {:?} starting from face {}",
                    //     valid_idx, point, face_idx
                    // );
                }
                return Some(valid_idx);
            }
        }
    }

    if debug {
        // println!(
        //     "No valid face found for point {:?} starting from face {}",
        //     point, face_idx
        // );
    }

    if face_idx >= mesh.faces.len() || mesh.faces[face_idx].half_edge == usize::MAX {
        if debug {
            // println!("Face index {} is out of bounds", face_idx);
        }
        return None; // Invalid face index
    }

    if point_in_face_simple(mesh, face_idx, point) {
        if debug {
            // println!("Point {:?} is in face {}", point, face_idx);
        }
        return Some(face_idx);
    }
    if let Some(split_map) = mesh.face_split_map.get(&face_idx) {
        if debug {
            // println!(
            //     "Face {} is not valid, checking split map for new faces",
            //     face_idx
            // );
        }
        for &new_face_idx in &split_map.new_faces {
            if let Some(valid_idx) = find_valid_face(mesh, new_face_idx, point, debug) {
                if debug {
                    // println!(
                    //     "Found valid face {} for point {:?} starting from face {}",
                    //     valid_idx, point, face_idx
                    // );
                }
                return Some(valid_idx);
            }
        }
    }
    if debug {
        // println!(
        //     "No valid face found for point {:?} starting from face {}",
        //     point, face_idx
        // );
    }
    None
}

/// Checks if two vertices have an edge between them in the mesh.
fn are_vertices_connected<T: Scalar>(mesh: &Mesh<T, 3>, vertex_a: usize, vertex_b: usize) -> bool {
    if vertex_a >= mesh.vertices.len() || vertex_b >= mesh.vertices.len() {
        return false; // Invalid vertex indices
    }
    for half_edge in &mesh.half_edges {
        if half_edge.vertex == vertex_a && half_edge.twin != usize::MAX {
            if mesh.half_edges[half_edge.twin].vertex == vertex_b {
                return true; // Found an edge between the vertices
            }
        }
    }
    false // No edge found between the vertices
}

fn process_segment_simple<T: Scalar>(
    mesh: &mut Mesh<T, 3>,
    intersection_segments: &mut Vec<IntersectionSegment<T, 3>>,
    segment_idx: usize,
    chain: &mut Vec<usize>,
    chain_idx: usize,
    tolerance: T,
) where
    Point<T, 3>: PointOps<T, 3, Vector = Vector<T, 3>>,
    Vector<T, 3>: VectorOps<T, 3, Cross = Vector<T, 3>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // println!("BEGINNING PROCESSING SEGMENT {}", segment_idx);
    let segment = &intersection_segments[segment_idx];
    let vertex_a;
    let vertex_b;

    if let Some(valid_face_idx) =
        find_valid_face(mesh, segment.faces_a[0], &segment.segment.a, false)
    {
        // println!("FOUND ON FIRST TRY");
        let split_result = mesh.split_or_find_vertex_on_face(valid_face_idx, &segment.segment.a);
        if let Some(split_result) = split_result {
            vertex_a = split_result.new_vertex;
        } else {
            panic!(
                "Failed to split or find vertex on face {} for segment end {:?}",
                segment.faces_a[0], segment.segment.b
            );
            // mesh.split_or_find_vertex_on_face(segment.face_a, &segment.segment.a);
        }
    } else {
        panic!(
            "Failed to find valid face for segment end {:?} on face {}",
            segment.segment.b, segment.faces_a[0]
        );
        // mesh.split_or_find_vertex_on_face(segment.face_a, &segment.segment.a);
    }

    if let Some(valid_face_idx) =
        find_valid_face(mesh, segment.faces_a[0], &segment.segment.b, false)
    {
        // println!("FOUND ON SECOND TRY");
        let split_result = mesh.split_or_find_vertex_on_face(valid_face_idx, &segment.segment.b);
        if let Some(split_result) = split_result {
            vertex_b = split_result.new_vertex;

            if !split_result.new_faces.is_empty() {
                // Update the intersection segment faces to reflect the new split
                let he = mesh.half_edge_between(vertex_a, vertex_b);
                if let Some(he) = he {
                    intersection_segments[segment_idx].faces_a[0] =
                        mesh.half_edges[he].face.unwrap_or(usize::MAX);
                    intersection_segments[segment_idx].faces_a[1] = mesh.half_edges
                        [mesh.half_edges[he].twin]
                        .face
                        .unwrap_or(usize::MAX);
                } else {
                    // If no direct half-edge found, use the new faces from split result
                    if split_result.new_faces.len() >= 2 {
                        intersection_segments[segment_idx].faces_a[0] = split_result.new_faces[0];
                        intersection_segments[segment_idx].faces_a[1] = split_result.new_faces[1];
                    }
                }
                // let he = mesh.half_edge_between(vertex_a, vertex_b);
                // let he = he.expect("No half-edge found between vertices");

                // // Filter 2 faces that share the half-edge
                // intersection_segments[segment_idx].faces_a = [
                //     new_faces.2,
                //     mesh.half_edges[mesh.half_edges[he].twin]
                //         .face
                //         .unwrap_or(usize::MAX),
                // ];
                // // intersection_segments[segment_idx].faces_a = [new_faces[0], new_faces[1]];
            } else {
                // get the faces between the two vertices
                let he = mesh.half_edge_between(vertex_a, vertex_b);
                let he = he.expect("No half-edge found between vertices");
                intersection_segments[segment_idx].faces_a[0] =
                    mesh.half_edges[he].face.unwrap_or(usize::MAX);
                intersection_segments[segment_idx].faces_a[1] = mesh.half_edges
                    [mesh.half_edges[he].twin]
                    .face
                    .unwrap_or(usize::MAX);
            }
        } else {
            panic!(
                "Failed to split or find vertex on face {} for segment end {:?}",
                segment.faces_a[0], segment.segment.b
            );
            // mesh.split_or_find_vertex_on_face(segment.face_a, &segment.segment.b);
        }
    } else {
        panic!(
            "Failed to find valid face for segment end {:?} on face {}",
            segment.segment.b, segment.faces_a[0]
        );
        // mesh.split_or_find_vertex_on_face(segment.face_a, &segment.segment.b);
    }

    // println!("Vertices are: {} and {}", vertex_a, vertex_b);

    if vertex_a != usize::MAX
        && vertex_b != usize::MAX
        && !are_vertices_connected(mesh, vertex_a, vertex_b)
    {
        // println!("******** CARVING **********");
        // Carve segment from vertex_a to vertex_b
        let success = carve_segment_to_vertex(
            mesh,
            vertex_a,
            vertex_b,
            intersection_segments,
            segment_idx,
            chain,
            chain_idx,
        );

        if !success {
            // println!(
            //     "Warning: Failed to carve segment from vertex {} to vertex {}",
            //     vertex_a, vertex_b
            // );
        }
    }
}

fn point_in_face_simple<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_idx: usize,
    point: &Point<T, N>,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let face_vertices = mesh.face_vertices(face_idx);
    if face_vertices.len() != 3 {
        return false;
    }

    let v0 = &mesh.vertices[face_vertices[0]].position;
    let v1 = &mesh.vertices[face_vertices[1]].position;
    let v2 = &mesh.vertices[face_vertices[2]].position;

    point_in_or_on_triangle(point, v0, v1, v2)
}

fn remove_invalidated_faces<T: Scalar, const N: usize>(mesh: &mut Mesh<T, N>) {
    let original_count = mesh.faces.len();

    // Filter out invalidated faces
    let mut new_faces = Vec::new();
    let mut face_mapping = vec![None; mesh.faces.len()];

    for (old_idx, face) in mesh.faces.iter().enumerate() {
        if face.half_edge != usize::MAX {
            let new_idx = new_faces.len();
            face_mapping[old_idx] = Some(new_idx);
            new_faces.push(face.clone());
        }
    }

    // Update half-edge face references
    for he in &mut mesh.half_edges {
        if let Some(ref mut face_ref) = he.face {
            if let Some(new_face_idx) = face_mapping[*face_ref] {
                *face_ref = new_face_idx;
            } else {
                he.face = None; // Face was invalidated
            }
        }
    }

    // Replace faces
    mesh.faces = new_faces;

    // println!(
    //     "Removed {} invalidated faces ({} -> {} faces)",
    //     original_count - mesh.faces.len(),
    //     original_count,
    //     mesh.faces.len()
    // );
}

// Enhanced version that maintains mesh quality
impl<T: Scalar, const N: usize> Mesh<T, N> {
    /// Add vertex with automatic deduplication
    pub fn add_vertex_deduplicated(&mut self, position: Point<T, N>, tolerance: T) -> usize
    where
        Point<T, N>: PointOps<T, N>,
    {
        if let Some(existing) = self.find_nearby_vertex(&position, tolerance) {
            existing
        } else {
            self.add_vertex(position)
        }
    }

    /// Validate mesh topology after modifications
    pub fn validate_topology(&self) -> bool {
        // Check face validity
        for (i, face) in self.faces.iter().enumerate() {
            if face.half_edge == usize::MAX {
                continue; // Invalidated face, skip
            }

            if face.half_edge >= self.half_edges.len() {
                // println!(
                //     "Invalid face {}: half_edge index {} >= {}",
                //     i,
                //     face.half_edge,
                //     self.half_edges.len()
                // );
                return false;
            }

            // Validate face vertices
            if let Some(vertices) = get_face_vertices_safe(self, i) {
                if vertices.len() != 3 {
                    // println!(
                    //     "Invalid face {}: {} vertices instead of 3",
                    //     i,
                    //     vertices.len()
                    // );
                    return false;
                }

                for &vertex_idx in &vertices {
                    if vertex_idx >= self.vertices.len() {
                        // println!(
                        //     "Invalid face {}: vertex index {} >= {}",
                        //     i,
                        //     vertex_idx,
                        //     self.vertices.len()
                        // );
                        return false;
                    }
                }
            } else {
                // println!("Invalid face {}: cannot extract vertices", i);
                return false;
            }
        }

        true
    }
}

/// Helper function with improved error handling
fn get_face_vertices_safe<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_idx: usize,
) -> Option<Vec<usize>> {
    if face_idx >= mesh.faces.len() || mesh.faces[face_idx].half_edge == usize::MAX {
        return None;
    }

    // Use existing safe method
    mesh.face_vertices_safe(face_idx)
}

fn carve_segment_to_vertex<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    start_vertex: usize,
    target_vertex: usize,
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    segment_idx: usize,
    chain: &mut Vec<usize>,
    chain_idx: usize,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let tolerance = T::tolerance();
    let segment = &intersection_segments[segment_idx];

    // **FAST PATH: Validation**
    if start_vertex >= mesh.vertices.len() || target_vertex >= mesh.vertices.len() {
        return false;
    }

    if start_vertex == target_vertex {
        return true;
    }

    if segment.segment.length() < tolerance {
        return false;
    }

    // println!(
    //     "DEBUG: Carving from vertex {} to vertex {}",
    //     start_vertex, target_vertex
    // );

    // **STRATEGY 1: Direct edge exists**
    if mesh
        .half_edge_between(start_vertex, target_vertex)
        .is_some()
    {
        // println!("DEBUG: Direct edge already exists");
        return true;
    }

    // **STRATEGY 2: Shared face connection**
    if let Some(shared_face) = find_shared_face_iterative(mesh, start_vertex, target_vertex) {
        // println!("DEBUG: Found shared face {}", shared_face);
        return create_edge_in_shared_face(mesh, shared_face, start_vertex, target_vertex);
    }

    // **STRATEGY 3: Adjacent face connection**
    if let Some(connection_path) = find_adjacent_face_connection(mesh, start_vertex, target_vertex)
    {
        // println!("DEBUG: Found adjacent face connection");
        return mesh
            .find_edge_intersection(
                connection_path.start_face,
                intersection_segments,
                segment_idx,
                chain,
                chain_idx,
                start_vertex,
            )
            .is_some();
    }

    // **STRATEGY 4: Direct geometric connection (non-recursive)**
    // println!("DEBUG: Using direct geometric connection");
    create_direct_geometric_connection(mesh, start_vertex, target_vertex)
}

/// Find shared face using iterative approach (no recursion)
fn find_shared_face_iterative<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    vertex_a: usize,
    vertex_b: usize,
) -> Option<usize> {
    for (face_idx, face) in mesh.faces.iter().enumerate() {
        if face.half_edge == usize::MAX {
            continue;
        }

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
    if face.half_edge == usize::MAX || face.half_edge >= mesh.half_edges.len() {
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

        if current_he >= mesh.half_edges.len() {
            return None;
        }

        result.push(mesh.half_edges[current_he].vertex);

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

/// Find connection through adjacent faces (non-recursive)
fn find_adjacent_face_connection<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    start_vertex: usize,
    target_vertex: usize,
) -> Option<AdjacentFaceConnection> {
    // Find all faces containing start_vertex
    let start_faces = find_faces_containing_vertex_safe(mesh, start_vertex);
    let target_faces = find_faces_containing_vertex_safe(mesh, target_vertex);

    // Look for faces that share an edge
    for &start_face in &start_faces {
        for &target_face in &target_faces {
            if start_face == target_face {
                continue; // Same face already handled
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

        if face.half_edge == usize::MAX {
            continue;
        }

        if let Some(face_vertices) = get_face_vertices_with_timeout(mesh, face_idx, 50) {
            if face_vertices.contains(&vertex) {
                faces.push(face_idx);
            }
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
    let vertices_a = get_face_vertices_with_timeout(mesh, face_a, 50)?;
    let vertices_b = get_face_vertices_with_timeout(mesh, face_b, 50)?;

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

/// Create edge in shared face (atomic operation)
fn create_edge_in_shared_face<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    face_idx: usize,
    vertex_a: usize,
    vertex_b: usize,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let face_vertices = get_face_vertices_with_timeout(mesh, face_idx, 50);
    if face_vertices.is_none() {
        return false;
    }
    let face_vertices = face_vertices.unwrap();

    if face_vertices.len() != 3 {
        return false;
    }

    // Find third vertex
    let vertex_c = face_vertices
        .iter()
        .find(|&&v| v != vertex_a && v != vertex_b)
        .copied();

    if vertex_c.is_none() {
        return false;
    }
    let vertex_c = vertex_c.unwrap();

    // Validate triangle is not degenerate
    let pos_a = &mesh.vertices[vertex_a].position;
    let pos_b = &mesh.vertices[vertex_b].position;
    let pos_c = &mesh.vertices[vertex_c].position;

    if !is_triangle_non_degenerate(pos_a, pos_b, pos_c) {
        return false;
    }

    // **ATOMIC OPERATION: Replace face**
    mesh.faces[face_idx].half_edge = usize::MAX;
    // println!("invalidating face {}", face_idx);

    // println!("*********************** create_edge_in_shared_face  ************************");

    // Add exactly one triangle with correct orientation
    if preserve_face_orientation(&face_vertices, vertex_a, vertex_b, vertex_c) {
        mesh.add_triangle(vertex_a, vertex_b, vertex_c);
    } else {
        mesh.add_triangle(vertex_b, vertex_a, vertex_c);
    }

    // println!("DEBUG: Created edge in shared face {}", face_idx);
    true
}

/// Direct geometric connection (non-recursive fallback)
fn create_direct_geometric_connection<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    vertex_a: usize,
    vertex_b: usize,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Find any face containing vertex_a
    let faces_a = find_faces_containing_vertex_safe(mesh, vertex_a);
    if faces_a.is_empty() {
        return false;
    }

    let face_idx = faces_a[0];
    let face_vertices = get_face_vertices_with_timeout(mesh, face_idx, 50);
    if face_vertices.is_none() {
        return false;
    }
    let face_vertices = face_vertices.unwrap();

    if face_vertices.len() != 3 {
        return false;
    }

    // Find other vertices in the face
    let other_vertices: Vec<usize> = face_vertices
        .iter()
        .filter(|&&v| v != vertex_a)
        .copied()
        .collect();

    if other_vertices.len() != 2 {
        return false;
    }

    // **ATOMIC OPERATION: Replace face with fan triangulation**
    mesh.faces[face_idx].half_edge = usize::MAX;
    // println!("invalidating face {}", face_idx);

    // println!(
    //     "*********************** create_direct_geometric_connection  ************************"
    // );

    // Create triangles connecting to target vertex
    mesh.add_triangle(vertex_a, vertex_b, other_vertices[0]);
    mesh.add_triangle(vertex_b, other_vertices[1], other_vertices[0]);
    mesh.add_triangle(vertex_a, other_vertices[1], vertex_b);

    // println!("DEBUG: Created direct geometric connection");
    true
}

/// Check if triangle is non-degenerate
fn is_triangle_non_degenerate<T: Scalar, const N: usize>(
    pos_a: &Point<T, N>,
    pos_b: &Point<T, N>,
    pos_c: &Point<T, N>,
) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Check edge lengths
    if pos_a.distance_to(pos_b) < T::edge_degeneracy_threshold()
        || pos_b.distance_to(pos_c) < T::edge_degeneracy_threshold()
        || pos_c.distance_to(pos_a) < T::edge_degeneracy_threshold()
    {
        return false;
    }

    // Check area
    let edge1 = (pos_b - pos_a).as_vector();
    let edge2 = (pos_c - pos_a).as_vector();
    let cross = edge1.cross(&edge2);

    cross.norm() > T::area_degeneracy_threshold()
}

/// Preserve face orientation when creating new triangle
fn preserve_face_orientation(
    face_vertices: &[usize],
    vertex_a: usize,
    vertex_b: usize,
    vertex_c: usize,
) -> bool {
    let pos_a = face_vertices
        .iter()
        .position(|&v| v == vertex_a)
        .unwrap_or(0);
    let pos_b = face_vertices
        .iter()
        .position(|&v| v == vertex_b)
        .unwrap_or(1);
    let pos_c = face_vertices
        .iter()
        .position(|&v| v == vertex_c)
        .unwrap_or(2);

    // Maintain cyclic order
    ((pos_a + 1) % 3 == pos_b && (pos_b + 1) % 3 == pos_c)
        || ((pos_a + 2) % 3 == pos_c && (pos_c + 1) % 3 == pos_b)
}

fn compute_boundary_loop_score<T: Scalar, const N: usize>(
    loop_segments: &Vec<usize>,
    segments: &Vec<IntersectionSegment<T, N>>,
) -> f64
where
    Point<T, N>: PointOps<T, N>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if loop_segments.is_empty() {
        return 0.0;
    }

    // 1. Length score (longer loops are generally better for outer boundaries)
    let length_score = loop_segments.len() as f64 * 10.0;

    // 2. Geometric perimeter score
    let total_length: f64 = loop_segments
        .iter()
        .map(|&seg_idx| segments[seg_idx].segment.length().to_f64().unwrap())
        .sum();
    let perimeter_score = total_length * 5.0;

    // 3. Regularity score (prefer loops with more uniform segment lengths)
    let avg_length = total_length / loop_segments.len() as f64;
    let variance: f64 = loop_segments
        .iter()
        .map(|&seg_idx| {
            let len = segments[seg_idx].segment.length().to_f64().unwrap();
            (len - avg_length).powi(2)
        })
        .sum::<f64>()
        / loop_segments.len() as f64;

    let regularity_score = if variance < 1e-10 {
        5.0
    } else {
        5.0 / (1.0 + variance)
    };

    // 4. Closure bonus (complete loops are much better than partial paths)
    let closure_bonus = if is_closed_loop(loop_segments, segments) {
        50.0
    } else {
        0.0
    };

    // 5. Topological validity score for flood fill classification
    let topology_score = if is_manifold_boundary_loop(loop_segments, segments) {
        100.0 // High bonus for valid manifold boundaries
    } else {
        20.0 // Lower score for non-manifold boundaries
    };

    length_score + perimeter_score + regularity_score + closure_bonus + topology_score
}

fn is_manifold_boundary_loop<T: Scalar, const N: usize>(
    loop_segments: &Vec<usize>,
    segments: &Vec<IntersectionSegment<T, N>>,
) -> bool
where
    Point<T, N>: PointOps<T, N>,
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
    T: Clone,
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
            let ratio = coord.clone() / tolerance.clone();
            let ratio_f64 = ratio.to_f64().unwrap();
            ratio_f64.round() as i64
        })
        .collect();
    format!("{:?}", quantized_coords)
}

// Pre-build face adjacency graph to eliminate recursion during flood fill
fn build_face_adjacency_graph<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
) -> HashMap<usize, Vec<usize>> {
    let mut adjacency_graph = HashMap::new();

    for face_idx in 0..mesh.faces.len() {
        if mesh.faces[face_idx].half_edge == usize::MAX {
            continue;
        }

        let mut adjacent_faces = Vec::new();

        // Use iterative approach with safety limits
        if let Some(half_edges) = get_face_half_edges_iterative(mesh, face_idx) {
            for he_idx in half_edges {
                if he_idx < mesh.half_edges.len() {
                    let twin_idx = mesh.half_edges[he_idx].twin;

                    if twin_idx != usize::MAX && twin_idx < mesh.half_edges.len() {
                        if let Some(twin_face) = mesh.half_edges[twin_idx].face {
                            if twin_face != face_idx
                                && twin_face < mesh.faces.len()
                                && mesh.faces[twin_face].half_edge != usize::MAX
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

    // println!("Built adjacency graph for {} faces", adjacency_graph.len());
    adjacency_graph
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
    if face.half_edge == usize::MAX {
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

fn build_robust_boundary_map<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    intersection_segments: &[IntersectionSegment<T, N>],
    loop_segments: &[Vec<usize>],
) -> HashMap<usize, usize>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let mut boundary_map = HashMap::new();

    // println!("Number of loop segments: {}", loop_segments.len());

    // Mark all face pairs that have intersection segments between them
    for loop_segs in loop_segments {
        // println!("Processing loop with {} segments", loop_segs.len());
        for &seg_idx in loop_segs {
            if seg_idx < intersection_segments.len() {
                let seg = &intersection_segments[seg_idx];

                let face_a_1 = find_valid_face(mesh, seg.faces_a[0], &seg.segment.a, true);
                let face_a_2 = find_valid_face(mesh, seg.faces_a[1], &seg.segment.b, true);

                let face_a_1 = face_a_1.expect("Failed to find valid face A1");
                let face_a_2 = face_a_2.expect("Failed to find valid face A2");

                boundary_map.insert(face_a_1, face_a_2);
                boundary_map.insert(face_a_2, face_a_1);
            }
        }
    }

    // println!(
    //     "Built robust boundary set with {} entries",
    //     boundary_map.len()
    // );
    boundary_map
}

fn classify_faces_inside_intersection_loops<T: Scalar, const N: usize>(
    a: &Mesh<T, N>,
    b: &Mesh<T, N>,
    intersection_segments: &[IntersectionSegment<T, N>],
    loop_segments: &[Vec<usize>],
) -> Vec<bool>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // build adjacency & boundary‐map
    let adj = build_face_adjacency_graph(a);
    let boundary_map = build_robust_boundary_map(a, intersection_segments, loop_segments);

    let mut state: Vec<bool> = vec![false; a.faces.len()];

    let tree_b = AabbTree::<T, 3, _, _>::build(
        (0..b.faces.len())
            .map(|i| {
                let aabb_n = b.face_aabb(i);
                // Convert Aabb<T, N, Point<T, N>> to Aabb<T, 3, Point<T, 3>>
                let aabb3 = Aabb::<T, 3, Point<T, 3>>::from_points(
                    &Point::<T, 3>::from_vals([
                        aabb_n.min()[0].clone(),
                        aabb_n.min()[1].clone(),
                        aabb_n.min()[2].clone(),
                    ]),
                    &Point::<T, 3>::from_vals([
                        aabb_n.max()[0].clone(),
                        aabb_n.max()[1].clone(),
                        aabb_n.max()[2].clone(),
                    ]),
                );
                (aabb3, i)
            })
            .collect::<Vec<(Aabb<T, 3, Point<T, 3>>, usize)>>(),
    );

    // 3) pick a seed face that lies inside B
    let seed = boundary_map
        .keys()
        .copied()
        .find(|&f| {
            f < a.faces.len() && a.faces[f].half_edge != usize::MAX && {
                let c = a.face_centroid(f).0;
                let c3 = Point::<T, 3>::from_vals([c[0].clone(), c[1].clone(), c[2].clone()]);
                b.point_in_mesh(&tree_b, &c3)
            }
        })
        .unwrap_or(0);
    state[seed] = true;

    // 4) iterative flood‐fill without crossing the boundary_map
    let mut visited = vec![false; a.faces.len()];
    let mut queue = VecDeque::new();
    visited[seed] = true;
    queue.push_back(seed);

    while let Some(curr) = queue.pop_front() {
        if let Some(neighbors) = adj.get(&curr) {
            for &nbr in neighbors {
                if visited[nbr] {
                    continue;
                }
                // skip if this edge is in the boundary_map
                if boundary_map.get(&curr) == Some(&nbr) || boundary_map.get(&nbr) == Some(&curr) {
                    continue;
                }
                visited[nbr] = true;
                state[nbr] = true;
                queue.push_back(nbr);
            }
        }
    }

    // for (k, v) in boundary_map {
    //     state[k] = true;
    //     state[v] = true;
    // }

    state
}

// Helper: Find closest face to point without recursion
fn find_closest_face_to_point<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_candidates: &[usize],
    target_point: &Point<T, N>,
) -> Option<usize>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let mut closest_face = None;
    let mut min_distance = None;

    for &face_idx in face_candidates {
        if mesh.faces[face_idx].half_edge == usize::MAX {
            continue;
        }

        let face_centroid = mesh.face_centroid(face_idx);
        let centroid_point = face_centroid.0;
        let distance = centroid_point.distance_to(target_point);

        if min_distance.as_ref().is_none() || distance < *min_distance.as_ref().unwrap() {
            min_distance = Some(distance);
            closest_face = Some(face_idx);
        }
    }

    closest_face
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

    // println!(
    //     "Edge length: {}, Split point distance: {}, Min edge length: {}, Min split distance: {}",
    //     edge_length.to_f64().unwrap(),
    //     split_point_distance.to_f64().unwrap(),
    //     min_edge_length.to_f64().unwrap(),
    //     min_split_distance.to_f64().unwrap()
    // );

    edge_length > &min_edge_length.clone()
        && split_point_distance > &min_split_distance.clone()
        && (edge_length - &split_point_distance) > min_split_distance
}
