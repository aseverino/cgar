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
    collections::{HashMap, HashSet},
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

#[derive(Debug, Clone)]
pub struct Mesh<T: Scalar, const N: usize> {
    pub vertices: Vec<Vertex<T, N>>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<Face>,

    pub edge_map: HashMap<(usize, usize), usize>,
    vertex_spatial_hash: HashMap<(i64, i64, i64), Vec<usize>>,
}

pub struct SplitEdgeResult {
    new_vertex: usize,
    new_faces: Option<Vec<usize>>,
    invalidated_faces: Option<Vec<usize>>,
}

impl<T: Scalar, const N: usize> Mesh<T, N> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            half_edges: Vec::new(),
            faces: Vec::new(),
            edge_map: HashMap::new(),
            vertex_spatial_hash: HashMap::new(),
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

    /// Remove unused vertices and remap face indices
    // pub fn remove_unused_vertices(&mut self) {
    //     // Find vertices that are actually referenced by faces
    //     let mut used_vertices = std::collections::HashSet::new();
    //     for face_idx in 0..self.faces.len() {
    //         let face_verts = self.face_vertices(face_idx);
    //         for &v_idx in &face_verts {
    //             used_vertices.insert(v_idx);
    //         }
    //     }

    //     // Remove unused vertices in reverse order (highest index first)
    //     let mut unused_vertices: Vec<usize> = (0..self.vertices.len())
    //         .filter(|idx| !used_vertices.contains(idx))
    //         .collect();
    //     unused_vertices.sort_by(|a, b| b.cmp(a)); // Reverse order

    //     for &vertex_idx in &unused_vertices {
    //         self.remove_vertex(vertex_idx);
    //     }
    // }
    pub fn remove_unused_vertices(&mut self) {
        // 1. Find used vertices
        let mut used = vec![false; self.vertices.len()];
        for face_idx in 0..self.faces.len() {
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

        // 3. Remap faces
        let mut new_mesh = Mesh::new();
        for v in &new_vertices {
            new_mesh.add_vertex(v.position.clone());
        }
        for face_idx in 0..self.faces.len() {
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
    ) -> Option<SplitEdgeResult>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        // println!("split_or_find_vertex_on_face");
        let tolerance = T::from(1e-6);

        // 1. Check if position matches an existing vertex on this face (still fast)
        let face_vertices = self.face_vertices(face);
        for &vi in &face_vertices {
            if self.vertices[vi].position.distance_to(pos) < tolerance {
                println!("Found existing vertex {} at position {:?}", vi, pos);
                return Some(SplitEdgeResult {
                    new_vertex: vi,
                    new_faces: None,
                    invalidated_faces: None,
                });
            }
        }

        // 2. Use spatial hash to find nearby vertices (much faster than full scan)
        if let Some(existing_vi) = self.find_nearby_vertex(pos, tolerance.clone()) {
            println!(
                "Found existing vertex {} near position {:?}",
                existing_vi, pos
            );
            return Some(SplitEdgeResult {
                new_vertex: existing_vi,
                new_faces: None,
                invalidated_faces: None,
            });
        }

        // 3. Check if position lies on an edge of this face
        for &he in &self.face_half_edges(face) {
            let src = self.half_edges[self.half_edges[he].prev].vertex;
            let dst = self.half_edges[he].vertex;
            let edge_start = &self.vertices[src].position;
            let edge_end = &self.vertices[dst].position;

            if point_on_segment(edge_start, edge_end, pos) {
                if let Ok(split_edge_result) = self.split_edge(he, pos.clone()) {
                    println!(
                        "Inserted new vertex {} on edge ({}, {}) at position {:?}",
                        split_edge_result.new_vertex, src, dst, pos
                    );
                    return Some(split_edge_result);
                } else {
                    println!(
                        "Failed to split edge ({}, {}) at position {:?}",
                        src, dst, pos
                    );
                    return Some(SplitEdgeResult {
                        new_vertex: self.add_vertex(pos.clone()),
                        new_faces: None,
                        invalidated_faces: None,
                    });
                }
            }
        }

        // 4. Position is inside the face - split the face into triangles
        if let Some(new_vertex) = self.find_or_insert_vertex_on_face(face, pos) {
            // println!("recursive!");
            return Some(new_vertex);
        }

        for f in self.faces_containing_point(pos) {
            if let Some(new_vertex) = self.find_or_insert_vertex_on_face(f, pos) {
                return Some(new_vertex);
            }
        }
        // .into_iter()
        // .next()
        // .and_then(|f| self.find_or_insert_vertex_on_face(f, pos))
        // .unwrap_or_else(|| {
        //     // If no suitable vertex found, add a new vertex at the position
        //     println!(
        //         "No suitable vertex found, adding new vertex at position {:?}",
        //         pos
        //     );
        //     return SplitEdgeResult {
        //         new_vertex: self.add_vertex(pos.clone()),
        //         new_faces: None,
        //         invalidated_faces: None,
        //     };
        // });

        // 5. Fallback: just add the vertex
        // println!("Fallback: Adding vertex at position {:?}", pos);
        // return SplitEdgeResult {
        //     new_vertex: self.add_vertex(pos.clone()),
        //     new_faces: None,
        //     invalidated_faces: None,
        // };

        println!(
            "No suitable vertex found, adding new vertex at position {:?}",
            pos
        );

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
    pub fn add_edge_if_not_exists(&mut self, vi0: usize, vi1: usize) -> bool
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        // Check if edge already exists
        if self.half_edge_between(vi0, vi1).is_some() {
            return false; // Edge already exists
        }

        // Find faces that contain both vertices
        let mut shared_faces = Vec::new();
        for face_idx in 0..self.faces.len() {
            if self.faces[face_idx].half_edge == usize::MAX {
                continue; // Skip invalidated faces
            }
            let face_verts = self.face_vertices(face_idx);
            if face_verts.contains(&vi0) && face_verts.contains(&vi1) {
                shared_faces.push(face_idx);
            }
        }

        match shared_faces.len() {
            0 => {
                self.carve_segment_across_faces(vi0, vi1);
                false
            }
            1 => {
                // Vertices are on the same face - split the face
                let face = shared_faces[0];
                self.split_segment_by_indices(face, vi0, vi1, false);
                true
            }
            _ => {
                // Multiple shared faces - edge already exists implicitly
                // This is typically an error case
                eprintln!("Warning: Vertices share multiple faces");
                false
            }
        }
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
        let grid_size = 1e-5; // Adjust based on your precision needs
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

        // let n_rays = 12;
        // for _ in 0..n_rays {
        //     let dir: Vector<T, 3> = random_unit_vector();
        //     if let Some(is_inside) = self.cast_ray(p, &dir, tree) {
        //         if is_inside {
        //             inside_count += 1;
        //         }
        //         total_rays += 1;
        //     }
        // }

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
                if t > T::from(1e-10) {
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
                (t.clone() - lt.clone()).abs() > T::from(1e-8)
            }) {
                filtered_hits.push(t.clone());
                last_t = Some(t);
            }
        }

        Some(filtered_hits.len() % 2 == 1)
    }

    // pub fn point_in_mesh(&self, _tree: &AabbTree<T, 3, Point<T, 3>, usize>, p: &Point<T, 3>) -> bool
    // where
    //     T: Scalar,
    //     for<'a> &'a T: Sub<&'a T, Output = T>
    //         + Mul<&'a T, Output = T>
    //         + Add<&'a T, Output = T>
    //         + Div<&'a T, Output = T>,
    // {
    //     // 0) First check if point is exactly ON the mesh surface
    //     let distance_to_surface = self.point_to_mesh_distance(_tree, p);
    //     if distance_to_surface.is_zero() {
    //         return false; // Points ON the surface are not "inside"
    //     }

    //     let mut inside_count = 0;
    //     let mut total_rays = 0;

    //     let n_rays = 32;
    //     for _ in 0..n_rays {
    //         let dir: Vector<T, 3> = random_unit_vector();
    //         if let Some(is_inside) = self.cast_ray(p, &dir) {
    //             if is_inside {
    //                 inside_count += 1;
    //             }
    //             total_rays += 1;
    //         }
    //     }

    //     // Majority vote: if most rays say "inside", then it's inside
    //     total_rays > 0 && inside_count > total_rays / 2
    // }

    // fn cast_ray(&self, p: &Point<T, 3>, dir: &Vector<T, 3>) -> Option<bool>
    // where
    //     T: Scalar,
    //     for<'a> &'a T: Sub<&'a T, Output = T>
    //         + Mul<&'a T, Output = T>
    //         + Add<&'a T, Output = T>
    //         + Div<&'a T, Output = T>,
    // {
    //     let orig = p.coords();
    //     let mut hits: Vec<T> = Vec::new();

    //     for fi in 0..self.faces.len() {
    //         let vs_idxs = self.face_vertices(fi);
    //         let vs: [&Point<T, 3>; 3] = [
    //             &Point::<T, 3>::from_vals([
    //                 self.vertices[vs_idxs[0]].position[0].clone(),
    //                 self.vertices[vs_idxs[0]].position[1].clone(),
    //                 self.vertices[vs_idxs[0]].position[2].clone(),
    //             ]),
    //             &Point::<T, 3>::from_vals([
    //                 self.vertices[vs_idxs[1]].position[0].clone(),
    //                 self.vertices[vs_idxs[1]].position[1].clone(),
    //                 self.vertices[vs_idxs[1]].position[2].clone(),
    //             ]),
    //             &Point::<T, 3>::from_vals([
    //                 self.vertices[vs_idxs[2]].position[0].clone(),
    //                 self.vertices[vs_idxs[2]].position[1].clone(),
    //                 self.vertices[vs_idxs[2]].position[2].clone(),
    //             ]),
    //         ];

    //         // Use Möller–Trumbore with better precision handling
    //         if let Some(t) = self.ray_triangle_intersection(&orig, dir, vs) {
    //             if t > T::from(1e-10) {
    //                 // Ignore hits very close to ray origin
    //                 hits.push(t);
    //             }
    //         }
    //     }

    //     if hits.is_empty() {
    //         return None; // Ray didn't hit anything
    //     }

    //     // Sort and remove near-duplicates more carefully
    //     hits.sort_by(|a, b| a.partial_cmp(b).unwrap());

    //     let mut filtered_hits = Vec::new();
    //     let mut last_t = None;

    //     for t in hits {
    //         if last_t.as_ref().map_or(true, |lt: &T| {
    //             (t.clone() - lt.clone()).abs() > T::from(1e-8)
    //         }) {
    //             filtered_hits.push(t.clone());
    //             last_t = Some(t);
    //         }
    //     }

    //     Some(filtered_hits.len() % 2 == 1)
    // }

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

        // --- 6) stitch up face f1 to be the triangle (d, c, v) ---
        // We pick the cycle [he_e, he_d, he_f] so that dests are [d, c, v]:
        self.half_edges[he_e].next = he_d;
        self.half_edges[he_d].next = he_f;
        self.half_edges[he_f].next = he_e;

        self.half_edges[he_d].prev = he_e;
        self.half_edges[he_f].prev = he_d;
        self.half_edges[he_e].prev = he_f;

        self.faces[f0].half_edge = he_e;

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

    pub fn split_edge(
        &mut self,
        he: usize,
        pos: Point<T, N>,
    ) -> Result<SplitEdgeResult, &'static str> {
        //println!("split_edge: he={}", he);
        let prev = self.half_edges[he].prev;
        let u = self.half_edges[prev].vertex;
        let v = self.half_edges[he].vertex;

        // Check if we already have this position
        if pos == self.vertices[u].position {
            return Ok(SplitEdgeResult {
                new_vertex: u,
                new_faces: None,
                invalidated_faces: None,
            });
        }
        if pos == self.vertices[v].position {
            return Ok(SplitEdgeResult {
                new_vertex: v,
                new_faces: None,
                invalidated_faces: None,
            });
        }

        // Add new vertex
        let w = self.vertices.len();
        self.vertex_spatial_hash
            .entry(self.position_to_hash_key(&pos))
            .or_default()
            .push(w);

        self.vertices.push(Vertex::new(pos.clone()));

        // Find all faces containing edge u-v (or v-u)
        let mut affected_faces = Vec::new();
        for face_idx in 0..self.faces.len() {
            if self.faces[face_idx].half_edge == usize::MAX {
                continue; // Skip removed faces
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

        // For each affected face, replace it with two new faces
        let mut new_face_indices = Vec::new();
        let mut faces_to_remove = Vec::new();
        let mut new_face_index = self.faces.len();
        for (face_idx, a, b, c) in affected_faces {
            faces_to_remove.push(face_idx);

            // Remove old face by marking for removal
            self.faces[face_idx].half_edge = usize::MAX;

            // Add two new triangles: (a, w, c) and (w, b, c)
            let f1 = new_face_index;
            new_face_index += 1;
            let f2 = new_face_index;
            new_face_index += 1;
            new_face_indices.push((f1, [a, w, c]));
            new_face_indices.push((f2, [w, b, c]));
        }

        // Remove marked faces
        // self.faces.retain(|f| f.half_edge != usize::MAX);

        // Add new faces and half-edges, updating connectivity
        for (face_idx, verts) in &new_face_indices {
            let base_idx = self.half_edges.len();
            let edge_vertices = [
                (verts[0], verts[1]),
                (verts[1], verts[2]),
                (verts[2], verts[0]),
            ];
            let mut edge_indices = [0; 3];

            // Create the 3 new half-edges
            for (i, &(from, to)) in edge_vertices.iter().enumerate() {
                let mut he = HalfEdge::new(to);
                he.face = Some(*face_idx);
                let idx = base_idx + i;

                // Find twin edge (to → from)
                if let Some(&twin_idx) = self.edge_map.get(&(to, from)) {
                    he.twin = twin_idx;
                    self.half_edges[twin_idx].twin = idx;
                } else {
                    he.twin = usize::MAX;
                }

                self.edge_map.insert((from, to), idx);
                self.half_edges.push(he);
                edge_indices[i] = idx;
            }

            // Link next/prev
            self.half_edges[edge_indices[0]].next = edge_indices[1];
            self.half_edges[edge_indices[0]].prev = edge_indices[2];
            self.half_edges[edge_indices[1]].next = edge_indices[2];
            self.half_edges[edge_indices[1]].prev = edge_indices[0];
            self.half_edges[edge_indices[2]].next = edge_indices[0];
            self.half_edges[edge_indices[2]].prev = edge_indices[1];

            // Attach half-edge to vertices
            self.vertices[verts[0]]
                .half_edge
                .get_or_insert(edge_indices[0]);
            self.vertices[verts[1]]
                .half_edge
                .get_or_insert(edge_indices[1]);
            self.vertices[verts[2]]
                .half_edge
                .get_or_insert(edge_indices[2]);

            self.faces.push(Face::new(edge_indices[0]));
        }

        // Remove old edge from edge_map
        self.edge_map.remove(&(u, v));
        self.edge_map.remove(&(v, u));

        Ok(SplitEdgeResult {
            new_vertex: w,
            new_faces: Some(new_face_indices.iter().map(|(idx, _)| *idx).collect()),
            invalidated_faces: Some(faces_to_remove),
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

    pub fn carve_segment_across_faces(&mut self, p_idx: usize, q_idx: usize)
    where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let p_pos = self.vertices[p_idx].position.clone();
        let q_pos = self.vertices[q_idx].position.clone();

        // CHECK FOR DEGENERATE SEGMENT FIRST
        if p_pos.distance_to(&q_pos) < T::from(EPS) {
            eprintln!("WARNING: Degenerate segment detected, skipping carve");
            return;
        }

        // Find all faces containing the starting point
        let mut current_faces = self.faces_containing_point(&p_pos);
        if current_faces.is_empty() {
            eprintln!("ERROR: No face contains starting point");
            return;
        }

        let mut curr_idx = p_idx;
        let mut visited_faces = std::collections::HashSet::new();
        let mut iteration_count = 0;
        let max_iterations = self.faces.len() * 2; // Safety limit

        loop {
            iteration_count += 1;
            if iteration_count > max_iterations {
                eprintln!(
                    "ERROR: Carve exceeded maximum iterations, stopping to prevent infinite loop"
                );
                break;
            }

            // Try each candidate face
            let mut advanced = false;
            for &face in &current_faces.clone() {
                if visited_faces.contains(&face) {
                    continue;
                }
                visited_faces.insert(face);

                // Check if target is on this face
                if self.face_vertices(face).contains(&q_idx) {
                    self.split_segment_by_indices(face, curr_idx, q_idx, true);
                    return;
                }

                // Strategy 1: Try to advance along existing colinear edges
                if let Some((next_idx, next_face)) =
                    self.advance_along_colinear_edge(face, &p_pos, &q_pos, curr_idx, q_idx)
                {
                    curr_idx = next_idx;
                    current_faces = vec![next_face];
                    advanced = true;
                    break;
                }

                // Strategy 2: Find edge intersections with robust geometry
                if let Some((intersection_vertex, next_face)) =
                    self.find_edge_intersection(face, &p_pos, &q_pos, curr_idx)
                {
                    curr_idx = intersection_vertex;
                    current_faces = vec![next_face];
                    advanced = true;
                    break;
                }
            }

            if advanced {
                continue;
            }

            // Strategy 3: Direct face splitting as fallback
            let face = current_faces[0];
            self.split_segment_by_indices(face, curr_idx, q_idx, true);
            break;
        }
    }

    /// Strategy 1: Advance along existing colinear edges
    fn advance_along_colinear_edge(
        &self,
        face: usize,
        p_pos: &Point<T, N>,
        q_pos: &Point<T, N>,
        curr_idx: usize,
        q_idx: usize,
    ) -> Option<(usize, usize)>
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let segment_dir = (q_pos - p_pos).as_vector();

        for &he in &self.face_half_edges(face) {
            let src = self.half_edges[self.half_edges[he].prev].vertex;
            let dst = self.half_edges[he].vertex;

            if src != curr_idx {
                continue; // Must start from current vertex
            }

            let edge_start = &self.vertices[src].position;
            let edge_end = &self.vertices[dst].position;
            let edge_dir = (edge_end - edge_start).as_vector();

            // Check if edge is colinear with segment
            if self.are_colinear_with_overlap(&segment_dir, &edge_dir) {
                // Check if edge advances toward target
                let progress = (&self.vertices[dst].position - p_pos)
                    .as_vector()
                    .dot(&segment_dir);
                let total = segment_dir.dot(&segment_dir);

                if progress > T::zero() && progress < total {
                    if dst == q_idx {
                        return Some((dst, face)); // Reached target
                    }

                    // Find adjacent face through this edge
                    let twin_he = self.half_edges[he].twin;

                    if twin_he != usize::MAX {
                        if let Some(next_face) = self.half_edges[twin_he].face {
                            return Some((dst, next_face));
                        }
                    }
                }
            }
        }

        None
    }

    /// Strategy 2: Robust edge intersection with multiple fallback methods
    fn find_edge_intersection(
        &mut self,
        face: usize,
        p_pos: &Point<T, N>,
        q_pos: &Point<T, N>,
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
        let mut best_intersection = None;
        let mut best_t = T::one();

        for &he in &self.face_half_edges(face) {
            let src = self.half_edges[self.half_edges[he].prev].vertex;
            let dst = self.half_edges[he].vertex;

            // Skip edges connected to current vertex
            if src == curr_idx || dst == curr_idx {
                continue;
            }

            let edge_start = &self.vertices[src].position;
            let edge_end = &self.vertices[dst].position;

            // Use your existing robust intersection computation
            if let Some((intersection_point, t)) =
                self.compute_intersection(p_pos, q_pos, edge_start, edge_end)
            {
                let eps = T::from(1e-6);
                if t > eps && t < T::one() - eps && t < best_t {
                    best_t = t.clone();
                    best_intersection = Some((he, intersection_point));
                }
            }
        }

        if let Some((he, intersection_point)) = best_intersection {
            // Split the edge and advance
            if let Ok(split_edge_result) = self.split_edge(he, intersection_point) {
                let new_vertex_idx = split_edge_result.new_vertex;
                // TODO: Optimize
                // IMPORTANT: After split_edge, half-edge indices may be invalid!
                // Re-find faces containing the new vertex
                let new_vertex_pos = self.vertices[new_vertex_idx].position.clone();
                let containing_faces = self.faces_containing_point(&new_vertex_pos);

                if !containing_faces.is_empty() {
                    // Pick a face that's not the original one we were processing
                    let next_face = containing_faces[0];
                    return Some((new_vertex_idx, next_face));
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
        let eps = T::from(1e-12);

        if &denom.abs() < &eps {
            return None; // Lines are parallel
        }

        let t = (&d2_dot_d2 * &d1_dot_r - &d1_dot_d2 * &d2_dot_r) / denom.clone();
        let s = (&d1_dot_d1 * &d2_dot_r - &d1_dot_d2 * &d1_dot_r) / denom;

        // Check if intersection is within both line segments
        let eps_bounds = T::from(1e-10);
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

        if segment_2d_len_sq < T::from(1e-16) || edge_2d_len_sq < T::from(1e-16) {
            return None; // Degenerate projection
        }

        // **USE PARAMETRIC 2D INTERSECTION (NOT segment_intersect_2d)**
        let dir_segment = [&q2[0] - &p2[0], &q2[1] - &p2[1]];
        let dir_edge = [&b2[0] - &a2[0], &b2[1] - &a2[1]];

        // Solve: p2 + t*dir_segment = a2 + s*dir_edge
        let denom = &dir_segment[0] * &dir_edge[1] - &dir_segment[1] * &dir_edge[0];

        if denom.abs() < T::from(1e-12) {
            return None; // Parallel in 2D
        }

        let diff = [&a2[0] - &p2[0], &a2[1] - &p2[1]];
        let t = (&diff[0] * &dir_edge[1] - &diff[1] * &dir_edge[0]) / denom.clone();
        let s = (&diff[0] * &dir_segment[1] - &diff[1] * &dir_segment[0]) / denom;

        // Check bounds
        let eps = T::from(1e-10);
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
        let eps = T::from(1e-12);

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

            if distance < T::from(1e-6) {
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
    ) {
        // 1) If there’s already a half‐edge vi0→vi1 on this face, split that edge
        if let Some(he) = self.find_half_edge_on_face(face, vi0, vi1) {
            let _ = self.split_edge(he, self.vertices[vi1].position.clone());
        } else {
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
    ) -> Option<SplitEdgeResult>
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
                return Some(SplitEdgeResult {
                    new_vertex: vi,
                    new_faces: None,
                    invalidated_faces: None,
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
                let split_edge_result = self.split_edge(he, p.clone()).unwrap();
                return Some(split_edge_result);
            }
        }

        // 3. If p is strictly inside this triangular face, split it into 3 (RARE case).
        let a = &self.vertices[vs[0]].position;
        let b = &self.vertices[vs[1]].position;
        let c = &self.vertices[vs[2]].position;
        if point_in_or_on_triangle(p, a, b, c) {
            // 3.a add new vertex
            let w = self.add_vertex(p.clone());

            // 3.b get the original face's half-edge before invalidation
            let original_he = self.faces[face].half_edge;

            // 3.c invalidate the original face
            self.faces[face].half_edge = usize::MAX;

            // 3.d create 3 new faces incrementally
            let new_face_1 = self.add_triangle(vs[0], vs[1], w); // (a,b,w)
            let new_face_2 = self.add_triangle(vs[1], vs[2], w); // (b,c,w)
            let new_face_3 = self.add_triangle(vs[2], vs[0], w); // (c,a,w)

            return Some(SplitEdgeResult {
                new_vertex: w,
                new_faces: Some(vec![new_face_1, new_face_2, new_face_3]),
                invalidated_faces: Some(vec![face]),
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
    let e = T::from(1e-9);
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
        // 1) Clone & prepare
        let mut a = self.clone();
        let mut b = other.clone();
        a.build_boundary_loops();
        b.build_boundary_loops();

        // 2) Pre-split A against B, handling both proper and coplanar intersections
        let tree_b_pre = AabbTree::build((0..b.faces.len()).map(|i| (b.face_aabb(i), i)).collect());
        let mut segments: Vec<([usize; 2], Segment<T, 3>)> = Vec::new();

        let mut fa = 0;
        while fa < a.faces.len() {
            if a.faces[fa].half_edge == usize::MAX {
                fa += 1;
                continue; // Skip invalid faces
            }
            let mut candidates = Vec::new();
            tree_b_pre.query(&a.face_aabb(fa), &mut candidates);

            let pa_idx = a.face_vertices(fa);
            let pa_vec: Vec<Point<T, 3>> = pa_idx
                .into_iter()
                .map(|vi| a.vertices[vi].position.clone())
                .collect();
            let pa: [Point<T, 3>; 3] = pa_vec.try_into().expect("Expected 3 vertices");

            for &fb in &candidates {
                if a.faces[fa].half_edge == usize::MAX {
                    continue; // Face may have been invalidated at this point
                }
                let pb_idx = b.face_vertices(*fb);
                let pb_vec: Vec<Point<T, 3>> = pb_idx
                    .into_iter()
                    .map(|vi| b.vertices[vi].position.clone())
                    .collect();
                let pb: [Point<T, 3>; 3] = pb_vec.try_into().expect("Expected 3 vertices");

                // Before expensive triangle-triangle intersection:
                let aabb_a = a.face_aabb(fa);
                let aabb_b = b.face_aabb(*fb);

                if !aabb_a.intersects(&aabb_b) {
                    continue; // Skip if bounding boxes don't even overlap
                }

                if let Some(s) = tri_tri_intersection(&pa, &pb) {
                    let segment_length = s.length();

                    // **Filter out degenerate/point intersections**
                    if segment_length.is_positive() {
                        boolean_split(&mut a, fa, &s);
                    }
                    continue;
                }

                let e1a = &pa[1] - &pa[0];
                let e2a = &pa[2] - &pa[0];
                let e1b = &pb[1] - &pb[0];
                let e2b = &pb[2] - &pb[0];
                let n_a = e1a.as_vector().cross(&e2a.as_vector());
                let n_b = e1b.as_vector().cross(&e2b.as_vector());

                let n_dot = n_a.dot(&n_b);
                let n_norm_sq_a = n_a.dot(&n_a);
                let n_norm_sq_b = n_b.dot(&n_b);

                if n_norm_sq_a.is_zero() || n_norm_sq_b.is_zero() {
                    continue;
                }

                let cos_angle = &n_dot / &(n_norm_sq_a.sqrt() * n_norm_sq_b.sqrt());

                if cos_angle.abs() > T::from(0.999) {
                    let diff: Point<T, 3> = &pa[0] - &pb[0];
                    let plane_distance = n_a.dot(&diff.as_vector()).abs() / n_norm_sq_a.sqrt();

                    if plane_distance < T::from(1e-6) {
                        if let Some(s) = tri_tri_intersection(&pa, &pb) {
                            if s.length().is_positive() {
                                let is_duplicate = segments.iter().any(|([_, _], existing)| {
                                    (s.a.distance_to(&existing.a).is_zero()
                                        && s.b.distance_to(&existing.b).is_zero())
                                        || (s.a.distance_to(&existing.b).is_zero()
                                            && s.b.distance_to(&existing.a).is_zero())
                                });

                                if !is_duplicate {
                                    boolean_split(&mut a, fa, &s);
                                }
                            }
                        }
                    }
                }
            }
            fa += 1;
        }

        a.faces.retain(|f| f.half_edge != usize::MAX);

        let _ = write_obj(&a, "/mnt/v/cgar_meshes/a.obj");
        println!("GOT HERE");

        a.build_boundary_loops();
        b.build_boundary_loops();

        println!("GOT HERE 2");

        // 1. Remove duplicate vertices
        let mut vertex_dedup_map = std::collections::HashMap::new();
        let mut unique_vertices = Vec::new();
        let mut seen_positions = std::collections::HashMap::new();

        for (old_idx, vertex) in a.vertices.iter().enumerate() {
            let pos = &vertex.position;
            let pos_key = (
                (pos.coords()[0].to_f64().unwrap() * 1e6) as i64,
                (pos.coords()[1].to_f64().unwrap() * 1e6) as i64,
                (pos.coords()[2].to_f64().unwrap() * 1e6) as i64,
            );

            if let Some(&existing_idx) = seen_positions.get(&pos_key) {
                vertex_dedup_map.insert(old_idx, existing_idx);
            } else {
                let new_idx = unique_vertices.len();
                unique_vertices.push(vertex.clone());
                vertex_dedup_map.insert(old_idx, new_idx);
                seen_positions.insert(pos_key, new_idx);
            }
        }

        println!("GOT HERE 3");

        // 2. Rebuild mesh with deduplicated vertices
        let mut final_mesh = Mesh::new();
        for vertex in unique_vertices {
            final_mesh.add_vertex(vertex.position);
        }

        // 3. Add faces with remapped vertex indices, removing duplicates
        let mut face_set = std::collections::HashSet::new();

        for face_idx in 0..a.faces.len() {
            let vs = a.face_vertices(face_idx);
            let remapped_vs = [
                vertex_dedup_map[&vs[0]],
                vertex_dedup_map[&vs[1]],
                vertex_dedup_map[&vs[2]],
            ];

            // Skip degenerate triangles
            if remapped_vs[0] == remapped_vs[1]
                || remapped_vs[1] == remapped_vs[2]
                || remapped_vs[2] == remapped_vs[0]
            {
                continue;
            }

            // Create canonical face representation for deduplication
            let mut sorted_face = remapped_vs.clone();
            sorted_face.sort();
            let face_key = (sorted_face[0], sorted_face[1], sorted_face[2]);

            if face_set.insert(face_key) {
                final_mesh.add_triangle(remapped_vs[0], remapped_vs[1], remapped_vs[2]);
            }
        }

        println!("GOT HERE 4");

        a = final_mesh;

        // 4) Build classification trees
        let tree_a = AabbTree::build((0..a.faces.len()).map(|i| (a.face_aabb(i), i)).collect());
        let tree_b = AabbTree::build((0..b.faces.len()).map(|i| (b.face_aabb(i), i)).collect());

        // Prepare
        let mut result = Mesh::new();
        let mut vid_map = HashMap::new();

        // 6a) Copy A verts
        for (i, v) in a.vertices.iter().enumerate() {
            let ni = result.add_vertex(v.position.clone());
            vid_map.insert((VertexSource::A, i), ni);
        }

        println!("GOT HERE 5");

        let mut b_f64 = Mesh::<CgarF64, 3>::new();
        copy_face_graph::<T, CgarF64>(&b, &mut b_f64);

        let tree_b_f64 = AabbTree::<CgarF64, 3, Point<CgarF64, 3>, usize>::build(
            (0..b_f64.faces.len())
                .map(|i| (b_f64.face_aabb(i), i))
                .collect::<Vec<(Aabb<CgarF64, 3, Point<CgarF64, 3>>, usize)>>(),
        );

        let start = Instant::now();
        let centroids: Vec<Point3<CgarF64>> = (0..a.faces.len())
            .map(|fa| {
                let c2 = a.face_centroid(fa);
                Point3::<CgarF64>::from_vals([
                    c2.coords()[0].to_f64().unwrap(),
                    c2.coords()[1].to_f64().unwrap(),
                    c2.coords()[2].to_f64().unwrap(),
                ])
            })
            .collect();
        println!("Centroid computation took: {:?}", start.elapsed());

        // Pre-compute bounds for fast rejection
        let b_bounds = b_f64.compute_mesh_bounds();

        let face_aabbs: Vec<Aabb<CgarF64, 3, Point<CgarF64, 3>>> = (0..a.faces.len())
            .map(|fa| {
                let aabb = a.face_aabb(fa);
                Aabb::<CgarF64, 3, Point<CgarF64, 3>>::from_points(
                    &Point::<CgarF64, 3>::from_vals([
                        aabb.min()[0].to_f64().unwrap(),
                        aabb.min()[1].to_f64().unwrap(),
                        aabb.min()[2].to_f64().unwrap(),
                    ]),
                    &Point::<CgarF64, 3>::from_vals([
                        aabb.max()[0].to_f64().unwrap(),
                        aabb.max()[1].to_f64().unwrap(),
                        aabb.max()[2].to_f64().unwrap(),
                    ]),
                )
            })
            .collect();

        let start = Instant::now();
        let face_classifications = fast_bulk_classification::<CgarF64>(
            &centroids,
            &face_aabbs,
            &b_f64,
            &tree_b_f64,
            &b_bounds,
            &op,
            false,
        );

        println!("Batch classification took: {:?}", start.elapsed());

        // Apply classifications
        for (fa, keep) in face_classifications.iter().enumerate() {
            if *keep {
                let vs = a.face_vertices(fa);
                result.add_triangle(
                    vid_map[&(VertexSource::A, vs[0])],
                    vid_map[&(VertexSource::A, vs[1])],
                    vid_map[&(VertexSource::A, vs[2])],
                );
            }
        }

        // // Classify A faces
        // for fa in 0..a.faces.len() {
        //     println!("Processing A face {}/{}", i, a.faces.len());
        //     i += 1;
        //     // compute centroid
        //     let c2 = a.face_centroid(fa);
        //     let cen = Point3::<CgarF64>::from_vals([
        //         c2.coords()[0].to_f64().unwrap(),
        //         c2.coords()[1].to_f64().unwrap(),
        //         c2.coords()[2].to_f64().unwrap(),
        //     ]);

        //     let start = Instant::now();
        //     let inside_b = b_f64.point_in_mesh(&tree_b_f64, &cen);
        //     let duration = start.elapsed();
        //     println!("Point in mesh took: {:?}", duration);

        //     let distance_to_b = if op == BooleanOp::Difference && !inside_b {
        //         // Only compute distance for faces that might be on boundary
        //         let start = Instant::now();
        //         let ret = b_f64.point_to_mesh_distance(&tree_b_f64, &cen);
        //         let duration = start.elapsed();
        //         println!("Distance to mesh took: {:?}", duration);
        //         ret
        //     } else {
        //         (1.0).into() // Dummy
        //     };

        //     let on_bnd = op == BooleanOp::Difference && distance_to_b.is_negative_or_zero();
        //     let keep = match op {
        //         BooleanOp::Union => !inside_b,
        //         BooleanOp::Intersection => inside_b,
        //         BooleanOp::Difference => {
        //             if inside_b {
        //                 test_inside += 1;
        //                 false // Always remove faces inside B
        //             } else if on_bnd {
        //                 false
        //             } else {
        //                 true // Keep faces outside B
        //             }
        //         }
        //     };
        //     if keep {
        //         let vs = a.face_vertices(fa);
        //         result.add_triangle(
        //             vid_map[&(VertexSource::A, vs[0])],
        //             vid_map[&(VertexSource::A, vs[1])],
        //             vid_map[&(VertexSource::A, vs[2])],
        //         );
        //         test_added += 1;
        //     }
        // }
        println!("GOT HERE 6");

        // 6b) Handle B according to op
        match op {
            BooleanOp::Union => {
                // copy B verts
                for (i, v) in b.vertices.iter().enumerate() {
                    let ni = result.add_vertex(v.position.clone());
                    vid_map.insert((VertexSource::B, i), ni);
                }
                // classify B faces
                for fb in 0..b.faces.len() {
                    let c2 = b.face_centroid(fb);
                    let cen = c2.into();
                    if !a.point_in_mesh(&tree_a, &cen) {
                        let vs = b.face_vertices(fb);
                        result.add_triangle(
                            vid_map[&(VertexSource::B, vs[0])],
                            vid_map[&(VertexSource::B, vs[1])],
                            vid_map[&(VertexSource::B, vs[2])],
                        );
                    }
                }

                result.remove_unused_vertices();
            }
            BooleanOp::Intersection => {
                // copy B verts
                for (i, v) in b.vertices.iter().enumerate() {
                    let ni = result.add_vertex(v.position.clone());
                    vid_map.insert((VertexSource::B, i), ni);
                }
                // classify B faces
                for fb in 0..b.faces.len() {
                    let c2 = b.face_centroid(fb);
                    let cen = c2.into();
                    if a.point_in_mesh(&tree_a, &cen) {
                        let vs = b.face_vertices(fb);
                        result.add_triangle(
                            vid_map[&(VertexSource::B, vs[0])],
                            vid_map[&(VertexSource::B, vs[1])],
                            vid_map[&(VertexSource::B, vs[2])],
                        );
                    }
                }

                result.remove_unused_vertices();
            }
            BooleanOp::Difference => {
                // 1) Copy B’s vertices into result
                for (i, v) in b.vertices.iter().enumerate() {
                    let ni = result.add_vertex(v.position.clone());
                    vid_map.insert((VertexSource::B, i), ni);
                }
                println!("GOT HERE 6.1");

                // Convert mesh A to CgarF64 for robust classification
                let mut a_f64 = Mesh::<CgarF64, 3>::new();
                copy_face_graph::<T, CgarF64>(&a, &mut a_f64);
                println!("GOT HERE 6.2");

                let tree_a_f64 = AabbTree::<CgarF64, 3, Point<CgarF64, 3>, usize>::build(
                    (0..a_f64.faces.len())
                        .map(|i| (a_f64.face_aabb(i), i))
                        .collect::<Vec<(Aabb<CgarF64, 3, Point<CgarF64, 3>>, usize)>>(),
                );
                println!("GOT HERE 6.3");

                // Prepare centroids and AABBs for B faces
                let b_centroids: Vec<Point3<CgarF64>> = (0..b.faces.len())
                    .map(|fb| {
                        let c2 = b.face_centroid(fb);
                        Point3::<CgarF64>::from_vals([
                            c2.coords()[0].to_f64().unwrap(),
                            c2.coords()[1].to_f64().unwrap(),
                            c2.coords()[2].to_f64().unwrap(),
                        ])
                    })
                    .collect();

                let b_face_aabbs: Vec<Aabb<CgarF64, 3, Point<CgarF64, 3>>> = (0..b.faces.len())
                    .map(|fb| {
                        let aabb = b.face_aabb(fb);
                        Aabb::<CgarF64, 3, Point<CgarF64, 3>>::from_points(
                            &Point::<CgarF64, 3>::from_vals([
                                aabb.min()[0].to_f64().unwrap(),
                                aabb.min()[1].to_f64().unwrap(),
                                aabb.min()[2].to_f64().unwrap(),
                            ]),
                            &Point::<CgarF64, 3>::from_vals([
                                aabb.max()[0].to_f64().unwrap(),
                                aabb.max()[1].to_f64().unwrap(),
                                aabb.max()[2].to_f64().unwrap(),
                            ]),
                        )
                    })
                    .collect();

                let a_bounds = a_f64.compute_mesh_bounds();

                let b_face_classifications = fast_bulk_classification::<CgarF64>(
                    &b_centroids,
                    &b_face_aabbs,
                    &a_f64,
                    &tree_a_f64,
                    &a_bounds,
                    &op,
                    true,
                );

                println!("GOT HERE 6.4");

                for (fb, keep) in b_face_classifications.iter().enumerate() {
                    if *keep {
                        let vs = b.face_vertices(fb);
                        // Flip for outward normals
                        result.add_triangle(
                            vid_map[&(VertexSource::B, vs[2])],
                            vid_map[&(VertexSource::B, vs[1])],
                            vid_map[&(VertexSource::B, vs[0])],
                        );
                    }
                }

                println!("GOT HERE 6.5");

                // 3) remove floating vertices
                result.remove_unused_vertices();

                println!("GOT HERE 6.6");

                // result.build_boundary_loops();
                // 6b) Add B’s interior faces (flipped for outward normals)
                // 1) copy B’s vertices into result
                // for (i, v) in b.vertices.iter().enumerate() {
                //     let ni = result.add_vertex(v.position.clone());
                //     vid_map.insert((VertexSource::B, i), ni);
                // }
                // println!("GOT HERE 6.1");

                // let mut a_f64 = Mesh::<CgarF64, 3>::new();
                // copy_face_graph::<T, CgarF64>(&a, &mut a_f64);
                // println!("GOT HERE 6.2");

                // let tree_a_f64 = AabbTree::<CgarF64, 3, Point<CgarF64, 3>, usize>::build(
                //     (0..a_f64.faces.len())
                //         .map(|i| (a_f64.face_aabb(i), i))
                //         .collect::<Vec<(Aabb<CgarF64, 3, Point<CgarF64, 3>>, usize)>>(),
                // );
                // println!("GOT HERE 6.3");

                // // 2) classify B–faces: keep those whose centroids lie inside A
                // for fb in 0..b.faces.len() {
                //     println!("Processing B face {}/{}", fb, b.faces.len());
                //     let start = Instant::now();
                //     let cen: Point<T, 3> = b.face_centroid(fb).into();
                //     let cen_f64 = Point3::<CgarF64>::from_vals([
                //         cen.coords()[0].to_f64().unwrap(),
                //         cen.coords()[1].to_f64().unwrap(),
                //         cen.coords()[2].to_f64().unwrap(),
                //     ]);
                //     println!("Time to compute centroid: {:?}", start.elapsed());
                //     let start = Instant::now();
                //     // inside‐test
                //     if a_f64.point_in_mesh(&tree_a_f64, &cen_f64) {
                //         println!("Time to test point in mesh: {:?}", start.elapsed());
                //         // distance to A’s surface
                //         let start = Instant::now();
                //         let dist = a_f64.point_to_mesh_distance(&tree_a_f64, &cen_f64);
                //         println!("Time to compute distance: {:?}", start.elapsed());
                //         if dist.is_positive() {
                //             let vs = b.face_vertices(fb);
                //             result.add_triangle(
                //                 vid_map[&(VertexSource::B, vs[2])],
                //                 vid_map[&(VertexSource::B, vs[1])],
                //                 vid_map[&(VertexSource::B, vs[0])],
                //             );
                //         }
                //     }
                // }

                // // 3) remove floating vertices
                // result.remove_unused_vertices();
            }
        }

        println!("GOT HERE 7");

        // for union & intersection we still want to remove accidental duplicates,
        // but for difference we must *not* strip away our outer hull:
        match op {
            BooleanOp::Union | BooleanOp::Intersection => {
                remove_duplicate_faces(&mut result);
                result.build_boundary_loops();
                result
            }
            BooleanOp::Difference => {
                // just cap the interior holes
                result
            }
        }
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
    let ab = b - a;
    let ap = p - a;
    let ab_dot_ab = ab.as_vector().dot(&ab.as_vector());
    let ab_dot_ap = ab.as_vector().dot(&ap.as_vector());
    // Project and check that projection is within [0, 1] with some tolerance
    if ab_dot_ab == T::zero() {
        return a.distance_to(p).is_zero();
    }
    let t = ab_dot_ap / ab_dot_ab;
    if t.is_negative() || t > T::one() {
        return false;
    }
    // Closest point on segment
    let closest = a + &ab.as_vector().scale(&t).0;
    closest.distance_to(p).is_zero()
}

/// Remove duplicate faces from mesh
fn remove_duplicate_faces<T: Scalar, const N: usize>(mesh: &mut Mesh<T, N>)
where
    T: Scalar,
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
{
    let mut unique_faces = std::collections::HashSet::new();
    let mut faces_to_keep = Vec::new();

    for face_idx in 0..mesh.faces.len() {
        let face_vertices = mesh.face_vertices(face_idx);

        // Create a canonical representation (sorted vertices)
        let mut sorted_face = [face_vertices[0], face_vertices[1], face_vertices[2]];
        sorted_face.sort();

        if unique_faces.insert(sorted_face) {
            faces_to_keep.push([face_vertices[0], face_vertices[1], face_vertices[2]]);
        }
    }

    // Rebuild mesh with unique faces
    let mut new_mesh = Mesh::new();

    // Copy all vertices
    for vertex in &mesh.vertices {
        new_mesh.add_vertex(vertex.position.clone());
    }

    // Add unique faces
    for face in faces_to_keep {
        new_mesh.add_triangle(face[0], face[1], face[2]);
    }

    *mesh = new_mesh;
}

fn random_unit_vector<T>() -> Vector<T, 3>
where
    T: Scalar, // your existing numeric trait
{
    let mut rng = rand::rng();

    // 1) sample u = cos(φ) uniformly in [−1,1]
    let u: f64 = rng.random_range(-1.0..=1.0);

    // 2) sample θ uniformly in [0, 2π)
    let theta: f64 = rng.random_range(0.0..2.0 * PI);

    // 3) convert back to (x,y,z)
    let sqrt_one_minus_u2 = (1.0 - u * u).sqrt();
    let x = sqrt_one_minus_u2 * theta.cos();
    let y = sqrt_one_minus_u2 * theta.sin();
    let z = u;

    Vector::from_vals([T::from(x), T::from(y), T::from(z)])
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

fn boolean_split<T: Scalar>(mesh: &mut Mesh<T, 3>, fa: usize, s: &Segment<T, 3>)
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    //println!("boolean_split");
    let split_edge_result_a = mesh.split_or_find_vertex_on_face(fa, &s.a);
    if split_edge_result_a.is_none() {
        return;
    }
    let split_edge_result_a = split_edge_result_a.expect("Failed to split edge");
    let mut split_edge_result_b = None;

    println!(
        "Invalidated faces: {:?}",
        split_edge_result_a.invalidated_faces
    );

    if let Some(new_faces) = split_edge_result_a.new_faces {
        if new_faces.len() > 100 {
            panic!("Too many new faces created: {}", new_faces.len());
        }
        println!("New faces created: {:?}", new_faces);
        for f in new_faces {
            if mesh.faces[f].half_edge == usize::MAX {
                continue; // Skip invalidated faces
            }
            split_edge_result_b = mesh.split_or_find_vertex_on_face(f, &s.b);
            if split_edge_result_b.is_some() {
                break;
            }
        }
    } else {
        println!("No new faces created, trying to split edge B directly.");
        split_edge_result_b = mesh.split_or_find_vertex_on_face(fa, &s.b);
        if split_edge_result_b.is_none() {
            println!("Failed to split edge B directly.");
            return; // If we can't split edge B, we can't continue
        }
    }

    if let Some(split_edge_result_b) = split_edge_result_b {
        // Add edge only if it doesn't already exist
        mesh.add_edge_if_not_exists(
            split_edge_result_a.new_vertex,
            split_edge_result_b.new_vertex,
        );
    }
}

pub fn copy_face_graph<T1: Scalar, T2: Scalar>(mesh_a: &Mesh<T1, 3>, mesh_b: &mut Mesh<T2, 3>)
where
    T1: Scalar + Into<T2>,
    T2: Scalar,
    Point<T1, 3>: PointOps<T1, 3, Vector = Vector<T1, 3>>,
    Point<T2, 3>: PointOps<T2, 3, Vector = Vector<T2, 3>>,
{
    for v in &mesh_a.vertices {
        let converted_point = Point::<T2, 3>::from_vals([
            v.position[0].clone().into(),
            v.position[1].clone().into(),
            v.position[2].clone().into(),
        ]);
        mesh_b.add_vertex(converted_point);
    }
    for f in 0..mesh_a.faces.len() {
        let vs = mesh_a.face_vertices(f);
        mesh_b.add_triangle(vs[0], vs[1], vs[2]);
    }
}

/// Check if a point is far from the mesh bounds
pub fn point_far_from_bounds<T: Scalar>(
    point: &Point<T, 3>,
    bounds: &Aabb<T, 3, Point<T, 3>>,
    threshold_distance: &T,
) -> bool
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Compute distance from point to AABB
    let mut distance_squared = T::zero();

    for i in 0..3 {
        let coord = &point[i];
        let min_bound = &bounds.min()[i];
        let max_bound = &bounds.max()[i];

        if coord < min_bound {
            let diff = min_bound - coord;
            distance_squared += &(&diff * &diff);
        } else if coord > max_bound {
            let diff = coord - max_bound;
            distance_squared += &(&diff * &diff);
        }
        // If coord is within [min_bound, max_bound], distance contribution is 0
    }

    let distance = distance_squared.sqrt();
    distance > *threshold_distance
}

fn fast_bulk_classification<T: Scalar>(
    centroids: &[Point<T, 3>],
    face_aabbs: &Vec<Aabb<CgarF64, 3, Point<CgarF64, 3>>>, // Add this argument
    mesh: &Mesh<T, 3>,
    tree: &AabbTree<T, 3, Point<T, 3>, usize>,
    bounds: &Aabb<CgarF64, 3, Point<CgarF64, 3>>,
    op: &BooleanOp,
    is_b: bool,
) -> Vec<bool>
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Compute bounding box diagonal
    let min = bounds.min();
    let max = bounds.max();
    let diag = (&(&max[0] - &min[0]).abs().pow(2)
        + &(&max[1] - &min[1]).abs().pow(2)
        + (&max[2] - &min[2]).abs().pow(2))
    .sqrt();

    // Compute average face area
    let avg_area = mesh.average_face_area();
    // Estimate average face "width" (sqrt of area)
    let avg_width = avg_area.sqrt();

    println!("Average face width: {}", avg_width);

    // Target: average face covers ~2 grid cells per axis
    let grid_size = if avg_width > 0.0 {
        ((diag.0 / avg_width) * 2.0).ceil() as usize
    } else {
        1
    };

    let grid_size = grid_size.clamp(1, 256);

    println!("Adaptive grid size: {}", grid_size);

    let mut grid_classifications = HashMap::new();
    let mut face_classifications = Vec::with_capacity(centroids.len());

    for (i, centroid) in centroids.iter().enumerate() {
        println!("Processing face {} of {}", i, centroids.len());
        let face_aabb = &face_aabbs[i];
        if let Some(cell) = face_aabb_in_single_grid_cell::<CgarF64>(face_aabb, bounds, grid_size) {
            // println!("Face {} is fully inside grid cell {:?}", i, cell);
            // Face is fully inside one grid cell
            if let Some(&classification) = grid_classifications.get(&cell) {
                println!("Skipped!");
                face_classifications.push(classification);
            } else {
                let cell_center = grid_cell_center(&cell, bounds, grid_size);
                let inside_b = mesh.point_in_mesh(tree, &cell_center);
                let keep = match op {
                    BooleanOp::Union => !inside_b,
                    BooleanOp::Intersection => inside_b,
                    BooleanOp::Difference => {
                        if is_b {
                            // For B faces: keep if inside A and not on boundary
                            if inside_b {
                                let ret = mesh.point_to_mesh_distance(tree, &centroid);
                                ret.is_positive()
                            } else {
                                false
                            }
                        } else {
                            // For A faces: keep if outside B and not on boundary
                            if inside_b {
                                false
                            } else {
                                let ret = mesh.point_to_mesh_distance(tree, &centroid);
                                ret.is_positive()
                            }
                        }
                    }
                };
                grid_classifications.insert(cell, keep);
                face_classifications.push(keep);
            }
        } else {
            println!("crosses borders");
            // println!("Face {} crosses grid cell boundaries", i);
            // Face crosses grid border, classify individually
            let inside_b = mesh.point_in_mesh(tree, centroid);
            // println!("Centroid {} inside B: {}", i, inside_b);
            let keep = match op {
                BooleanOp::Union => !inside_b,
                BooleanOp::Intersection => inside_b,
                BooleanOp::Difference => {
                    if is_b {
                        // For B faces: keep if inside A and not on boundary
                        if inside_b {
                            let ret = mesh.point_to_mesh_distance(tree, &centroid);
                            ret.is_positive()
                        } else {
                            false
                        }
                    } else {
                        // For A faces: keep if outside B and not on boundary
                        if inside_b {
                            false
                        } else {
                            let ret = mesh.point_to_mesh_distance(tree, &centroid);
                            ret.is_positive()
                        }
                    }
                }
            };
            face_classifications.push(keep);
        }
    }

    println!("finished!");

    face_classifications
}

fn face_in_single_grid_cell<T: Scalar>(
    face_vertices: &[Point<T, 3>],
    bounds: &Aabb<T, 3, Point<T, 3>>,
    grid_size: usize,
) -> Option<(usize, usize, usize)> {
    let mut cell = None;
    for v in face_vertices {
        let c = map_point_to_grid_cell(v, bounds, grid_size);
        if let Some(existing) = cell {
            if existing != c {
                return None; // Face crosses grid border
            }
        } else {
            cell = Some(c);
        }
    }
    cell // All vertices in the same cell
}

fn map_point_to_grid_cell<T: Scalar>(
    point: &Point<T, 3>,
    bounds: &Aabb<T, 3, Point<T, 3>>,
    grid_size: usize,
) -> (usize, usize, usize) {
    let size = T::from(grid_size as f64);
    let min = bounds.min();
    let max = bounds.max();

    let x = if max[0] == min[0] {
        grid_size / 2
    } else {
        ((point[0].clone() - min[0].clone()) / (max[0].clone() - min[0].clone()) * size.clone())
            .to_f64()
            .unwrap()
            .clamp(0.0, (grid_size - 1) as f64) as usize
    };
    let y = if max[1] == min[1] {
        grid_size / 2
    } else {
        ((point[1].clone() - min[1].clone()) / (max[1].clone() - min[1].clone()) * size.clone())
            .to_f64()
            .unwrap()
            .clamp(0.0, (grid_size - 1) as f64) as usize
    };
    let z = if max[2] == min[2] {
        grid_size / 2
    } else {
        ((point[2].clone() - min[2].clone()) / (max[2].clone() - min[2].clone()) * size.clone())
            .to_f64()
            .unwrap()
            .clamp(0.0, (grid_size - 1) as f64) as usize
    };

    (x, y, z)
}

fn grid_cell_center<T: Scalar>(
    grid_cell: &(usize, usize, usize),
    bounds: &Aabb<CgarF64, 3, Point<CgarF64, 3>>,
    grid_size: usize,
) -> Point<T, 3>
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let (gx, gy, gz) = *grid_cell;
    let grid_size_t = CgarF64::from(grid_size as f64);
    let half = CgarF64::from(0.5);

    let min = bounds.min();
    let max = bounds.max();

    let world_x = if max[0] == min[0] {
        min[0].clone()
    } else {
        let norm_x = &(&CgarF64::from(gx as f64) + &half) / &grid_size_t;
        &min[0] + &(&norm_x * &(&max[0] - &min[0]))
    };
    let world_y = if max[1] == min[1] {
        min[1].clone()
    } else {
        let norm_y = &(&CgarF64::from(gy as f64) + &half) / &grid_size_t;
        &min[1] + &(&norm_y * &(&max[1] - &min[1]))
    };
    let world_z = if max[2] == min[2] {
        min[2].clone()
    } else {
        let norm_z = &(&CgarF64::from(gz as f64) + &half) / &grid_size_t;
        &min[2] + &(&norm_z * &(&max[2] - &min[2]))
    };

    Point::<T, 3>::from_vals([world_x, world_y, world_z])
}

fn single_ray_test<T: Scalar>(
    mesh: &Mesh<T, 3>,
    tree: &AabbTree<T, 3, Point<T, 3>, usize>,
    point: &Point<T, 3>,
) -> bool
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Use a deterministic ray direction to avoid randomness
    let dir = Vector::<T, 3>::from_vals([T::from(1.0), T::from(0.0), T::from(0.0)]);

    if let Some(is_inside) = mesh.cast_ray(point, &dir, tree) {
        is_inside
    } else {
        false // Default to outside if ray misses everything
    }
}

fn face_aabb_in_single_grid_cell<T: Scalar>(
    face_aabb: &Aabb<T, 3, Point<T, 3>>,
    bounds: &Aabb<T, 3, Point<T, 3>>,
    grid_size: usize,
) -> Option<(usize, usize, usize)> {
    let min_cell = map_point_to_grid_cell(&face_aabb.min(), bounds, grid_size);
    let max_cell = map_point_to_grid_cell(&face_aabb.max(), bounds, grid_size);
    if min_cell == max_cell {
        Some(min_cell)
    } else {
        None
    }
}
