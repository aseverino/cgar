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
    numeric::{cgar_f64::CgarF64, scalar::Scalar},
    operations::Zero,
};
use num_traits::Float;
use rand::prelude::*;

use super::{face::Face, half_edge::HalfEdge, vertex::Vertex};
use core::panic;
use std::{
    array::from_fn,
    collections::{HashMap, HashSet},
    ops::{Add, Div, Mul, Sub},
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
}

impl<T: Scalar, const N: usize> Mesh<T, N> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            half_edges: Vec::new(),
            faces: Vec::new(),
            edge_map: HashMap::new(),
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
    pub fn remove_unused_vertices(&mut self) {
        // Find vertices that are actually referenced by faces
        let mut used_vertices = std::collections::HashSet::new();
        for face_idx in 0..self.faces.len() {
            let face_verts = self.face_vertices(face_idx);
            for &v_idx in &face_verts {
                used_vertices.insert(v_idx);
            }
        }

        // Remove unused vertices in reverse order (highest index first)
        let mut unused_vertices: Vec<usize> = (0..self.vertices.len())
            .filter(|idx| !used_vertices.contains(idx))
            .collect();
        unused_vertices.sort_by(|a, b| b.cmp(a)); // Reverse order

        for &vertex_idx in &unused_vertices {
            self.remove_vertex(vertex_idx);
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
        self.vertices.push(Vertex::new(position));
        idx
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

        let n_rays = 8; // ← Reduce from 32 to 8 rays
        for _ in 0..n_rays {
            let dir: Vector<T, 3> = random_unit_vector();
            if let Some(is_inside) = self.cast_ray(p, &dir, tree) {
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
        let far_point = Point::<T, 3>::from_vals([
            &p[0] + &(&dir[0] * &T::from(1000.0)),
            &p[1] + &(&dir[1] * &T::from(1000.0)),
            &p[2] + &(&dir[2] * &T::from(1000.0)),
        ]);
        let ray_aabb = Aabb::from_points(p, &far_point);

        // Query tree for faces that intersect ray
        let mut candidate_faces = Vec::new();
        tree.query(&ray_aabb, &mut candidate_faces);

        // Test only candidate faces (not all faces!)
        for &fi in &candidate_faces {
            let vs_idxs = self.face_vertices(*fi);
            let vs: [&Point<T, 3>; 3] = [
                &Point::<T, 3>::from_vals([
                    self.vertices[vs_idxs[0]].position[0].clone(),
                    self.vertices[vs_idxs[0]].position[1].clone(),
                    self.vertices[vs_idxs[0]].position[2].clone(),
                ]),
                &Point::<T, 3>::from_vals([
                    self.vertices[vs_idxs[1]].position[0].clone(),
                    self.vertices[vs_idxs[1]].position[1].clone(),
                    self.vertices[vs_idxs[1]].position[2].clone(),
                ]),
                &Point::<T, 3>::from_vals([
                    self.vertices[vs_idxs[2]].position[0].clone(),
                    self.vertices[vs_idxs[2]].position[1].clone(),
                    self.vertices[vs_idxs[2]].position[2].clone(),
                ]),
            ];

            if let Some(t) = self.ray_triangle_intersection(&p.coords(), dir, vs) {
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

    /// Robust ray-triangle intersection using Möller-Trumbore algorithm
    fn ray_triangle_intersection(
        &self,
        ray_origin: &[T; 3],
        ray_dir: &Vector<T, 3>,
        triangle: [&Point<T, 3>; 3],
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

    pub fn split_edge(&mut self, he: usize, pos: Point<T, N>) -> Result<usize, &'static str> {
        let prev = self.half_edges[he].prev;
        let u = self.half_edges[prev].vertex;
        let v = self.half_edges[he].vertex;

        // Check if we already have this position
        if pos == self.vertices[u].position {
            return Ok(u);
        }
        if pos == self.vertices[v].position {
            return Ok(v);
        }

        // Add new vertex
        let w = self.vertices.len();
        self.vertices.push(Vertex::new(pos.clone()));

        // Find all faces containing edge u-v (or v-u)
        let mut affected_faces = Vec::new();
        for face_idx in 0..self.faces.len() {
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
        for (face_idx, a, b, c) in affected_faces {
            faces_to_remove.push(face_idx);

            // Remove old face by marking for removal
            self.faces[face_idx].half_edge = usize::MAX;

            // Add two new triangles: (a, w, c) and (w, b, c)
            let f1 = self.faces.len();
            let f2 = self.faces.len() + 1;
            new_face_indices.push((f1, [a, w, c]));
            new_face_indices.push((f2, [w, b, c]));
        }

        // Remove marked faces
        self.faces.retain(|f| f.half_edge != usize::MAX);

        // Add new faces and half-edges, updating connectivity
        for (face_idx, verts) in new_face_indices {
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
                he.face = Some(face_idx);
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

        Ok(w)
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
        if p_pos.distance_to(&q_pos) < T::from(1e-10) {
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
            if let Ok(new_vertex_idx) = self.split_edge(he, intersection_point) {
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
                if f != face {
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
        _tree: &AabbTree<T, 3, Point<T, 3>, usize>,
        p: &Point3<T>,
    ) -> T
    where
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut min_d2 = T::from(std::f64::INFINITY);
        for fi in 0..self.faces.len() {
            let vs = self.face_vertices(fi);
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
            min_d2 = min_d2.min(d2);
        }
        min_d2.sqrt()
    }

    /// Given a face and a target point, finds an existing vertex (if close)
    /// or splits the appropriate edge and returns the new vertex index.
    pub fn find_or_insert_vertex_on_face(&mut self, face: usize, p: &Point<T, N>) -> Option<usize>
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
                return Some(vi);
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
                let new_vi = self.split_edge(he, p.clone()).unwrap();
                return Some(new_vi);
            }
        }

        // 3. If p is strictly inside this triangular face, split it into 3 (RARE case).
        let a = &self.vertices[vs[0]].position;
        let b = &self.vertices[vs[1]].position;
        let c = &self.vertices[vs[2]].position;
        if point_in_or_on_triangle(p, a, b, c) {
            // 3.a add new vertex
            let w = self.add_vertex(p.clone());

            // 3.b remove old face by marking its half‐edge
            self.faces[face].half_edge = usize::MAX;

            // 3.c collect all other faces, rebuild this one into 3 triangles
            let mut new_tris = Vec::new();
            for f in 0..self.faces.len() {
                if f != face {
                    let fv = self.face_vertices(f);
                    new_tris.push([fv[0], fv[1], fv[2]]);
                }
            }
            // split (a,b,c) into (a,b,w), (b,c,w), (c,a,w)
            new_tris.push([vs[0], vs[1], w]);
            new_tris.push([vs[1], vs[2], w]);
            new_tris.push([vs[2], vs[0], w]);

            // rebuild mesh (vertices unchanged)
            let old_positions = self
                .vertices
                .iter()
                .map(|v| v.position.clone())
                .collect::<Vec<_>>();
            *self = Mesh::new();
            for pos in old_positions {
                self.add_vertex(pos);
            }
            for t in new_tris {
                self.add_triangle(t[0], t[1], t[2]);
            }

            return Some(w);
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
        let mut segments: Vec<(usize, usize, Segment<T, 3>)> = Vec::new();
        for fa in 0..a.faces.len() {
            let mut candidates = Vec::new();
            tree_b_pre.query(&a.face_aabb(fa), &mut candidates);

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
                    .map(|vi| b.vertices[vi].position.clone()) // ← FIX: Use b.vertices not a.vertices
                    .collect();
                let pb: [Point<T, 3>; 3] = pb_vec.try_into().expect("Expected 3 vertices");

                // Try proper intersection first
                if let Some(s) = tri_tri_intersection(&pa, &pb) {
                    let segment_length = s.length();
                    // **FILTER DEGENERATE SEGMENTS**
                    if segment_length.is_positive() {
                        segments.push((fa, *fb, s));
                    }
                } else {
                    // **LIMIT COPLANAR PROCESSING**
                    let e1a = &pa[1] - &pa[0];
                    let e2a = &pa[2] - &pa[0];
                    let e1b = &pb[1] - &pb[0];
                    let e2b = &pb[2] - &pb[0];
                    let n_a = e1a.as_vector().cross(&e2a.as_vector());
                    let n_b = e1b.as_vector().cross(&e2b.as_vector());

                    if n_a.dot(&n_b).abs() > T::from_num_den(1, 1) {
                        let diff: Point<T, 3> = &pa[0] - &pb[0];
                        if n_a.dot(&diff.as_vector()).abs().is_zero() {
                            // **COPLANAR CASE: Use tri_tri_intersection**
                            if let Some(segment) = tri_tri_intersection(&pa, &pb) {
                                if segment.length().is_positive() {
                                    // Check if this segment already exists
                                    let is_duplicate = segments.iter().any(|(_, _, existing)| {
                                        (segment.a.distance_to(&existing.a).is_zero()
                                            && segment.b.distance_to(&existing.b).is_zero())
                                            || (segment.a.distance_to(&existing.b).is_zero()
                                                && segment.b.distance_to(&existing.a).is_zero())
                                    });

                                    if !is_duplicate {
                                        segments.push((fa, *fb, segment));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut a_and_b_arr = [&mut a, &mut b];

        // 3) Split on both meshes
        for &(_orig_fa, _orig_fb, ref s) in &segments {
            for mesh in a_and_b_arr.iter_mut() {
                let fa_p_candidates = mesh.faces_containing_point(&s.a);

                let mut pi = None;
                for &face in &fa_p_candidates {
                    pi = mesh.find_or_insert_vertex_on_face(face, &s.a);
                    if pi.is_some() {
                        break;
                    }
                }
                let pi = pi.expect("no face contains p in A");

                let fa_q_candidates = mesh.faces_containing_point(&s.b);
                let mut qi = None;
                for &face in &fa_q_candidates {
                    qi = mesh.find_or_insert_vertex_on_face(face, &s.b);
                    if qi.is_some() {
                        break;
                    }
                }
                let qi = qi.expect("no face contains q in A");

                // check if endpoints have an edge between them
                if mesh.half_edge_between(pi, qi).is_some() {
                    continue; // Skip this segment if edge already exists
                }

                let mut shared_face = None;
                for &face in &fa_p_candidates {
                    if mesh.face_vertices(face).contains(&qi) {
                        shared_face = Some(face);
                        break;
                    }
                }
                if let Some(face) = shared_face {
                    mesh.split_segment_by_indices(face, pi, qi, true);
                } else {
                    mesh.carve_segment_across_faces(pi, qi);
                }

                remove_duplicate_faces(mesh);
            }
        }

        a.build_boundary_loops();
        b.build_boundary_loops();

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

        let mut test_inside = 0;
        let mut test_boundary = 0;
        let mut test_added = 0;
        // Classify A faces
        for fa in 0..a.faces.len() {
            // compute centroid
            let c2 = a.face_centroid(fa);
            let cen = c2.into();
            let inside_b = b.point_in_mesh(&tree_b, &cen);
            let distance_to_b = b.point_to_mesh_distance(&tree_b, &cen);

            let on_bnd = op == BooleanOp::Difference && distance_to_b.is_negative_or_zero();
            let keep = match op {
                BooleanOp::Union => !inside_b,
                BooleanOp::Intersection => inside_b,
                BooleanOp::Difference => {
                    if inside_b {
                        test_inside += 1;
                        false // Always remove faces inside B
                    } else if on_bnd {
                        false
                    } else {
                        true // Keep faces outside B
                    }
                }
            };
            if keep {
                let vs = a.face_vertices(fa);
                result.add_triangle(
                    vid_map[&(VertexSource::A, vs[0])],
                    vid_map[&(VertexSource::A, vs[1])],
                    vid_map[&(VertexSource::A, vs[2])],
                );
                test_added += 1;
            }
        }

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
                // result.build_boundary_loops();
                // 6b) Add B’s interior faces (flipped for outward normals)
                // 1) copy B’s vertices into result
                for (i, v) in b.vertices.iter().enumerate() {
                    let ni = result.add_vertex(v.position.clone());
                    vid_map.insert((VertexSource::B, i), ni);
                }
                // 2) classify B–faces: keep those whose centroids lie inside A
                for fb in 0..b.faces.len() {
                    let cen: Point<T, 3> = b.face_centroid(fb).into();
                    // inside‐test
                    if a.point_in_mesh(&tree_a, &cen) {
                        // distance to A’s surface
                        let dist = a.point_to_mesh_distance(&tree_a, &cen);
                        if dist.is_positive() {
                            let vs = b.face_vertices(fb);
                            result.add_triangle(
                                vid_map[&(VertexSource::B, vs[2])],
                                vid_map[&(VertexSource::B, vs[1])],
                                vid_map[&(VertexSource::B, vs[0])],
                            );
                        }
                    }
                }

                // 3) remove floating vertices
                result.remove_unused_vertices();
            }
        }

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
