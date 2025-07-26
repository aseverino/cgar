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
    half_edge_split_map: HashMap<usize, (usize, usize)>,
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

    /// Safe version of face_vertices that validates indices
    pub fn face_vertices_safe(&self, f: usize) -> Option<Vec<usize>> {
        get_face_vertices_safe(self, f)
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
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
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
            let prev = find_valid_half_edge(self, self.half_edges[he].prev);
            let src = self.half_edges[prev].vertex;
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

                if let Ok(split_edge_result) = self.split_edge(aabb_tree, he, &pos.clone()) {
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
        if let Some(split_edge_result) = self.find_or_insert_vertex_on_face(aabb_tree, face, pos) {
            // // println!("recursive!");
            return Some(split_edge_result);
        }

        for f in self.faces_containing_point_aabb(aabb_tree, pos) {
            if let Some(split_edge_result) = self.find_or_insert_vertex_on_face(aabb_tree, f, pos) {
                return Some(split_edge_result);
            }
        }

        // println!(
        //     "No suitable vertex found, adding new vertex at position {:?}",
        //     pos
        // );

        return None;
    }

    pub fn build_face_tree_fast(&self) -> AabbTree<T, N, Point<T, N>, usize>
    where
        T: Scalar,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut face_aabbs = Vec::with_capacity(self.faces.len());

        let start = Instant::now();
        for (face_idx, face) in self.faces.iter().enumerate() {
            if face.removed {
                continue;
            }

            let he0 = find_valid_half_edge(self, face.half_edge);
            let he1 = find_valid_half_edge(self, self.half_edges[he0].next);
            let he2 = find_valid_half_edge(self, self.half_edges[he1].next);

            if he1 >= self.half_edges.len()
                || he2 >= self.half_edges.len()
                || self.half_edges[he2].next != he0
            {
                continue;
            }

            let v0_idx = self.half_edges[he0].vertex;
            let v1_idx = self.half_edges[he1].vertex;
            let v2_idx = self.half_edges[he2].vertex;

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
        // println!("TEST 1 {:.2?}", start.elapsed());

        let start = Instant::now();
        let tree = AabbTree::build(face_aabbs);
        // println!("TEST 2 {:.2?}", start.elapsed());

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
        if face.half_edge == usize::MAX || self.half_edges[face.half_edge].removed {
            // Return degenerate AABB for invalid faces
            let origin = Point::<T, N>::from_vals(from_fn(|_| T::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        // Direct 3-vertex access for triangular faces (no loop traversal)
        let he0 = find_valid_half_edge(self, face.half_edge);
        let he1 = find_valid_half_edge(self, self.half_edges[he0].next);
        let he2 = find_valid_half_edge(self, self.half_edges[he1].next);

        // Safety checks
        if he1 >= self.half_edges.len() || he2 >= self.half_edges.len() {
            let origin = Point::<T, N>::from_vals(from_fn(|_| T::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        // Verify it's a triangle (optional safety check)
        if find_valid_half_edge(self, self.half_edges[he2].next) != he0 {
            let origin = Point::<T, N>::from_vals(from_fn(|_| T::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        let v0_idx = self.half_edges[he0].vertex;
        let v1_idx = self.half_edges[he1].vertex;
        let v2_idx = self.half_edges[he2].vertex;

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
            + Div<&'a T, Output = T>,
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
            println!("NEEDED RECONSTRUCTION 1");
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
                *aabb_tree = self.build_face_tree_fast();
                candidates = Vec::new();
                aabb_tree.query(&query_aabb, &mut candidates);

                if candidates.is_empty() {
                    return Vec::new();
                }
            }
        }

        let mut result = Vec::with_capacity(candidates.len().min(8)); // Pre-allocate reasonable size

        // **ALSO FIXED: Use same tolerance for geometric tests**
        for &face_idx in &candidates {
            let face = find_valid_face(self, *face_idx, p, false).unwrap();
            let he0 = self.faces[face].half_edge;

            if he0 >= self.half_edges.len() {
                continue;
            }

            let he1 = find_valid_half_edge(self, self.half_edges[he0].next);
            let he2 = find_valid_half_edge(self, self.half_edges[he1].next);

            if he1 >= self.half_edges.len()
                || he2 >= self.half_edges.len()
                || find_valid_half_edge(self, self.half_edges[he2].next) != he0
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

            if point_in_triangle_optimized(p, a, b, c) {
                result.push(*face_idx);
            }
        }

        // **OPTIMIZED: Tree rebuild detection**
        if result.is_empty() && !candidates.is_empty() {
            println!("NEEDED RECONSTRUCTION 2");
            // Candidates existed but none contained point - possible stale tree
            *aabb_tree = self.build_face_tree_fast();
            return self.faces_containing_point_aabb(aabb_tree, p);
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
            let t = find_valid_half_edge(self, self.half_edges[h].twin);
            // Now that every edge has a twin (real or ghost), we never hit usize::MAX
            h = find_valid_half_edge(self, self.half_edges[t].next);
            if h == start {
                break;
            }
        }
        result
    }

    /// Returns true if vertex `v` has any outgoing ghost edge (face == None).
    pub fn is_boundary_vertex(&self, v: usize) -> bool {
        self.outgoing_half_edges(v).into_iter().any(|he| {
            self.half_edges[find_valid_half_edge(self, he)]
                .face
                .is_none()
        })
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
            .map(|&he_idx| self.half_edges[find_valid_half_edge(self, he_idx)].vertex)
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
                let prev = self.half_edges[find_valid_half_edge(self, he)].prev;
                let twin = find_valid_half_edge(self, self.half_edges[prev].twin);
                he = if twin != usize::MAX { twin } else { prev };
                if he == start {
                    break;
                }
            }

            // 2) Filter to *just* the boundary edges
            let boundary_cycle: Vec<usize> = hole_cycle
                .into_iter()
                .filter(|&bhe| {
                    bhe < original_count
                        && find_valid_half_edge(self, self.half_edges[bhe].twin) == usize::MAX
                })
                .collect();

            // 3) Spawn one ghost per boundary half-edge
            let mut ghosts = Vec::with_capacity(boundary_cycle.len());
            for &bhe in &boundary_cycle {
                let origin = {
                    let prev = find_valid_half_edge(self, self.half_edges[bhe].prev);
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
        let start = find_valid_half_edge(self, self.faces[f].half_edge);
        let mut h = start;
        loop {
            result.push(h);
            h = find_valid_half_edge(self, self.half_edges[h].next);

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
            .map(|he| self.half_edges[find_valid_half_edge(self, he)].vertex)
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

    pub fn split_edge(
        &mut self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        he: usize,
        pos: &Point<T, N>,
    ) -> Result<SplitResult, &'static str>
    where
        T: Scalar,
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let he_ab = find_valid_half_edge(self, he);
        let he_ba = find_valid_half_edge(self, self.half_edges[he_ab].twin);
        let he_ca = find_valid_half_edge(self, self.half_edges[he_ab].prev);
        let he_bc = find_valid_half_edge(self, self.half_edges[he_ab].next);
        let he_ad = find_valid_half_edge(self, self.half_edges[he_ba].next);
        let he_db = find_valid_half_edge(self, self.half_edges[he_ad].next);

        let ex_he_ac = find_valid_half_edge(self, self.half_edges[he_ca].twin);
        let ex_he_bd = find_valid_half_edge(self, self.half_edges[he_db].twin);
        let ex_he_da = find_valid_half_edge(self, self.half_edges[he_ad].twin);
        let ex_he_cb = find_valid_half_edge(self, self.half_edges[he_bc].twin);

        let a = self.half_edges[he_ca].vertex;
        let b = self.half_edges[he_ab].vertex;
        let c = self.half_edges[he_bc].vertex;
        let d = self.half_edges[he_ad].vertex;

        let edge_start = &self.vertices[a].position;
        let edge_end = &self.vertices[b].position;
        let edge_length = edge_start.distance_to(edge_end);

        let split_point_distance_from_start = edge_start.distance_to(pos);
        let split_point_distance_from_end = edge_end.distance_to(pos);

        // Check if we already have this position at existing vertices
        if pos == edge_start {
            return Ok(SplitResult {
                kind: SplitResultKind::NoSplit,
                new_vertex: a,
                new_faces: Vec::new(),
            });
        }
        if pos == edge_end {
            return Ok(SplitResult {
                kind: SplitResultKind::NoSplit,
                new_vertex: b,
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
                    new_vertex: a,
                    new_faces: Vec::new(),
                });
            } else {
                return Ok(SplitResult {
                    kind: SplitResultKind::NoSplit,
                    new_vertex: b,
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

        // let mut new_face_results = Vec::new();
        let original_face_1 = self.half_edges[he_ab].face.unwrap();
        let original_face_2 = self.half_edges[he_ba].face.unwrap();

        // 1. Build the new faces (4 in total) and create their half-edges (no twins for now)
        let face_1_subface_1_verts = [a, w, c];
        let face_1_subface_2_verts = [w, b, c];
        let face_2_subface_1_verts = [w, a, d];
        let face_2_subface_2_verts = [b, w, d];

        let face_1_subface_1_idx = self.faces.len();
        let face_1_subface_2_idx = self.faces.len() + 1;
        let face_2_subface_1_idx = self.faces.len() + 2;
        let face_2_subface_2_idx = self.faces.len() + 3;

        // Create face 1 subface 1
        let base_he_idx_face_1_sub_1 = self.half_edges.len();
        let edge_vertices_face_1_sub_1 = [
            (face_1_subface_1_verts[0], face_1_subface_1_verts[1]), // a -> w
            (face_1_subface_1_verts[1], face_1_subface_1_verts[2]), // w -> c
            (face_1_subface_1_verts[2], face_1_subface_1_verts[0]), // c -> a
        ];

        // Create half-edges for face 1 subface 1
        for (_i, &(_from, to)) in edge_vertices_face_1_sub_1.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(face_1_subface_1_idx);
            self.half_edges.push(he);
        }

        // Link face 1 subface 1 half-edges
        let edge_indices_face_1_sub_1 = [
            base_he_idx_face_1_sub_1,
            base_he_idx_face_1_sub_1 + 1,
            base_he_idx_face_1_sub_1 + 2,
        ];
        self.half_edges[edge_indices_face_1_sub_1[0]].next = edge_indices_face_1_sub_1[1];
        self.half_edges[edge_indices_face_1_sub_1[0]].prev = edge_indices_face_1_sub_1[2];
        self.half_edges[edge_indices_face_1_sub_1[1]].next = edge_indices_face_1_sub_1[2];
        self.half_edges[edge_indices_face_1_sub_1[1]].prev = edge_indices_face_1_sub_1[0];
        self.half_edges[edge_indices_face_1_sub_1[2]].next = edge_indices_face_1_sub_1[0];
        self.half_edges[edge_indices_face_1_sub_1[2]].prev = edge_indices_face_1_sub_1[1];

        ///////////////////////
        // Create face 1 subface 2
        let base_he_idx_face_1_sub_2 = self.half_edges.len();
        let edge_vertices_face_1_sub_2 = [
            (face_1_subface_2_verts[0], face_1_subface_2_verts[1]), // w -> b
            (face_1_subface_2_verts[1], face_1_subface_2_verts[2]), // b -> c
            (face_1_subface_2_verts[2], face_1_subface_2_verts[0]), // c -> w
        ];

        // Create half-edges for face 1 subface 2
        for (_i, &(_from, to)) in edge_vertices_face_1_sub_2.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(face_1_subface_2_idx);
            self.half_edges.push(he);
        }

        // Link face 1 subface 2 half-edges
        let edge_indices_face_1_sub_2 = [
            base_he_idx_face_1_sub_2,
            base_he_idx_face_1_sub_2 + 1,
            base_he_idx_face_1_sub_2 + 2,
        ];
        self.half_edges[edge_indices_face_1_sub_2[0]].next = edge_indices_face_1_sub_2[1];
        self.half_edges[edge_indices_face_1_sub_2[0]].prev = edge_indices_face_1_sub_2[2];
        self.half_edges[edge_indices_face_1_sub_2[1]].next = edge_indices_face_1_sub_2[2];
        self.half_edges[edge_indices_face_1_sub_2[1]].prev = edge_indices_face_1_sub_2[0];
        self.half_edges[edge_indices_face_1_sub_2[2]].next = edge_indices_face_1_sub_2[0];
        self.half_edges[edge_indices_face_1_sub_2[2]].prev = edge_indices_face_1_sub_2[1];

        ///////////////////////////////
        // Create face 2 subface 1
        let base_he_idx_face_2_sub_1 = self.half_edges.len();
        let edge_vertices_face_2_sub_1 = [
            (face_2_subface_1_verts[0], face_2_subface_1_verts[1]), // w -> a
            (face_2_subface_1_verts[1], face_2_subface_1_verts[2]), // a -> d
            (face_2_subface_1_verts[2], face_2_subface_1_verts[0]), // d -> w
        ];

        // Create half-edges for face 2 subface 1
        for (_i, &(_from, to)) in edge_vertices_face_2_sub_1.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(face_2_subface_1_idx);
            self.half_edges.push(he);
        }

        // Link face 2 subface 1 half-edges
        let edge_indices_face_2_sub_1 = [
            base_he_idx_face_2_sub_1,
            base_he_idx_face_2_sub_1 + 1,
            base_he_idx_face_2_sub_1 + 2,
        ];
        self.half_edges[edge_indices_face_2_sub_1[0]].next = edge_indices_face_2_sub_1[1];
        self.half_edges[edge_indices_face_2_sub_1[0]].prev = edge_indices_face_2_sub_1[2];
        self.half_edges[edge_indices_face_2_sub_1[1]].next = edge_indices_face_2_sub_1[2];
        self.half_edges[edge_indices_face_2_sub_1[1]].prev = edge_indices_face_2_sub_1[0];
        self.half_edges[edge_indices_face_2_sub_1[2]].next = edge_indices_face_2_sub_1[0];
        self.half_edges[edge_indices_face_2_sub_1[2]].prev = edge_indices_face_2_sub_1[1];

        ///////////////////////////////
        // Create face 2 subface 2
        let base_he_idx_face_2_sub_2 = self.half_edges.len();
        let edge_vertices_face_2_sub_2 = [
            (face_2_subface_2_verts[0], face_2_subface_2_verts[1]), // b -> w
            (face_2_subface_2_verts[1], face_2_subface_2_verts[2]), // w -> d
            (face_2_subface_2_verts[2], face_2_subface_2_verts[0]), // d -> b
        ];

        // Create half-edges for face 2 subface 2
        for (_i, &(_from, to)) in edge_vertices_face_2_sub_2.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(face_2_subface_2_idx);
            self.half_edges.push(he);
        }

        // Link face 2 subface 2 half-edges
        let edge_indices_face_2_sub_2 = [
            base_he_idx_face_2_sub_2,
            base_he_idx_face_2_sub_2 + 1,
            base_he_idx_face_2_sub_2 + 2,
        ];
        self.half_edges[edge_indices_face_2_sub_2[0]].next = edge_indices_face_2_sub_2[1];
        self.half_edges[edge_indices_face_2_sub_2[0]].prev = edge_indices_face_2_sub_2[2];
        self.half_edges[edge_indices_face_2_sub_2[1]].next = edge_indices_face_2_sub_2[2];
        self.half_edges[edge_indices_face_2_sub_2[1]].prev = edge_indices_face_2_sub_2[0];
        self.half_edges[edge_indices_face_2_sub_2[2]].next = edge_indices_face_2_sub_2[0];
        self.half_edges[edge_indices_face_2_sub_2[2]].prev = edge_indices_face_2_sub_2[1];

        // Add the 4 faces
        self.faces.push(Face::new(edge_indices_face_1_sub_1[0]));
        self.faces.push(Face::new(edge_indices_face_1_sub_2[0]));
        self.faces.push(Face::new(edge_indices_face_2_sub_1[0]));
        self.faces.push(Face::new(edge_indices_face_2_sub_2[0]));

        // Set w's half-edge
        self.vertices[w].half_edge = Some(base_he_idx_face_1_sub_1); // a -> w

        // Remove old edges from edge map
        self.edge_map.remove(&(a, b));
        self.edge_map.remove(&(b, a));

        // Add new edges to the edge map
        self.edge_map.insert((a, w), base_he_idx_face_1_sub_1);
        self.edge_map.insert((w, c), base_he_idx_face_1_sub_1 + 1);
        self.edge_map.insert((c, a), base_he_idx_face_1_sub_1 + 2);

        self.edge_map.insert((w, b), base_he_idx_face_1_sub_2);
        self.edge_map.insert((b, c), base_he_idx_face_1_sub_2 + 1);
        self.edge_map.insert((c, w), base_he_idx_face_1_sub_2 + 2);

        self.edge_map.insert((w, a), base_he_idx_face_2_sub_1);
        self.edge_map.insert((a, d), base_he_idx_face_2_sub_1 + 1);
        self.edge_map.insert((d, w), base_he_idx_face_2_sub_1 + 2);

        self.edge_map.insert((b, w), base_he_idx_face_2_sub_2);
        self.edge_map.insert((w, d), base_he_idx_face_2_sub_2 + 1);
        self.edge_map.insert((d, b), base_he_idx_face_2_sub_2 + 2);

        // Update the unchanged bordering faces' twins
        self.half_edges[ex_he_ac].twin = base_he_idx_face_1_sub_1 + 2; // c -> a
        self.half_edges[ex_he_bd].twin = base_he_idx_face_2_sub_2 + 2; // d -> b
        self.half_edges[ex_he_da].twin = base_he_idx_face_2_sub_1 + 1; // a -> d
        self.half_edges[ex_he_cb].twin = base_he_idx_face_1_sub_2 + 1; // b -> c

        self.half_edges[base_he_idx_face_1_sub_1 + 2].twin = ex_he_ac; // a -> c
        self.half_edges[base_he_idx_face_2_sub_2 + 2].twin = ex_he_bd; // b -> d
        self.half_edges[base_he_idx_face_2_sub_1 + 1].twin = ex_he_da; // d -> a
        self.half_edges[base_he_idx_face_1_sub_2 + 1].twin = ex_he_cb; // c -> b

        // Update the split half-edge related twins
        self.half_edges[base_he_idx_face_1_sub_1].twin = base_he_idx_face_2_sub_1;
        self.half_edges[base_he_idx_face_2_sub_1].twin = base_he_idx_face_1_sub_1;

        self.half_edges[base_he_idx_face_1_sub_2].twin = base_he_idx_face_2_sub_2;
        self.half_edges[base_he_idx_face_2_sub_2].twin = base_he_idx_face_1_sub_2;

        // Set internal new edge's twins
        self.half_edges[base_he_idx_face_1_sub_1 + 1].twin = base_he_idx_face_1_sub_2 + 2; // w -> c
        self.half_edges[base_he_idx_face_1_sub_2 + 2].twin = base_he_idx_face_1_sub_1 + 1; // c -> w

        self.half_edges[base_he_idx_face_2_sub_1 + 2].twin = base_he_idx_face_2_sub_2 + 1; // w -> d
        self.half_edges[base_he_idx_face_2_sub_2 + 1].twin = base_he_idx_face_2_sub_1 + 2; // d -> w

        // Mark face 1 as removed and add to face_split_map
        self.faces[original_face_1].removed = true; // Mark as removed
        let face_split = FaceSplitMap {
            face: original_face_1,
            new_faces: vec![face_1_subface_1_idx, face_1_subface_2_idx],
        };
        self.face_split_map.insert(original_face_1, face_split);

        // Mark face 2 as removed and add to face_split_map
        self.faces[original_face_2].removed = true; // Mark as removed
        let face_split = FaceSplitMap {
            face: original_face_2,
            new_faces: vec![face_2_subface_1_idx, face_2_subface_2_idx],
        };
        self.face_split_map.insert(original_face_2, face_split);

        // Mark old half-edges as removed
        self.half_edges[he_ab].removed = true; // Mark as removed
        self.half_edges[he_ba].removed = true; // Mark as removed
        self.half_edges[he_ca].removed = true; // Mark as removed
        self.half_edges[he_bc].removed = true; // Mark as removed
        self.half_edges[he_ad].removed = true; // Mark as removed
        self.half_edges[he_db].removed = true; // Mark as removed

        self.half_edge_split_map
            .insert(he_ab, (base_he_idx_face_1_sub_1, base_he_idx_face_1_sub_2));
        self.half_edge_split_map
            .insert(he_ba, (base_he_idx_face_2_sub_1, base_he_idx_face_2_sub_2));

        self.half_edge_split_map
            .insert(he_ca, (base_he_idx_face_1_sub_1 + 2, usize::MAX));
        self.half_edge_split_map
            .insert(he_bc, (base_he_idx_face_1_sub_2 + 1, usize::MAX));
        self.half_edge_split_map
            .insert(he_ad, (base_he_idx_face_2_sub_1 + 1, usize::MAX));
        self.half_edge_split_map
            .insert(he_db, (base_he_idx_face_2_sub_2 + 2, usize::MAX));

        let split_result = SplitResult {
            kind: SplitResultKind::SplitEdge,
            new_vertex: w,
            new_faces: vec![
                face_1_subface_1_idx,
                face_1_subface_2_idx,
                face_2_subface_1_idx,
                face_2_subface_2_idx,
            ],
        };

        for &new_face_idx in &split_result.new_faces {
            let face_aabb = self.face_aabb(new_face_idx);
            aabb_tree.insert(face_aabb, new_face_idx);
        }

        aabb_tree.invalidate(&original_face_1);
        aabb_tree.invalidate(&original_face_2);

        Ok(split_result)

        // self.vertices[*c].half_edge = Some(base_he_idx_face_1_sub_1 + 1); // w -> c
        // self.vertices[*a].half_edge = Some(base_he_idx_face_1_sub_1 + 2); // c -> a
        // self.vertices[*b].half_edge = Some(base_he_idx_2); // w -> b

        // let mut prepared_maps = HashMap::new();

        // // 2. Map the to-be-removed half-edges to their replacements
        // for (i, (_face_idx, _a, _b, _c)) in affected_faces.iter().enumerate() {
        //     let new_faces_from_this = new_face_results[i];
        //     let _face_1_idx = new_faces_from_this.0;
        //     let base_he_idx_1 = new_faces_from_this.1;
        //     let _face_2_idx = new_faces_from_this.2;
        //     let base_he_idx_2 = new_faces_from_this.3;

        //     // Let's get the half-edges from the old face that were not split
        //     let old_he_face_1 = find_valid_half_edge(self, self.half_edges[he_to_split].next); // b -> c
        //     let old_he_face_2 = find_valid_half_edge(self, self.half_edges[old_he_face_1].next); // c -> a

        //     // Now let's get the one half-edge from the edge of the old face that was split
        //     // and map it to both new half-edges connecting to vertex w (a -> w and w -> b).
        //     prepared_maps.insert(he_to_split, (base_he_idx_1, base_he_idx_2));

        //     // Now we map the two half-edges from old face that were not split
        //     prepared_maps.insert(old_he_face_1, (base_he_idx_2, usize::MAX));
        //     prepared_maps.insert(old_he_face_2, (base_he_idx_1, usize::MAX));

        //     // We can already tie up the half-edges that border not-modified faces

        //     // Update face_x_sub_1's third half-edge (c -> a) to twin the old face's twin half-edge (a -> c)
        //     self.half_edges[base_he_idx_1 + 2].twin = self.half_edges[old_he_face_2].twin;

        //     // Update face_x_sub_2's second half-edge (b -> c) to twin the old face's twin half-edge (c -> b)
        //     self.half_edges[base_he_idx_2 + 1].twin = self.half_edges[old_he_face_1].twin;
        // }

        // // 3. Add twins to the new half-edges, but don't make their pointed twins point back to them
        // let new_subfaces_from_face_1 = new_face_results[0];

        // // let face_1_subface_1_idx = new_subfaces_from_face_1.0;
        // let face_1_subface_1_base_he_idx = new_subfaces_from_face_1.1;
        // // let face_1_subface_2_idx = new_subfaces_from_face_1.2;
        // let face_1_subface_2_base_he_idx = new_subfaces_from_face_1.3;

        // let new_subfaces_from_face_2 = new_face_results[1];
        // // let face_2_subface_1_idx = new_subfaces_from_face_2.0;
        // let face_2_subface_1_base_he_idx = new_subfaces_from_face_2.1;
        // // let face_2_subface_2_idx = new_subfaces_from_face_2.2;
        // let face_2_subface_2_base_he_idx = new_subfaces_from_face_2.3;

        // // Link face 1 half-edges to their respective twins

        // // Old face 1 a -> w (now face_1_sub_1) connects with Old face 2 w -> b (now face_2_sub_2)
        // self.half_edges[face_1_subface_1_base_he_idx].twin = face_2_subface_2_base_he_idx;
        // self.half_edges[face_2_subface_2_base_he_idx].twin = face_1_subface_1_base_he_idx;

        // // Old face 1 w -> b (now face_1_sub_2) connects with Old face 2 a -> w (now face_2_sub_1)
        // self.half_edges[face_1_subface_2_base_he_idx].twin = face_2_subface_1_base_he_idx;
        // self.half_edges[face_2_subface_1_base_he_idx].twin = face_1_subface_2_base_he_idx;

        // // Update face_x_sub_1's second half-edge (w -> c) to point to face_x_sub_2's third half-edge (c -> w)
        // self.half_edges[face_1_subface_1_base_he_idx + 1].twin = face_1_subface_2_base_he_idx + 2;

        // // Update face_x_sub_2's third half-edge (c -> w) to point to face_x_sub_1's second half-edge (w -> c)
        // self.half_edges[face_1_subface_2_base_he_idx + 2].twin = face_1_subface_1_base_he_idx + 1;

        // // 4. Mark old faces as removed
        // for (i, (face_idx, a, b, c)) in affected_faces.iter().enumerate() {
        //     self.faces[*face_idx].removed = true;
        //     let face_split_map = FaceSplitMap {
        //         face: *face_idx,
        //         new_faces: vec![new_face_results[i].0, new_face_results[i].2],
        //     };
        //     self.face_split_map.insert(*face_idx, face_split_map);
        // }

        // for (old_he, (new_he_1, new_he_2)) in &prepared_maps {
        //     // If the new half-edge is usize::MAX, it means it was not used
        //     self.half_edge_split_map
        //         .insert(*old_he, (*new_he_1, *new_he_2));
        // }

        // // 5. Insert the prepared_maps into half_edge_split_map and mark old half-edges as removed
        // for (old_he, (new_he_1, new_he_2)) in prepared_maps {
        //     // If the new half-edge is usize::MAX, it means it was not used
        //     self.half_edges[old_he].removed = true;
        //     self.half_edge_split_map
        //         .insert(old_he, (new_he_1, new_he_2));
        // }

        // // Remove old edge from edge_map
        // self.edge_map.remove(&(u, v));
        // self.edge_map.remove(&(v, u));

        // // 6. Update the edge_map with the new half-edges
        // for (i, (_face_idx, a, b, c)) in affected_faces.iter().enumerate() {
        //     let new_faces_from_this = new_face_results[i];
        //     // let face_1_idx = new_faces_from_this.0;
        //     let base_he_idx_1 = new_faces_from_this.1;
        //     // let face_2_idx = new_faces_from_this.2;
        //     let base_he_idx_2 = new_faces_from_this.3;

        //     // Update edge_map with the new half-edges
        //     self.edge_map.insert((*a, w), base_he_idx_1);
        //     self.edge_map.insert((w, *b), base_he_idx_2);

        //     self.edge_map.insert((*b, *c), base_he_idx_2 + 1);
        //     self.edge_map
        //         .insert((*c, *b), self.half_edges[base_he_idx_2 + 1].twin);

        //     self.edge_map.insert((*c, *a), base_he_idx_1 + 2);
        //     self.edge_map
        //         .insert((*a, *c), self.half_edges[base_he_idx_1 + 2].twin);

        //     self.edge_map.insert((w, *c), base_he_idx_1 + 1);
        //     self.edge_map.insert((*c, w), base_he_idx_2 + 2);
        // }

        // let split_result = SplitResult {
        //     kind: SplitResultKind::SplitEdge,
        //     new_vertex: w,
        //     new_faces: new_face_results
        //         .iter()
        //         .flat_map(|&(face_idx_1, _, face_idx_2, _)| [face_idx_1, face_idx_2])
        //         .collect(),
        // };

        // for &new_face_idx in &split_result.new_faces {
        //     let face_aabb = self.face_aabb(new_face_idx);
        //     aabb_tree.insert(face_aabb, new_face_idx);
        // }

        // Ok(split_result)
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
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
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
            let src = self.half_edges[find_valid_half_edge(self, self.half_edges[he].prev)].vertex;
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
            println!(
                "Found intersection at face {}: {:?}",
                face, intersection_point
            );
            if let Ok(split_edge_result) = self.split_edge(aabb_tree, he, &intersection_point) {
                let new_vertex_idx = split_edge_result.new_vertex;
                let new_vertex_pos = self.vertices[new_vertex_idx].position.clone();
                println!("Running faces_containing_point_aabb");
                let mut containing_faces =
                    self.faces_containing_point_aabb(aabb_tree, &new_vertex_pos);
                if let Some(&new_face_b) = containing_faces.get(0) {
                    println!("Found intersection at vertex: {}", new_vertex_idx);
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

    /// Given a face and a target point, finds an existing vertex (if close)
    /// or splits the appropriate edge and returns the new vertex index.
    pub fn find_or_insert_vertex_on_face(
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
        if self.faces[face].removed {
            panic!("Cannot find or insert vertex on a removed face");
        }
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
                let split_edge_result = self.split_edge(aabb_tree, he, &p.clone()).unwrap();
                return Some(split_edge_result);
            }
        }

        // 3. If p is strictly inside this triangular face, split it into 3 (RARE case).
        let a = &self.vertices[vs[0]].position;
        let b = &self.vertices[vs[1]].position;
        let c = &self.vertices[vs[2]].position;
        if point_in_or_on_triangle(p, a, b, c) {
            // println!("Point is inside the triangle, splitting... RARE CASE *********************");

            let a = vs[0];
            let b = vs[1];
            let c = vs[2];

            // 3.a add new vertex
            let w = self.add_vertex(p.clone());

            let face_1_idx = self.faces.len();
            let face_2_idx = self.faces.len() + 1;
            let face_3_idx = self.faces.len() + 2;

            let tri1_verts = [a, w, c];
            let tri2_verts = [w, b, c];
            let tri3_verts = [a, b, w];

            // Create face 1
            let base_he_idx_1 = self.half_edges.len();
            let edge_vertices_1 = [
                (tri1_verts[0], tri1_verts[1]), // a → w
                (tri1_verts[1], tri1_verts[2]), // w → c
                (tri1_verts[2], tri1_verts[0]), // c → a
            ];

            // Create half-edges for face 1
            for (_i, &(_from, to)) in edge_vertices_1.iter().enumerate() {
                let mut he = HalfEdge::new(to);
                he.face = Some(face_1_idx);
                self.half_edges.push(he);
            }

            // Link face 1 half-edges
            let edge_indices_1 = [base_he_idx_1, base_he_idx_1 + 1, base_he_idx_1 + 2];
            self.half_edges[edge_indices_1[0]].next = edge_indices_1[1];
            self.half_edges[edge_indices_1[0]].prev = edge_indices_1[2];
            self.half_edges[edge_indices_1[1]].next = edge_indices_1[2];
            self.half_edges[edge_indices_1[1]].prev = edge_indices_1[0];
            self.half_edges[edge_indices_1[2]].next = edge_indices_1[0];
            self.half_edges[edge_indices_1[2]].prev = edge_indices_1[1];

            // Create face 2
            let base_he_idx_2 = self.half_edges.len();
            let edge_vertices_2 = [
                (tri2_verts[0], tri2_verts[1]), // w → b
                (tri2_verts[1], tri2_verts[2]), // b → c
                (tri2_verts[2], tri2_verts[0]), // c → w
            ];

            // Create half-edges for face 2
            for (_i, &(_from, to)) in edge_vertices_2.iter().enumerate() {
                let mut he = HalfEdge::new(to);
                he.face = Some(face_2_idx);
                self.half_edges.push(he);
            }

            // Link face 2 half-edges
            let edge_indices_2 = [base_he_idx_2, base_he_idx_2 + 1, base_he_idx_2 + 2];
            self.half_edges[edge_indices_2[0]].next = edge_indices_2[1];
            self.half_edges[edge_indices_2[0]].prev = edge_indices_2[2];
            self.half_edges[edge_indices_2[1]].next = edge_indices_2[2];
            self.half_edges[edge_indices_2[1]].prev = edge_indices_2[0];
            self.half_edges[edge_indices_2[2]].next = edge_indices_2[0];
            self.half_edges[edge_indices_2[2]].prev = edge_indices_2[1];

            // Create face 3
            let base_he_idx_3 = self.half_edges.len();
            let edge_vertices_3 = [
                (tri3_verts[0], tri3_verts[1]), // a → b
                (tri3_verts[1], tri3_verts[2]), // b → w
                (tri3_verts[2], tri3_verts[0]), // w → a
            ];

            // Create half-edges for face 3
            for (_i, &(_from, to)) in edge_vertices_3.iter().enumerate() {
                let mut he = HalfEdge::new(to);
                he.face = Some(face_3_idx);
                self.half_edges.push(he);
            }

            // Link face 3 half-edges
            let edge_indices_3 = [base_he_idx_3, base_he_idx_3 + 1, base_he_idx_3 + 2];
            self.half_edges[edge_indices_3[0]].next = edge_indices_3[1];
            self.half_edges[edge_indices_3[0]].prev = edge_indices_3[2];
            self.half_edges[edge_indices_3[1]].next = edge_indices_3[2];
            self.half_edges[edge_indices_3[1]].prev = edge_indices_3[0];
            self.half_edges[edge_indices_3[2]].next = edge_indices_3[0];
            self.half_edges[edge_indices_3[2]].prev = edge_indices_3[1];

            self.faces.push(Face::new(edge_indices_1[0]));
            self.faces.push(Face::new(edge_indices_2[0]));
            self.faces.push(Face::new(edge_indices_3[0]));

            self.half_edges[base_he_idx_1].face = Some(face_1_idx);
            self.half_edges[base_he_idx_1 + 1].face = Some(face_1_idx);
            self.half_edges[base_he_idx_1 + 2].face = Some(face_1_idx);

            self.half_edges[base_he_idx_2].face = Some(face_2_idx);
            self.half_edges[base_he_idx_2 + 1].face = Some(face_2_idx);
            self.half_edges[base_he_idx_2 + 2].face = Some(face_2_idx);

            self.half_edges[base_he_idx_3].face = Some(face_3_idx);
            self.half_edges[base_he_idx_3 + 1].face = Some(face_3_idx);
            self.half_edges[base_he_idx_3 + 2].face = Some(face_3_idx);

            let old_he_1 = self.edge_map.get(&(a, b)).unwrap();
            let old_he_2 = self.edge_map.get(&(b, c)).unwrap();
            let old_he_3 = self.edge_map.get(&(c, a)).unwrap();

            self.vertices[w].half_edge = Some(base_he_idx_1); // a -> w
            self.vertices[a].half_edge = Some(base_he_idx_1 + 2); // c -> a
            self.vertices[b].half_edge = Some(base_he_idx_2); // w -> b
            self.vertices[c].half_edge = Some(base_he_idx_1 + 1); // w -> c

            let external_twin_1 = find_valid_half_edge(self, self.half_edges[*old_he_1].twin);
            self.half_edges[external_twin_1].twin = base_he_idx_3;
            self.half_edges[base_he_idx_3].twin = external_twin_1;

            let external_twin_2 = find_valid_half_edge(self, self.half_edges[*old_he_2].twin);
            self.half_edges[external_twin_2].twin = base_he_idx_2 + 1;
            self.half_edges[base_he_idx_2 + 1].twin = external_twin_2;

            let external_twin_3 = find_valid_half_edge(self, self.half_edges[*old_he_3].twin);
            self.half_edges[external_twin_3].twin = base_he_idx_1 + 2;
            self.half_edges[base_he_idx_1 + 2].twin = external_twin_3;

            // internal twins
            // a -> w and w -> a, respectively
            self.half_edges[base_he_idx_1].twin = base_he_idx_3 + 2;
            self.half_edges[base_he_idx_3 + 2].twin = base_he_idx_1;

            // w -> c and c -> w, respectively
            self.half_edges[base_he_idx_1 + 1].twin = base_he_idx_2 + 2;
            self.half_edges[base_he_idx_2 + 2].twin = base_he_idx_1 + 1;

            // c -> a and a -> c, respectively
            self.half_edges[base_he_idx_2].twin = base_he_idx_3 + 1;
            self.half_edges[base_he_idx_3 + 1].twin = base_he_idx_2;

            self.faces[face].removed = true;
            self.half_edges[*old_he_1].removed = true;
            self.half_edges[*old_he_2].removed = true;
            self.half_edges[*old_he_3].removed = true;

            self.half_edge_split_map
                .insert(*old_he_1, (base_he_idx_1, usize::MAX));
            self.half_edge_split_map
                .insert(*old_he_2, (base_he_idx_2 + 1, usize::MAX));
            self.half_edge_split_map
                .insert(*old_he_3, (base_he_idx_1 + 2, usize::MAX));

            let face_split_map = FaceSplitMap {
                face: face,
                new_faces: vec![face_1_idx, face_2_idx, face_3_idx],
            };
            self.face_split_map.insert(face, face_split_map);

            self.edge_map.insert((a, w), base_he_idx_1);
            self.edge_map.insert((w, c), base_he_idx_1 + 1);
            self.edge_map.insert((c, a), base_he_idx_1 + 2);

            self.edge_map.insert((w, b), base_he_idx_2);
            self.edge_map.insert((b, c), base_he_idx_2 + 1);
            self.edge_map.insert((c, w), base_he_idx_2 + 2);

            self.edge_map.insert((a, b), base_he_idx_3);
            self.edge_map.insert((b, w), base_he_idx_3 + 1);
            self.edge_map.insert((w, a), base_he_idx_3 + 2);

            let split_result = SplitResult {
                kind: SplitResultKind::SplitFace,
                new_vertex: w,
                new_faces: vec![face_1_idx, face_2_idx, face_3_idx],
            };

            for &new_face_idx in &split_result.new_faces {
                let face_aabb = self.face_aabb(new_face_idx);
                aabb_tree.insert(face_aabb, new_face_idx);
            }

            aabb_tree.invalidate(&face);

            // remove_invalidated_faces(self);
            // self.remove_unused_vertices();
            // let _ = write_obj(&self, "/mnt/v/cgar_meshes/vamooo.obj");
            // panic!("a");

            return Some(split_result);
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

impl<T: Scalar, const N: usize> BooleanImpl<T> for Mesh<T, N>
where
    T: Scalar,
    CgarF64: From<T>,
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn boolean(&self, other: &Mesh<T, N>, op: BooleanOp) -> Mesh<T, N> {
        let mut a = self.clone();
        let mut b = other.clone();

        let start = Instant::now();
        a.build_boundary_loops();
        b.build_boundary_loops();
        println!("Boundary loops built in {:.2?}", start.elapsed());

        // 1. Collect ALL intersection segments
        let mut intersection_segments_a = Vec::new();
        let mut intersection_segments_b = Vec::new();
        let start = Instant::now();
        let mut tree_a = a.build_face_tree_fast();
        let mut tree_b = b.build_face_tree_fast();
        println!("Total AABB computation: {:.2?}", start.elapsed());

        let start = Instant::now();
        for fa in 0..a.faces.len() {
            let mut candidates = Vec::new();
            tree_b.query(&a.face_aabb(fa), &mut candidates);

            let pa_idx = a.face_vertices(fa);
            let pa_vec: Vec<Point<T, 3>> = pa_idx
                .into_iter()
                .map(|vi| {
                    Point::<T, 3>::from_vals([
                        a.vertices[vi].position[0].clone(),
                        a.vertices[vi].position[1].clone(),
                        a.vertices[vi].position[2].clone(),
                    ])
                })
                .collect();
            let pa: [Point<T, 3>; 3] = pa_vec.try_into().expect("Expected 3 vertices");

            for &fb in &candidates {
                let pb_idx = b.face_vertices(*fb);
                let pb_vec: Vec<Point<T, 3>> = pb_idx
                    .into_iter()
                    .map(|vi| {
                        Point::<T, 3>::from_vals([
                            b.vertices[vi].position[0].clone(),
                            b.vertices[vi].position[1].clone(),
                            b.vertices[vi].position[2].clone(),
                        ])
                    })
                    .collect();
                let pb: [Point<T, 3>; 3] = pb_vec.try_into().expect("Expected 3 vertices");

                if let Some(s) = tri_tri_intersection(&pa, &pb) {
                    let segment_n = Segment::<T, N>::new(
                        &Point::<T, N>::from_vals(from_fn(|i| s.a[i].clone())),
                        &Point::<T, N>::from_vals(from_fn(|i| s.b[i].clone())),
                    );
                    if segment_n.length().is_positive() {
                        intersection_segments_a.push(IntersectionSegment::new(
                            segment_n,
                            [fa, usize::MAX],
                            [*fb, usize::MAX],
                        ));
                    }
                }
            }
        }
        println!(
            "A Intersection segments collected in {:.2?}",
            start.elapsed()
        );

        let start = Instant::now();
        for fb in 0..b.faces.len() {
            let mut candidates = Vec::new();
            tree_a.query(&b.face_aabb(fb), &mut candidates);

            let pa_idx = b.face_vertices(fb);
            let pa_vec: Vec<Point<T, 3>> = pa_idx
                .into_iter()
                .map(|vi| {
                    Point::<T, 3>::from_vals([
                        b.vertices[vi].position[0].clone(),
                        b.vertices[vi].position[1].clone(),
                        b.vertices[vi].position[2].clone(),
                    ])
                })
                .collect();
            let pa: [Point<T, 3>; 3] = pa_vec.try_into().expect("Expected 3 vertices");

            for &fa in &candidates {
                let pb_idx = a.face_vertices(*fa);
                let pb_vec: Vec<Point<T, 3>> = pb_idx
                    .into_iter()
                    .map(|vi| {
                        Point::<T, 3>::from_vals([
                            a.vertices[vi].position[0].clone(),
                            a.vertices[vi].position[1].clone(),
                            a.vertices[vi].position[2].clone(),
                        ])
                    })
                    .collect();
                let pb: [Point<T, 3>; 3] = pb_vec.try_into().expect("Expected 3 vertices");

                if let Some(s) = tri_tri_intersection(&pa, &pb) {
                    let segment_n = Segment::<T, N>::new(
                        &Point::<T, N>::from_vals(from_fn(|i| s.a[i].clone())),
                        &Point::<T, N>::from_vals(from_fn(|i| s.b[i].clone())),
                    );
                    if segment_n.length().is_positive() {
                        intersection_segments_b.push(IntersectionSegment::new(
                            segment_n,
                            [fb, usize::MAX],
                            [*fa, usize::MAX],
                        ));
                    }
                }
            }
        }
        println!(
            "A Intersection segments collected in {:.2?}",
            start.elapsed()
        );

        let start = Instant::now();
        filter_coplanar_intersections(&mut intersection_segments_a, &a, &b);
        filter_coplanar_intersections(&mut intersection_segments_b, &b, &a);
        println!("Coplanar intersections filtered in {:.2?}", start.elapsed());

        // 2. Link intersection segments into chains/graphs
        let start = Instant::now();
        let chain_roots_a = link_intersection_segments(&mut intersection_segments_a);
        let chain_roots_b = link_intersection_segments(&mut intersection_segments_b);
        println!(
            "Intersection segments linked into chains in {:.2?}",
            start.elapsed()
        );

        // 3. Split faces along intersection curves
        let start = Instant::now();
        let chains_a = split_mesh_along_intersection_curves(
            &mut a,
            &mut tree_a,
            &mut intersection_segments_a,
            &chain_roots_a,
        );
        println!(
            "A Faces split along intersection curves in {:.2?}",
            start.elapsed()
        );

        let start = Instant::now();
        let chains_b = split_mesh_along_intersection_curves(
            &mut b,
            &mut tree_b,
            &mut intersection_segments_b,
            &chain_roots_b,
        );
        println!(
            "B Faces split along intersection curves in {:.2?}",
            start.elapsed()
        );

        // a.faces.retain(|f| f.half_edge != usize::MAX);

        // let mut test = a.clone();
        // 4. Rebuild and cleanup

        // let _ = write_obj(&test, "/mnt/v/cgar_meshes/a.obj");

        // for i in &chains[0] {
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

        // let intersection = &intersection_segments[*i];
        // println!("{}: {:?}", i, intersection);
        // println!("------------------------------");
        // }

        // remove_invalidated_faces(&mut test);
        // test.remove_unused_vertices();
        // test.build_boundary_loops();

        // let _ = write_obj(&test, "/mnt/v/cgar_meshes/a_2.obj");

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
            classify_faces_inside_intersection_loops(&a, &b, &intersection_segments_a, &chains_a);

        // remove_invalidated_faces(&mut a);

        for (fa, inside) in a_classifications.iter().enumerate() {
            if a.faces[fa].removed {
                continue;
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
                    &intersection_segments_b,
                    &chains_b,
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
                    &intersection_segments_b,
                    &chains_b,
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
                let b_classifications = classify_faces_inside_intersection_loops(
                    &b,
                    &a,
                    &intersection_segments_b,
                    &chains_b,
                );
                for (fb, keep) in b_classifications.iter().enumerate() {
                    if *keep {
                        let vs = b.face_vertices(fb);
                        // Flip face orientation for caps
                        result.add_triangle(
                            vid_map[&(VertexSource::B, vs[2])],
                            vid_map[&(VertexSource::B, vs[1])],
                            vid_map[&(VertexSource::B, vs[0])],
                        );
                    }
                }
            }
        }

        result.remove_unused_vertices();
        remove_invalidated_faces(&mut result);
        // result.build_boundary_loops();
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

fn split_mesh_along_intersection_curves<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    chain_roots: &Vec<usize>,
) -> Vec<Vec<usize>>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
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

            process_segment_simple(
                mesh,
                aabb_tree,
                intersection_segments,
                seg_idx,
                &mut chain,
                i,
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

fn find_valid_half_edge<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    half_edge_idx: usize,
) -> usize {
    let mut current_idx = half_edge_idx;
    let mut visited = std::collections::HashSet::new();
    let max_depth = 50; // Prevent infinite loops

    for _ in 0..max_depth {
        // Cycle detection
        if !visited.insert(current_idx) {
            // Already visited - choose first valid replacement to break cycle
            if let Some(&(he1, he2)) = mesh.half_edge_split_map.get(&current_idx) {
                if he1 < mesh.half_edges.len() && !mesh.half_edges[he1].removed {
                    return he1;
                }
                if he2 != usize::MAX && he2 < mesh.half_edges.len() && !mesh.half_edges[he2].removed
                {
                    return he2;
                }
            }
            break;
        }

        // Check if current half-edge is valid
        if current_idx < mesh.half_edges.len() && !mesh.half_edges[current_idx].removed {
            return current_idx;
        }

        // Look up replacement in split map
        if let Some(&(replacement_a, replacement_b)) = mesh.half_edge_split_map.get(&current_idx) {
            if replacement_b == usize::MAX {
                // Single replacement - follow it
                current_idx = replacement_a;
                continue;
            }

            // Multiple replacements - choose first valid one
            if replacement_a < mesh.half_edges.len() && !mesh.half_edges[replacement_a].removed {
                return replacement_a;
            }
            if replacement_b < mesh.half_edges.len() && !mesh.half_edges[replacement_b].removed {
                return replacement_b;
            }

            // Both potentially invalid - follow first one to continue search
            current_idx = replacement_a;
        } else {
            // No replacement found
            break;
        }
    }

    panic!(
        "No valid half-edge found for index {} after {} iterations in mesh with {} half-edges",
        half_edge_idx,
        max_depth,
        mesh.half_edges.len()
    );
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
    if face_idx < mesh.faces.len() && !mesh.faces[face_idx].removed {
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

    // if face_idx >= mesh.faces.len() || mesh.faces[face_idx].half_edge == usize::MAX {
    //     if debug {
    //         // println!("Face index {} is out of bounds", face_idx);
    //     }
    //     return None; // Invalid face index
    // }

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
fn are_vertices_connected<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    vertex_a: usize,
    vertex_b: usize,
) -> bool {
    if vertex_a >= mesh.vertices.len() || vertex_b >= mesh.vertices.len() {
        return false; // Invalid vertex indices
    }
    for half_edge in &mesh.half_edges {
        if half_edge.vertex == vertex_a && half_edge.twin != usize::MAX {
            if mesh.half_edges[find_valid_half_edge(mesh, half_edge.twin)].vertex == vertex_b {
                return true; // Found an edge between the vertices
            }
        }
    }
    false // No edge found between the vertices
}

fn process_segment_simple<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    segment_idx: usize,
    chain: &mut Vec<usize>,
    chain_idx: usize,
) where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let segment = &intersection_segments[segment_idx];
    let vertex_a;
    let vertex_b;

    if let Some(valid_face_idx) =
        find_valid_face(mesh, segment.faces_a[0], &segment.segment.a, false)
    {
        let start = Instant::now();
        let split_result =
            mesh.split_or_find_vertex_on_face(aabb_tree, valid_face_idx, &segment.segment.a);
        // println!("Split or find vertex took {:?}", start.elapsed());
        if let Some(split_result) = split_result {
            vertex_a = split_result.new_vertex;
        } else {
            panic!(
                "Failed to split or find vertex on face {} for segment end {:?}",
                segment.faces_a[0], segment.segment.b
            );
        }
    } else {
        panic!(
            "Failed to find valid face for segment end {:?} on face {}",
            segment.segment.b, segment.faces_a[0]
        );
    }

    // let start = Instant::now();
    if let Some(valid_face_idx) =
        find_valid_face(mesh, segment.faces_a[0], &segment.segment.b, false)
    {
        let split_result =
            mesh.split_or_find_vertex_on_face(aabb_tree, valid_face_idx, &segment.segment.b);
        if let Some(split_result) = split_result {
            vertex_b = split_result.new_vertex;

            if !split_result.new_faces.is_empty() {
                let he = mesh.half_edge_between(vertex_a, vertex_b);
                if let Some(mut he) = he {
                    he = find_valid_half_edge(mesh, he);
                    let twin = find_valid_half_edge(mesh, mesh.half_edges[he].twin);
                    intersection_segments[segment_idx].faces_a[0] =
                        mesh.half_edges[he].face.unwrap_or(usize::MAX);
                    intersection_segments[segment_idx].faces_a[1] =
                        mesh.half_edges[twin].face.unwrap_or(usize::MAX);
                } else {
                    // If no direct half-edge found, use the new faces from split result
                    if split_result.new_faces.len() >= 2 {
                        intersection_segments[segment_idx].faces_a[0] = split_result.new_faces[0];
                        intersection_segments[segment_idx].faces_a[1] = split_result.new_faces[1];
                    }
                }
            } else {
                let he = mesh.half_edge_between(vertex_a, vertex_b);
                if let Some(he) = he {
                    intersection_segments[segment_idx].faces_a[0] =
                        mesh.half_edges[he].face.unwrap_or(usize::MAX);
                    intersection_segments[segment_idx].faces_a[1] = mesh.half_edges
                        [mesh.half_edges[he].twin]
                        .face
                        .unwrap_or(usize::MAX);
                }
            }
        } else {
            panic!(
                "Failed to split or find vertex on face {} for segment end {:?}",
                segment.faces_a[0], segment.segment.b
            );
        }
    } else {
        panic!(
            "Failed to find valid face for segment end {:?} on face {}",
            segment.segment.b, segment.faces_a[0]
        );
    }
    // println!("Part Two took {:?}", start.elapsed());

    if vertex_a != usize::MAX
        && vertex_b != usize::MAX
        && !are_vertices_connected(mesh, vertex_a, vertex_b)
    {
        // remove_invalidated_faces(mesh);
        // mesh.remove_unused_vertices();
        // let _ = write_obj(&mesh, "/mnt/v/cgar_meshes/vamooo.obj");
        // panic!("a");

        // println!("******** CARVING **********");
        // Carve segment from vertex_a to vertex_b
        let start = Instant::now();
        let success = carve_segment_to_vertex(
            mesh,
            aabb_tree,
            vertex_a,
            vertex_b,
            intersection_segments,
            segment_idx,
            chain,
            chain_idx,
        );
        println!("Carving took {:?}", start.elapsed());

        // remove_invalidated_faces(mesh);
        // mesh.remove_unused_vertices();
        // let _ = write_obj(&mesh, "/mnt/v/cgar_meshes/vamooo.obj");
        // panic!("a");

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

    println!("num of faces before removal: {}", original_count);

    for (old_idx, face) in mesh.faces.iter().enumerate() {
        if !face.removed {
            let new_idx = new_faces.len();
            face_mapping[old_idx] = Some(new_idx);
            new_faces.push(face.clone());
        }
    }

    // Update half-edge face references
    for he in &mut mesh.half_edges {
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

    println!("num of faces after removal: {}", new_faces.len());

    // Replace faces
    mesh.faces = new_faces;

    // println!(
    //     "Removed {} invalidated faces ({} -> {} faces)",
    //     original_count - mesh.faces.len(),
    //     original_count,
    //     mesh.faces.len()
    // );
}

/// Helper function with improved error handling
fn get_face_vertices_safe<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_idx: usize,
) -> Option<Vec<usize>> {
    // if face_idx >= mesh.faces.len() || mesh.faces[face_idx].half_edge == usize::MAX {
    //     return None;
    // }

    // Use existing safe method
    mesh.face_vertices_safe(face_idx)
}

fn carve_segment_to_vertex<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
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
    println!("CARVING............");
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
        println!("STRATEGY 1");
        // println!("DEBUG: Direct edge already exists");
        return true;
    }

    // **STRATEGY 2: Shared face connection**
    if let Some(shared_face) = find_shared_face_iterative(mesh, start_vertex, target_vertex) {
        println!("STRATEGY 2");
        // println!("DEBUG: Found shared face {}", shared_face);
        return create_edge_in_shared_face(mesh, shared_face, start_vertex, target_vertex);
    }

    // **STRATEGY 3: Adjacent face connection**
    if let Some(connection_path) = find_adjacent_face_connection(mesh, start_vertex, target_vertex)
    {
        println!("STRATEGY 3");
        println!("FOUND ADJACENT FACE CONNECTION: {:?}", connection_path);
        return mesh
            .find_edge_intersection(
                aabb_tree,
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
        // if face.half_edge == usize::MAX {
        //     continue;
        // }

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
    let mut current_he = find_valid_half_edge(mesh, start_he);
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

        current_he = find_valid_half_edge(mesh, mesh.half_edges[current_he].next);
        if current_he >= mesh.half_edges.len() || mesh.half_edges[current_he].removed {
            return None;
        }

        if current_he == start_he {
            break;
        }
    }

    Some(result)
}

fn find_face_containing_vertex_from_face_split_map<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    root_face: usize,
    vertex: usize,
) -> usize {
    let mut current_face = root_face;
    let mut visited = HashSet::new();
    let max_depth = 100; // Prevent infinite recursion
    let mut depth = 0;

    while depth < max_depth {
        if visited.contains(&current_face) {
            break; // Avoid cycles
        }
        visited.insert(current_face);

        // Check if current face is valid (not removed) and contains the vertex
        if current_face < mesh.faces.len() && !mesh.faces[current_face].removed {
            // Check if the face contains the vertex
            if let Some(face_vertices) = get_face_vertices_with_timeout(mesh, current_face, 50) {
                if face_vertices.contains(&vertex) {
                    return current_face;
                }
            }
        }

        // Look in face split map for replacement faces
        if let Some(split_info) = mesh.face_split_map.get(&current_face) {
            // Try to find a split face that contains the vertex
            let mut found_replacement = false;
            for &new_face in &split_info.new_faces {
                if new_face < mesh.faces.len() && !mesh.faces[new_face].removed {
                    if let Some(face_vertices) = get_face_vertices_with_timeout(mesh, new_face, 50)
                    {
                        if face_vertices.contains(&vertex) {
                            current_face = new_face;
                            found_replacement = true;
                            break;
                        }
                    }
                }
            }

            if found_replacement {
                depth += 1;
                continue;
            }

            // If no specific face contains the vertex, try the first valid one as fallback
            for &new_face in &split_info.new_faces {
                if new_face < mesh.faces.len() && !mesh.faces[new_face].removed {
                    current_face = new_face;
                    found_replacement = true;
                    break;
                }
            }

            if found_replacement {
                depth += 1;
                continue;
            }
        }

        // No valid replacement found
        break;
    }

    // Fallback: return original face if nothing better found
    root_face
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

    println!("!!!!!!!!!!!! create_edge_in_shared_face !!!!!!!!!!!!");
    mesh.faces[face_idx].removed = true;
    mesh.face_split_map
        .entry(face_idx)
        .or_default()
        .new_faces
        .push(mesh.faces.len());

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

    println!("!!!!!!!!!!!! create_direct_geometric_connection !!!!!!!!!!!!");
    // **ATOMIC OPERATION: Replace face with fan triangulation**
    mesh.faces[face_idx].removed = true;
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
        // if mesh.faces[face_idx].half_edge == usize::MAX {
        //     continue;
        // }

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
    if face.half_edge == usize::MAX || mesh.half_edges[face.half_edge].removed {
        return None;
    }

    let start_he = face.half_edge;
    if start_he >= mesh.half_edges.len() {
        return None;
    }

    let mut result = Vec::new();
    let mut current_he = find_valid_half_edge(mesh, start_he);
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

        current_he = find_valid_half_edge(mesh, mesh.half_edges[current_he].next);
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
            f < a.faces.len() && !a.faces[f].removed && {
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

fn point_in_triangle_optimized<T: Scalar, const N: usize>(
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
    // **OPTIMIZED: Single-pass barycentric coordinates**
    let v0 = (c - a).as_vector();
    let v1 = (b - a).as_vector();
    let v2 = (p - a).as_vector();

    // **OPTIMIZED: Precompute dot products**
    let dot00 = v0.dot(&v0);
    let dot01 = v0.dot(&v1);
    let dot11 = v1.dot(&v1);
    let dot02 = v0.dot(&v2);
    let dot12 = v1.dot(&v2);

    // **OPTIMIZED: Single division**
    let denom = &dot00 * &dot11 - &dot01 * &dot01;
    if denom.abs().is_zero() {
        return false; // Degenerate triangle
    }

    let inv_denom = T::one() / denom;
    let u = (&dot11 * &dot02 - &dot01 * &dot12) * inv_denom.clone();
    let v = (&dot00 * &dot12 - &dot01 * &dot02) * inv_denom;

    // **OPTIMIZED: Tolerance-based boundary check**
    let eps = T::tolerance();
    u >= -eps.clone() && v >= -eps.clone() && (&u + &v) <= (T::one() + eps)
}
