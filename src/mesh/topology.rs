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

use std::{
    array::from_fn,
    collections::HashSet,
    ops::{Add, Div, Mul, Sub},
};

use smallvec::SmallVec;

use crate::{
    geometry::{
        Aabb, AabbTree,
        plane::Plane,
        point::{Point, PointOps},
        segment::Segment,
        spatial_element::SpatialElement,
        util::*,
        vector::*,
    },
    impl_mesh,
    mesh::basic_types::{Mesh, PairRing, PointInMeshResult, RayCastResult, VertexRing},
    numeric::scalar::Scalar,
    operations::Zero,
};

impl_mesh! {
    pub fn face_normal(&self, face_idx: usize) -> Vector<T, N> where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>, {
        let face_vertices = self
            .face_vertices(face_idx)
            .map(|v| &self.vertices[v].position);
        let edge1 = (face_vertices[1] - face_vertices[0]).as_vector();
        let edge2 = (face_vertices[2] - face_vertices[0]).as_vector();
        edge1.cross(&edge2).normalized()
    }

    pub fn source(&self, he: usize) -> usize {
        self.half_edges[self.half_edges[he].prev].vertex
    }

    pub fn target(&self, he: usize) -> usize {
        self.half_edges[he].vertex
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

    pub fn plane_from_face(&self, face_idx: usize) -> Plane<T, N> where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,{
        let verts = self.face_vertices(face_idx); // [usize; 3]
        let v0 = &self.vertices[verts[0]].position;
        let v1 = &self.vertices[verts[1]].position;
        let v2 = &self.vertices[verts[2]].position;

        Plane::from_points(v0, v1, v2)
    }

    /// Find existing vertex near position using spatial hash
    fn find_nearby_vertex(&self, pos: &Point<T, N>, tolerance: T) -> Option<usize>
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

    pub fn faces_containing_point_aabb(
        &self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        p: &Point<T, N>,
    ) -> Vec<usize>
    where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>, {
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

    /// Return the centroid of face `f` as a Vec<f64> of length = dimensions().
    /// Currently works for any dimension, but returns a flat Vec.
    pub fn face_centroid(&self, f: usize) -> Vector<T, N>
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
    {
        match N {
            2 => self.face_area_2d(f),
            3 => self.face_area_3d(f),
            _ => panic!("face_area only supports 2D and 3D"),
        }
    }

    fn face_area_2d(&self, f: usize) -> T
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

    #[inline]
    pub fn face_vertices(&self, f: usize) -> [usize; 3] {
        let he0 = self.faces[f].half_edge;
        let he1 = self.half_edges[he0].next;
        let he2 = self.half_edges[he1].next;
        debug_assert_eq!(self.half_edges[he2].next, he0);

        let a = self.half_edges[self.half_edges[he0].prev].vertex; // TAIL of he0
        let b = self.half_edges[he0].vertex;                       // HEAD of he0
        let c = self.half_edges[he1].vertex;                       // HEAD of he1
        [a, b, c]
    }

    /// Collect unique faces incident to `v` (with real faces only)
    pub fn incident_faces(&self, v: usize) -> std::collections::HashSet<usize> {
        use std::collections::HashSet;
        let mut faces = HashSet::new();
        let ring = self.vertex_ring_ccw(v);
        for &fopt in &ring.faces_ccw {
            if let Some(f) = fopt {
                if !self.faces[f].removed { faces.insert(f); }
            }
        }
        faces
    }

    pub fn point_on_half_edge(&self, he: usize, p: &Point<T, N>) -> Option<T>
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

    pub fn point_in_mesh(
        &self,
        tree: &AabbTree<T, 3, Point<T, 3>, usize>,
        p: &Point<T, 3>,
    ) -> PointInMeshResult
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
                if !self.faces[face_idx].removed {
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
    where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
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
    ) -> (usize, T, T) where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
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

        match (closest_he, closest_t, closest_u) {
            (Some(he), Some(t), Some(u)) => (he, t, u),
            _ => panic!("Ray did not intersect any edge from start vertex"),
        }
    }

    pub fn half_edge_between(&self, vi0: usize, vi1: usize) -> Option<usize> {
        self.edge_map.get(&(vi0, vi1)).copied()
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
                loop {
                    if cur == i {
                        break;
                    }
                    cur = self.half_edges[cur].next;
                    assert!(cur != self.faces[f].half_edge, "he {} not in face {}", i, f);
                }
            }
        }

        let mut edge_set = HashSet::new();
        for (_i, he) in self.half_edges.iter().enumerate() {
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

    // Iterative face half-edge traversal to eliminate recursion
    pub fn get_face_half_edges_iterative(
        &self,
        face_idx: usize,
    ) -> Option<Vec<usize>> {
        if face_idx >= self.faces.len() {
            return None;
        }

        let face = &self.faces[face_idx];
        if face.half_edge == usize::MAX || self.half_edges[face.half_edge].removed {
            return None;
        }

        let start_he = face.half_edge;
        if start_he >= self.half_edges.len() {
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

            if current_he >= self.half_edges.len() {
                return None;
            }

            result.push(current_he);

            current_he = self.half_edges[current_he].next;
            if current_he >= self.half_edges.len() {
                return None;
            }

            if current_he == start_he {
                break;
            }
        }

        Some(result)
    }

    #[inline]
    pub fn rot_ccw_around_vertex(&self, h: usize) -> usize {
        // CCW around the *origin* (tail) of h. Works for interior and border
        // once boundary loops have valid prev/next.
        let prev = self.half_edges[h].prev;
        self.half_edges[prev].twin
    }

    pub fn vertex_ring_ccw(&self, v: usize) -> VertexRing {
        use std::collections::HashSet;

        // --- Seed: ensure we start from an OUTGOING half-edge of v ---
        let mut seed = match self.vertices[v].half_edge {
            Some(h) => h,
            None => {
                // No incident half-edge recorded — return an empty, harmless ring.
                return VertexRing {
                    center: v,
                    halfedges_ccw: vec![],
                    neighbors_ccw: vec![],
                    faces_ccw: vec![],
                    is_border: true,
                };
            }
        };
        if self.source(seed) != v {
            seed = self.half_edges[seed].twin;
        }

        // If the cached seed is removed, rotate to find a live outgoing spoke.
        // If *all* spokes are removed, return an empty ring.
        {
            let mut cur = seed;
            let mut seen = HashSet::new();
            while self.half_edges[cur].removed && seen.insert(cur) {
                cur = self.rot_ccw_around_vertex(cur);
                if cur == seed { break; }
            }
            if self.half_edges[cur].removed {
                return VertexRing {
                    center: v,
                    halfedges_ccw: vec![],
                    neighbors_ccw: vec![],
                    faces_ccw: vec![],
                    is_border: true,
                };
            }
            seed = cur;
        }

        // --- Traverse once around v in CCW order ---
        let mut halfedges = Vec::new();
        let mut neighbors = Vec::new();
        let mut faces     = Vec::new();
        let mut is_border = false;

        let start = seed;
        let mut cur = start;
        let mut seen = HashSet::new();

        loop {
            // Emit only live spokes (but still rotate across anything, since wiring is valid)
            if !self.half_edges[cur].removed {
                // neighbor is the head of the half-edge
                neighbors.push(self.half_edges[cur].vertex);

                // face entry is None for border or if the face is marked removed
                let f = if self.face_ok(cur) { self.half_edges[cur].face } else { None };
                if f.is_none() { is_border = true; }
                faces.push(f);

                halfedges.push(cur);
            }

            let nxt = self.rot_ccw_around_vertex(cur);
            if nxt == start { break; }

            // Safety guard against accidental loops if wiring is corrupt
            if !seen.insert(nxt) { break; }

            cur = nxt;
        }

        VertexRing {
            center: v,
            halfedges_ccw: halfedges,
            neighbors_ccw: neighbors,
            faces_ccw: faces,
            is_border,
        }
    }

    #[inline]
    pub fn face_ok(&self, h: usize) -> bool {
        match self.half_edges[h].face {
            Some(f) => !self.faces[f].removed,
            None => false,
        }
    }

    pub fn ring_pair(&self, v0: usize, v1: usize) -> Option<PairRing> {
        if v0 == v1 { return None; }

        // Exact directed incidences
        let he_v0v1 = self.half_edge_between(v0, v1)?;
        let he_v1v0 = self.half_edge_between(v1, v0)?;

        let ring0 = self.vertex_ring_ccw(v0);
        let ring1 = self.vertex_ring_ccw(v1);

        // Index by the EXACT half-edge ids
        let idx_v1_in_ring0 = ring0.halfedges_ccw.iter().position(|&h| h == he_v0v1)?;
        let idx_v0_in_ring1 = ring1.halfedges_ccw.iter().position(|&h| h == he_v1v0)?;

        // (Optional sanity while debugging)
        debug_assert_eq!(ring0.neighbors_ccw[idx_v1_in_ring0], v1);
        debug_assert_eq!(ring1.neighbors_ccw[idx_v0_in_ring1], v0);

        // Third (opposite) vertex only if the incident face exists and is not removed
        let opposite_a = if self.face_ok(he_v0v1) {
            let hn = self.half_edges[he_v0v1].next;
            Some(self.half_edges[hn].vertex)
        } else { None };

        let opposite_b = if self.face_ok(he_v1v0) {
            let hn = self.half_edges[he_v1v0].next;
            Some(self.half_edges[hn].vertex)
        } else { None };

        use std::collections::HashSet;
        let set0: HashSet<_> = ring0.neighbors_ccw.iter().copied().filter(|&x| x != v1).collect();
        let set1: HashSet<_> = ring1.neighbors_ccw.iter().copied().filter(|&x| x != v0).collect();

        let is_border_edge = !(self.face_ok(he_v0v1) && self.face_ok(he_v1v0));

        Some(PairRing {
            v0, v1,
            ring0, ring1,
            idx_v1_in_ring0: Some(idx_v1_in_ring0),
            idx_v0_in_ring1: Some(idx_v0_in_ring1),
            opposite_a,
            opposite_b,
            common_neighbors: set0.intersection(&set1).copied().collect(),
            union_neighbors:  set0.union(&set1).copied().collect(),
            is_border_edge,
        })
    }

    /// Convenience: the classic triangle-mesh link condition for collapsing edge (v0,v1).
    /// Accepts borders: on border, common neighbors must be exactly {a} or {b}.
    pub fn collapse_link_condition_triangle(&self, v0: usize, v1: usize) -> bool {
        let Some(pr) = self.ring_pair(v0, v1) else { return false; };

        // Gather expected opposite set (ignoring None)
        let mut expected = std::collections::HashSet::new();
        if let Some(a) = pr.opposite_a { expected.insert(a); }
        if let Some(b) = pr.opposite_b { expected.insert(b); }

        // Intersection of neighbor sets should equal expected
        pr.common_neighbors == expected
    }
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
