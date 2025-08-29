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
    ops::{Add, Div, Mul, Neg, Sub},
};

use ahash::AHashSet;
use smallvec::SmallVec;

use crate::{
    geometry::{
        Aabb, AabbTree,
        plane::Plane,
        point::{Point, PointOps},
        segment::{Segment, SegmentOps},
        spatial_element::SpatialElement,
        tri_tri_intersect::{TriTriIntersectionResult, tri_tri_intersection},
        util::*,
        vector::*,
    },
    impl_mesh,
    kernel::{self, predicates::TrianglePoint},
    mesh::basic_types::{Mesh, PairRing, PointInMeshResult, RayCastResult, Triangle, VertexRing},
    numeric::{
        cgar_f64::CgarF64,
        scalar::{RefInto, Scalar},
    },
    operations::Zero,
};

/// - Inside:    (f, usize::MAX, usize::MAX, 0)
/// - OnEdge:    (f, he_of_f_edge, usize::MAX, u in [0,1] along that half-edge)
/// - OnVertex:  (f, usize::MAX, vertex_id, 0)
#[derive(Debug)]
pub enum FindFaceResult<T> {
    Inside { f: usize, bary: (T, T, T) },
    OnEdge { f: usize, he: usize, u: T },
    OnVertex { f: usize, v: usize },
}

/// Kind of self-intersection detected between two faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfIntersectionKind {
    // Non-coplanar triangles cross along a segment of positive length
    NonCoplanarCrossing,
    // Coplanar triangles have overlapping positive area (polygon overlap)
    CoplanarAreaOverlap,
    // Coplanar triangles overlap along a segment (not just a shared mesh edge)
    CoplanarEdgeOverlap,
}

/// A detected self-intersection/overlap between faces `a` and `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelfIntersection {
    pub a: usize,
    pub b: usize,
    pub kind: SelfIntersectionKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VertexRayResult<T: Scalar> {
    /// Ray intersects opposite edge at distance t from vertex, at position u along the edge
    EdgeIntersection {
        half_edge: usize,
        distance: T,
        edge_parameter: T,
    },
    /// Ray is collinear with an edge of the triangle
    CollinearWithEdge(usize),
}

impl_mesh! {
    pub fn face_normal(&self, face_idx: usize) -> Vector<T, N> where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>, {
        let face_vertices = self
            .face_vertices(face_idx)
            .map(|v| &self.vertices[v].position);
        let edge1 = (face_vertices[1] - face_vertices[0]).as_vector();
        let edge2 = (face_vertices[2] - face_vertices[0]).as_vector();
        edge1.cross(&edge2)
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
        let verts = self.face_vertices(face_idx); // [usize; N]
        let v0 = &self.vertices[verts[0]].position;
        let v1 = &self.vertices[verts[1]].position;
        let v2 = &self.vertices[verts[2]].position;

        Plane::from_points(v0, v1, v2)
    }

    /// Find existing vertex near position using spatial hash
    // fn find_nearby_vertex(&self, pos: &Point<T, N>, tolerance: T) -> Option<usize>
    // {
    //     let center_key = self.position_to_hash_key(pos);

    //     // Check center cell and 26 neighboring cells (3x3x3 grid)
    //     for dx in -1..=1 {
    //         for dy in -1..=1 {
    //             for dz in -1..=1 {
    //                 let key = (center_key.0 + dx, center_key.1 + dy, center_key.2 + dz);

    //                 if let Some(candidates) = self.vertex_spatial_hash.get(&key) {
    //                     for &vi in candidates {
    //                         if vi < self.vertices.len()
    //                             && self.vertices[vi].position.distance_to(pos) < tolerance
    //                         {
    //                             return Some(vi);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     None
    // }

    pub fn faces_containing_point_aabb(
        &self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        p: &Point<T, N>,
    ) -> Vec<usize>
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Add<&'a T, Output = T>
            + Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>,
    {
        // 1) tight query box around p (avoid `.into()`; it’s a no-op but may hide clones)
        let tol  = T::query_tolerance();
        let qmin = Point::<T, N>::from_vals(std::array::from_fn(|i| &p[i] - &tol));
        let qmax = Point::<T, N>::from_vals(std::array::from_fn(|i| &p[i] + &tol));
        let query_aabb = Aabb::new(qmin, qmax);

        // 2) gather candidates
        let mut candidates = Vec::new();
        aabb_tree.query_valid(&query_aabb, &mut candidates);

        // Optional very-wide retry BEFORE rebuilding (keep constants exact)
        if candidates.is_empty() {
            let big = &T::tolerance() * &T::from(100); // exact 100
            let qmin2 = Point::<T, N>::from_vals(std::array::from_fn(|i| &p[i] - &big));
            let qmax2 = Point::<T, N>::from_vals(std::array::from_fn(|i| &p[i] + &big));
            let large = Aabb::from_points(&qmin2, &qmax2);
            aabb_tree.query_valid(&large, &mut candidates);
            // Avoid rebuilding here unless you really must. Rebuilding with LazyExact is expensive.
            // Prefer a periodic/explicit rebuild based on needs_rebuild().
        }

        if candidates.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(candidates.len().min(8));

        // helpers only used when N==3; bail early otherwise
        if N != 3 {
            // fallback: keep your old kernel path if you support 2D as well
            for &face_idx in &candidates {
                let he0 = self.faces[*face_idx].half_edge;
                if he0 >= self.half_edges.len() { continue; }
                let he1 = self.half_edges[he0].next;
                let he2 = self.half_edges[he1].next;
                if he2 >= self.half_edges.len() || self.half_edges[he2].next != he0 { continue; }

                let a = &self.vertices[self.half_edges[he0].vertex].position;
                let b = &self.vertices[self.half_edges[he1].vertex].position;
                let c = &self.vertices[self.half_edges[he2].vertex].position;

                if kernel::point_in_or_on_triangle(p, a, b, c) == TrianglePoint::In {
                    result.push(*face_idx);
                }
            }
            return result;
        }

        // 3D fast path with zero duplicated math, squared plane test, sign-only decisions
        let tol2 = &tol * &tol;

        for &face_idx in &candidates {
            // (A) lightweight topology checks — ideally validate once at build!
            let he0 = self.faces[*face_idx].half_edge;
            if he0 >= self.half_edges.len() { continue; }
            let he1 = self.half_edges[he0].next;
            let he2 = self.half_edges[he1].next;
            if he1 >= self.half_edges.len() || he2 >= self.half_edges.len() || self.half_edges[he2].next != he0 { continue; }

            let v0 = self.half_edges[he0].vertex;
            let v1 = self.half_edges[he1].vertex;
            let v2 = self.half_edges[he2].vertex;
            if v0 >= self.vertices.len() || v1 >= self.vertices.len() || v2 >= self.vertices.len() { continue; }

            let a = &self.vertices[v0].position;
            let b = &self.vertices[v1].position;
            let c = &self.vertices[v2].position;

            // (B) geometry (single pass)
            let ab = (b - a).as_vector();
            let ac = (c - a).as_vector();
            let ap = (p - a).as_vector();

            let n   = ab.cross(&ac);
            let n2  = n.dot(&n);
            if n2.is_zero() { continue; } // degenerate face

            // plane distance squared: (n·ap)^2  ?  (tol^2) * (n·n)
            let d_plane = n.dot(&ap);
            let d2      = &d_plane * &d_plane;
            let rhs     = &tol2 * &n2;
            if (&d2 - &rhs).is_positive() { continue; }

            // edge functions on projected point: compute two, derive the third
            let bp = (p - b).as_vector();
            // let cp = (p - c).as_vector();
            let bc = (c - b).as_vector();
            // let ca = (a - c).as_vector();

            let e0 = ab.cross(&ap).dot(&n);
            let e1 = bc.cross(&bp).dot(&n);
            let e2 = &(&n2 - &e0) - &e1; // identity: e0 + e1 + e2 = n·n

            // inside iff all same sign or zero; treat boundary as "not strictly inside"
            // (If you want to count boundary as inside, change the checks to >= 0.)
            let z0 = e0.is_zero(); let z1 = e1.is_zero(); let z2 = e2.is_zero();
            let neg = (!z0 && e0.is_negative()) || (!z1 && e1.is_negative()) || (!z2 && e2.is_negative());
            let pos = (!z0 && e0.is_positive()) || (!z1 && e1.is_positive()) || (!z2 && e2.is_positive());

            if !(neg && pos) && !(z0 || z1 || z2) {
                // all nonzero and same sign -> strictly inside
                result.push(*face_idx);
            }
            // else: discard or record separately if you want “on edge/vertex”
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
            2 => self.face_area_2(f),
            3 => self.face_area_3(f),
            _ => panic!("face_area only supports 2D and 3D"),
        }
    }

    fn face_area_2(&self, f: usize) -> T
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

        let ab_2 = Vector::<T, 2>::from_vals([ab[0].clone(), ab[1].clone()]);
        let ac_2 = Vector::<T, 2>::from_vals([ac[0].clone(), ac[1].clone()]);

        let cross_product = ab_2.cross(&ac_2);
        cross_product.abs() / T::from_num_den(2, 1)
    }

    fn face_area_3(&self, f: usize) -> T
    {
        let face_vertices = self.face_vertices(f);
        assert!(face_vertices.len() == 3, "face_area only works for triangular faces");

        let a = &self.vertices[face_vertices[0]].position;
        let b = &self.vertices[face_vertices[1]].position;
        let c = &self.vertices[face_vertices[2]].position;

        let ab = (b - a).as_vector();
        let ac = (c - a).as_vector();

        let ab_3 = Vector::<T, 3>::from_vals([ab[0].clone(), ab[1].clone(), ab[2].clone()]);
        let ac_3 = Vector::<T, 3>::from_vals([ac[0].clone(), ac[1].clone(), ac[2].clone()]);

        let cross = ab_3.cross(&ac_3);
        // area^2 = ||cross||^2 / 4
        cross.norm2() / T::from_num_den(4, 1)
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
    pub fn incident_faces(&self, v: usize) -> AHashSet<usize> {
        let mut faces = AHashSet::new();
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

        kernel::point_u_on_segment(start, end, p)
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
        tree: &AabbTree<T, N, Point<T, N>, usize>,
        p: &Point<T, N>,
    ) -> PointInMeshResult where T: Into<T>, Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        let mut inside_count = 0;
        let mut total_rays = 0;
        let mut on_surface = false;

        let arr_rays = [
            [T::one(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero()],
            [T::zero(), T::zero(), T::one()],
            [T::from(-1.0), T::zero(), T::zero()],
            [T::zero(), T::from(-1.0), T::zero()],
            [T::zero(), T::zero(), T::from(-1.0)],
        ];

        let rays = vec![
            Vector::<T, N>::from_vals(from_fn(|i| arr_rays[0][i].clone())),
            Vector::<T, N>::from_vals(from_fn(|i| arr_rays[1][i].clone())),
            Vector::<T, N>::from_vals(from_fn(|i| arr_rays[2][i].clone())),
            Vector::<T, N>::from_vals(from_fn(|i| arr_rays[3][i].clone())),
            Vector::<T, N>::from_vals(from_fn(|i| arr_rays[4][i].clone())),
            Vector::<T, N>::from_vals(from_fn(|i| arr_rays[5][i].clone())),
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
        p: &Point<T, N>,
        dir: &Vector<T, N>,
        tree: &AabbTree<T, N, Point<T, N>, usize>,
    ) -> Option<RayCastResult>
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
        for<'a> &'a T: Add<&'a T, Output = T>
            + Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        use std::cmp::Ordering;

        let mut hits: Vec<T> = Vec::new();
        let mut touches_surface = false;

        // Create ray AABB for tree query using approximate conversion (no exact)
        let far_multiplier = T::from(1000.0);
        let far_point = Point::<T, N>::from_vals(std::array::from_fn(|i| {
            let tmp = &p[i] + &(&dir[i] * &far_multiplier);
            tmp.into()
        }));
        let p_f = Point::<T, N>::from_vals(std::array::from_fn(|i| {
            p[i].clone().into()
        }));
        let ray_aabb = Aabb::<T, N, Point<T, N>>::from_points(&p_f, &far_point);

        // Query tree for faces that intersect ray
        let mut candidate_faces = Vec::new();
        tree.query_valid(&ray_aabb, &mut candidate_faces);

        let tol = T::tolerance();

        // Test only candidate faces
        for &fi in &candidate_faces {
            let vs_idxs = self.face_vertices(*fi);
            let vs = [
                &self.vertices[vs_idxs[0]].position,
                &self.vertices[vs_idxs[1]].position,
                &self.vertices[vs_idxs[2]].position,
            ];

            if N == 3 {
                if let Some(t) = ray_triangle_intersection(p, dir, std::array::from_fn(|i| vs[i])) {
                    let at = t.abs();
                    if (&at - &tol).is_negative_or_zero() {
                        touches_surface = true;
                    } else if (&t - &tol).is_positive() {
                        hits.push(t);
                    }
                }
            }
        }

        if hits.is_empty() {
            return None;
        }

        // Sort using sign-based comparator to avoid PartialOrd exact
        hits.sort_by(|a, b| {
            let d = a - b;
            if d.is_negative() {
                Ordering::Less
            } else if d.is_positive() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        });

        if touches_surface {
            Some(RayCastResult::OnSurface)
        } else {
            // Deduplicate by tolerance using sign checks
            let mut filtered_hits = 0usize;
            let mut last_t: Option<T> = None;

            for t in hits.into_iter() {
                if let Some(ref lt) = last_t {
                    let diff = (&t - lt).abs();
                    if (&diff - &tol).is_positive() {
                        filtered_hits += 1;
                        last_t = Some(t);
                    }
                } else {
                    filtered_hits += 1;
                    last_t = Some(t);
                }
            }

            Some(if filtered_hits % 2 == 1 {
                RayCastResult::Inside
            } else {
                RayCastResult::Outside
            })
        }
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
                if self.faces[face_idx].removed {
                    panic!("The structure here should always be valid.");
                }
                result.push(face_idx); // valid half_edges should always point to valid faces
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

    /// Cast a ray from a vertex within a triangle face along the geodesic direction.
    /// Preserves the surface curvature relationship of the direction vector.
    pub fn cast_ray_from_vertex_in_triangle_3(
        &self,
        face: usize,
        vertex: usize,
        direction: &Vector<T, N>,
    ) -> Option<VertexRayResult<T>>
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        let vs = self.face_vertices(face);

        // Find vertex position in face and identify opposite edge
        let vertex_idx = vs.iter().position(|&v| v == vertex)?;
        let (opp_v1, opp_v2) = match vertex_idx {
            0 => (vs[1], vs[2]), // vertex is A, opposite edge is BC
            1 => (vs[2], vs[0]), // vertex is B, opposite edge is CA
            2 => (vs[0], vs[1]), // vertex is C, opposite edge is AB
            _ => return None,
        };

        let p0 = &self.vertices[vertex].position;
        let p1 = &self.vertices[opp_v1].position;
        let p2 = &self.vertices[opp_v2].position;

        // Calculate face normal and tangent vectors
        let vs_tri = [&self.vertices[vs[0]].position, &self.vertices[vs[1]].position, &self.vertices[vs[2]].position];
        let e1 = (vs_tri[1] - vs_tri[0]).as_vector();
        let e2 = (vs_tri[2] - vs_tri[0]).as_vector();
        let face_normal = e1.cross(&e2);

        let n2 = face_normal.dot(&face_normal);
        if n2.is_zero() {
            return None; // degenerate triangle
        }

        // Compute discrete metric tensor for geodesic preservation
        let e1_norm2 = e1.dot(&e1);
        let e2_norm2 = e2.dot(&e2);
        let e1_dot_e2 = e1.dot(&e2);

        let tol = T::tolerance();
        if e1_norm2.is_zero() || e2_norm2.is_zero() {
            return None; // degenerate edges
        }

        // Geodesic direction via parallel transport
        // Transform direction using the metric tensor to preserve geodesic properties
        let d_dot_e1 = direction.dot(&e1);
        let d_dot_e2 = direction.dot(&e2);

        // Metric determinant
        let metric_det = &e1_norm2 * &e2_norm2 - &e1_dot_e2 * &e1_dot_e2;
        if metric_det.abs() <= &tol * &tol {
            return None; // singular metric
        }

        // Geodesic-preserving projection using inverse metric tensor
        let inv_metric_det = T::one() / metric_det;
        let coeff1 = &(&d_dot_e1 * &e2_norm2 - &d_dot_e2 * &e1_dot_e2) * &inv_metric_det;
        let coeff2 = &(&d_dot_e2 * &e1_norm2 - &d_dot_e1 * &e1_dot_e2) * &inv_metric_det;

        let geodesic_dir = &e1.scale(&coeff1) + &e2.scale(&coeff2);

        // Validate geodesic direction magnitude
        let tol2 = &tol * &tol;
        if geodesic_dir.dot(&geodesic_dir) <= tol2 {
            return None; // geodesic direction too small
        }

        // Check collinearity with edges from the vertex
        let adjacent_edges = match vertex_idx {
            0 => [(vs[0], vs[1]), (vs[0], vs[2])],
            1 => [(vs[1], vs[2]), (vs[1], vs[0])],
            2 => [(vs[2], vs[0]), (vs[2], vs[1])],
            _ => return None,
        };

        for &(v_start, v_end) in &adjacent_edges {
            let edge_vec = (&self.vertices[v_end].position - &self.vertices[v_start].position).as_vector();
            let cross = geodesic_dir.cross(&edge_vec);

            if cross.dot(&cross) <= tol2 {
                if let Some(he) = self.half_edge_between(v_start, v_end) {
                    let dot = geodesic_dir.dot(&edge_vec);
                    if dot > tol {
                        return Some(VertexRayResult::CollinearWithEdge(he));
                    }
                }
                if let Some(he) = self.half_edge_between(v_end, v_start) {
                    let reverse_edge_vec = -edge_vec;
                    let dot = geodesic_dir.dot(&reverse_edge_vec);
                    if dot > tol {
                        return Some(VertexRayResult::CollinearWithEdge(he));
                    }
                }
            }
        }

        // Ray-segment intersection using geodesic direction
        let edge_vec = (p2 - p1).as_vector();
        let rhs = (p1 - p0).as_vector();

        let det = geodesic_dir.cross(&edge_vec).dot(&face_normal);
        if det.abs() <= tol {
            return None; // ray parallel to opposite edge
        }

        let inv_det = T::one() / det;
        let t = &rhs.cross(&edge_vec).dot(&face_normal) * &inv_det;
        let u = &geodesic_dir.cross(&rhs).dot(&face_normal) * &inv_det;

        // Validate solution
        if t <= tol || u < T::zero() || u > T::one() {
            println!("t is: {:?}", t);
            println!("Original direction: {:?}", direction);
            println!("Direction magnitude: {:?}", direction.dot(direction));
            println!("hit is behind or at start point");
            return None;
        }

        // Find the half-edge corresponding to the opposite edge
        let he_fwd = self.half_edge_between(opp_v1, opp_v2);
        let he_bwd = self.half_edge_between(opp_v2, opp_v1);

        let he = match (he_fwd, he_bwd) {
            (Some(h0), Some(h1)) => {
                if self.half_edges[h0].face == Some(face) { h0 }
                else if self.half_edges[h1].face == Some(face) { h1 }
                else { return None; }
            }
            (Some(h0), None) => {
                if self.half_edges[h0].face == Some(face) { h0 } else { return None; }
            }
            (None, Some(h1)) => {
                if self.half_edges[h1].face == Some(face) { h1 } else { return None; }
            }
            _ => return None,
        };

        // Adjust u parameter to match half-edge orientation
        let he_src = self.half_edges[self.half_edges[he].twin].vertex;
        let he_dst = self.half_edges[he].vertex;

        let u_param = if he_src == opp_v1 && he_dst == opp_v2 {
            u
        } else if he_src == opp_v2 && he_dst == opp_v1 {
            T::one() - u
        } else {
            return None;
        };

        Some(VertexRayResult::EdgeIntersection {
            half_edge: he,
            distance: t,
            edge_parameter: u_param,
        })
    }


    /// Cast a ray from a vertex within a triangle face along the given direction.
    /// Returns either an intersection with the opposite edge or indicates collinearity with an edge.
    pub fn cast_ray_from_vertex_in_triangle(
        &self,
        face: usize,
        vertex: usize,
        direction: &Vector<T, N>,
    ) -> Option<VertexRayResult<T>>
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        let vs = self.face_vertices(face);

        // Find vertex position in face and identify opposite edge
        let vertex_idx = vs.iter().position(|&v| v == vertex)?;
        let (opp_v1, opp_v2) = match vertex_idx {
            0 => (vs[1], vs[2]), // vertex is A, opposite edge is BC
            1 => (vs[2], vs[0]), // vertex is B, opposite edge is CA
            2 => (vs[0], vs[1]), // vertex is C, opposite edge is AB
            _ => return None,
        };

        let p0 = &self.vertices[vertex].position;
        let p1 = &self.vertices[opp_v1].position;
        let p2 = &self.vertices[opp_v2].position;

        // Calculate face normal
        let vs_tri = [&self.vertices[vs[0]].position, &self.vertices[vs[1]].position, &self.vertices[vs[2]].position];
        let face_normal = {
            let e1 = (vs_tri[1] - vs_tri[0]).as_vector();
            let e2 = (vs_tri[2] - vs_tri[0]).as_vector();
            e1.cross(&e2)
        };

        let n2 = face_normal.dot(&face_normal);
        if n2.is_zero() {
            println!("degenerate triangle");
            return None; // degenerate triangle
        }

        // Project direction onto the face plane
        let dot_dn = direction.dot(&face_normal);
        let projected_dir = direction - &face_normal.scale(&(&dot_dn / &n2));

        // Check if projected direction is essentially zero
        let tol = T::tolerance();
        let tol2 = &tol * &tol;
        if projected_dir.dot(&projected_dir) <= tol2 {
            println!("direction is perpendicular to face plane");
            return None; // direction is perpendicular to face plane
        }

        // Check collinearity with edges from the vertex
        let adjacent_edges = match vertex_idx {
            0 => [(vs[0], vs[1]), (vs[0], vs[2])], // edges from vertex A
            1 => [(vs[1], vs[2]), (vs[1], vs[0])], // edges from vertex B
            2 => [(vs[2], vs[0]), (vs[2], vs[1])], // edges from vertex C
            _ => return None,
        };

        for &(v_start, v_end) in &adjacent_edges {
            let edge_vec = (&self.vertices[v_end].position - &self.vertices[v_start].position).as_vector();
            let cross = projected_dir.cross(&edge_vec);

            // Check if vectors are collinear (cross product near zero)
            if cross.dot(&cross) <= tol2 {
                // Find the half-edge for this edge direction
                if let Some(he) = self.half_edge_between(v_start, v_end) {
                    // Ensure the direction is forward along the edge (not backward)
                    let dot = projected_dir.dot(&edge_vec);
                    if dot > tol {
                        return Some(VertexRayResult::CollinearWithEdge(he));
                    }
                }
                // Check reverse direction
                if let Some(he) = self.half_edge_between(v_end, v_start) {
                    let reverse_edge_vec = -edge_vec;
                    let dot = projected_dir.dot(&reverse_edge_vec);
                    if dot > tol {
                        return Some(VertexRayResult::CollinearWithEdge(he));
                    }
                }
            }
        }

        // Ray-segment intersection using projected direction
        let edge_vec = (p2 - p1).as_vector();
        let rhs = (p1 - p0).as_vector();

        // Solve 2x2 system using cross products projected onto face normal
        let det = projected_dir.cross(&edge_vec).dot(&face_normal);

        if det.abs() <= tol {
            println!("Starting pos: {:?}", &self.vertices[vertex].position);
            println!("Original direction: {:?}", direction);
            println!("Direction magnitude: {:?}", direction.dot(direction));
            println!("half-edge endpoints: {:?}", (&self.vertices[opp_v1].position, &self.vertices[opp_v2].position));
            println!("-> ray parallel to opposite edge");
            return None; // ray parallel to opposite edge
        }

        let inv_det = T::one() / det;
        let t = &rhs.cross(&edge_vec).dot(&face_normal) * &inv_det;
        let u = &rhs.cross(&projected_dir).dot(&face_normal) * &inv_det;

        // Validate solution
        if t <= tol {
            println!("Starting pos: {:?}", &self.vertices[vertex].position);
            println!("t is: {:?}", t);
            println!("Original direction: {:?}", direction);
            println!("Direction magnitude: {:?}", direction.dot(direction));
            println!("half-edge endpoints: {:?}", (&self.vertices[opp_v1].position, &self.vertices[opp_v2].position));
            println!("-> hit is behind or at start point");
            return None; // hit is behind or at start point
        }

        if u < T::zero() || u > T::one() {
            println!("Starting pos: {:?}", &self.vertices[vertex].position);
            println!("u is: {:?}", u);
            println!("Original direction: {:?}", direction);
            println!("Direction magnitude: {:?}", direction.dot(direction));
            println!("half-edge endpoints: {:?}", (&self.vertices[opp_v1].position, &self.vertices[opp_v2].position));
            println!("-> hit is outside segment");
            return None; // hit is outside segment
        }

        // Find the half-edge corresponding to the opposite edge
        let he_fwd = self.half_edge_between(opp_v1, opp_v2);
        let he_bwd = self.half_edge_between(opp_v2, opp_v1);

        let he = match (he_fwd, he_bwd) {
            (Some(h0), Some(h1)) => {
                // Prefer the half-edge whose face is the query face
                if self.half_edges[h0].face == Some(face) {
                    h0
                } else if self.half_edges[h1].face == Some(face) {
                    h1
                } else {
                    println!("no matching half-edge found for face 1 - {}", face);
                    return None;
                }
            }
            (Some(h0), None) => {
                if self.half_edges[h0].face == Some(face) {
                    h0
                } else {
                    println!("no matching half-edge found for face 2 - {}", face);
                    return None;
                }
            }
            (None, Some(h1)) => {
                if self.half_edges[h1].face == Some(face) {
                    h1
                } else {
                    println!("no matching half-edge found for face 3 - {}", face);
                    return None;
                }
            }
            _ => { println!("no matching half-edge found for face 4 - {}", face); return None },
        };

        // Adjust u parameter to match half-edge orientation
        let he_src = self.half_edges[self.half_edges[he].twin].vertex;
        let he_dst = self.half_edges[he].vertex;

        let u_param = if he_src == opp_v1 && he_dst == opp_v2 {
            u // half-edge goes v1->v2, u is correct
        } else if he_src == opp_v2 && he_dst == opp_v1 {
            T::one() - u // half-edge goes v2->v1, flip u
        } else {
            println!("topology mismatch");
            return None; // topology mismatch
        };

        Some(VertexRayResult::EdgeIntersection {
            half_edge: he,
            distance: t,
            edge_parameter: u_param,
        })
    }

    pub fn get_first_half_edge_intersection_on_face(
        &self,
        face: usize,
        from: &Point<T, N>,
        direction: &Vector<T, N>,
    ) -> Option<(usize, T, T)>
    where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        // 0) Quick sanity: face must be valid and triangular
        if self.faces[face].removed {
            return None;
        }

        let hes = self.face_half_edges(face);
        if hes.len() != 3 {
            // This routine is triangle-only.
            return None;
        }
        if self.half_edges[hes[0]].removed
            || self.half_edges[hes[1]].removed
            || self.half_edges[hes[2]].removed
        {
            return None;
        }

        // 1) Plane data
        let plane = self.plane_from_face(face);
        let origin = plane.origin();            // a point on the plane
        let origin = Point::<T, N>::from_vals(from_fn(|i| origin[i].clone()));
        let (u3, v3) = plane.basis();           // two independent in-plane vectors
        let n3 = u3.cross(&v3);                 // plane normal (not necessarily unit)

        // Guard against degenerate plane (area ~ 0)
        let tol = T::tolerance();
        let tol2 = &tol * &tol;
        let n2 = n3.dot(&n3);
        if n2 <= tol2 {
            // Degenerate face: skip
            return None;
        }

        // 2) Intersect 3D ray with plane: from + t_plane * direction
        let w = (from - &origin).as_vector();
        let num  = -w.dot(&n3);
        let den  = direction.dot(&n3);

        // Helper: near-zero test
        let near_zero = |x: &T| -> bool {
            let mtol = -tol.clone();
            x >= &mtol && x <= &tol
        };

        let mut start_on_plane = Point::<T, N>::from_vals(from_fn(|i| from[i].clone()));
        let mut dir_in_plane3  = direction.clone();

        if near_zero(&den) {
            // Ray parallel to plane
            if !near_zero(&num) {
                // Off-plane and parallel => never meets the plane
                return None;
            }
            // In-plane ray: remove any residual normal component (robust)
            // dir_in_plane3 = direction - proj_n(direction)
            let k = &direction.dot(&n3) / &n2; // this is ~0 but keeps consistency
            dir_in_plane3 = &dir_in_plane3 - &n3.scale(&k);
            start_on_plane = from.clone();
        } else {
            // Proper intersection: advance origin to the hit point on the plane
            let t_plane = &num / &den;

            // FIX 1: accept starts exactly on the plane (t≈0); only reject if strictly behind more than tol
            if &t_plane < &(-tol.clone()) {
                return None;
            }
            // Clamp tiny negative t to zero to stay numerically stable
            let t_clamped = if t_plane > T::zero() { t_plane } else { T::zero() };

            // start_on_plane = from + t_clamped * direction
            start_on_plane = (&from.as_vector() + &direction.scale(&t_clamped)).0;

            // Only the in-plane component should drive the boundary hit
            let k = &direction.dot(&n3) / &n2;
            dir_in_plane3 = &dir_in_plane3 - &n3.scale(&k);
        }

        // 3) Project to a 2D basis on the plane
        let project2 = |p: &Point<T, N>| -> Point<T, 2> {
            let d = (p - &origin).as_vector();
            Point::<T, 2>::new([d.dot(&u3), d.dot(&v3)])
        };
        let projectv2 = |v: &Vector<T, N>| -> Vector<T, 2> {
            Vector::<T, 2>::new([v.dot(&u3), v.dot(&v3)])
        };

        let from2 = project2(&start_on_plane);
        let dir2  = projectv2(&dir_in_plane3);

        // If projected direction is (near) zero, there is no forward in-plane march
        if dir2.dot(&dir2) <= tol2 {
            return None;
        }

        // 4) Build 2D endpoints for each half-edge using the half-edge's TAIL -> HEAD (prev.vertex -> he.vertex)
        let mut best: Option<(usize, T, T)> = None; // (he_idx, t_in_plane, u_on_segment)

        for &he_idx in &hes {
            let he = &self.half_edges[he_idx];

            // FIX 2: use geometric edge of he = (tail -> head) = (prev.vertex -> he.vertex)
            let src = self.half_edges[he.prev].vertex; // tail of he
            let dst = he.vertex;                       // head of he

            let a3 = &self.vertices[src].position;
            let b3 = &self.vertices[dst].position;

            let a2 = project2(a3);
            let b2 = project2(b3);

            if let Some((t_hit, u_seg)) =
                ray_segment_intersection_2d_robust(&from2, &dir2, &a2, &b2, &tol)
            {
                if t_hit >= tol {
                    match &mut best {
                        None => best = Some((he_idx, t_hit, u_seg)),
                        Some((best_he, best_t, best_u)) => {
                            if &t_hit < best_t {
                                *best_he = he_idx;
                                *best_t = t_hit;
                                *best_u = u_seg;
                            } else if near_zero(&(&t_hit - best_t)) && he_idx < *best_he {
                                *best_he = he_idx;
                                *best_t = t_hit;
                                *best_u = u_seg;
                            }
                        }
                    }
                }
            }
        }

        best
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
            if let Some(u) = kernel::point_u_on_segment(ps, pd, point) {
                return Some((he, u));
            }
        }
        None
    }

    #[inline]
    pub fn face_edges(&self, f: usize) -> (usize, usize, usize) {
        let e0 = self.faces[f].half_edge;
        let e1 = self.half_edges[e0].next;
        let e2 = self.half_edges[e1].next;
        debug_assert_eq!(self.half_edges[e2].next, e0);
        (e0, e1, e2) // e0: a->b, e1: b->c, e2: c->a
    }

    // Map bary zero to (edge, u) on face f:
    // l2==0 => edge a-b -> e0, u=l1;  l0==0 => edge b-c -> e1, u=l2;  l1==0 => edge c-a -> e2, u=l0.
    pub fn edge_and_u_from_bary_zero(&self, f: usize, l0: &T, l1: &T, l2: &T) -> Option<(usize, T)> {
        let (e0,e1,e2) = self.face_edges(f);
        if l2.is_zero() { return Some((e0, l1.clone())); }
        if l0.is_zero() { return Some((e1, l2.clone())); }
        if l1.is_zero() { return Some((e2, l0.clone())); }
        None
    }

    // Two bary zeros => vertex id on face f.
    pub fn vertex_from_bary_zero(&self, f: usize, l0: &T, l1: &T, l2: &T) -> Option<usize> {
        let [a,b,c] = self.face_vertices(f);
        if l1.is_zero() && l2.is_zero() { return Some(a); }
        if l2.is_zero() && l0.is_zero() { return Some(b); }
        if l0.is_zero() && l1.is_zero() { return Some(c); }
        None
    }

    pub fn location_on_face(&self, f: usize, p: &Point<T, N>)  -> FindFaceResult<T> {
        let zero = T::zero();
        let one  = T::one();

        let [i0, i1, i2] = self.face_vertices(f);
        let (l0, l1, l2) = barycentric_coords(
                p,
                &self.vertices[i0].position,
                &self.vertices[i1].position,
                &self.vertices[i2].position,
            ).unwrap();

        // OnEdge (exactly one zero)
        let zc = l0.is_zero() as u8 + l1.is_zero() as u8 + l2.is_zero() as u8;
        if zc == 1 {
            if let Some((he_guess, mut u_bary)) = self.edge_and_u_from_bary_zero(f, &l0, &l1, &l2) {
                if u_bary.is_negative() { u_bary = zero.clone(); }
                if (&u_bary - &one).is_positive() { u_bary = one.clone(); }
                return FindFaceResult::OnEdge { f, he: he_guess, u: u_bary };
            }
        }

        // OnVertex (exactly two zeros)
        if zc >= 2 {
            if let Some(v_id) = self.vertex_from_bary_zero(f, &l0, &l1, &l2) {
                return FindFaceResult::OnVertex { f, v: v_id };
            }
            // Fallback if ambiguous: treat as inside
            return FindFaceResult::Inside { f, bary: (l0, l1, l2) };
        }

        // Inside (no zeros, all non-negative)
        return FindFaceResult::Inside { f, bary: (l0, l1, l2) };
    }

    /// Returns (face_id, half_edge_id, vertex_id, u).
    /// Order of detection priority: OnEdge -> OnVertex -> Inside.
    /// - Inside:    (f, usize::MAX, usize::MAX, 0)
    /// - OnEdge:    (f, he_of_f_edge, usize::MAX, u in [0,1] along that half-edge)
    /// - OnVertex:  (f, usize::MAX, vertex_id, 0)
    pub fn find_valid_face(&self, start_face: usize, point: &Point<T, N>) -> FindFaceResult<T>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        Vector<T, 3>: VectorOps<T, 3, Cross = Vector<T, 3>>,
        for<'a> &'a T:
            Sub<&'a T, Output = T> +
            Mul<&'a T, Output = T> +
            Add<&'a T, Output = T> +
            Div<&'a T, Output = T>,
    {
        use std::collections::VecDeque;

        let zero = T::zero();
        let one  = T::one();

        #[inline]
        fn live_face_degenerate<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, f: usize) -> bool
        where
            Point<TS, M>: PointOps<TS, M, Vector = Vector<TS, M>>,
            Vector<TS, M>: VectorOps<TS, M>,
            Vector<TS, 3>: VectorOps<TS, 3, Cross = Vector<TS, 3>>,
            for<'a> &'a TS: Add<&'a TS, Output = TS>
                        + Sub<&'a TS, Output = TS>
                        + Mul<&'a TS, Output = TS>,
        {
            // vertices (A,B,C) of face f
            let [i0, i1, i2] = {
                let e0 = mesh.faces[f].half_edge;
                let e1 = mesh.half_edges[e0].next;
                let a = mesh.half_edges[mesh.half_edges[e0].prev].vertex; // tail(e0)
                let b = mesh.half_edges[e0].vertex;                       // head(e0)
                let c = mesh.half_edges[e1].vertex;                       // head(e1)
                [a, b, c]
            };

            // zero-length edge => degenerate
            let ab = (&mesh.vertices[i1].position - &mesh.vertices[i0].position).as_vector();
            let ac = (&mesh.vertices[i2].position - &mesh.vertices[i0].position).as_vector();
            let bc = (&mesh.vertices[i2].position - &mesh.vertices[i1].position).as_vector();
            if ab.dot(&ab).is_zero() || ac.dot(&ac).is_zero() || bc.dot(&bc).is_zero() {
                return true;
            }

            // collinear A,B,C => degenerate (3D-safe; for 2D z=0 it still works)
            let ab3 = ab.0.as_vector_3();
            let ac3 = ac.0.as_vector_3();
            let n = ab3.cross(&ac3);               // compute once
            // exact test with LazyExact: only true if *exactly* collinear
            n[0].is_zero() && n[1].is_zero() && n[2].is_zero()
        }

        // Exact barycentrics (l0,l1,l2) with respect to face f, using current vertex positions
        let bary_live = |f: usize| -> (T, T, T) {
            let [i0,i1,i2] = self.face_vertices(f);
            barycentric_coords(
                point,
                &self.vertices[i0].position,
                &self.vertices[i1].position,
                &self.vertices[i2].position,
            ).unwrap()
        };

        // neighbors of a live face via twins (skip borders / removed faces)
        let push_live_neighbors = |mesh: &Mesh<T, N>, f: usize, q: &mut VecDeque<usize>, seen: &AHashSet<usize>| {
            let (e0,e1,e2) = self.face_edges(f);
            for e in [e0,e1,e2] {
                let tw = mesh.half_edges[e].twin;
                if let Some(fnbr) = mesh.half_edges[tw].face {
                    if !mesh.faces[fnbr].removed && !seen.contains(&fnbr) {
                        q.push_back(fnbr);
                    }
                }
            }
        };

        // --- BFS over descendants + neighbors ---
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut seen: AHashSet<usize> = AHashSet::new();
        q.push_back(start_face);

        let mut steps = 0usize;
        while let Some(f) = q.pop_front() {
            if !seen.insert(f) { continue; }
            steps += 1;
            debug_assert!(steps < 1_000_000, "find_valid_face: excessive traversal");

            if !self.faces[f].removed {
                // Barycentric-driven membership and priority: Edge -> Vertex -> Inside
                let (l0, l1, l2) = bary_live(f);
                let neg = |x: &T| -> bool { x.is_negative() };

                if neg(&l0) || neg(&l1) || neg(&l2) {
                    // Off this face; explore neighbors
                    push_live_neighbors(self, f, &mut q, &seen);
                    continue;
                }

                return self.location_on_face(f, point);
            }

            // Removed face: descend via recorded triangles (use current vertex positions for test)
            if let Some(mapping) = self.face_split_map.get(&f) {
                let mut interior: Vec<usize> = Vec::new();
                let mut boundary:  Vec<usize> = Vec::new();

                for tri in &mapping.new_faces {
                    let [i0,i1,i2] = tri.vertices;
                    let (l0,l1,l2) = barycentric_coords(
                        point,
                        &self.vertices[i0].position,
                        &self.vertices[i1].position,
                        &self.vertices[i2].position,
                    ).unwrap();

                    let neg = |x: &T| -> bool { x.is_negative() };
                    if neg(&l0) || neg(&l1) || neg(&l2) {
                        continue; // Off
                    }

                    let zc = l0.is_zero() as u8 + l1.is_zero() as u8 + l2.is_zero() as u8;
                    if zc >= 1 {
                        boundary.push(tri.face_idx); // OnEdge or OnVertex -> boundary
                    } else {
                        interior.push(tri.face_idx); // Inside
                    }
                }

                for fc in interior { if !seen.contains(&fc) { q.push_front(fc); } }
                for fc in boundary  { if !seen.contains(&fc) { q.push_back(fc); } }
                for tri in &mapping.new_faces {
                    let fc = tri.face_idx;
                    if !seen.contains(&fc) { q.push_back(fc); }
                }
                continue;
            }
        }

        // --- Global fallback: scan all live faces and pick the first match by priority ---
        let mut best_edge:   Option<(usize, usize, T)> = None; // (f, he, u)
        let mut best_vertex: Option<(usize, usize)>    = None; // (f, v_id)
        let mut best_inside: Option<(usize, (T, T, T))> = None; // (f, bary)

        for f in 0..self.faces.len() {
            if self.faces[f].removed { continue; }
            if live_face_degenerate(self, f) { continue; }

            let (l0,l1,l2) = bary_live(f);
            let neg = |x: &T| -> bool { x.is_negative() };
            if neg(&l0) || neg(&l1) || neg(&l2) { continue; }

            let zc = l0.is_zero() as u8 + l1.is_zero() as u8 + l2.is_zero() as u8;

            if zc == 1 && best_edge.is_none() {
                if let Some((he_guess, mut u_bary2)) = self.edge_and_u_from_bary_zero(f, &l0, &l1, &l2) {
                    if u_bary2.is_negative() { u_bary2 = zero.clone(); }
                    if (&u_bary2 - &one).is_positive() { u_bary2 = one.clone(); }
                    best_edge = Some((f, he_guess, u_bary2));
                    break; // edge has highest priority
                }
            } else if zc >= 2 && best_vertex.is_none() {
                if let Some(v_id) = self.vertex_from_bary_zero(f, &l0, &l1, &l2) {
                    best_vertex = Some((f, v_id));
                }
            } else if zc == 0 && best_inside.is_none() {
                best_inside = Some((f, (l0, l1, l2)));
            }
        }

        if let Some((f, he, u)) = best_edge   { return FindFaceResult::OnEdge { f, he, u }; }
        if let Some((f, v))     = best_vertex { return FindFaceResult::OnVertex { f, v }; }
        if let Some((f, bary))  = best_inside { return FindFaceResult::Inside { f, bary }; }

        panic!("find_valid_face: could not locate a face/edge/vertex for the point starting from {}", start_face);
    }

    /// Returns (face_id, half_edge_id, vertex_id, u).
    /// - Inside:    (f, usize::MAX, usize::MAX, 0)
    /// - OnEdge:    (f, he_of_f_edge, usize::MAX, u in [0,1] along that half-edge)
    /// - OnVertex:  (f, usize::MAX, vertex_id, 0)
    pub fn find_valid_face_working_but_weird(&self, start_face: usize, point: &Point<T, N>) -> (usize, usize, usize, T)
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        Vector<T, 3>: VectorOps<T, 3, Cross = Vector<T, 3>>,
        for<'a> &'a T:
            Sub<&'a T, Output = T> +
            Mul<&'a T, Output = T> +
            Add<&'a T, Output = T> +
            Div<&'a T, Output = T>,
    {
        use std::collections::VecDeque;

        let zero = T::zero();
        let one  = T::one();

        #[inline]
        fn get_face_cycle<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, f: usize) -> (usize, usize, usize) {
            let e0 = mesh.faces[f].half_edge;
            let e1 = mesh.half_edges[e0].next;
            let e2 = mesh.half_edges[e1].next;
            debug_assert_eq!(mesh.half_edges[e2].next, e0);
            (e0, e1, e2) // e0: a->b, e1: b->c, e2: c->a
        }

        #[inline]
        fn face_vertices<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, f: usize) -> [usize; 3] {
            let (e0, e1, _e2) = get_face_cycle(mesh, f);
            let a = mesh.half_edges[mesh.half_edges[e0].prev].vertex; // origin(e0)
            let b = mesh.half_edges[e0].vertex;                       // target(e0)
            let c = mesh.half_edges[e1].vertex;                       // target(e1)
            [a, b, c]
        }

        #[inline]
        fn live_face_degenerate<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, f: usize) -> bool
        where
            Point<TS, M>: PointOps<TS, M, Vector = Vector<TS, M>>,
            Vector<TS, M>: VectorOps<TS, M>,
            Vector<TS, 3>: VectorOps<TS, 3, Cross = Vector<TS, 3>>,
            for<'a> &'a TS: Add<&'a TS, Output = TS> + Sub<&'a TS, Output = TS> + Mul<&'a TS, Output = TS>,
        {
            let [i0,i1,i2] = face_vertices(mesh, f);
            let u = (&mesh.vertices[i1].position - &mesh.vertices[i0].position).as_vector_3();
            let v = (&mesh.vertices[i2].position - &mesh.vertices[i0].position).as_vector_3();
            let n = u.cross(&v);
            let n2 = n.dot(&n);
            n2.is_zero()
        }

        let classify_live = |f: usize| -> TrianglePoint {
            let [i0,i1,i2] = face_vertices(self, f);
            kernel::point_in_or_on_triangle(
                point,
                &self.vertices[i0].position,
                &self.vertices[i1].position,
                &self.vertices[i2].position,
            )
        };

        // IMPORTANT: use exact barycentrics with LazyExact
        let bary_live = |f: usize| -> (T, T, T) {
            let [i0,i1,i2] = face_vertices(self, f);
            barycentric_coords(
                point,
                &self.vertices[i0].position,
                &self.vertices[i1].position,
                &self.vertices[i2].position,
            ).unwrap()
        };

        // Map bary zeros to (half-edge, u) on face f:
        // l2==0 => edge a-b -> e0, u=l1;  l0==0 => edge b-c -> e1, u=l2;  l1==0 => edge c-a -> e2, u=l0.
        let edge_and_u_from_bary_zero = |f: usize, l0: &T, l1: &T, l2: &T| -> Option<(usize, T)> {
            let (e0,e1,e2) = get_face_cycle(self, f);
            if l2.is_zero() { return Some((e0, l1.clone())); }
            if l0.is_zero() { return Some((e1, l2.clone())); }
            if l1.is_zero() { return Some((e2, l0.clone())); }
            None
        };

        // Map bary zeros to vertex id on face f:
        // (l1==0 && l2==0) -> vertex a; (l2==0 && l0==0) -> b; (l0==0 && l1==0) -> c.
        let vertex_from_bary_zero = |f: usize, l0: &T, l1: &T, l2: &T| -> Option<usize> {
            let [a,b,c] = face_vertices(self, f);
            if l1.is_zero() && l2.is_zero() { return Some(a); }
            if l2.is_zero() && l0.is_zero() { return Some(b); }
            if l0.is_zero() && l1.is_zero() { return Some(c); }
            None
        };

        // neighbors of a live face via twins (skip borders / removed faces)
        let push_live_neighbors = |mesh: &Mesh<T, N>, f: usize, q: &mut VecDeque<usize>, seen: &AHashSet<usize>| {
            let (e0,e1,e2) = get_face_cycle(mesh, f);
            for e in [e0,e1,e2] {
                let tw = mesh.half_edges[e].twin;
                if let Some(fnbr) = mesh.half_edges[tw].face {
                    if !mesh.faces[fnbr].removed && !seen.contains(&fnbr) {
                        q.push_back(fnbr);
                    }
                }
            }
        };

        // --- BFS over descendants + neighbors ---
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut seen: AHashSet<usize> = AHashSet::new();
        q.push_back(start_face);

        let mut steps = 0usize;
        while let Some(f) = q.pop_front() {
            if !seen.insert(f) { continue; }
            steps += 1;
            debug_assert!(steps < 1_000_000, "find_valid_face: excessive traversal");

            if !self.faces[f].removed {
                if live_face_degenerate(self, f) {
                    push_live_neighbors(self, f, &mut q, &seen);
                    continue;
                }

                match classify_live(f) {
                    TrianglePoint::In => {
                        return (f, usize::MAX, usize::MAX, zero);
                    }
                    TrianglePoint::OnEdge => {
                        let (l0,l1,l2) = bary_live(f);
                        if let Some((he, mut u)) = edge_and_u_from_bary_zero(f, &l0, &l1, &l2) {
                            if u < zero { u = zero.clone(); }
                            if u > one  { u = one.clone(); }
                            return (f, he, usize::MAX, u);
                        }
                        // If mapping failed (shouldn't in exact mode), treat as inside
                        return (f, usize::MAX, usize::MAX, zero);
                    }
                    TrianglePoint::OnVertex => {
                        let (l0,l1,l2) = bary_live(f);
                        if let Some(v_id) = vertex_from_bary_zero(f, &l0, &l1, &l2) {
                            return (f, usize::MAX, v_id, zero);
                        }
                        // Fallback: inside
                        return (f, usize::MAX, usize::MAX, zero);
                    }
                    TrianglePoint::Off => {
                        // Explore neighbors; the correct owner might be adjacent
                        push_live_neighbors(self, f, &mut q, &seen);
                        continue;
                    }
                }
            }

            // Removed face: descend via recorded triangles (use current vertex positions for test)
            if let Some(mapping) = self.face_split_map.get(&f) {
                let mut interior: Vec<usize> = Vec::new();
                let mut boundary:  Vec<usize> = Vec::new();

                for tri in &mapping.new_faces {
                    let [i0,i1,i2] = tri.vertices;
                    let cls = kernel::point_in_or_on_triangle(
                        point,
                        &self.vertices[i0].position,
                        &self.vertices[i1].position,
                        &self.vertices[i2].position,
                    );
                    match cls {
                        TrianglePoint::In => interior.push(tri.face_idx),
                        TrianglePoint::OnEdge | TrianglePoint::OnVertex => boundary.push(tri.face_idx),
                        TrianglePoint::Off => {}
                    }
                }

                for fc in interior { if !seen.contains(&fc) { q.push_front(fc); } }
                for fc in boundary  { if !seen.contains(&fc) { q.push_back(fc); } }
                for tri in &mapping.new_faces {
                    let fc = tri.face_idx;
                    if !seen.contains(&fc) { q.push_back(fc); }
                }
                continue;
            }
        }

        // --- Global fallback: scan all live faces and pick best match ---
        let mut best_inside: Option<usize> = None;
        let mut best_edge:   Option<(usize, usize, T)> = None;
        let mut best_vertex: Option<usize> = None;

        for f in 0..self.faces.len() {
            if self.faces[f].removed { continue; }
            if live_face_degenerate(self, f) { continue; }

            match classify_live(f) {
                TrianglePoint::In => { best_inside = Some(f); break; }
                TrianglePoint::OnEdge => {
                    let (l0,l1,l2) = bary_live(f);
                    if let Some((he, mut u)) = edge_and_u_from_bary_zero(f, &l0, &l1, &l2) {
                        if u < zero { u = zero.clone(); }
                        if u > one  { u = one.clone(); }
                        if best_edge.is_none() { best_edge = Some((f, he, u)); }
                    }
                }
                TrianglePoint::OnVertex => {
                    if best_vertex.is_none() { best_vertex = Some(f); }
                }
                TrianglePoint::Off => {}
            }
        }

        if let Some(f) = best_inside { return (f, usize::MAX, usize::MAX, zero); }
        if let Some((f,he,u)) = best_edge { return (f, he, usize::MAX, u); }
        if let Some(f) = best_vertex {
            // Identify which vertex
            let (l0,l1,l2) = bary_live(f);
            if let Some(v_id) = vertex_from_bary_zero(f, &l0, &l1, &l2) {
                return (f, usize::MAX, v_id, zero);
            }
            return (f, usize::MAX, usize::MAX, zero);
        }

        panic!("find_valid_face: could not locate a face/edge/vertex for the point starting from {}", start_face);
    }

    /// Returns (face_id, half_edge_id, u). For Inside/OnVertex: (f, usize::MAX, 0).
    /// For OnEdge: (f, he_of_f_edge, u in [0,1] along that half-edge).
    pub fn find_valid_face_almost_working(&self, start_face: usize, point: &Point<T, N>) -> (usize, usize, T)
    {
        use std::collections::VecDeque;

        let zero = T::zero();
        let one  = T::one();

        #[inline]
        fn get_face_cycle<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, f: usize) -> (usize, usize, usize) {
            let e0 = mesh.faces[f].half_edge;
            let e1 = mesh.half_edges[e0].next;
            let e2 = mesh.half_edges[e1].next;
            debug_assert_eq!(mesh.half_edges[e2].next, e0);
            (e0, e1, e2) // e0: a->b, e1: b->c, e2: c->a
        }

        #[inline]
        fn face_vertices<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, f: usize) -> [usize; 3] {
            let (e0, e1, _e2) = get_face_cycle(mesh, f);
            let a = mesh.half_edges[mesh.half_edges[e0].prev].vertex; // origin(e0)
            let b = mesh.half_edges[e0].vertex;                       // target(e0)
            let c = mesh.half_edges[e1].vertex;                       // target(e1)
            [a, b, c]
        }

        #[inline]
        fn live_face_degenerate<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, f: usize) -> bool
        where
            Point<TS, M>: PointOps<TS, M, Vector = Vector<TS, M>>,
            Vector<TS, M>: VectorOps<TS, M>,
            Vector<TS, 3>: VectorOps<TS, 3, Cross = Vector<TS, 3>>,
            for<'a> &'a TS:
                Add<&'a TS, Output = TS> + Sub<&'a TS, Output = TS> + Mul<&'a TS, Output = TS>,
        {
            let [i0,i1,i2] = face_vertices(mesh, f);
            let u = (&mesh.vertices[i1].position - &mesh.vertices[i0].position).as_vector_3();
            let v = (&mesh.vertices[i2].position - &mesh.vertices[i0].position).as_vector_3();
            let cross_product = u.cross(&v);
            let n2 = cross_product.dot(&cross_product);
            n2.is_zero()
        }

        let classify_live = |f: usize| -> TrianglePoint {
            let [i0,i1,i2] = face_vertices(self, f);
            kernel::point_in_or_on_triangle(
                point,
                &self.vertices[i0].position,
                &self.vertices[i1].position,
                &self.vertices[i2].position,
            )
        };

        let bary_live = |f: usize| -> (T, T, T) {
            // let [i0,i1,i2] = face_vertices(self, f);
            self.barycentric_coords_on_face(
                f,
                point
            ).unwrap()
        };

        // l2==0 => on edge a-b -> e0, u=l1; l0==0 => on edge b-c -> e1, u=l2; l1==0 => on edge c-a -> e2, u=l0.
        let edge_and_u_from_bary_zero = |f: usize, l0: &T, l1: &T, l2: &T| -> Option<(usize, T)> {
            let (e0,e1,e2) = get_face_cycle(self, f);
            if l2.is_zero() { return Some((e0, l1.clone())); }
            if l0.is_zero() { return Some((e1, l2.clone())); }
            if l1.is_zero() { return Some((e2, l0.clone())); }
            None
        };

        // neighbors of a live face via twins (skip borders / removed faces)
        let push_live_neighbors = |mesh: &Mesh<T, N>, f: usize, q: &mut VecDeque<usize>, seen: &AHashSet<usize>| {
            let (e0,e1,e2) = get_face_cycle(mesh, f);
            for e in [e0,e1,e2] {
                let tw = mesh.half_edges[e].twin;
                if let Some(fnbr) = mesh.half_edges[tw].face {
                    if !mesh.faces[fnbr].removed && !seen.contains(&fnbr) {
                        q.push_back(fnbr);
                    }
                }
            }
        };

        // --- BFS over descendants + neighbors ---
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut seen: AHashSet<usize> = AHashSet::new();
        q.push_back(start_face);

        let mut steps = 0usize;
        while let Some(f) = q.pop_front() {
            if !seen.insert(f) { continue; }
            steps += 1;
            debug_assert!(steps < 1_000_000, "find_valid_face: excessive traversal");

            if !self.faces[f].removed {
                if live_face_degenerate(self, f) {
                    // degenerate triangle — don't classify, just flow to neighbors
                    push_live_neighbors(self, f, &mut q, &seen);
                    continue;
                }

                match classify_live(f) {
                    TrianglePoint::In | TrianglePoint::OnVertex => {
                        return (f, usize::MAX, zero);
                    }
                    TrianglePoint::OnEdge => {
                        let (l0,l1,l2) = bary_live(f);
                        if let Some((he, mut u)) = edge_and_u_from_bary_zero(f, &l0, &l1, &l2) {
                            if u < zero { u = zero.clone(); }
                            if u > one  { u = one.clone(); }
                            return (f, he, u);
                        }
                        // If edge mapping failed, treat as inside
                        return (f, usize::MAX, zero);
                    }
                    TrianglePoint::Off => {
                        // We’re in the wrong live face; explore all live neighbors.
                        push_live_neighbors(self, f, &mut q, &seen);
                        continue;
                    }
                }
            }

            // Removed face: expand to children at this split level
            if let Some(mapping) = self.face_split_map.get(&f) {
                // Classify against recorded triangles; enqueue interior first, then boundary;
                // but also enqueue all children (as low-priority) so we won't dead-end on ties.
                let mut interior: Vec<usize> = Vec::new();
                let mut boundary:  Vec<usize> = Vec::new();

                for tri in &mapping.new_faces {
                    let [i0,i1,i2] = tri.vertices;
                    let cls = kernel::point_in_or_on_triangle(
                        point,
                        &self.vertices[i0].position,
                        &self.vertices[i1].position,
                        &self.vertices[i2].position,
                    );
                    match cls {
                        TrianglePoint::In => interior.push(tri.face_idx),
                        TrianglePoint::OnEdge | TrianglePoint::OnVertex => boundary.push(tri.face_idx),
                        TrianglePoint::Off => {}
                    }
                }

                for fc in interior { if !seen.contains(&fc) { q.push_front(fc); } }
                for fc in boundary  { if !seen.contains(&fc) { q.push_back(fc); } }
                for tri in &mapping.new_faces {
                    let fc = tri.face_idx;
                    if !seen.contains(&fc) { q.push_back(fc); }
                }
                continue;
            }

            // Removed but no mapping — data hole: try neighbors of any surviving twins
            // (best-effort salvage)
            // If you want, you can also log this condition for debugging.
        }

        println!("find_valid_face: Global fallback");
        // --- Global fallback: scan all live faces and pick the best match ---
        let mut best_inside: Option<usize> = None;
        let mut best_edge:   Option<(usize, usize, T)> = None;
        let mut best_vertex: Option<usize> = None;

        for f in 0..self.faces.len() {
            if self.faces[f].removed { continue; }
            if live_face_degenerate(self, f) { continue; }

            match classify_live(f) {
                TrianglePoint::In => { best_inside = Some(f); break; }
                TrianglePoint::OnEdge => {
                    let (l0,l1,l2) = bary_live(f);
                    if let Some((he, mut u)) = edge_and_u_from_bary_zero(f, &l0, &l1, &l2) {
                        if u < zero { u = zero.clone(); }
                        if u > one  { u = one.clone(); }
                        if best_edge.is_none() { best_edge = Some((f, he, u)); }
                    }
                }
                TrianglePoint::OnVertex => {
                    if best_vertex.is_none() { best_vertex = Some(f); }
                }
                TrianglePoint::Off => {}
            }
        }

        if let Some(f) = best_inside { return (f, usize::MAX, zero); }
        if let Some(res) = best_edge   { return res; }
        if let Some(f) = best_vertex { return (f, usize::MAX, zero); }

        panic!("find_valid_face: could not locate a face/edge for the point starting from {}", start_face);
    }

    pub fn find_valid_half_edge(&self, he_start: usize, point: &Point<T, N>) -> usize
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T:
            Sub<&'a T, Output = T> +
            Mul<&'a T, Output = T> +
            Add<&'a T, Output = T> +
            Div<&'a T, Output = T>,
    {
        use std::collections::VecDeque;

        let zero = T::zero();
        let one  = T::one();

        #[inline]
        fn origin_of<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, he: usize) -> usize {
            mesh.half_edges[ mesh.half_edges[he].twin ].vertex
        }
        #[inline]
        fn target_of<TS: Scalar, const M: usize>(mesh: &Mesh<TS, M>, he: usize) -> usize {
            mesh.half_edges[he].vertex
        }

        // Return (u_raw, d2). d2 uses clamped u, u_raw is unclamped.
        // Uses sign-based predicates to avoid forcing exact on LazyExact.
        let proj_u_and_dist2 = |a: &Point<T, N>, p: &Point<T, N>, b: &Point<T, N>| -> (T, T) {
            let ab = (b - a).as_vector();
            let ap = (p - a).as_vector();
            let denom = ab.dot(&ab);
            if denom.is_zero() {
                // degenerate edge → distance to 'a'
                return (zero.clone(), ap.norm2());
            }
            let u_raw = ap.dot(&ab) / denom;
            // Clamp without PartialOrd
            let u_clamped =
                if u_raw.is_negative() { zero.clone() }
                else if (&u_raw - &one).is_positive() { one.clone() }
                else { u_raw.clone() };
            // closest point on segment = a + ab * u_clamped
            let closest = a + &ab.scale(&u_clamped).0;
            let d2 = (&(point - &closest).as_vector()).norm2();
            (u_raw, d2)
        };

        // Exact containment on a directed half-edge (including endpoints).
        // Bounds checks avoid PartialOrd to keep laziness.
        let contains_exact = |he: usize| -> Option<T> {
            let v0 = origin_of(self, he);
            let v1 = target_of(self, he);
            let a = &self.vertices[v0].position;
            let b = &self.vertices[v1].position;
            let (u_raw, d2) = proj_u_and_dist2(a, point, b);
            if d2.is_zero()
                && !u_raw.is_negative()
                && (&u_raw - &one).is_negative_or_zero()
            {
                Some(u_raw)
            } else {
                None
            }
        };

        // Keep the best (closest) live descendant as a fallback if no exact containment is found.
        let mut best_live: Option<(usize, T)> = None; // (he, d2)

        // BFS over descendants through half_edge_split_map
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut seen: AHashSet<usize> = AHashSet::new();
        q.push_back(he_start);

        let mut steps = 0usize;
        while let Some(he) = q.pop_front() {
            if !seen.insert(he) { continue; }
            steps += 1;
            debug_assert!(steps < 1_000_000, "find_valid_half_edge: excessive traversal");

            if !self.half_edges[he].removed {
                // Live candidate: exact containment?
                if let Some(_u) = contains_exact(he) {
                    return he;
                }
                // Track nearest descendant (use clamped distance)
                let v0 = origin_of(self, he);
                let v1 = target_of(self, he);
                let a = &self.vertices[v0].position;
                let b = &self.vertices[v1].position;
                let (_u_raw, d2) = proj_u_and_dist2(a, point, b);
                match &mut best_live {
                    None => best_live = Some((he, d2)),
                    Some((_bhe, bd2)) => {
                        if (&d2 - bd2).is_negative() {
                            *_bhe = he;
                            *bd2 = d2;
                        }
                    }
                }
                continue;
            }

            // Removed: expand to children if present
            if let Some(&(c0, c1)) = self.half_edge_split_map.get(&he) {
                if c0 != usize::MAX { q.push_back(c0); }
                if c1 != usize::MAX { q.push_back(c1); }
                continue;
            }
            // Removed but no mapping: nothing to enqueue; just continue.
        }

        // No exact-containing descendant found. Fall back to closest live descendant if we have one.
        if let Some((he, _d2)) = best_live {
            return he;
        }

        // Last resort: locate via the face that contains the point and use that edge.
        if let Some(f_anchor) = self.half_edges[he_start].face.or(self.half_edges[self.half_edges[he_start].twin].face) {
            match self.find_valid_face(f_anchor, point) {
                FindFaceResult::OnEdge { he, .. } => {
                    return he;
                },
                FindFaceResult::OnVertex { f, .. } => {
                    return self.half_edges[self.faces[f].half_edge].twin;
                },
                FindFaceResult::Inside { f, .. } => {
                    // Choose the boundary edge of that face that is closest.
                    let (e0, e1, e2) = {
                        let e0 = self.faces[f].half_edge;
                        let e1 = self.half_edges[e0].next;
                        let e2 = self.half_edges[e1].next;
                        (e0, e1, e2)
                    };
                    let mut best = (e0, T::from(1_000_000_000));
                    for e in [e0, e1, e2] {
                        let v0 = origin_of(self, e);
                        let v1 = target_of(self, e);
                        let a = &self.vertices[v0].position;
                        let b = &self.vertices[v1].position;
                        let (_u_raw, d2) = proj_u_and_dist2(a, point, b);
                        if (&d2 - &best.1).is_negative() {
                            best = (e, d2);
                        }
                    }
                    return best.0;
                },
            }
        }

        panic!("find_valid_half_edge: Half-edge {} is removed and unmapped (no descendants, no anchor)", he_start);
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

        if kernel::point_in_or_on_triangle(point, v0, v1, v2) != TrianglePoint::Off {
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

    pub fn vertex_touches_face(&self, v: usize, face_id: usize) -> bool {
        if v >= self.vertices.len() { return false; }
        // follow outgoing ring
        if let Some(mut h) = self.vertices[v].half_edge {
            let start = h;
            loop {
                let he = &self.half_edges[h];
                if he.face == Some(face_id) { return true; }
                h = he.twin;
                if h == usize::MAX { break; }
                h = self.half_edges[h].next;
                if h == start { break; }
            }
        }
        false
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

        let mut edge_set = AHashSet::new();
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

    pub fn validate_half_edges(&self) {
        let mut edge_set = AHashSet::new();
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
            let mut seen = AHashSet::new();
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
        let mut seen = AHashSet::new();

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

        let set0: AHashSet<_> = ring0.neighbors_ccw.iter().copied().filter(|&x| x != v1).collect();
        let set1: AHashSet<_> = ring1.neighbors_ccw.iter().copied().filter(|&x| x != v0).collect();

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
        let mut expected = AHashSet::new();
        if let Some(a) = pr.opposite_a { expected.insert(a); }
        if let Some(b) = pr.opposite_b { expected.insert(b); }

        // Intersection of neighbor sets should equal expected
        pr.common_neighbors == expected
    }

    /// Compute area-squared (2*Area)^2 of face `f` if vertex `mv` moved to `p_star`.
    #[inline]
    pub fn area2x4_after_move(
        &self,
        f: usize,
        mv: usize,
        p_star: &Point<T, N>,
    ) -> T
    where     Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        let [i, j, k] = self.face_vertices(f);
        let p = |idx: usize| -> &Point<T, N> {
            if idx == mv {
                p_star
            } else {
                &self.vertices[idx].position
            }
        };
        let ab = (p(j) - p(i)).as_vector();
        let ac = (p(k) - p(i)).as_vector();
        let n = ab.cross(&ac);
        n.dot(&n)
    }

    /// Find all self-intersections or overlaps among the mesh faces.
    /// Requires N == 3 (triangle meshes in 3D).
    pub fn find_self_intersections(&self) -> Vec<SelfIntersection>
    where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    Segment<T, N>: SegmentOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>
    {
        if N != 3 {
            // Only 3D triangle meshes are supported
            return Vec::new();
        }

        use crate::geometry::tri_tri_intersect::{tri_tri_intersection, TriTriIntersectionResult};

        // Build a face AABB tree (in CgarF64) for broad-phase culling
        let face_boxes: Vec<(Aabb<T, N, Point<T, N>>, usize)> = (0..self.faces.len())
            .filter(|&fi| !self.faces[fi].removed)
            .map(|fi| {
                let aabb_t = self.face_aabb(fi);
                let min = aabb_t.min();
                let max = aabb_t.max();
                let box3 = Aabb::<T, N, Point<T, N>>::from_points(
                    &Point::<T, N>::from_vals(from_fn(|i| min[i].clone())),
                    &Point::<T, N>::from_vals(from_fn(|i| max[i].clone())),
                );
                (box3, fi)
            })
            .collect();

        let tree = AabbTree::<T, N, Point<T, N>, usize>::build(face_boxes);

        let tol = T::tolerance();
        let tol2 = &tol * &tol;

        let mut out = Vec::new();
        let mut candidates = Vec::new();

        // Helper: face vertices as [&Point; 3]
        let tri_points = |f: usize| -> [&Point<T, N>; 3] {
            let vs = self.face_vertices(f);
            [
                &self.vertices[vs[0]].position,
                &self.vertices[vs[1]].position,
                &self.vertices[vs[2]].position,
            ]
        };

        // Helper: check if faces share an (undirected) mesh edge
        let share_edge = |fa: usize, fb: usize| -> bool {
            if self.are_faces_adjacent(fa, fb) { return true; }
            let va = self.face_vertices(fa);
            let vb = self.face_vertices(fb);
            let ea = [
                (va[0].min(va[1]), va[0].max(va[1])),
                (va[1].min(va[2]), va[1].max(va[2])),
                (va[2].min(va[0]), va[2].max(va[0])),
            ];
            let eb = [
                (vb[0].min(vb[1]), vb[0].max(vb[1])),
                (vb[1].min(vb[2]), vb[1].max(vb[2])),
                (vb[2].min(vb[0]), vb[2].max(vb[0])),
            ];
            ea.iter().any(|e| eb.contains(e))
        };

        // Sweep all faces; for each face, query overlapping boxes and test only fb > fa to avoid duplicates
        for fa in 0..self.faces.len() {
            if self.faces[fa].removed {
                continue;
            }

            // Query with the face AABB slightly dilated by tolerance
            let aabb_t = self.face_aabb(fa);
            let min = aabb_t.min();
            let max = aabb_t.max();
            let query = Aabb::<T, N, Point<T, N>>::from_points(
                &Point::<T, N>::from_vals(from_fn(|i| (&min[i] - &T::tolerance()))),
                &Point::<T, N>::from_vals(from_fn(|i| (&max[i] + &T::tolerance()))),
            );

            candidates.clear();
            tree.query_valid(&query, &mut candidates);

            if candidates.is_empty() {
                continue;
            }

            let pa = tri_points(fa);

            for &fb in &candidates {
                let fb = *fb;
                if fb <= fa {
                    continue; // avoid duplicates and self
                }
                if self.faces[fb].removed {
                    continue;
                }

                let pb = tri_points(fb);

                match tri_tri_intersection(&pa, &pb) {
                    TriTriIntersectionResult::Proper(seg) => {
                        if seg.length2() > tol2 {
                            println!("Non-coplanar crossing detected between faces {} and {}", fa, fb);
                            out.push(SelfIntersection {
                                a: fa, b: fb,
                                kind: SelfIntersectionKind::NonCoplanarCrossing,
                            });
                        }
                    }

                    TriTriIntersectionResult::Coplanar(seg) => {
                        // Skip adjacency also here
                        if share_edge(fa, fb) { continue; }
                        if seg.length2() > tol2 {
                            let fa_vs = self.face_vertices(fa);
                            let fb_vs = self.face_vertices(fb);

                            println!("tol2: {:?}", tol2);

                            println!("Coplanar edge overlap detected between faces {} and {}", fa, fb);
                            println!("\n    FA Vertices:\n        {:?}\n        {:?}\n        {:?}", self.vertices[fa_vs[0]].position, self.vertices[fa_vs[1]].position, self.vertices[fa_vs[2]].position);
                            println!("\n    FB Vertices:\n        {:?}\n        {:?}\n        {:?}", self.vertices[fb_vs[0]].position, self.vertices[fb_vs[1]].position, self.vertices[fb_vs[2]].position);
                            out.push(SelfIntersection {
                                a: fa, b: fb,
                                kind: SelfIntersectionKind::CoplanarEdgeOverlap,
                            });
                        }
                    }

                    TriTriIntersectionResult::CoplanarPolygon(poly) => {
                        // Skip adjacency here too (your split pairs fall here spuriously)
                        if share_edge(fa, fb) { continue; }

                        // Degeneracy filter: require ≥3 unique points & area > eps
                        if coplanar_polygon_has_area(&poly, &tol) {
                            println!("Coplanar area overlap detected between faces {} and {}", fa, fb);
                            out.push(SelfIntersection {
                                a: fa, b: fb,
                                kind: SelfIntersectionKind::CoplanarAreaOverlap,
                            });
                        }
                    }

                    _ => {}
                }
            }
        }

        out
    }

    /// True if the mesh contains any self-intersection or coplanar overlap.
    pub fn has_self_intersections(&self) -> bool
    where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
    {
        !self.find_self_intersections().is_empty()
    }

}

pub fn ray_segment_intersection_2d<T>(
    o: &crate::geometry::point::Point<T, 2>,
    d: &crate::geometry::vector::Vector<T, 2>,
    a: &crate::geometry::point::Point<T, 2>,
    b: &crate::geometry::point::Point<T, 2>,
) -> Option<(T, T)>
where
    T: crate::numeric::scalar::Scalar + PartialOrd + Clone,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // Ray direction must be non-zero
    let rx = d[0].clone();
    let ry = d[1].clone();
    let rr = &rx * &rx + &ry * &ry;
    if rr.is_zero() {
        return None;
    }

    // Segment direction s = b - a
    let sx = &b[0] - &a[0];
    let sy = &b[1] - &a[1];

    // Vector from ray origin to segment start q = a - o
    let qx = &a[0] - &o[0];
    let qy = &a[1] - &o[1];

    // Cross products
    let rxs = &rx * &sy - &ry * &sx;
    let qxr = &qx * &ry - &qy * &rx; // cross(q, r)
    let qxs = &qx * &sy - &qy * &sx; // cross(q, s)

    // Parallel?
    if rxs.is_zero() {
        // Collinear if q x r == 0 (use kernel-backed collinearity to be safe)
        // Faster short-circuit via cross check, then confirm with kernel to avoid false positives.
        let collinear = qxr.is_zero()
            && kernel::are_collinear(o, &(Point::<T, 2>::from([&o[0] + &rx, &o[1] + &ry])), a)
            && kernel::are_collinear(o, &(Point::<T, 2>::from([&o[0] + &rx, &o[1] + &ry])), b);

        if !collinear {
            return None; // parallel disjoint
        }

        // Collinear overlap: project endpoints onto the ray to get [t0, t1] on the ray
        // t = ((p - o) · r) / |r|^2
        let t0 = &(&qx * &rx + &qy * &ry) / &rr;
        let t1 = &t0 + &(&(&sx * &rx + &sy * &ry) / &rr);

        let (tmin, tmax) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        if tmax < T::zero() {
            return None; // entire segment lies behind the ray
        }

        // Hit point is the closest point along the ray that overlaps the segment
        let t_hit = if tmin >= T::zero() { tmin } else { T::zero() };

        // Compute intersection point p_hit = o + t_hit * r
        let px = &o[0] + &(&rx * &t_hit);
        let py = &o[1] + &(&ry * &t_hit);
        let p_hit = crate::geometry::point::Point::<T, 2>::from([px, py]);

        // Get segment parameter u in [0,1] using the kernel helper (handles degenerate [a==b] cleanly)
        if let Some(u) = crate::kernel::point_u_on_segment(a, b, &p_hit) {
            return Some((t_hit, u));
        }
        return None;
    }

    // Skew lines: unique intersection
    // Solve:
    //   t = cross(q, s) / cross(r, s)
    //   u = cross(q, r) / cross(r, s)
    let t = &qxs / &rxs;
    let u = &qxr / &rxs;

    if t < T::zero() || u < T::zero() || u > T::one() {
        return None;
    }
    Some((t, u))
}

#[derive(Debug)]
enum RayTriCore<T> {
    // Unique solution: the ray intersects the triangle's plane with well-defined (t,u,v)
    Skew { t: T, u: T, v: T },
    // Ray direction lies in the triangle plane; 'coplanar' reports if origin also lies in that plane
    Parallel { coplanar: bool },
    // Triangle is degenerate (edges don't span a 2D plane)
    Degenerate,
}

/// Robust ray-triangle intersection using Möller-Trumbore algorithm
// ...existing code...
fn ray_triangle_intersection<T: Scalar, const N: usize>(
    ray_origin: &Point<T, N>,
    ray_dir: &Vector<T, N>,
    triangle: [&Point<T, N>; N],
) -> Option<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    if N != 3 {
        panic!("Currently, only 3 dimensions are supported.");
    }

    let one = T::one();

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

    // Use sign-based checks to preserve laziness
    if u.is_negative() || (&u - &one).is_positive() {
        return None; // Intersection outside triangle
    }

    // Cross product: s x edge1
    let q = Vector::<T, 3>::from_vals([
        &s[1] * &edge1[2] - &s[2] * &edge1[1],
        &s[2] * &edge1[0] - &s[0] * &edge1[2],
        &s[0] * &edge1[1] - &s[1] * &edge1[0],
    ]);

    // Calculate v parameter
    let v = &f * &ray_dir.0.as_vector_3().dot(&q);

    let sum_uv = &u + &v;
    if v.is_negative() || (&sum_uv - &one).is_positive() {
        return None; // Intersection outside triangle
    }

    // Distance along ray
    let t = &f * &edge2.dot(&q);
    Some(t)
}

/// Robust 2D ray–segment intersection:
/// Solves p + t*r = a + u*e, with t >= 0 and u in (ε, 1+ε] to own the vertex once.
/// Returns (t, u) if an intersection exists.
fn ray_segment_intersection_2d_robust<T>(
    p: &Point<T, 2>,
    r: &Vector<T, 2>,
    a: &Point<T, 2>,
    b: &Point<T, 2>,
    eps: &T,
) -> Option<(T, T)>
where
    T: Scalar + PartialOrd + Clone,
    Vector<T, 2>: VectorOps<T, 2, Cross = T>,
    for<'a> &'a T: core::ops::Sub<&'a T, Output = T>
        + core::ops::Mul<&'a T, Output = T>
        + core::ops::Add<&'a T, Output = T>
        + core::ops::Div<&'a T, Output = T>
        + core::ops::Neg<Output = T>,
{
    let e = (b - a).as_vector();
    let w = (a - p).as_vector();

    let denom = r.cross(&e); // scalar
    let t_num = w.cross(&e);
    let u_num = w.cross(&r);

    let near_zero = |x: &T| -> bool {
        let meps = -(eps.clone());
        x >= &meps && x <= &eps
    };

    // Non-parallel case
    if !near_zero(&denom) {
        let t = &t_num / &denom;
        let u = &u_num / &denom;

        // Accept forward t and "own the vertex once":
        //   u in (eps, 1 + eps]  => exclude ~0; include ~1
        if &t >= &eps && &u > &eps && &u <= &(&T::one() + &eps) {
            return Some((t, u));
        }
        return None;
    }

    // Parallel or collinear
    // If not collinear (w not perpendicular to both r and e), there's no hit.
    // Collinearity: r × (a - p) ≈ 0 AND r × e ≈ 0  (the latter is denom≈0 already)
    if !near_zero(&w.cross(r)) {
        return None;
    }

    // Collinear overlap handling:
    // Project (a-p) and (b-p) on r to get t-parameters; take the smallest t >= eps.
    let r2 = r.dot(r);
    if near_zero(&r2) {
        // Ray has no direction: no meaningful intersection
        return None;
    }

    let t0 = &(a.as_vector() - p.as_vector()).dot(r) / &r2;
    let t1 = &(b.as_vector() - p.as_vector()).dot(r) / &r2;

    let (t_enter, t_exit) = if t0 <= t1 {
        (t0.clone(), t1.clone())
    } else {
        (t1.clone(), t0.clone())
    };

    // Intersection with the ray exists iff t_exit >= eps
    if &t_exit < &eps {
        return None;
    }

    // First contact along the ray:
    let t_hit = if &t_enter >= &eps { t_enter } else { t_exit };

    // Recover u on [0,1] for (a->b)
    let e2 = e.dot(&e);
    if near_zero(&e2) {
        // Degenerate segment
        return None;
    }
    // Point on segment: q = p + t_hit*r; u = ((q - a)·e)/(e·e)
    let q = &(p.as_vector() + r.scale(&t_hit)).0;
    let u = &(&q.as_vector() - &a.as_vector()).dot(&e) / &e2;

    // Apply the same ownership rule for the vertex:
    if &t_hit >= &eps && &u > &eps && &u <= &(&T::one() + &eps) {
        return Some((t_hit, u));
    }
    None
}

fn coplanar_polygon_has_area<T: Scalar, const N: usize>(segs: &[Segment<T, N>], tol: &T) -> bool
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>
        + core::ops::Sub<&'a T, Output = T>
        + core::ops::Mul<&'a T, Output = T>,
{
    // Collect unique endpoints (within tol)
    let mut uniq: Vec<Point<T, N>> = Vec::new();
    for s in segs {
        for p in [&s.a, &s.b] {
            let mut dup = false;
            for q in &uniq {
                if &(&(p - q).as_vector()).norm2() <= tol {
                    dup = true;
                    break;
                }
            }
            if !dup {
                uniq.push(p.clone());
            }
        }
    }
    if uniq.len() < 3 {
        return false;
    }

    // Threshold on area^2 ~ (tol^2)^2 = tol^4, like your previous version
    let tol2 = tol * tol;
    let area_thresh2 = &tol2 * &tol2;

    // If any triple of points is non-collinear by more than tol, we have area.
    for i in 0..uniq.len() - 2 {
        for j in i + 1..uniq.len() - 1 {
            let v1 = (&uniq[j] - &uniq[i]).as_vector();
            for k in j + 1..uniq.len() {
                let v2 = (&uniq[k] - &uniq[i]).as_vector();
                let cross = v1.cross(&v2);
                let cross2 = cross.dot(&cross); // = 4 * (triangle_area)^2
                if cross2 > area_thresh2 {
                    return true;
                }
            }
        }
    }
    false
}
