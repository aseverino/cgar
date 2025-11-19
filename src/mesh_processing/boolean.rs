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
    collections::VecDeque,
    hash::Hash,
    ops::{Add, Div, Mul, Neg, Sub},
    time::Instant,
};

use ahash::{AHashMap, AHashSet};

use crate::{
    geometry::{
        Aabb, AabbTree,
        plane::{Plane, PlaneOps},
        point::{Point, PointOps},
        segment::{Segment, SegmentOps},
        spatial_element::SpatialElement,
        tri_tri_intersect::{
            ContactOnTri, TriPrecomp, TriTriIntersectionDetailed, segment_intersect_2d,
            tri_tri_intersection_with_precomp_detailed,
        },
        vector::{Cross2, Cross3, Vector, VectorOps, vector_cross},
    },
    mesh::{
        basic_types::{Mesh, PointInMeshResult, VertexSource},
        intersection_segment::{IntersectionEndPoint, IntersectionSegment},
    },
    mesh_processing::batching::{FaceJobUV, build_face_pslgs_2, build_face_pslgs_3},
    numeric::{
        cgar_f64::CgarF64,
        // lazy_exact::ENABLE_PANIC_ON_EXACT,
        scalar::{RefInto, Scalar},
    },
    operations::triangulation::delaunay::Delaunay,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ApproxPointKey {
    qx: i64,
    qy: i64,
    qz: i64, // use only N you need
}

pub fn point_key<T: Scalar, const N: usize>(p: &Point<T, N>) -> ApproxPointKey {
    let tol = T::point_merge_threshold();

    let tol_approx: CgarF64 = (&tol).ref_into();
    let s = 1.0 / tol_approx.0;

    // Read coords via approx, then quantize
    let q = |i: usize| -> i64 {
        let ai: CgarF64 = (&p[i]).ref_into();
        (ai.0 * s).round() as i64
    };

    ApproxPointKey {
        qx: q(0),
        qy: if N > 1 { q(1) } else { 0 },
        qz: if N > 2 { q(2) } else { 0 },
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EndPointHandle {
    pub segment_idx: usize,
    pub endpoint_idx: usize, // 0 or 1
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitType {
    Edge,
    Face,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BooleanOp {
    Union,
    Intersection,
    Difference,
}

impl<T: Scalar, const N: usize> Mesh<T, N>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    Plane<T, N>: PlaneOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
{
    fn classify_faces_3(
        &mut self,
        other: &Mesh<T, N>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
        coplanar_triangles: &Vec<CoplanarTriangle>,
        include_on_surface: bool,
    ) -> Vec<bool> {
        // build adjacency & boundary‐map
        let tree_b = AabbTree::<T, N, _, _>::build(
            (0..other.faces.len())
                .map(|i| {
                    let aabb_n = other.face_aabb(i);
                    let min = aabb_n.min();
                    let max = aabb_n.max();
                    // Convert Aabb<T, N, Point<T, N>> to Aabb<T, 3, Point<T, 3>>
                    let aabb3 = Aabb::<T, N, Point<T, N>>::from_points(
                        &Point::<T, N>::from_vals(from_fn(|i| min[i].clone())),
                        &Point::<T, N>::from_vals(from_fn(|i| max[i].clone())),
                    );
                    (aabb3, i)
                })
                .collect::<Vec<(Aabb<T, N, Point<T, N>>, usize)>>(),
        );

        let mut visited = vec![false; self.faces.len()];
        let mut inside = vec![false; self.faces.len()];

        // Non-coplanar intersections first.
        if intersection_segments.is_empty() {
            for f in 0..self.faces.len() {
                if self.faces[f].removed {
                    continue;
                }
                let c = self.face_centroid_fast(f);
                match other.point_in_mesh_3(&tree_b, &c) {
                    PointInMeshResult::Inside => inside[f] = true,
                    PointInMeshResult::OnSurface if include_on_surface => inside[f] = true,
                    _ => {}
                }
            }
            return inside;
        } else {
            let mut boundary_faces = AHashSet::new();
            for seg in intersection_segments {
                let v0 = seg.a.resulting_vertex.unwrap();
                let v1 = seg.b.resulting_vertex.unwrap();

                if let Some(&he) = self.edge_map.get(&(v0, v1)) {
                    if let Some(f0) = self.half_edges[he].face {
                        boundary_faces.insert(f0);
                    }
                    if let Some(f1) = self.half_edges[self.half_edges[he].twin].face {
                        boundary_faces.insert(f1);
                    }
                }
            }

            // 3) pick a seed face that lies inside B
            let seed_result = Self::get_seed_face_3(
                &self,
                &other,
                &tree_b,
                intersection_segments,
                &boundary_faces,
                include_on_surface,
            );

            if seed_result.is_none() {
                for (f, face) in self.faces.iter().enumerate() {
                    if face.removed {
                        continue;
                    }
                    let c = self.face_centroid_fast(f);
                    match other.point_in_mesh_3(&tree_b, &c) {
                        PointInMeshResult::Inside => inside[f] = true,
                        PointInMeshResult::OnSurface if include_on_surface => inside[f] = true,
                        _ => {}
                    }
                }
                return inside;
            }

            let (_seed_intersection_idx, selected_face) = seed_result.unwrap();

            let mut face_pairs: AHashMap<usize, Vec<usize>> = AHashMap::new();
            for seg_idx in 0..intersection_segments.len() {
                let seg = &intersection_segments[seg_idx];
                if seg[0].resulting_vertex == seg[1].resulting_vertex {
                    continue; // degenerate segment
                }
                let he = self
                    .edge_map
                    .get(&(
                        seg.a.resulting_vertex.unwrap(),
                        seg.b.resulting_vertex.unwrap(),
                    ))
                    .expect("Edge must exist in edge_map");

                let f0 = self.half_edges[*he]
                    .face
                    .expect("Half-edge must have a face");
                let f1 = self.half_edges
                    [self.find_valid_half_edge(self.half_edges[*he].twin, &seg.segment.b)]
                .face
                .expect("Half-edge must have a face");

                face_pairs.entry(f0).or_default().push(f1);
                face_pairs.entry(f1).or_default().push(f0);
            }

            // 4) iterative flood‐fill without crossing the boundary_map
            let mut queue = VecDeque::new();

            // seed found as before
            visited[selected_face] = true;
            inside[selected_face] = true;
            queue.push_back(selected_face);
            let mut num_visited = 1;

            while let Some(curr) = queue.pop_front() {
                for nbr in self.adjacent_faces(curr) {
                    if visited[nbr] {
                        continue;
                    }

                    let paired = face_pairs.get(&curr);
                    // skip exactly the neighbor that comes from the same segment split
                    if let Some(pv) = paired {
                        if pv.iter().any(|&pf| pf == nbr) {
                            // this nbr is the other half of a segment splitting curr
                            continue;
                        }
                    }

                    // otherwise, it's a genuine inside‐region adjacency
                    visited[nbr] = true;
                    inside[nbr] = true;
                    num_visited += 1;
                    queue.push_back(nbr);
                }
            }

            println!(
                "Flood-fill visited {} faces out of {}",
                num_visited,
                self.faces.len()
            );
        }

        if include_on_surface {
            // Now do co-planar intersections.
            for triangle in coplanar_triangles {
                let face = self.face_from_vertices(
                    triangle.verts[0],
                    triangle.verts[1],
                    triangle.verts[2],
                );
                if face != usize::MAX {
                    inside[face] = true;
                }
            }
        }

        inside
    }

    fn classify_faces_2(
        &self,
        other: &Mesh<T, N>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
    ) -> Vec<bool> {
        let mut inside = vec![false; self.faces.len()];

        if intersection_segments.is_empty() {
            for (face_idx, face) in self.faces.iter().enumerate() {
                if face.removed {
                    continue;
                }
                let centroid = self.face_centroid_fast(face_idx);
                inside[face_idx] = other.point_in_mesh_2(&centroid.as_point_2());
            }
            return inside;
        }

        let mut boundary_faces = AHashSet::new();
        for seg in intersection_segments {
            let v0 = seg.a.resulting_vertex.unwrap();
            let v1 = seg.b.resulting_vertex.unwrap();

            if let Some(&he) = self.edge_map.get(&(v0, v1)) {
                if let Some(f0) = self.half_edges[he].face {
                    boundary_faces.insert(f0);
                }
                if let Some(f1) = self.half_edges[self.half_edges[he].twin].face {
                    boundary_faces.insert(f1);
                }
            }
        }

        // Find seed face
        let seed_result =
            Self::get_seed_face_2(self, other, intersection_segments, &boundary_faces);

        if seed_result.is_none() {
            for (face_idx, face) in self.faces.iter().enumerate() {
                if face.removed {
                    continue;
                }
                let centroid = self.face_centroid_fast(face_idx);
                inside[face_idx] = other.point_in_mesh_2(&centroid.as_point_2());
            }
            return inside;
        }

        let (_seed_intersection_idx, seed_face) = seed_result.unwrap();

        // Build face connectivity avoiding intersection boundaries
        let mut face_pairs: AHashMap<usize, Vec<usize>> = AHashMap::new();
        for seg in intersection_segments {
            let v0 = seg.a.resulting_vertex.unwrap();
            let v1 = seg.b.resulting_vertex.unwrap();

            if let Some(&he) = self.edge_map.get(&(v0, v1)) {
                if let (Some(f0), Some(f1)) = (
                    self.half_edges[he].face,
                    self.half_edges[self.half_edges[he].twin].face,
                ) {
                    face_pairs.entry(f0).or_default().push(f1);
                    face_pairs.entry(f1).or_default().push(f0);
                }
            }
        }

        // Flood fill
        let mut visited = vec![false; self.faces.len()];
        let mut queue = VecDeque::new();

        visited[seed_face] = true;
        inside[seed_face] = true;
        queue.push_back(seed_face);

        while let Some(curr) = queue.pop_front() {
            for nbr in self.adjacent_faces(curr) {
                if visited[nbr] {
                    continue;
                }

                if let Some(pairs) = face_pairs.get(&curr) {
                    if pairs.contains(&nbr) {
                        continue;
                    }
                }

                visited[nbr] = true;
                inside[nbr] = true;
                queue.push_back(nbr);
            }
        }

        inside
    }

    pub fn get_mesh_intersections_3(
        &self,
        other: &Mesh<T, N>,
    ) -> (
        Vec<IntersectionSegment<T, N>>,
        Vec<IntersectionSegment<T, N>>,
    ) {
        let start = Instant::now();
        let tree_b = other.build_face_tree();
        println!("Total AABB computation: {:.2?}", start.elapsed());
        let mut intersection_segments_a = Vec::new();
        let mut intersection_segments_b = Vec::new();
        let intersection_segments = [&mut intersection_segments_a, &mut intersection_segments_b];
        let meshes = [&self, &other];

        let mut candidates = Vec::with_capacity(64);
        let mut ends_vec = Vec::with_capacity(64);

        for fa in 0..self.faces.len() {
            candidates.clear();
            tree_b.query(&self.face_aabb(fa), &mut candidates);

            if candidates.is_empty() {
                continue;
            }

            let pa_idx = self.face_vertices(fa);
            let pa: [&Point<T, N>; 3] = from_fn(|i| &self.vertices[pa_idx[i]].position);
            let pre_a = TriPrecomp::new(&pa);

            for &fb in &candidates {
                ends_vec.clear();

                let pb_idx = other.face_vertices(*fb);
                let pb: [&Point<T, N>; 3] = from_fn(|i| &other.vertices[pb_idx[i]].position);
                let pre_b = TriPrecomp::new(&pb);

                let vertices_indices = [&pa_idx, &pb_idx];
                let faces = [fa, *fb];

                let mut coplanar = false;

                match tri_tri_intersection_with_precomp_detailed(&pa, &pb, &pre_a, &pre_b) {
                    TriTriIntersectionDetailed::Proper { ends } => {
                        ends_vec.push(ends);
                    }
                    TriTriIntersectionDetailed::Coplanar { segs } => {
                        for ends in segs {
                            ends_vec.push(ends);
                        }
                        coplanar = true;
                    }
                    _ => {}
                }

                for ends in &ends_vec {
                    for mesh_x in 0..2 {
                        let on_x = if mesh_x == 0 {
                            [&ends[0].on_p, &ends[1].on_p]
                        } else {
                            [&ends[0].on_q, &ends[1].on_q]
                        };
                        let mut intersection_endpoint_0 =
                            IntersectionEndPoint::<T, N>::new_default();
                        let mut intersection_endpoint_1 =
                            IntersectionEndPoint::<T, N>::new_default();
                        let intersection_endpoints =
                            [&mut intersection_endpoint_0, &mut intersection_endpoint_1];

                        for end_x in 0..2 {
                            match &on_x[end_x] {
                                ContactOnTri::Vertex(i) => {
                                    intersection_endpoints[end_x].vertex_hint =
                                        Some([vertices_indices[mesh_x][*i], usize::MAX]);
                                }
                                ContactOnTri::Edge { e, u } => {
                                    intersection_endpoints[end_x].vertex_hint = Some([
                                        vertices_indices[mesh_x][e.0],
                                        vertices_indices[mesh_x][e.1],
                                    ]);

                                    let he = meshes[mesh_x].edge_map[&(
                                        vertices_indices[mesh_x][e.0],
                                        vertices_indices[mesh_x][e.1],
                                    )];

                                    let f1 = meshes[mesh_x].half_edges[he].face.unwrap();
                                    let f2 = meshes[mesh_x].half_edges
                                        [meshes[mesh_x].half_edges[he].twin]
                                        .face
                                        .unwrap();

                                    intersection_endpoints[end_x].faces_hint = Some([f1, f2]);
                                    intersection_endpoints[end_x].half_edge_hint = Some(he);
                                    intersection_endpoints[end_x].half_edge_u_hint =
                                        Some(u.clone());
                                }
                                ContactOnTri::Interior { bary } => {
                                    intersection_endpoints[end_x].faces_hint =
                                        Some([faces[mesh_x], usize::MAX]);
                                    intersection_endpoints[end_x].barycentric_hint =
                                        Some(bary.clone());
                                }
                            }
                        }
                        intersection_segments[mesh_x].push(IntersectionSegment::new(
                            intersection_endpoint_0,
                            intersection_endpoint_1,
                            &Segment::new(&ends[0].point, &ends[1].point),
                            faces[mesh_x],
                            coplanar,
                        ));
                    }
                }
            }
        }
        (intersection_segments_a, intersection_segments_b)
    }

    pub fn get_mesh_intersections_2(
        &self,
        other: &Mesh<T, N>,
    ) -> (
        Vec<IntersectionSegment<T, N>>,
        Vec<IntersectionSegment<T, N>>,
    ) {
        let tree_b = other.build_face_tree();
        let mut intersection_segments_a = Vec::new();
        let mut intersection_segments_b = Vec::new();

        let mut candidates = Vec::with_capacity(64);

        // Track coplanar segment endpoints per half-edge to filter redundant point intersections
        let mut coplanar_endpoints_a = AHashSet::<(usize, i64)>::new();
        let mut coplanar_endpoints_b = AHashSet::<(usize, i64)>::new();

        let tol = T::point_merge_threshold();
        let tol_f64 = RefInto::<CgarF64>::ref_into(&tol).0.max(1e-12);
        let one = T::one();

        for fa in 0..self.faces.len() {
            candidates.clear();
            tree_b.query(&self.face_aabb(fa), &mut candidates);

            if candidates.is_empty() {
                continue;
            }

            let ea_idx = self.face_vertices(fa);
            let na = ea_idx.len();

            for &fb in &candidates {
                let eb_idx = other.face_vertices(*fb);
                let nb = eb_idx.len();

                for i in 0..na {
                    let va0 = ea_idx[i];
                    let va1 = ea_idx[(i + 1) % na];
                    let edge_a = ordered(va0, va1);

                    let pa0 = &self.vertices[va0].position;
                    let pa1 = &self.vertices[va1].position;

                    let p2_a0 = Point::<T, 2>::from_vals(from_fn(|k| pa0[k].clone()));
                    let p2_a1 = Point::<T, 2>::from_vals(from_fn(|k| pa1[k].clone()));

                    for j in 0..nb {
                        let vb0 = eb_idx[j];
                        let vb1 = eb_idx[(j + 1) % nb];
                        let edge_b = ordered(vb0, vb1);

                        let pb0 = &other.vertices[vb0].position;
                        let pb1 = &other.vertices[vb1].position;

                        let p2_b0 = Point::<T, 2>::from_vals(from_fn(|k| pb0[k].clone()));
                        let p2_b1 = Point::<T, 2>::from_vals(from_fn(|k| pb1[k].clone()));

                        if let Some(seg_2d) = segment_intersect_2d(&p2_a0, &p2_a1, &p2_b0, &p2_b1) {
                            let p_start = Point::<T, N>::from_vals(from_fn(|k| {
                                if k < 2 {
                                    seg_2d.a[k].clone()
                                } else {
                                    T::zero()
                                }
                            }));
                            let p_end = Point::<T, N>::from_vals(from_fn(|k| {
                                if k < 2 {
                                    seg_2d.b[k].clone()
                                } else {
                                    T::zero()
                                }
                            }));

                            let is_point = p_start == p_end;
                            let coplanar = !is_point;

                            let edge_a_geom = Segment::new(pa0, pa1);
                            let edge_b_geom = Segment::new(pb0, pb1);

                            let u_a_start = edge_a_geom.parameter_of_point(&p_start);
                            let u_a_end = edge_a_geom.parameter_of_point(&p_end);
                            let u_b_start = edge_b_geom.parameter_of_point(&p_start);
                            let u_b_end = edge_b_geom.parameter_of_point(&p_end);

                            let is_endpoint_a_start =
                                &u_a_start <= &tol || &u_a_start >= &(&one - &tol);
                            let is_endpoint_a_end = &u_a_end <= &tol || &u_a_end >= &(&one - &tol);
                            let is_endpoint_b_start =
                                &u_b_start <= &tol || &u_b_start >= &(&one - &tol);
                            let is_endpoint_b_end = &u_b_end <= &tol || &u_b_end >= &(&one - &tol);

                            // For mesh A
                            let dominated_by_coplanar_a = if is_point {
                                if let Some(&he) = self.edge_map.get(&(va0, va1)) {
                                    let bucket = bucket_u(
                                        RefInto::<CgarF64>::ref_into(&u_a_start).0,
                                        tol_f64,
                                    );
                                    coplanar_endpoints_a.contains(&(he, bucket))
                                } else {
                                    false
                                }
                            } else {
                                false
                            };

                            if !dominated_by_coplanar_a {
                                let a_start_interior = !is_endpoint_a_start;
                                let a_end_interior = !is_endpoint_a_end;

                                if a_start_interior || a_end_interior || coplanar {
                                    if !(is_point && is_endpoint_a_start) {
                                        let mut ep0 = IntersectionEndPoint::<T, N>::new_default();
                                        let mut ep1 = IntersectionEndPoint::<T, N>::new_default();

                                        ep0.vertex_hint = Some([va0, va1]);
                                        ep1.vertex_hint = Some([va0, va1]);

                                        if let Some(&he) = self.edge_map.get(&(va0, va1)) {
                                            ep0.half_edge_hint = Some(he);
                                            ep1.half_edge_hint = Some(he);

                                            if coplanar {
                                                let bucket_start = bucket_u(
                                                    RefInto::<CgarF64>::ref_into(&u_a_start).0,
                                                    tol_f64,
                                                );
                                                let bucket_end = bucket_u(
                                                    RefInto::<CgarF64>::ref_into(&u_a_end).0,
                                                    tol_f64,
                                                );
                                                coplanar_endpoints_a.insert((he, bucket_start));
                                                coplanar_endpoints_a.insert((he, bucket_end));
                                            }
                                        }

                                        ep0.half_edge_u_hint = Some(u_a_start.clone());
                                        ep1.half_edge_u_hint = Some(u_a_end.clone());

                                        intersection_segments_a.push(IntersectionSegment::new(
                                            ep0,
                                            ep1,
                                            &Segment::new(&p_start, &p_end),
                                            fa,
                                            coplanar,
                                        ));
                                    }
                                }
                            }

                            // For mesh B
                            let dominated_by_coplanar_b = if is_point {
                                if let Some(&he) = other.edge_map.get(&(vb0, vb1)) {
                                    let bucket = bucket_u(
                                        RefInto::<CgarF64>::ref_into(&u_b_start).0,
                                        tol_f64,
                                    );
                                    coplanar_endpoints_b.contains(&(he, bucket))
                                } else {
                                    false
                                }
                            } else {
                                false
                            };

                            if !dominated_by_coplanar_b {
                                let b_start_interior = !is_endpoint_b_start;
                                let b_end_interior = !is_endpoint_b_end;

                                if b_start_interior || b_end_interior || coplanar {
                                    if !(is_point && is_endpoint_b_start) {
                                        let mut ep0 = IntersectionEndPoint::<T, N>::new_default();
                                        let mut ep1 = IntersectionEndPoint::<T, N>::new_default();

                                        ep0.vertex_hint = Some([vb0, vb1]);
                                        ep1.vertex_hint = Some([vb0, vb1]);

                                        if let Some(&he) = other.edge_map.get(&(vb0, vb1)) {
                                            ep0.half_edge_hint = Some(he);
                                            ep1.half_edge_hint = Some(he);

                                            if coplanar {
                                                let bucket_start = bucket_u(
                                                    RefInto::<CgarF64>::ref_into(&u_b_start).0,
                                                    tol_f64,
                                                );
                                                let bucket_end = bucket_u(
                                                    RefInto::<CgarF64>::ref_into(&u_b_end).0,
                                                    tol_f64,
                                                );
                                                coplanar_endpoints_b.insert((he, bucket_start));
                                                coplanar_endpoints_b.insert((he, bucket_end));
                                            }
                                        }

                                        ep0.half_edge_u_hint = Some(u_b_start.clone());
                                        ep1.half_edge_u_hint = Some(u_b_end.clone());

                                        intersection_segments_b.push(IntersectionSegment::new(
                                            ep0,
                                            ep1,
                                            &Segment::new(&p_start, &p_end),
                                            *fb,
                                            coplanar,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        (intersection_segments_a, intersection_segments_b)
    }

    pub fn corefine_and_boolean(&mut self, other: &mut Mesh<T, N>, op: BooleanOp) -> Mesh<T, N>
    where
        Vector<T, 2>: Cross2<T>,
        Vector<T, 3>: Cross3<T>,
    {
        let mut a = self;
        let mut b = other;

        let start = Instant::now();
        let (mut intersection_segments_a, mut intersection_segments_b) = {
            if N == 3 {
                a.get_mesh_intersections_3(&b)
            } else {
                a.get_mesh_intersections_2(&b)
            }
        };

        println!("Intersections created in {:.2?}", start.elapsed());

        // Remove duplicate segments from both lists.
        let start = Instant::now();
        remove_duplicate_and_invalid_segments(&mut intersection_segments_a);
        remove_duplicate_and_invalid_segments(&mut intersection_segments_b);
        println!("Removed duplicates in {:.2?}", start.elapsed());

        println!("Splits on A: {}", intersection_segments_a.len());
        println!("{:?}", intersection_segments_a);
        let start = Instant::now();
        let _created_a =
            allocate_vertices_for_splits_no_topology(&mut a, &mut intersection_segments_a);

        // ENABLE_PANIC_ON_EXACT.store(true, std::sync::atomic::Ordering::Relaxed);

        println!("Allocated vertices on A in {:.2?}", start.elapsed());
        let start = Instant::now();
        let jobs_a = if N == 3 {
            build_face_pslgs_3(a, &intersection_segments_a)
        } else {
            build_face_pslgs_2(a, &intersection_segments_a)
        };
        println!("Built faces PSLGS on A in {:.2?}", start.elapsed());

        // Build CDT per compacted job (example)
        let start = Instant::now();

        let cdts_a = Delaunay::build_batch_with_constraints_bowyer_watson(&jobs_a, a);

        println!("Built Bowyer-Watson on A in {:.2?}", start.elapsed());
        let start = Instant::now();
        rewrite_faces_from_cdt_batch(a, &jobs_a, &cdts_a);
        println!("Rewrite on A faces done in {:.2?}", start.elapsed());

        println!("Splits on B: {}", intersection_segments_b.len());
        let start = Instant::now();
        let _created_b =
            allocate_vertices_for_splits_no_topology(&mut b, &mut intersection_segments_b);
        println!("Allocated vertices on B in {:.2?}", start.elapsed());
        let start = Instant::now();
        let jobs_b = if N == 3 {
            build_face_pslgs_3(b, &intersection_segments_b)
        } else {
            build_face_pslgs_2(b, &intersection_segments_b)
        };
        println!("Built faces PSLGS on B in {:.2?}", start.elapsed());

        // Build CDT per compacted job (example)
        let start = Instant::now();
        let cdts_b = Delaunay::build_batch_with_constraints_bowyer_watson(&jobs_b, b);
        println!("Built Bowyer-Watson on B in {:.2?}", start.elapsed());
        let start = Instant::now();
        rewrite_faces_from_cdt_batch(b, &jobs_b, &cdts_b);
        println!("Rewrite on B faces done in {:.2?}", start.elapsed());

        intersection_segments_a.retain(|segment| !segment.invalidated);
        intersection_segments_b.retain(|segment| !segment.invalidated);

        // 6. Create result mesh
        let mut result = Mesh::new();
        let mut vid_map = AHashMap::new();

        // Add A vertices
        for (i, v) in a.vertices.iter().enumerate() {
            let ni = result.add_vertex(v.position.clone());
            vid_map.insert((VertexSource::A, i), ni);
        }

        // Build chains
        let start = Instant::now();
        let mut a_coplanars = Self::build_links(&a, &mut intersection_segments_a);
        let mut b_coplanars = Self::build_links(&b, &mut intersection_segments_b);
        println!("Chains built in {:.2?}", start.elapsed());

        // Classify faces using topological method
        let start = Instant::now();
        let a_classification = {
            if N == 3 {
                a.classify_faces_3(&b, &mut intersection_segments_a, &mut a_coplanars, true)
            } else {
                a.classify_faces_2(&b, &intersection_segments_a)
            }
        };

        println!(
            "A faces classified inside intersection loops in {:.2?}",
            start.elapsed()
        );

        let inside = match op {
            BooleanOp::Union => false,
            BooleanOp::Intersection => true,
            BooleanOp::Difference => false,
        };

        let include_on_surface = match op {
            BooleanOp::Union => true,
            BooleanOp::Intersection => true,
            BooleanOp::Difference => false,
        };

        for (fa, fa_inside) in a_classification.iter().enumerate() {
            if a.faces[fa].removed {
                continue;
            }
            if *fa_inside == inside {
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

                let b_classification = if N == 3 {
                    b.classify_faces_3(
                        &a,
                        &mut intersection_segments_b,
                        &mut b_coplanars,
                        include_on_surface,
                    )
                } else {
                    b.classify_faces_2(&a, &intersection_segments_b)
                };

                for (fb, inside) in b_classification.iter().enumerate() {
                    if b.faces[fb].removed {
                        continue;
                    }
                    if !*inside {
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

                let b_classification = if N == 3 {
                    b.classify_faces_3(
                        &a,
                        &mut intersection_segments_b,
                        &mut b_coplanars,
                        include_on_surface,
                    )
                } else {
                    b.classify_faces_2(&a, &intersection_segments_b)
                };
                for (fb, inside) in b_classification.iter().enumerate() {
                    if b.faces[fb].removed {
                        continue;
                    }
                    if *inside {
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

                let start = Instant::now();
                let b_classification = if N == 3 {
                    b.classify_faces_3(
                        &a,
                        &mut intersection_segments_b,
                        &mut b_coplanars,
                        include_on_surface,
                    )
                } else {
                    b.classify_faces_2(&a, &intersection_segments_b)
                };

                println!("B CLASSIFICATION:");
                for (fb, inside) in b_classification.iter().enumerate() {
                    if b.faces[fb].removed {
                        continue;
                    }
                    if *inside {
                        let vs = b.face_vertices(fb);
                        // Flip face orientation for caps
                        result.add_triangle(
                            vid_map[&(VertexSource::B, vs[2])],
                            vid_map[&(VertexSource::B, vs[1])],
                            vid_map[&(VertexSource::B, vs[0])],
                        );
                    }
                }
                println!(
                    "B faces classified inside intersection loops in {:.2?}",
                    start.elapsed()
                );
            }
        }

        result.remove_duplicate_vertices();
        result.remove_unused_vertices();
        result.remove_invalidated_faces();

        // let result_edges: Vec<[usize; 2]> = result.edge_map.keys().map(|&(u, v)| [u, v]).collect();
        // let t_junctions =
        //     find_x_vertices_on_y_edges(&result, &result, &result_edges, &result_edges);

        // println!("t_junctions: {:?}", t_junctions);

        result
    }

    /// Build clean links between "intersection_segments" in-place. Removing coplanars from the collection.
    /// Returns a separate vector of vectors of intersections that are coplanar and form graphs.
    /// The coplanar segments are grouped by a shared plane.
    fn build_links(
        mesh: &Mesh<T, N>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    ) -> Vec<CoplanarTriangle> {
        if intersection_segments.is_empty() {
            return Vec::new();
        }

        if N == 2 {
            let mut vertex_to_segments: AHashMap<usize, Vec<usize>> = AHashMap::new();

            for (seg_idx, seg) in intersection_segments.iter().enumerate() {
                let [v0, v1] = [
                    seg.a.resulting_vertex.unwrap(),
                    seg.b.resulting_vertex.unwrap(),
                ];
                vertex_to_segments.entry(v0).or_default().push(seg_idx);
                vertex_to_segments.entry(v1).or_default().push(seg_idx);
            }

            for (seg_idx, seg) in intersection_segments.iter_mut().enumerate() {
                let [v0, v1] = [
                    seg.a.resulting_vertex.unwrap(),
                    seg.b.resulting_vertex.unwrap(),
                ];
                let mut connected_segments = AHashSet::new();

                if let Some(segments) = vertex_to_segments.get(&v0) {
                    for &other_idx in segments {
                        if other_idx != seg_idx {
                            connected_segments.insert(other_idx);
                        }
                    }
                }
                if let Some(segments) = vertex_to_segments.get(&v1) {
                    for &other_idx in segments {
                        if other_idx != seg_idx {
                            connected_segments.insert(other_idx);
                        }
                    }
                }

                seg.links.extend(connected_segments);
            }

            return Vec::new();
        }

        // First let's drain the coplanar segments into a separate structure
        let mut coplanar_groups: AHashMap<Plane<T, N>, Vec<IntersectionSegment<T, N>>> =
            AHashMap::new();

        println!(
            "intersection_segments.len() = {}",
            intersection_segments.len()
        );
        let mut i = 0;
        while i < intersection_segments.len() {
            if intersection_segments[i].coplanar {
                let seg = intersection_segments.remove(i);
                let edge = mesh.edge_map.get(&(
                    seg.a.resulting_vertex.unwrap(),
                    seg.b.resulting_vertex.unwrap(),
                ));
                if let Some(half_edge) = edge {
                    if let Some(face_0) = mesh.half_edges[*half_edge].face {
                        let plane_key = mesh.plane_from_face(face_0).canonicalized();
                        coplanar_groups
                            .entry(plane_key)
                            .or_default()
                            .push(seg.clone());
                    }
                    if let Some(face_1) = mesh.half_edges[mesh.half_edges[*half_edge].twin].face {
                        let plane_key = mesh.plane_from_face(face_1).canonicalized();
                        coplanar_groups.entry(plane_key).or_default().push(seg);
                    }
                }
            } else {
                i += 1;
            }
        }
        println!(
            "intersection_segments.len() = {}",
            intersection_segments.len()
        );

        let mut vertex_to_segments: AHashMap<usize, Vec<usize>> = AHashMap::new();

        for (seg_idx, seg) in intersection_segments.iter().enumerate() {
            let [v0, v1] = [
                seg.a.resulting_vertex.unwrap(),
                seg.b.resulting_vertex.unwrap(),
            ];
            vertex_to_segments.entry(v0).or_default().push(seg_idx);
            vertex_to_segments.entry(v1).or_default().push(seg_idx);
        }

        // === NON-COPLANAR SEGMENTS: Build double-linked structure ===
        for (seg_idx, seg) in intersection_segments.iter_mut().enumerate() {
            let [v0, v1] = [
                seg.a.resulting_vertex.unwrap(),
                seg.b.resulting_vertex.unwrap(),
            ];
            let mut connected_segments = AHashSet::new();

            if let Some(segments) = vertex_to_segments.get(&v0) {
                for &other_idx in segments {
                    if other_idx != seg_idx {
                        connected_segments.insert(other_idx);
                    }
                }
            }
            if let Some(segments) = vertex_to_segments.get(&v1) {
                for &other_idx in segments {
                    if other_idx != seg_idx {
                        connected_segments.insert(other_idx);
                    }
                }
            }

            let mut links: Vec<usize> = connected_segments.into_iter().collect();
            links.sort_unstable();

            seg.links.extend_from_slice(&links);
        }

        let mut triangles_per_group: Vec<CoplanarTriangle> = Vec::new();
        for (_plane, group) in &coplanar_groups {
            if group.len() >= 3 {
                let triangles = extract_triangles_in_group(mesh, group);
                triangles_per_group.extend(triangles);
            }
        }

        // panic!("testing");

        triangles_per_group
    }

    fn get_seed_face_3(
        a: &Mesh<T, N>,
        b: &Mesh<T, N>,
        tree_b: &AabbTree<T, N, Point<T, N>, usize>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
        boundary_faces: &AHashSet<usize>,
        include_on_surface: bool,
    ) -> Option<(usize, usize)> {
        for (seg_idx, seg) in intersection_segments.iter().enumerate() {
            let v0 = seg.a.resulting_vertex.unwrap();
            let v1 = seg.b.resulting_vertex.unwrap();

            if let Some(&he) = a.edge_map.get(&(v0, v1)) {
                for &face_id in &[
                    a.half_edges[he].face,
                    a.half_edges[a.half_edges[he].twin].face,
                ] {
                    if let Some(f) = face_id {
                        if boundary_faces.contains(&f) {
                            let centroid = a.face_centroid_fast(f);
                            match b.point_in_mesh_3(tree_b, &centroid) {
                                PointInMeshResult::Inside => return Some((seg_idx, f)),
                                PointInMeshResult::OnSurface if include_on_surface => {
                                    return Some((seg_idx, f));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        None
    }

    fn get_seed_face_2(
        a: &Mesh<T, N>,
        b: &Mesh<T, N>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
        boundary_faces: &AHashSet<usize>,
    ) -> Option<(usize, usize)> {
        for (seg_idx, seg) in intersection_segments.iter().enumerate() {
            let v0 = seg.a.resulting_vertex.unwrap();
            let v1 = seg.b.resulting_vertex.unwrap();

            if let Some(&he) = a.edge_map.get(&(v0, v1)) {
                for &face_id in &[
                    a.half_edges[he].face,
                    a.half_edges[a.half_edges[he].twin].face,
                ] {
                    if let Some(f) = face_id {
                        if boundary_faces.contains(&f) {
                            let centroid = a.face_centroid_fast(f);
                            if b.point_in_mesh_2(&centroid.as_point_2()) {
                                return Some((seg_idx, f));
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

pub fn remove_duplicate_and_invalid_segments<T: Scalar + Eq + Hash, const N: usize>(
    segments: &mut Vec<IntersectionSegment<T, N>>,
) where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Segment<T, N>: SegmentOps<T, N>,
{
    use ahash::AHashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Canonical, unordered hash-pair key (still verify equality on collisions)
    #[derive(Copy, Clone, Eq, PartialEq, Hash)]
    struct Key {
        a: u64,
        b: u64,
    }
    #[inline(always)]
    fn point_hash<TT: Hash>(p: &TT) -> u64 {
        let mut h = DefaultHasher::new();
        p.hash(&mut h);
        h.finish()
    }
    #[inline(always)]
    fn make_key<TT: Scalar, const NN: usize>(p: &Point<TT, NN>, q: &Point<TT, NN>) -> Key {
        let ha = point_hash(p);
        let hb = point_hash(q);
        if ha <= hb {
            Key { a: ha, b: hb }
        } else {
            Key { a: hb, b: ha }
        }
    }

    // For each hash key, keep a small bucket of indices whose endpoints truly match
    // (prevents false positives if different segments collide on hashes)
    let mut buckets: AHashMap<Key, smallvec::SmallVec<[usize; 4]>> = AHashMap::new();

    // We’ll mark invalidation after the scan
    let mut invalidate = vec![false; segments.len()];

    for (i, seg) in segments.iter().enumerate() {
        let a_ref = &seg.segment.a;
        let b_ref = &seg.segment.b;
        let key = make_key(a_ref, b_ref);

        let entry = buckets.entry(key).or_default();

        // Try to find a true duplicate in the bucket (check actual endpoints)
        let mut dup_of: Option<usize> = None;
        for &j in entry.iter() {
            let sj = &segments[j];
            let same = (sj.segment.a == *a_ref && sj.segment.b == *b_ref)
                || (sj.segment.a == *b_ref && sj.segment.b == *a_ref);
            if same {
                dup_of = Some(j);
                break;
            }
        }

        match dup_of {
            None => {
                // First time we see this undirected segment → keep it (for now)
                entry.push(i);
            }
            Some(j) => {
                // Duplicate found. Prefer COPLANAR over non-coplanar.
                match (segments[j].coplanar, seg.coplanar) {
                    (false, true) => {
                        // Invalidate the previous non-coplanar; keep current coplanar
                        invalidate[j] = true;
                        // replace j with i in the bucket so further dupes compare against the kept one
                        if let Some(slot) = entry.iter_mut().find(|slot| **slot == j) {
                            *slot = i;
                        }
                    }
                    _ => {
                        // Current should be invalidated (either both coplanar or previous is coplanar)
                        invalidate[i] = true;
                    }
                }
            }
        }
    }

    // Apply marks
    for (i, inv) in invalidate.into_iter().enumerate() {
        segments[i].invalidated = inv;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CoplanarTriangle {
    verts: [usize; 3], // (a,b,c) with a<b<c
}

fn ordered(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn extract_triangles_in_group<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    group: &[IntersectionSegment<T, N>],
) -> Vec<CoplanarTriangle>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) adjacency and one seg index per undirected edge
    let mut adj: AHashMap<usize, AHashSet<usize>> = AHashMap::new();
    let mut edge_to_seg: AHashMap<(usize, usize), usize> = AHashMap::new();

    for (ei, seg) in group.iter().enumerate() {
        let [v0, v1] = [
            seg.a.resulting_vertex.unwrap(),
            seg.b.resulting_vertex.unwrap(),
        ];
        adj.entry(v0).or_default().insert(v1);
        adj.entry(v1).or_default().insert(v0);
        edge_to_seg.entry(ordered(v0, v1)).or_insert(ei);
    }

    // quick accessor for point coords
    let get_p = |i: usize| &mesh.vertices[i].position;

    // 2) enumerate unique triangles a<b<c s.t. (a,b), (a,c), (b,c) are edges
    let mut verts: Vec<usize> = adj.keys().copied().collect();
    verts.sort_unstable();

    let mut out = Vec::new();

    for (_, &a) in verts.iter().enumerate() {
        // neighbors of a
        let mut na: Vec<usize> = adj[&a].iter().copied().collect();
        na.sort_unstable();

        for (bi, &b) in na.iter().enumerate() {
            if b <= a {
                continue;
            }

            // candidates c are neighbors of a with c>b
            for &c in na.iter().skip(bi + 1) {
                if c <= b {
                    continue;
                }

                // edge (b,c) must exist to close the triangle
                if !adj.get(&b).map_or(false, |nb| nb.contains(&c)) {
                    continue;
                }

                // 3) reject collinear (degenerate) triangles
                let ab = (get_p(b) - get_p(a)).as_vector();
                let ac = (get_p(c) - get_p(a)).as_vector();

                let cr = vector_cross(&ab, &ac);

                let area2 = cr.dot(&cr); // squared area
                if area2 == T::zero() {
                    // exact-zero ok for rationals
                    continue;
                }

                // 4) pick one segment index per edge
                let e_ab = *edge_to_seg.get(&ordered(a, b)).unwrap();
                let e_bc = *edge_to_seg.get(&ordered(b, c)).unwrap();
                let e_ca = *edge_to_seg.get(&ordered(c, a)).unwrap();

                // (paranoia) ensure distinct segment indices
                if e_ab == e_bc || e_bc == e_ca || e_ab == e_ca {
                    continue;
                }

                out.push(CoplanarTriangle { verts: [a, b, c] });
            }
        }
    }

    out
}

#[inline(always)]
fn bucket_u(u: f64, eps: f64) -> i64 {
    // Quantize to tolerance-sized bins
    ((u / eps).round() as i64).clamp(i64::MIN / 4, i64::MAX / 4)
}

/// 1) Allocate global vertices for all splits (edge OR face) without touching topology.
///    - Reuses your `ApproxPointKey` to dedup.
///    - Updates intersection endpoint vertex_hints in-place.
///    - Returns a map ApproxPointKey -> new global vertex id (for convenience).
pub fn allocate_vertices_for_splits_no_topology<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
) -> AHashMap<ApproxPointKey, usize>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    // Map keyed by (canonical_point_key, half_edge_hint_or_MAX, face_hint_or_MAX, endpoint_flag) -> allocated global vid.
    // endpoint_flag = 1 when the split is essentially at an edge endpoint (u ~= 0 or u ~= 1),
    //               = 0 for interior-of-edge, face-interior, or no-edge cases.
    let mut keyed_map: AHashMap<(ApproxPointKey, (usize, usize), [usize; 2], u8), usize> =
        AHashMap::default();

    // Returned convenience map: canonical_point_key -> one representative vid for *no-edge* bucket only
    let mut canonical_map: AHashMap<ApproxPointKey, usize> = AHashMap::default();
    let mut edge_param_map: AHashMap<((usize, usize), i64), usize> = AHashMap::default();

    // threshold for treating half_edge_u_hint as an endpoint
    let tol_t: T = T::point_merge_threshold();
    let tol_f64: f64 = RefInto::<CgarF64>::ref_into(&tol_t).0;

    for seg in intersection_segments.iter_mut() {
        if seg.invalidated {
            continue;
        }
        for (i, ep) in [&mut seg.a, &mut seg.b].iter().enumerate() {
            let ep_pos = &seg.segment[i];
            let canonical = point_key(&ep_pos);

            // Extract edge context
            let he_opt = ep.half_edge_hint;
            let edge_key: (usize, usize) = match he_opt {
                Some(h) => mesh
                    .canonical_edge_indices(h)
                    .unwrap_or((usize::MAX, usize::MAX)),
                None => (usize::MAX, usize::MAX),
            };

            // Extract face context
            let faces_opt = ep.faces_hint;
            let faces_key = faces_opt.unwrap_or([usize::MAX, usize::MAX]);

            // Determine u along the edge if available
            let u_opt_f64: Option<f64> = ep
                .half_edge_u_hint
                .as_ref()
                .map(|u_t| RefInto::<CgarF64>::ref_into(u_t).0);

            // endpoint_flag = 1 if u is near 0 or near 1 (use tol), else 0
            let endpoint_flag: u8 = match u_opt_f64 {
                Some(u) if (u <= tol_f64) || (u >= 1.0 - tol_f64) => 1,
                _ => 0,
            };
            let map_key = (canonical, edge_key, faces_key, endpoint_flag);

            // only accept an existing vertex_hint as authoritative when
            // it is no-edge (he_key == MAX) OR the endpoint is effectively at the edge endpoint (endpoint_flag == 1).
            if let Some(hint) = ep.vertex_hint {
                if hint[0] != usize::MAX
                    && (edge_key == (usize::MAX, usize::MAX) || endpoint_flag == 1)
                {
                    keyed_map.entry(map_key).or_insert(hint[0]);
                    // don't populate canonical_map here unless it's the no-edge case
                    if edge_key == (usize::MAX, usize::MAX) && faces_key == [usize::MAX, usize::MAX]
                    {
                        canonical_map.entry(canonical).or_insert(hint[0]);
                    }
                    continue;
                }
            }

            // 1b) If this map_key already decided, reuse
            if let Some(&_vid) = keyed_map.get(&map_key) {
                continue;
            }

            // 1c) If endpoint_flag==1 and we have half-edge hint, reuse the proper endpoint vertex:
            if endpoint_flag == 1 && he_opt.is_some() {
                let he = he_opt.unwrap();
                if he < mesh.half_edges.len() {
                    // decide whether this is source (u≈0) or dest (u≈1)
                    if let Some(u) = u_opt_f64 {
                        let chosen_vid = if u <= tol_f64 {
                            // source vertex = previous half-edge's vertex
                            let src_he = mesh.half_edges[he].prev;
                            mesh.half_edges[src_he].vertex
                        } else {
                            // dest vertex
                            mesh.half_edges[he].vertex
                        };
                        keyed_map.insert(map_key, chosen_vid);
                        // don't set canonical_map for edge-derived allocations
                        continue;
                    }
                }
            }

            // 1d) Interior-of-edge (he present, not near endpoints): one vertex per (undirected edge, canonical u bucket)
            if let (Some(he), Some(u_raw)) = (he_opt, u_opt_f64) {
                if endpoint_flag == 0 {
                    if let Some(ek) = mesh.canonical_edge_indices(he) {
                        let u_can = mesh.canonicalize_u_for_edge(he, ek, u_raw);
                        let ub = bucket_u(u_can, tol_f64);
                        let k = (ek, ub);

                        // Reuse if we’ve already allocated on this edge/param
                        let vid = *edge_param_map.entry(k).or_insert_with(|| {
                            let (vid, _existed) = mesh.get_or_insert_vertex(&ep_pos);
                            vid
                        });

                        // Record under your original keyed_map so write-back can find it with this endpoint’s map_key
                        keyed_map.insert(map_key, vid);
                        // no canonical_map update for edge-derived allocations
                        continue;
                    }
                }
            }

            // 1e) If face-interior (face_hint + barycentric_hint), allocate one vertex per (canonical,MAX,face_key,0)
            if faces_opt.is_some() && ep.barycentric_hint.is_some() {
                let map_key_face = (canonical, (usize::MAX, usize::MAX), faces_key, 0u8);
                if let Some(&_vid) = keyed_map.get(&map_key_face) {
                    // already allocated for this canonical/face combination
                    continue;
                }
                let new_vid = mesh.get_or_insert_vertex(&ep_pos);
                keyed_map.insert(map_key_face, new_vid.0);
                // do not set canonical_map here (avoid cross-face fallback)
                continue;
            }

            // 1f) No half-edge or face context -> bucket by canonical key (allocate if needed)
            let map_key_nocontext = (
                canonical,
                (usize::MAX, usize::MAX),
                [usize::MAX, usize::MAX],
                0u8,
            );
            if let Some(&vid) = keyed_map.get(&map_key_nocontext) {
                // already allocated for canonical/no-context
                canonical_map.entry(canonical).or_insert(vid);
                continue;
            }
            let new_vid = mesh.get_or_insert_vertex(&ep_pos);
            keyed_map.insert(map_key_nocontext, new_vid.0);
            canonical_map.entry(canonical).or_insert(new_vid.0);
        }
    }

    // 2) Write back vertex_hint for every endpoint referenced by splits.
    //    We set vertex_hint = [vid, usize::MAX] and clear half_edge_hint / u / face / barycentric.
    //    IMPORTANT: do not fallback to canonical_map for arbitrary (canonical,he,face) keys to avoid merging
    //    interior-of-different-half-edge or interior-of-different-face points into a single canonical representative.
    for seg in intersection_segments.iter_mut() {
        if seg.invalidated {
            continue;
        }

        for (i, ep) in [&mut seg.a, &mut seg.b].iter_mut().enumerate() {
            let ep_pos = &seg.segment[i];
            let canonical = point_key(&ep_pos);

            let he_opt = ep.half_edge_hint;
            let edge_key: (usize, usize) = match he_opt {
                Some(h) => mesh
                    .canonical_edge_indices(h)
                    .unwrap_or((usize::MAX, usize::MAX)),
                None => (usize::MAX, usize::MAX),
            };
            let faces_opt = ep.faces_hint;
            let faces_key = faces_opt.unwrap_or([usize::MAX, usize::MAX]);

            let u_opt_f64: Option<f64> = ep
                .half_edge_u_hint
                .as_ref()
                .map(|u_t| RefInto::<CgarF64>::ref_into(u_t).0);
            let endpoint_flag: u8 = match u_opt_f64 {
                Some(u) if (u <= tol_f64) || (u >= 1.0 - tol_f64) => 1,
                _ => 0,
            };

            let faces_key_for_map = if edge_key != (usize::MAX, usize::MAX) {
                [usize::MAX, usize::MAX]
            } else {
                faces_key
            };
            // prefer the exact (canonical,he,face,endpoint_flag) bucket; fall back to canonical/nocontext only
            let map_key = (canonical, edge_key, faces_key_for_map, endpoint_flag);
            let map_key_nocontext = (canonical, (usize::MAX, usize::MAX), [usize::MAX; 2], 0u8);

            let vid = keyed_map
                .get(&map_key)
                .copied()
                .or_else(|| keyed_map.get(&map_key_nocontext).copied())
                // do NOT use canonical_map as general fallback here
                .unwrap_or_else(|| {
                    // If still missing, allocate a fresh vertex for this exact endpoint to avoid accidental merges.
                    mesh.get_or_insert_vertex(&ep_pos).0
                });

            // Set vertex_hint to vid (slot 0) and clear other geometric hints
            ep.vertex_hint = Some([vid, usize::MAX]);
            if ep.half_edge_hint.is_none() && ep.faces_hint.is_none() {
                ep.faces_hint = Some([seg.initial_face_reference, usize::MAX]);
            }
            ep.half_edge_hint = None;
            ep.half_edge_u_hint = None;
            ep.barycentric_hint = None;
        }
    }

    // 3) Update each segment's resulting_vertices_pair from endpoint vertex_hint
    for seg in intersection_segments.iter_mut() {
        let a_vid = seg.a.vertex_hint.map(|h| h[0]).unwrap_or(usize::MAX);
        let b_vid = seg.b.vertex_hint.map(|h| h[0]).unwrap_or(usize::MAX);
        seg.a.resulting_vertex = Some(a_vid);
        seg.b.resulting_vertex = Some(b_vid);
    }

    canonical_map
}

pub fn rewrite_faces_from_cdt_batch<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    jobs: &[FaceJobUV<T>],
    cdts: &[Delaunay<T>],
) where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let total_triangles: usize = cdts.iter().map(|dt| dt.triangles.len()).sum();
    let mut face_ids = Vec::with_capacity(jobs.len());
    let mut triangles = Vec::with_capacity(total_triangles);

    for ((job, dt), _) in jobs.iter().zip(cdts.iter()).zip(0..) {
        face_ids.push(job.face_id);
        let verts_global = &job.verts_global;

        for t in &dt.triangles {
            triangles.push((verts_global[t.0], verts_global[t.1], verts_global[t.2]));
        }
    }

    mesh.remove_triangles_deferred(&face_ids);
    mesh.add_triangles_deferred(&triangles);
}
