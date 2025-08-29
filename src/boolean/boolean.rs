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
use smallvec::*;
use std::collections::hash_map::Entry;

use crate::{
    geometry::{
        Aabb, AabbTree,
        plane::Plane,
        point::{Point, PointOps},
        segment::{Segment, SegmentOps},
        spatial_element::SpatialElement,
        tri_tri_intersect::{
            ContactOnTri, EndpointInfo, TriPrecomp, TriTriIntersectionDetailed,
            TriTriIntersectionResult, tri_tri_intersection, tri_tri_intersection_with_precomp,
            tri_tri_intersection_with_precomp_detailed, tri_tri_intersection_with_precomp_p,
        },
        util::point_from_segment_and_u,
        vector::{Vector, VectorOps},
    },
    io::obj::write_obj,
    kernel,
    mesh::{
        basic_types::{Mesh, PointInMeshResult, VertexSource},
        intersection_segment::{IntersectionEndPoint, IntersectionSegment},
        topology::{FindFaceResult, VertexRayResult},
    },
    numeric::{cgar_f64::CgarF64, scalar::Scalar},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ApproxPointKey {
    qx: i64,
    qy: i64,
    qz: i64, // use only N you need
}

fn point_key<T: Scalar, const N: usize>(p: &Point<T, N>) -> ApproxPointKey {
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
    segment_idx: usize,
    endpoint_idx: usize, // 0 or 1
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitType {
    Edge,
    Face,
}

pub struct Splits<T: Scalar, const N: usize> {
    pub splits:
        AHashMap<ApproxPointKey, (Point<T, N>, SplitType, usize, SmallVec<[EndPointHandle; 2]>)>,
}

impl<T: Scalar, const N: usize> Splits<T, N> {
    pub fn new() -> Self {
        Self {
            splits: AHashMap::new(),
        }
    }
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
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    fn classify_faces_inside_intersection_loops(
        &mut self,
        other: &Mesh<T, N>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
        coplanar_triangles: &Vec<CoplanarTriangle>,
        include_on_surface: bool,
    ) -> Vec<bool> {
        // build adjacency & boundary‐map
        let adj = self.build_face_adjacency_graph();

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
                let c = self.face_centroid(f).0;
                match other.point_in_mesh(&tree_b, &c) {
                    PointInMeshResult::Inside => inside[f] = true,
                    PointInMeshResult::OnSurface if include_on_surface => inside[f] = true,
                    _ => {}
                }
            }
            return inside;
        } else {
            let boundary_map = self.build_boundary_map(intersection_segments);

            let mut boundary_faces = AHashSet::new();
            for &(a, b) in &boundary_map {
                boundary_faces.insert(a);
                boundary_faces.insert(b);
            }

            // 3) pick a seed face that lies inside B
            let (_seed_intersection_idx, selected_face) = Self::get_seed_face(
                &self,
                &other,
                &tree_b,
                intersection_segments,
                &boundary_faces,
                include_on_surface,
            );

            let mut face_pairs: AHashMap<usize, Vec<usize>> = AHashMap::new();
            for seg_idx in 0..intersection_segments.len() {
                let seg = &intersection_segments[seg_idx];
                let he = self
                    .edge_map
                    .get(&(
                        seg.resulting_vertices_pair[0],
                        seg.resulting_vertices_pair[1],
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
                if let Some(neighbors) = adj.get(&curr) {
                    // grab the list of “paired” faces for this curr
                    let paired = face_pairs.get(&curr);

                    for &nbr in neighbors {
                        if visited[nbr] {
                            continue;
                        }

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

    fn create_intersection_segment(
        mesh: &Mesh<T, N>,
        splits: &mut Splits<T, N>,
        container: &mut Vec<IntersectionSegment<T, N>>,
        segment: &Segment<T, N>,
        main_face: usize,
        coplanar: bool,
        skip_endpoint_b: bool,
    ) {
        // Reserve slot
        let seg_idx = container.len();
        container.push(IntersectionSegment::new(
            IntersectionEndPoint::new_default(),
            IntersectionEndPoint::new_default(),
            &segment,
            main_face,
            [usize::MAX, usize::MAX],
            coplanar,
        ));

        #[inline(always)]
        fn he_for_face_edge<T: Scalar, const N: usize>(
            mesh: &Mesh<T, N>,
            face: usize,
            u: usize,
            v: usize,
        ) -> usize {
            if let Some(&he) = mesh.edge_map.get(&(u, v)) {
                if mesh.half_edges[he].face == Some(face) {
                    return he;
                }
                let twin = mesh.half_edges[he].twin;
                if mesh.half_edges[twin].face == Some(face) {
                    return twin;
                }
            }
            if let Some(&he) = mesh.edge_map.get(&(v, u)) {
                if mesh.half_edges[he].face == Some(face) {
                    return he;
                }
                let twin = mesh.half_edges[he].twin;
                if mesh.half_edges[twin].face == Some(face) {
                    return twin;
                }
            }
            panic!(
                "Face edge half-edge not found for ({u},{v}) on face {face}\n{:?}\n{:?}",
                mesh.vertices[u].position, mesh.vertices[v].position
            );
        }

        // Precompute per-call constants/refs once
        let tol = T::point_merge_threshold();
        let tol2 = &tol * &tol;
        let [ia, ib, ic] = mesh.face_vertices(main_face);
        let pa = &mesh.vertices[ia].position;
        let pb = &mesh.vertices[ib].position;
        let pc = &mesh.vertices[ic].position;
        let he_ab = he_for_face_edge(mesh, main_face, ia, ib);
        let he_bc = he_for_face_edge(mesh, main_face, ib, ic);
        let he_ca = he_for_face_edge(mesh, main_face, ic, ia);

        // Precompute quantizer scale once
        let scale: CgarF64 = (&tol).ref_into();
        let q_scale = 1.0 / scale.0;

        #[inline(always)]
        fn attach_handle_or_classify<T: Scalar, const N: usize>(
            mesh: &Mesh<T, N>,
            splits: &mut Splits<T, N>,
            container: &mut Vec<IntersectionSegment<T, N>>,
            seg_idx: usize,
            endpoint_idx: usize,
            face: usize,
            pa: &Point<T, N>,
            pb: &Point<T, N>,
            pc: &Point<T, N>,
            ia: usize,
            ib: usize,
            ic: usize,
            he_ab: usize,
            he_bc: usize,
            he_ca: usize,
            tol: &T,
            tol2: &T,
            q_scale: f64,
            pos: &Point<T, N>,
        ) where
            Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
            Vector<T, N>: VectorOps<T, N>,
            for<'a> &'a T: Sub<&'a T, Output = T>
                + Mul<&'a T, Output = T>
                + Add<&'a T, Output = T>
                + Div<&'a T, Output = T>
                + Neg<Output = T>,
        {
            // Cache probe
            let key_q = point_key(pos);
            match splits.splits.entry(key_q) {
                Entry::Occupied(mut occ) => {
                    occ.get_mut().3.push(EndPointHandle {
                        segment_idx: seg_idx,
                        endpoint_idx,
                    });
                    return;
                }
                Entry::Vacant(_) => {}
            }

            // Vertex snaps (exact zero preferred to avoid LazyExact eval)
            let da = (pos - pa).as_vector().dot(&(pos - pa).as_vector());
            if da.is_zero() {
                container[seg_idx][endpoint_idx].vertex_hint = Some([ia, usize::MAX]);
                return;
            }
            let db = (pos - pb).as_vector().dot(&(pos - pb).as_vector());
            if db.is_zero() {
                container[seg_idx][endpoint_idx].vertex_hint = Some([ib, usize::MAX]);
                return;
            }
            let dc = (pos - pc).as_vector().dot(&(pos - pc).as_vector());
            if dc.is_zero() {
                container[seg_idx][endpoint_idx].vertex_hint = Some([ic, usize::MAX]);
                return;
            }

            #[inline(always)]
            fn try_edge<T: Scalar, const N: usize>(
                mesh: &Mesh<T, N>,
                splits: &mut Splits<T, N>,
                container: &mut Vec<IntersectionSegment<T, N>>,
                seg_idx: usize,
                endpoint_idx: usize,
                face: usize,
                pu_idx: usize,
                pv_idx: usize,
                he: usize,
                p: &Point<T, N>,
                tol: &T,
                tol2: &T,
                q_scale: f64,
            ) -> bool
            where
                Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
                Vector<T, N>: VectorOps<T, N>,
                for<'a> &'a T: Sub<&'a T, Output = T>
                    + Mul<&'a T, Output = T>
                    + Add<&'a T, Output = T>
                    + Div<&'a T, Output = T>
                    + Neg<Output = T>,
            {
                let pu = &mesh.vertices[pu_idx].position;
                let pv = &mesh.vertices[pv_idx].position;
                let uv = (pv - pu).as_vector();
                let uv2 = uv.dot(&uv);
                if uv2.is_zero() {
                    return false;
                }

                let up = (p - pu).as_vector();
                let t = &up.dot(&uv) / &uv2;

                // Range check with tol slack
                if (&t + tol).is_negative() || (&(&t - &T::one()) - tol).is_positive() {
                    return false;
                }

                // Distance check
                let closest = pu + &(uv.scale(&t)).0;
                let diff = (p - &closest).as_vector();
                let d2 = diff.dot(&diff);
                if (&d2 - tol2).is_positive() {
                    return false;
                }

                // Snap to endpoints if within tol
                if (&t).is_zero() {
                    container[seg_idx][endpoint_idx].vertex_hint = Some([pu_idx, usize::MAX]);
                    return true;
                }
                if (&(&t - &T::one())).is_zero() {
                    container[seg_idx][endpoint_idx].vertex_hint = Some([pv_idx, usize::MAX]);
                    return true;
                }

                // Record edge split hints
                container[seg_idx][endpoint_idx].vertex_hint = Some([
                    mesh.half_edges[he].vertex,
                    mesh.half_edges[mesh.half_edges[he].twin].vertex,
                ]);
                container[seg_idx][endpoint_idx].half_edge_hint = Some(he);
                container[seg_idx][endpoint_idx].half_edge_u_hint = Some(t.clone());

                // Quantize pu + uv*t without allocating a Point unless we insert
                let to_i64 = |val: &T| -> i64 {
                    let cf: CgarF64 = val.ref_into();
                    (cf.0 * q_scale).round() as i64
                };

                let mut qx = 0i64;
                let mut qy = 0i64;
                let mut qz = 0i64;
                for i in 0..N.min(3) {
                    let a = &pu[i];
                    let b = &pv[i];
                    let comp = a + &((&(b - a)) * &t);
                    let qi = to_i64(&comp);
                    if i == 0 {
                        qx = qi
                    } else if i == 1 {
                        qy = qi
                    } else {
                        qz = qi
                    }
                }

                let key_snap = ApproxPointKey {
                    qx,
                    qy: if N > 1 { qy } else { 0 },
                    qz: if N > 2 { qz } else { 0 },
                };

                match splits.splits.entry(key_snap) {
                    Entry::Occupied(mut occ) => {
                        occ.get_mut().3.push(EndPointHandle {
                            segment_idx: seg_idx,
                            endpoint_idx,
                        });
                    }
                    Entry::Vacant(vac) => {
                        // Allocate snap Point only on insert
                        let snap = pu + &(uv.scale(&t)).0;
                        vac.insert((snap, SplitType::Edge, he, {
                            let mut v: SmallVec<[EndPointHandle; 2]> = SmallVec::new();
                            v.push(EndPointHandle {
                                segment_idx: seg_idx,
                                endpoint_idx,
                            });
                            v
                        }));
                    }
                }
                true
            }

            // Try each edge once (precomputed half-edges)
            if try_edge(
                mesh,
                splits,
                container,
                seg_idx,
                endpoint_idx,
                face,
                ia,
                ib,
                he_ab,
                pos,
                tol,
                tol2,
                q_scale,
            ) {
                return;
            }
            if try_edge(
                mesh,
                splits,
                container,
                seg_idx,
                endpoint_idx,
                face,
                ib,
                ic,
                he_bc,
                pos,
                tol,
                tol2,
                q_scale,
            ) {
                return;
            }
            if try_edge(
                mesh,
                splits,
                container,
                seg_idx,
                endpoint_idx,
                face,
                ic,
                ia,
                he_ca,
                pos,
                tol,
                tol2,
                q_scale,
            ) {
                return;
            }

            // Interior face split
            container[seg_idx][endpoint_idx].face_hint = Some(face);
            splits.splits.insert(
                key_q,
                (pos.clone(), SplitType::Face, face, {
                    let mut v: SmallVec<[EndPointHandle; 2]> = SmallVec::new();
                    v.push(EndPointHandle {
                        segment_idx: seg_idx,
                        endpoint_idx,
                    });
                    v
                }),
            );
        }

        // Endpoint A
        attach_handle_or_classify(
            mesh, splits, container, seg_idx, 0, main_face, pa, pb, pc, ia, ib, ic, he_ab, he_bc,
            he_ca, &tol, &tol2, q_scale, &segment.a,
        );

        // Endpoint B
        if !skip_endpoint_b {
            attach_handle_or_classify(
                mesh, splits, container, seg_idx, 1, main_face, pa, pb, pc, ia, ib, ic, he_ab,
                he_bc, he_ca, &tol, &tol2, q_scale, &segment.b,
            );
        }
    }

    fn boolean_split(
        mesh: &mut Mesh<T, N>,
        tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        splits: &mut Splits<T, N>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    ) {
        for (_key, endpoint_tup) in &splits.splits {
            if endpoint_tup.1 == SplitType::Edge {
                let result = mesh
                    .split_edge(tree, endpoint_tup.2, &endpoint_tup.0, false)
                    .expect("Failed to split edge");
                for endpoint in endpoint_tup.3.iter() {
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .vertex_hint = Some([result.vertex, usize::MAX]);
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .half_edge_hint = None;
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .half_edge_u_hint = None;
                }
            } else {
                let result = mesh
                    .split_face(tree, endpoint_tup.2, &endpoint_tup.0, false)
                    .expect("Failed to split face");
                for endpoint in endpoint_tup.3.iter() {
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .vertex_hint = Some([result.vertex, usize::MAX]);
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx].face_hint =
                        None;
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .barycentric_hint = None;
                }
            }
        }
    }

    pub fn corefine_and_boolean(&mut self, other: &mut Mesh<T, N>, op: BooleanOp) -> Mesh<T, N> {
        let mut a = self;
        let mut b = other;

        // 1. Collect ALL intersection segments
        let mut intersection_segments_a = Vec::new();
        let mut intersection_segments_b = Vec::new();
        let intersection_segments = [&mut intersection_segments_a, &mut intersection_segments_b];

        let start = Instant::now();
        let mut tree_a = a.build_face_tree();
        let mut tree_b = b.build_face_tree();
        println!("Total AABB computation: {:.2?}", start.elapsed());

        let mut splits_a = Splits::new();
        let mut splits_b = Splits::new();

        let splits = [&mut splits_a, &mut splits_b];
        let meshes = [&a, &b];

        let start = Instant::now();
        let mut candidates = Vec::new();
        let mut ends_vec = Vec::new();

        for fa in 0..a.faces.len() {
            candidates.clear();
            tree_b.query(&a.face_aabb(fa), &mut candidates);

            let pa_idx = a.face_vertices(fa);
            let pa: [&Point<T, N>; 3] = from_fn(|i| &a.vertices[pa_idx[i]].position);
            let pre_a = TriPrecomp::new(&pa);

            for &fb in &candidates {
                ends_vec.clear();

                let pb_idx = b.face_vertices(*fb);
                let pb: [&Point<T, N>; 3] = from_fn(|i| &b.vertices[pb_idx[i]].position);
                let pre_b = TriPrecomp::new(&pb);

                let vertices_indices = [&pa_idx, &pb_idx];
                let faces = [fa, *fb];

                let mut split_idx_info = usize::MAX;

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
                    for mesh_i in 0..2 {
                        let on_x = if mesh_i == 0 {
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
                            let mut split_type = None;
                            match &on_x[end_x] {
                                ContactOnTri::Vertex(i) => {
                                    intersection_endpoints[end_x].vertex_hint =
                                        Some([vertices_indices[mesh_i][*i], usize::MAX]);
                                }
                                ContactOnTri::Edge { e, u } => {
                                    intersection_endpoints[end_x].vertex_hint = Some([
                                        vertices_indices[mesh_i][e.0],
                                        vertices_indices[mesh_i][e.1],
                                    ]);

                                    let he = meshes[mesh_i].edge_map[&(
                                        vertices_indices[mesh_i][e.0],
                                        vertices_indices[mesh_i][e.1],
                                    )];
                                    intersection_endpoints[end_x].half_edge_hint = Some(he);
                                    intersection_endpoints[end_x].half_edge_u_hint =
                                        Some(u.clone());

                                    split_type = Some(SplitType::Edge);
                                    split_idx_info = he;
                                }
                                ContactOnTri::Interior { bary } => {
                                    intersection_endpoints[end_x].face_hint = Some(faces[mesh_i]);
                                    intersection_endpoints[end_x].barycentric_hint =
                                        Some(bary.clone());

                                    split_type = Some(SplitType::Face);
                                    split_idx_info = faces[mesh_i];
                                }
                            }

                            if let Some(split_type) = split_type {
                                let key = point_key(&ends[end_x].point);
                                match splits[mesh_i].splits.entry(key) {
                                    Entry::Occupied(mut occ) => {
                                        occ.get_mut().3.push(EndPointHandle {
                                            segment_idx: intersection_segments[mesh_i].len(),
                                            endpoint_idx: end_x,
                                        });
                                        continue;
                                    }
                                    Entry::Vacant(_) => {}
                                }

                                match splits[mesh_i].splits.entry(key) {
                                    Entry::Occupied(mut occ) => {
                                        occ.get_mut().3.push(EndPointHandle {
                                            segment_idx: intersection_segments[mesh_i].len(),
                                            endpoint_idx: end_x,
                                        });
                                    }
                                    Entry::Vacant(vac) => {
                                        vac.insert((
                                            ends[end_x].point.clone(),
                                            split_type,
                                            split_idx_info,
                                            {
                                                let mut v: SmallVec<[EndPointHandle; 2]> =
                                                    SmallVec::new();
                                                v.push(EndPointHandle {
                                                    segment_idx: intersection_segments[mesh_i]
                                                        .len(),
                                                    endpoint_idx: end_x,
                                                });
                                                v
                                            },
                                        ));
                                    }
                                }
                            }
                        }
                        intersection_segments[mesh_i].push(IntersectionSegment::new(
                            intersection_endpoint_0,
                            intersection_endpoint_1,
                            &Segment::new(&ends[0].point, &ends[1].point),
                            faces[mesh_i],
                            [usize::MAX, usize::MAX],
                            coplanar,
                        ));
                    }
                }
            }
        }

        println!("Intersections created in: {:.2?}", start.elapsed());

        // Remove duplicate segments from both lists.
        // remove_duplicate_segments(&mut intersection_segments_a);
        // remove_duplicate_segments(&mut intersection_segments_b);

        println!("Splits on A ({})", splits_a.splits.len());
        let start = Instant::now();
        Self::boolean_split(
            &mut a,
            &mut tree_a,
            &mut splits_a,
            &mut intersection_segments_a,
        );
        println!("Splits done in {:.2?}", start.elapsed());

        println!("Splits on B ({})", splits_b.splits.len());
        let start = Instant::now();
        Self::boolean_split(
            &mut b,
            &mut tree_b,
            &mut splits_b,
            &mut intersection_segments_b,
        );
        println!("Splits done in {:.2?}", start.elapsed());

        let start = Instant::now();
        tree_a = a.build_face_tree();
        tree_b = b.build_face_tree();
        println!("Rebuilt the face trees in {:.2?}", start.elapsed());

        let mut intersection_by_edge_a = AHashMap::new();
        let mut intersection_by_edge_b = AHashMap::new();

        println!("Processing remaining segments A");
        let start = Instant::now();
        let mut i = 0;

        while i < intersection_segments_a.len() {
            process_segment_and_edge_map(
                &mut a,
                &mut intersection_segments_a,
                &mut intersection_by_edge_a,
                i,
                &mut splits_a,
                &mut tree_a,
            );
            i += 1;
        }

        println!("Processing remaining segments B");
        i = 0;
        while i < intersection_segments_b.len() {
            process_segment_and_edge_map(
                &mut b,
                &mut intersection_segments_b,
                &mut intersection_by_edge_b,
                i,
                &mut splits_b,
                &mut tree_b,
            );
            i += 1;
        }

        println!("Intersection segments processed in {:.2?}", start.elapsed());

        println!("Finding T-junctions");
        let start = Instant::now();
        let mut edges_a = Vec::new();
        // gather all edges from intersection A
        for seg in &intersection_segments_a {
            if seg.resulting_vertices_pair[0] != usize::MAX
                && seg.resulting_vertices_pair[1] != usize::MAX
            {
                edges_a.push(seg.resulting_vertices_pair);
            }
        }

        let mut edges_b = Vec::new();
        // gather all edges from intersection B
        for seg in &intersection_segments_b {
            if seg.resulting_vertices_pair[0] != usize::MAX
                && seg.resulting_vertices_pair[1] != usize::MAX
            {
                edges_b.push(seg.resulting_vertices_pair);
            }
        }

        println!(
            "Found {} edges on A and {} edges on B",
            edges_a.len(),
            edges_b.len()
        );

        // Naming is a bit confusing here, and it may appear at first that the arguments are inverted.
        // But this is correct. We are getting edges on A that are on a given B segment (edge).
        let t_junctions_a = find_x_vertices_on_y_edges(&b, &a, &edges_b, &edges_a);
        // Same as previous, but inverted.
        let t_junctions_b = find_x_vertices_on_y_edges(&a, &b, &edges_a, &edges_b);
        let duration = start.elapsed();
        println!("Finding T-junctions done in {:.2?}", duration);

        println!("Processing T-junctions");
        let start = Instant::now();
        for t in t_junctions_a {
            process_t_junction(
                &mut a,
                &mut b,
                &mut tree_a,
                &t,
                &mut intersection_segments_a,
                &mut intersection_by_edge_a,
            );
        }

        for t in t_junctions_b {
            process_t_junction(
                &mut b,
                &mut a,
                &mut tree_b,
                &t,
                &mut intersection_segments_b,
                &mut intersection_by_edge_b,
            );
        }
        let duration = start.elapsed();
        println!("Processing T-junctions done in {:.2?}", duration);

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
        let a_classification = a.classify_faces_inside_intersection_loops(
            &b,
            &mut intersection_segments_a,
            &mut a_coplanars,
            true,
        );
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

                let b_classification = b.classify_faces_inside_intersection_loops(
                    &a,
                    &mut intersection_segments_b,
                    &mut b_coplanars,
                    include_on_surface,
                );
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

                let b_classification = b.classify_faces_inside_intersection_loops(
                    &a,
                    &mut intersection_segments_b,
                    &mut b_coplanars,
                    include_on_surface,
                );
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
                let b_classification = b.classify_faces_inside_intersection_loops(
                    &a,
                    &mut intersection_segments_b,
                    &mut b_coplanars,
                    include_on_surface,
                );

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
                    seg.resulting_vertices_pair[0],
                    seg.resulting_vertices_pair[1],
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
            let [v0, v1] = seg.resulting_vertices_pair;
            vertex_to_segments.entry(v0).or_default().push(seg_idx);
            vertex_to_segments.entry(v1).or_default().push(seg_idx);
        }

        // === NON-COPLANAR SEGMENTS: Build double-linked structure ===
        for (seg_idx, seg) in intersection_segments.iter_mut().enumerate() {
            let [v0, v1] = seg.resulting_vertices_pair;
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

    fn check_for_process_segment_early_exit(
        mesh: &Mesh<T, N>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
        segment_idx: usize,
    ) -> bool {
        let mut vertex_ab = [usize::MAX; 2];
        // We check if endpoints of the segment are already represented by vertices.
        for i in 0..2 {
            if intersection_segments[segment_idx][i].vertex_hint.is_some()
                && intersection_segments[segment_idx][i]
                    .half_edge_hint
                    .is_none()
            {
                // If the vertex is already set and no half edge hint is defined, it means we can deterministically choose a vertex.
                vertex_ab[i] = intersection_segments[segment_idx][i].vertex_hint.unwrap()[0];
            }
        }

        // If they are, let's see if they are connected for an early exit.
        if vertex_ab[0] != usize::MAX && vertex_ab[1] != usize::MAX {
            if vertex_ab[0] == vertex_ab[1] {
                // Both endpoints are the same vertex, so we invalidate and skip processing.
                intersection_segments[segment_idx].invalidated = true;
                return true;
            } else {
                let he_ab = mesh.vertices_connection(vertex_ab[0], vertex_ab[1]);
                if he_ab != usize::MAX {
                    // Both endpoints are connected, so we can simply return.
                    intersection_segments[segment_idx].resulting_vertices_pair =
                        [vertex_ab[0], vertex_ab[1]];
                    return true;
                }
            }
        }

        false
    }

    fn process_segment(
        &mut self,
        edge_splits: &mut Splits<T, N>,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
        segment_idx: usize,
    ) {
        let mut vertex_ab = [usize::MAX, usize::MAX];

        if Self::check_for_process_segment_early_exit(self, intersection_segments, segment_idx) {
            return;
        }

        for i in 0..2 {
            if intersection_segments[segment_idx][i]
                .half_edge_hint
                .is_some()
                || intersection_segments[segment_idx][i].face_hint.is_some()
            {
                println!("{}, {:?}", i, intersection_segments[segment_idx]);
                panic!("Shouldn't be using half_edge or face hints at this point.");
            }
        }

        let starting_vertex = intersection_segments[segment_idx][0].vertex_hint.unwrap()[0];
        let direction = intersection_segments[segment_idx].segment.direction();

        let mut found = false;
        for f in self.faces_around_vertex(starting_vertex) {
            match self.cast_ray_from_vertex_in_triangle(f, starting_vertex, &direction) {
                Some(VertexRayResult::EdgeIntersection {
                    half_edge: he,
                    distance: _t,
                    edge_parameter: u,
                }) => {
                    found = true;
                    let start = self.half_edges[self.half_edges[he].prev].vertex;
                    let end = self.half_edges[he].vertex;
                    let new_pos = point_from_segment_and_u(
                        &self.vertices[start].position,
                        &self.vertices[end].position,
                        &u,
                    );
                    let split_result = self.split_edge(aabb_tree, he, &new_pos, false).unwrap();
                    let new_point = &self.vertices[split_result.vertex].position;

                    let new_segment =
                        Segment::new(&new_point, &intersection_segments[segment_idx].segment[1]);

                    let find_face_result = self.find_valid_face(f, &new_point);

                    if let FindFaceResult::OnVertex { f, v: _ } = find_face_result {
                        Self::create_intersection_segment(
                            self,
                            edge_splits,
                            intersection_segments,
                            &new_segment,
                            f,
                            intersection_segments[segment_idx].coplanar,
                            true,
                        );
                    } else {
                        panic!("Invalid face while processing segment.");
                    }

                    let seg = intersection_segments[segment_idx].b.clone();
                    let len = intersection_segments.len() - 1;
                    let new_intersection = &mut intersection_segments[len];
                    new_intersection.b = seg;

                    intersection_segments[segment_idx].segment[1] = new_point.clone();
                    intersection_segments[segment_idx].b.face_hint = None;
                    intersection_segments[segment_idx].b.half_edge_hint = None;
                    intersection_segments[segment_idx].b.half_edge_u_hint = None;
                    intersection_segments[segment_idx].b.vertex_hint =
                        Some([split_result.vertex, usize::MAX]);

                    intersection_segments[segment_idx].resulting_vertices_pair = [
                        intersection_segments[segment_idx].a.vertex_hint.unwrap()[0],
                        split_result.vertex,
                    ];

                    // new_intersection.resulting_vertices_pair = [
                    //     split_result.vertex,
                    //     intersection_segments[segment_idx].a.vertex_hint.unwrap()[0],
                    // ];
                }
                Some(VertexRayResult::CollinearWithEdge(he)) => {
                    println!("collinear with edge {:?}", he);
                    found = true;
                }
                _ => {}
            }

            if found {
                break;
            }
        }

        if !found {
            // for he in
            println!("not found, {:?}", intersection_segments[segment_idx]);

            let mut test_faces = Mesh::new();
            for f in self.faces_around_vertex(starting_vertex) {
                let mut vs = SmallVec::<[usize; 3]>::new();
                for v in self.face_vertices(f) {
                    vs.push(test_faces.add_vertex(self.vertices[v].position.clone()));
                }

                test_faces.add_triangle(vs[0], vs[1], vs[2]);
            }

            let _ = write_obj(&test_faces, "/mnt/v/cgar_meshes/test_faces.obj");

            let mut test_direction_1 = Mesh::new();
            test_direction_1.add_vertex(intersection_segments[segment_idx].segment.a.clone());
            let _ = write_obj(&test_direction_1, "/mnt/v/cgar_meshes/direction_1.obj");

            let mut test_direction_2 = Mesh::new();
            test_direction_2.add_vertex(intersection_segments[segment_idx].segment.b.clone());
            let _ = write_obj(&test_direction_2, "/mnt/v/cgar_meshes/direction_2.obj");

            panic!("Failed to split edge.");
        }
    }

    fn get_seed_face(
        a: &Mesh<T, N>,
        b: &Mesh<T, N>,
        tree_b: &AabbTree<T, N, Point<T, N>, usize>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
        boundary_faces: &AHashSet<usize>,
        include_on_surface: bool,
    ) -> (usize, usize) {
        let mut selected_face = usize::MAX;
        let seed_idx = (0..intersection_segments.len())
            .filter(|&seg_idx| {
                let seg = &intersection_segments[seg_idx];
                let vertex_a = seg.resulting_vertices_pair[0];
                let vertex_b = seg.resulting_vertices_pair[1];
                let he = a
                    .edge_map
                    .get(&(vertex_a, vertex_b))
                    .expect("Edge must exist in edge map");

                let face_0 = a.half_edges[*he].face.expect("Half-edge must have a face");
                if !a.faces[face_0].removed && boundary_faces.contains(&face_0) {
                    return true;
                }

                let face_1 = a.half_edges[a.half_edges[*he].twin]
                    .face
                    .expect("Half-edge must have a face");
                if !a.faces[face_1].removed && boundary_faces.contains(&face_1) {
                    return true;
                }

                false
            })
            .find(|&seg_idx| {
                let seg = &intersection_segments[seg_idx];
                let vertex_a = seg.resulting_vertices_pair[0];
                let vertex_b = seg.resulting_vertices_pair[1];
                let he = a
                    .edge_map
                    .get(&(vertex_a, vertex_b))
                    .expect("Edge must exist in edge map");

                let mut faces = [usize::MAX, usize::MAX];
                faces[0] = a.half_edges[*he].face.expect("Half-edge must have a face");
                faces[1] = a.half_edges[a.half_edges[*he].twin]
                    .face
                    .expect("Half-edge must have a face");

                for f in faces {
                    if a.faces[f].removed {
                        continue;
                    }
                    let c = a.face_centroid(f).0;

                    let point_in_mesh = b.point_in_mesh(&tree_b, &c);
                    if point_in_mesh == PointInMeshResult::Inside {
                        selected_face = f;
                        return true;
                    } else if point_in_mesh == PointInMeshResult::OnSurface && include_on_surface {
                        selected_face = f;
                        return true;
                    }
                }
                false
            })
            .expect("No seed face found inside B");

        (seed_idx, selected_face)
    }
}

pub fn remove_duplicate_segments<T: Scalar + Eq + Hash, const N: usize>(
    segments: &mut Vec<IntersectionSegment<T, N>>,
) {
    use std::hash::{Hash, Hasher};

    struct SegmentKey<'a, T: Scalar, const N: usize>(&'a Point<T, N>, &'a Point<T, N>);

    impl<'a, T: Scalar, const N: usize> PartialEq for SegmentKey<'a, T, N> {
        fn eq(&self, other: &Self) -> bool {
            (self.0 == other.0 && self.1 == other.1) || (self.0 == other.1 && self.1 == other.0)
        }
    }

    impl<'a, T: Scalar, const N: usize> Eq for SegmentKey<'a, T, N> {}

    impl<'a, T: Scalar, const N: usize> Hash for SegmentKey<'a, T, N> {
        fn hash<H: Hasher>(&self, state: &mut H) {
            let mut hasher_a = std::collections::hash_map::DefaultHasher::new();
            self.0.hash(&mut hasher_a);
            let ha = hasher_a.finish();

            let mut hasher_b = std::collections::hash_map::DefaultHasher::new();
            self.1.hash(&mut hasher_b);
            let hb = hasher_b.finish();

            (ha ^ hb).hash(state);
        }
    }

    let mut seen: AHashMap<u64, (usize, bool)> = AHashMap::new();
    let mut invalidation_updates = Vec::new();

    for (i, seg) in segments.iter().enumerate() {
        let key = SegmentKey(&seg.segment.a, &seg.segment.b);

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        match seen.get(&hash) {
            Some(&(existing_i, existing_is_coplanar)) => {
                // If current one is coplanar and previous was not, prefer current
                if seg.coplanar && !existing_is_coplanar {
                    invalidation_updates.push((existing_i, true));
                    invalidation_updates.push((i, false));
                    seen.insert(hash, (i, true));
                } else {
                    invalidation_updates.push((i, true));
                }
            }
            None => {
                invalidation_updates.push((i, false));
                seen.insert(hash, (i, seg.coplanar));
            }
        }
    }

    for (index, should_invalidate) in invalidation_updates {
        segments[index].invalidated = should_invalidate;
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
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    // 1) adjacency and one seg index per undirected edge
    let mut adj: AHashMap<usize, AHashSet<usize>> = AHashMap::new();
    let mut edge_to_seg: AHashMap<(usize, usize), usize> = AHashMap::new();

    for (ei, seg) in group.iter().enumerate() {
        let [u, v] = seg.resulting_vertices_pair;
        adj.entry(u).or_default().insert(v);
        adj.entry(v).or_default().insert(u);
        edge_to_seg.entry(ordered(u, v)).or_insert(ei);
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
                let cr = ab.cross(&ac); // 3D: normal vector
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

struct TJunction {
    a_vertex: usize,
    b_edge: [usize; 2],
}

/// Find all vertices from `a_edges` (by their vertex indices into `a_vertices`)
/// that lie on any geometric edge (segment) defined by `b_edges` (indices into `b_vertices`).
///
/// Returns every (a_vertex_index, (b_u, b_v)) pair such that:
///   1. The position of the a-vertex lies on segment (b_u,b_v) within T::tolerance()
///      (orthogonal distance to the segment <= tol).
///   2. The a-vertex is NOT within T::tolerance() of either endpoint of (b_u,b_v)
///      (discard near-endpoint hits to avoid duplicating endpoints).
/// A single a-vertex may appear multiple times if it lies on multiple distinct b-edges.
/// Order is stable: vertices follow first appearance while scanning `a_edges` left-to-right;
/// for each vertex, matching b-edges follow the order in `b_edges`.
fn find_x_vertices_on_y_edges<T: Scalar, const N: usize>(
    mesh_x: &Mesh<T, N>,
    mesh_y: &Mesh<T, N>,
    x_edges: &[[usize; 2]],
    y_edges: &[[usize; 2]],
) -> Vec<TJunction>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
{
    use crate::geometry::{Aabb, AabbTree, point::Point};

    if x_edges.is_empty() || y_edges.is_empty() {
        return Vec::new();
    }

    let tol = T::tolerance();
    let tol2 = &tol * &tol;

    // Collect unique X-vertices (stable order)
    let mut ordered_x_verts = Vec::new();
    let mut seen = AHashSet::with_capacity(x_edges.len() * 2);
    for &[u, v] in x_edges {
        if seen.insert(u) {
            ordered_x_verts.push(u);
        }
        if seen.insert(v) {
            ordered_x_verts.push(v);
        }
    }

    // Precompute per Y-edge data + build 3D AABB tree (the engine already uses 3D trees).
    // Each edge AABB is expanded by tol to ensure we don't miss near-segment hits.
    let mut edge_boxes: Vec<(Aabb<T, 3, Point<T, 3>>, usize)> = Vec::with_capacity(y_edges.len());
    let mut ab_vecs: Vec<Vector<T, N>> = Vec::with_capacity(y_edges.len());
    let mut ab_len2s: Vec<T> = Vec::with_capacity(y_edges.len());

    for (ei, &[u, v]) in y_edges.iter().enumerate() {
        let p0 = &mesh_y.vertices[u].position;
        let p1 = &mesh_y.vertices[v].position;

        // AABB in 3D with tolerance expansion
        let mut minx = p0[0].clone().min(p1[0].clone());
        let mut miny = p0[1].clone().min(p1[1].clone());
        let mut minz = p0[2].clone().min(p1[2].clone());
        let mut maxx = p0[0].clone().max(p1[0].clone());
        let mut maxy = p0[1].clone().max(p1[1].clone());
        let mut maxz = p0[2].clone().max(p1[2].clone());

        minx = &minx - &tol;
        miny = &miny - &tol;
        minz = &minz - &tol;
        maxx = &maxx + &tol;
        maxy = &maxy + &tol;
        maxz = &maxz + &tol;

        let bb = Aabb::<T, 3, Point<T, 3>>::from_points(
            &Point::<T, 3>::from_vals([minx, miny, minz]),
            &Point::<T, 3>::from_vals([maxx, maxy, maxz]),
        );
        edge_boxes.push((bb, ei));

        let ab = (p1 - p0).as_vector();
        ab_vecs.push(ab.clone());
        ab_len2s.push(ab.dot(&ab));
    }

    let tree_y = AabbTree::<T, 3, Point<T, 3>, usize>::build(edge_boxes);

    let mut out = Vec::new();
    let mut candidates = Vec::new();

    for &xv in &ordered_x_verts {
        let p = &mesh_x.vertices[xv].position;

        // Query box around p with tol in 3D
        let minq = Point::<T, 3>::from_vals([&p[0] - &tol, &p[1] - &tol, &p[2] - &tol]);
        let maxq = Point::<T, 3>::from_vals([&p[0] + &tol, &p[1] + &tol, &p[2] + &tol]);
        let query = Aabb::<T, 3, Point<T, 3>>::from_points(&minq, &maxq);

        candidates.clear();
        tree_y.query(&query, &mut candidates);

        for &ei in &candidates {
            let [u_idx, v_idx] = y_edges[*ei];
            let p0 = &mesh_y.vertices[u_idx].position;
            let p1 = &mesh_y.vertices[v_idx].position;

            // Skip degenerate edges
            let ab_len2 = &ab_len2s[*ei];
            if ab_len2.is_zero() {
                continue;
            }
            let ab_vec = &ab_vecs[*ei];

            // Quick parametric projection
            let ap = (p - p0).as_vector();
            let t = &ap.dot(ab_vec) / ab_len2;

            // Segment range test with slack (avoid PartialOrd on T)
            if (&t + &tol).is_negative() || (&(&t - &T::one()) - &tol).is_positive() {
                continue;
            }

            // Orthogonal distance test (approx-first)
            let closest = p0 + &(ab_vec.scale(&t)).0;
            let diff = (p - &closest).as_vector();
            let d2 = diff.dot(&diff);
            if (&d2 - &tol2).is_positive() {
                continue;
            }

            // Discard near-endpoint hits
            let d0 = (p - p0).as_vector().dot(&(p - p0).as_vector());
            if (&d0 - &tol2).is_negative_or_zero() {
                continue;
            }
            let d1 = (p - p1).as_vector().dot(&(p - p1).as_vector());
            if (&d1 - &tol2).is_negative_or_zero() {
                continue;
            }

            // Order edge indices (integer compare is fine)
            let (mut be0, mut be1) = (u_idx, v_idx);
            if be0 > be1 {
                std::mem::swap(&mut be0, &mut be1);
            }

            out.push(TJunction {
                a_vertex: xv,
                b_edge: [be0, be1],
            });
        }
    }

    out
}

fn process_segment_and_edge_map<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    intersections_edge_map: &mut AHashMap<(usize, usize), (usize, IntersectionSegment<T, N>)>,
    i: usize,
    splits: &mut Splits<T, N>,
    tree: &mut AabbTree<T, N, Point<T, N>, usize>,
) where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
{
    if intersection_segments[i].invalidated || !intersection_segments[i].split {
        return;
    }
    mesh.process_segment(splits, tree, intersection_segments, i);

    let mut edge_verts = intersection_segments[i].resulting_vertices_pair;

    if edge_verts[0] > edge_verts[1] {
        edge_verts = [edge_verts[1], edge_verts[0]];
    }

    if !intersection_segments[i].invalidated
        && edge_verts[0] != usize::MAX
        && edge_verts[1] != usize::MAX
    {
        intersections_edge_map.insert(
            (edge_verts[0], edge_verts[1]),
            (i, intersection_segments[i].clone()),
        );
    }
}

fn process_t_junction<T: Scalar, const N: usize>(
    mesh_x: &mut Mesh<T, N>,
    mesh_y: &mut Mesh<T, N>,
    tree_x: &mut AabbTree<T, N, Point<T, N>, usize>,
    t_junction: &TJunction,
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    intersections_edge_map: &mut AHashMap<(usize, usize), (usize, IntersectionSegment<T, N>)>,
) where
    T: Scalar,
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
{
    let edge = mesh_x
        .edge_map
        .get(&(t_junction.b_edge[0], t_junction.b_edge[1]))
        .expect("Edge must exist in edge_map");

    let mut segment = intersections_edge_map
        .get(&(t_junction.b_edge[0], t_junction.b_edge[1]))
        .expect("Segment must exist")
        .clone();
    let original_vertices_pair = segment.1.resulting_vertices_pair;
    let original_segment_b = segment.1.segment.b.clone();

    let split_result = mesh_x
        .split_edge(
            tree_x,
            *edge,
            &mesh_y.vertices[t_junction.a_vertex].position,
            false,
        )
        .unwrap();
    segment.1.b = IntersectionEndPoint::new(
        Some([t_junction.a_vertex, usize::MAX]),
        None,
        None,
        None,
        None,
    );
    segment.1.resulting_vertices_pair = [original_vertices_pair[0], split_result.vertex];

    segment.1.segment.b = mesh_y.vertices[t_junction.a_vertex].position.clone();

    let new_segment = IntersectionSegment::new(
        segment.1.b.clone(),
        IntersectionEndPoint::new(
            Some([original_vertices_pair[1], usize::MAX]),
            None,
            None,
            None,
            None,
        ),
        &Segment::new(&segment.1.segment.b, &original_segment_b),
        segment.1.initial_face_reference,
        [split_result.vertex, original_vertices_pair[1]],
        segment.1.coplanar,
    );

    intersections_edge_map.remove(&(original_vertices_pair[0], original_vertices_pair[1]));
    intersection_segments[segment.0] = segment.1.clone();

    let mut edge_verts = segment.1.resulting_vertices_pair;

    if edge_verts[0] > edge_verts[1] {
        edge_verts = [edge_verts[1], edge_verts[0]];
    }

    intersections_edge_map.insert((edge_verts[0], edge_verts[1]), segment.clone());

    let mut edge_verts = new_segment.resulting_vertices_pair;

    if edge_verts[0] > edge_verts[1] {
        edge_verts = [edge_verts[1], edge_verts[0]];
    }

    intersections_edge_map.insert(
        (edge_verts[0], edge_verts[1]),
        (intersection_segments.len(), new_segment.clone()),
    );

    intersection_segments.push(new_segment);
}
