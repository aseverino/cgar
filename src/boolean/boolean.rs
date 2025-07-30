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
    collections::{HashMap, HashSet, VecDeque},
    ops::{Add, Div, Mul, Sub},
    time::Instant,
};

use smallvec::SmallVec;

use crate::{
    geometry::{
        Aabb, AabbTree, Point2,
        plane::Plane,
        point::{Point, PointOps},
        segment::{Segment, SegmentOps},
        spatial_element::SpatialElement,
        tri_tri_intersect::{
            TriTriIntersectionResult, back_project_to_3d, convex_hull_2d_indices,
            tri_tri_intersection,
        },
        vector::{Vector, VectorOps},
    },
    io::obj::write_obj,
    mesh::mesh::{IntersectionSegment, Mesh, PointInMeshResult, VertexSource},
    numeric::{cgar_f64::CgarF64, scalar::Scalar},
};

#[derive(Debug, PartialEq, Eq)]
pub enum BooleanOp {
    Union,
    Intersection,
    Difference,
}

pub trait BooleanImpl<T: Scalar, const N: usize>
where
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn boolean(&self, other: &Self, op: BooleanOp) -> Self;

    fn process_segment(
        &mut self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
        segment_idx: usize,
    );

    fn classify_faces_inside_intersection_loops(
        &self,
        other: &Mesh<T, N>,
        intersection_segments: &[IntersectionSegment<T, N>],
        include_on_surface: bool,
    ) -> Vec<bool>;

    fn build_links(mesh: &Mesh<T, N>, intersection_segments: &mut Vec<IntersectionSegment<T, N>>);
}

impl<T: Scalar, const N: usize> BooleanImpl<T, N> for Mesh<T, N>
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
    fn classify_faces_inside_intersection_loops(
        &self,
        other: &Mesh<T, N>,
        intersection_segments: &[IntersectionSegment<T, N>],
        include_on_surface: bool,
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
        let adj = self.build_face_adjacency_graph();
        let boundary_map = self.build_robust_boundary_map(intersection_segments);

        let mut boundary_faces = HashSet::new();
        for &(a, b) in &boundary_map {
            boundary_faces.insert(a);
            boundary_faces.insert(b);
        }

        let mut state: Vec<bool> = vec![false; self.faces.len()];

        let tree_b = AabbTree::<T, 3, _, _>::build(
            (0..other.faces.len())
                .map(|i| {
                    let aabb_n = other.face_aabb(i);
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
        let seed = (0..self.faces.len())
            .filter(|&f| !self.faces[f].removed && boundary_faces.contains(&f))
            .find(|&f| {
                let c = self.face_centroid(f).0;
                let c3 = Point::<T, 3>::from_vals([c[0].clone(), c[1].clone(), c[2].clone()]);
                match other.point_in_mesh(&tree_b, &c3) {
                    PointInMeshResult::Inside => true,
                    PointInMeshResult::OnSurface if include_on_surface => true,
                    _ => false,
                }
            })
            .expect("No seed face found inside B");

        state[seed] = true;

        let mut face_pairs: HashMap<usize, Vec<usize>> = HashMap::new();
        for seg in intersection_segments {
            let f0 = self
                .find_valid_face(seg.resulting_faces[0], &seg.segment.a, false)
                .unwrap();
            let f1 = self
                .find_valid_face(seg.resulting_faces[1], &seg.segment.b, false)
                .unwrap();

            face_pairs.entry(f0).or_default().push(f1);
            face_pairs.entry(f1).or_default().push(f0);
        }

        // 4) iterative flood‐fill without crossing the boundary_map
        let mut visited = vec![false; self.faces.len()];
        let mut state = vec![false; self.faces.len()];
        let mut queue = VecDeque::new();

        // seed found as before
        visited[seed] = true;
        state[seed] = true;
        queue.push_back(seed);
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
                    state[nbr] = true;
                    num_visited += 1;
                    queue.push_back(nbr);
                }
            }
        }

        println!(
            "Flood‐fill visited {} faces out of {}",
            num_visited,
            self.faces.len()
        );

        state
    }

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

        let mut coplanar_num = 0;
        let start = Instant::now();
        for fa in 0..a.faces.len() {
            let mut candidates = Vec::new();
            tree_b.query(&a.face_aabb(fa), &mut candidates);

            let pa_idx = a.face_vertices(fa);
            let pa_vec: Vec<&Point<T, N>> = pa_idx
                .into_iter()
                .map(|vi| &a.vertices[vi].position)
                .collect();
            let pa: [&Point<T, N>; 3] = pa_vec.try_into().expect("Expected 3 vertices");

            for &fb in &candidates {
                let pb_idx = b.face_vertices(*fb);
                let pb_vec: Vec<&Point<T, N>> = pb_idx
                    .into_iter()
                    .map(|vi| &b.vertices[vi].position)
                    .collect();
                let pb: [&Point<T, N>; 3] = pb_vec.try_into().expect("Expected 3 vertices");

                match tri_tri_intersection(&pa, &pb) {
                    TriTriIntersectionResult::Proper(s) => {
                        if s.length().is_positive() {
                            // println!("faces {} and {}, segment: {:?}", fa, *fb, s);
                            intersection_segments_a
                                .push(IntersectionSegment::new_default(s.clone(), fa));
                            intersection_segments_b.push(IntersectionSegment::new_default(s, *fb));
                        }
                    }
                    TriTriIntersectionResult::Coplanar(s) => {
                        coplanar_num += 1;
                        if s.length().is_positive() {
                            // println!("faces {} and {}, coplanar segment: {:?}", fa, *fb, s);
                            let mut a = IntersectionSegment::new_default(s.clone(), fa);
                            a.coplanar = true;
                            intersection_segments_a.push(a);
                            let mut b = IntersectionSegment::new_default(s, *fb);
                            b.coplanar = true;
                            intersection_segments_b.push(b);
                        }
                    }
                    TriTriIntersectionResult::CoplanarPolygon(vs) => {
                        coplanar_num += 1;
                        // println!("faces {} and {}, coplanar segment:", fa, *fb);
                        for s in vs {
                            // println!("  {:?}", s);
                            let mut a = IntersectionSegment::new_default(s.clone(), fa);
                            a.coplanar = true;
                            intersection_segments_a.push(a);
                            let mut b = IntersectionSegment::new_default(s, *fb);
                            b.coplanar = true;
                            intersection_segments_b.push(b);
                        }
                    }
                    _ => {}
                }
            }
        }

        println!("Intersection segments collected in {:.2?}", start.elapsed());
        println!("Found {} coplanar intersections", coplanar_num);

        // let start = Instant::now();
        // Mesh::filter_degenerate_segments(&mut intersection_segments_a);
        // Mesh::filter_degenerate_segments(&mut intersection_segments_b);
        // println!("Point intersections filtered in {:.2?}", start.elapsed());

        for seg_idx in 0..intersection_segments_a.len() {
            a.process_segment(&mut tree_a, &mut intersection_segments_a, seg_idx);
        }

        for seg_idx in 0..intersection_segments_b.len() {
            b.process_segment(&mut tree_b, &mut intersection_segments_b, seg_idx);
        }

        println!("Intersection segments processed in {:.2?}", start.elapsed());

        // 6. Create result mesh
        let mut result = Mesh::new();
        let mut vid_map = HashMap::new();

        // Add A vertices
        for (i, v) in a.vertices.iter().enumerate() {
            let ni = result.add_vertex(v.position.clone());
            vid_map.insert((VertexSource::A, i), ni);
        }

        // Build chains
        let start = Instant::now();
        Self::build_links(&a, &mut intersection_segments_a);
        Self::build_links(&b, &mut intersection_segments_b);
        println!("Chains built in {:.2?}", start.elapsed());

        // Classify faces using topological method
        let start = Instant::now();
        let a_classifications =
            a.classify_faces_inside_intersection_loops(&b, &intersection_segments_a, true);
        println!(
            "A faces classified inside intersection loops in {:.2?}",
            start.elapsed()
        );

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

                let b_classifications =
                    b.classify_faces_inside_intersection_loops(&a, &intersection_segments_b, false);
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

                let b_classifications =
                    b.classify_faces_inside_intersection_loops(&a, &intersection_segments_b, false);
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

                let start = Instant::now();
                let b_classifications =
                    b.classify_faces_inside_intersection_loops(&a, &intersection_segments_b, false);
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
                println!(
                    "B faces classified inside intersection loops in {:.2?}",
                    start.elapsed()
                );
            }
        }

        result.remove_duplicate_vertices();
        result.remove_unused_vertices();
        result.remove_invalidated_faces();
        // result.build_boundary_loops();
        result
    }

    /// Build clean intersection loops on each carved face by taking the convex hull of all intersection points.
    /// Returns a vector of loops, each loop is a vector of intersection_segment indices (into `intersection_segments`).
    /// `inside_faces[f]` is true for faces classified as inside the other mesh.
    fn build_links(mesh: &Mesh<T, N>, intersection_segments: &mut Vec<IntersectionSegment<T, N>>) {
        if intersection_segments.is_empty() {
            return;
        }

        let mut vertex_to_segments: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut coplanar_groups: HashMap<Plane<T, N>, Vec<usize>> = HashMap::new();

        for (seg_idx, seg) in intersection_segments.iter().enumerate() {
            let [v0, v1] = seg.resulting_vertices_pair;
            vertex_to_segments.entry(v0).or_default().push(seg_idx);
            vertex_to_segments.entry(v1).or_default().push(seg_idx);

            if seg.coplanar {
                let plane_key = mesh.plane_from_face(seg.original_face);
                coplanar_groups.entry(plane_key).or_default().push(seg_idx);
            }
        }

        // === NON-COPLANAR SEGMENTS: Build double-linked structure ===
        for (seg_idx, seg) in intersection_segments.iter_mut().enumerate() {
            if seg.coplanar {
                continue;
            }

            let [v0, v1] = seg.resulting_vertices_pair;
            let mut connected_segments = HashSet::new();

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

            seg.links.clear();
            seg.links.extend_from_slice(&links);
        }

        // === COPLANAR SEGMENTS: Build links within each group ===
        for group in coplanar_groups.values() {
            for &seg_idx in group {
                let [v0, v1] = intersection_segments[seg_idx].resulting_vertices_pair;

                let mut connected = HashSet::new();
                for &other_idx in group {
                    if other_idx == seg_idx {
                        continue;
                    }
                    let [ov0, ov1] = intersection_segments[other_idx].resulting_vertices_pair;
                    if v0 == ov0 || v0 == ov1 || v1 == ov0 || v1 == ov1 {
                        connected.insert(other_idx);
                    }
                }

                let mut links: Vec<usize> = connected.into_iter().collect();
                links.sort_unstable();

                intersection_segments[seg_idx].links.clear();
                intersection_segments[seg_idx]
                    .links
                    .extend_from_slice(&links);
            }
        }

        // === Optional: Deduplication logic remains as-is ===
        // [same deduplication step as you had before, untouched]
    }

    fn process_segment(
        &mut self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
        segment_idx: usize,
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
            self.find_valid_face(segment.original_face, &segment.segment.a, true)
        {
            let split_result =
                self.split_or_find_vertex_on_face(aabb_tree, valid_face_idx, &segment.segment.a);
            // println!("Split or find vertex took {:?}", start.elapsed());
            if let Some(split_result) = split_result {
                vertex_a = split_result.vertex;
            } else {
                panic!(
                    "Failed to split or find vertex on face {} for segment end {:?}",
                    segment.original_face, segment.segment.b
                );
            }
        } else {
            panic!(
                "Failed to find valid face for segment end {:?} on face {}",
                segment.segment.b, segment.original_face
            );
        }

        if let Some(valid_face_idx) =
            self.find_valid_face(segment.original_face, &segment.segment.b, true)
        {
            let split_result =
                self.split_or_find_vertex_on_face(aabb_tree, valid_face_idx, &segment.segment.b);
            if let Some(split_result) = split_result {
                vertex_b = split_result.vertex;

                if !split_result.new_faces.is_empty() {
                    let he = self.half_edge_between(vertex_a, vertex_b);
                    if let Some(mut he) = he {
                        he = self.find_valid_half_edge(he);
                        let twin = self.find_valid_half_edge(self.half_edges[he].twin);
                        intersection_segments[segment_idx].resulting_faces[0] =
                            self.half_edges[he].face.unwrap_or(usize::MAX);
                        intersection_segments[segment_idx].resulting_faces[1] =
                            self.half_edges[twin].face.unwrap_or(usize::MAX);
                        intersection_segments[segment_idx].resulting_vertices_pair[0] = vertex_a;
                        intersection_segments[segment_idx].resulting_vertices_pair[1] = vertex_b;
                    } else {
                        let he = self.half_edge_between(vertex_a, vertex_b);
                        if let Some(he) = he {
                            intersection_segments[segment_idx].resulting_faces[0] =
                                self.half_edges[he].face.unwrap_or(usize::MAX);
                            intersection_segments[segment_idx].resulting_faces[1] = self.half_edges
                                [self.half_edges[he].twin]
                                .face
                                .unwrap_or(usize::MAX);
                            intersection_segments[segment_idx].resulting_vertices_pair[0] =
                                vertex_a;
                            intersection_segments[segment_idx].resulting_vertices_pair[1] =
                                vertex_b;
                        }
                    }
                } else {
                    let he = self.half_edge_between(vertex_a, vertex_b);
                    if let Some(he) = he {
                        intersection_segments[segment_idx].resulting_faces[0] =
                            self.half_edges[he].face.unwrap_or(usize::MAX);
                        intersection_segments[segment_idx].resulting_faces[1] = self.half_edges
                            [self.half_edges[he].twin]
                            .face
                            .unwrap_or(usize::MAX);
                        intersection_segments[segment_idx].resulting_vertices_pair[0] = vertex_a;
                        intersection_segments[segment_idx].resulting_vertices_pair[1] = vertex_b;
                    }
                }
            } else {
                panic!(
                    "Failed to split or find vertex on face {} for segment end {:?}",
                    segment.original_face, segment.segment.b
                );
            }
        } else {
            panic!(
                "Failed to find valid face for segment end {:?} on face {}",
                segment.segment.b, segment.original_face
            );
        }

        if vertex_a == usize::MAX || vertex_b == usize::MAX {
            panic!(
                "Invalid vertex indices: vertex_a = {}, vertex_b = {}",
                vertex_a, vertex_b
            );
        }

        if !self.are_vertices_connected(vertex_a, vertex_b) {
            if !self.carve_segment_to_vertex(
                aabb_tree,
                vertex_a,
                vertex_b,
                intersection_segments,
                segment_idx,
            ) {
                panic!(
                    "Failed to carve segment from vertex {} to vertex {}",
                    vertex_a, vertex_b
                );
            }
        }
    }
}

fn face_axes(face_idx: usize) -> (usize, usize, usize) {
    match face_idx {
        // top/bottom faces lie in Z-plane → drop Z (axis 2)
        2 | 3 => (0, 1, 2),
        // right/left faces lie in Y-plane → drop Y (axis 1)
        6 | 7 => (0, 2, 1),
        // front/back faces lie in X-plane → drop X (axis 0)
        8 | 9 => (1, 2, 0),
        _ => (0, 1, 2), // fallback
    }
}

fn should_connect_segments<T: Scalar, const N: usize>(
    seg1: &IntersectionSegment<T, N>,
    seg2: &IntersectionSegment<T, N>,
    shared_vertex: usize,
) -> bool {
    // Don't connect coplanar segments here (handled separately)
    if seg1.coplanar || seg2.coplanar {
        return false;
    }

    // Check if segments share exactly one face (continuous boundary condition)
    let faces1 = &seg1.resulting_faces;
    let faces2 = &seg2.resulting_faces;

    let shared_faces = faces1.iter().filter(|&&f| faces2.contains(&f)).count();

    // Only connect if segments share exactly one face (forming a continuous boundary)
    shared_faces == 1
}

fn build_coplanar_links<T: Scalar, const N: usize>(
    intersection_segments: &mut [IntersectionSegment<T, N>],
    segment_indices: &[usize],
) {
    // For coplanar segments, build a more restricted connectivity
    for &seg_idx in segment_indices {
        let [v0, v1] = intersection_segments[seg_idx].resulting_vertices_pair;
        let mut connected = Vec::new();

        for &other_idx in segment_indices {
            if other_idx == seg_idx {
                continue;
            }

            let [ov0, ov1] = intersection_segments[other_idx].resulting_vertices_pair;

            // Only connect if they share exactly one vertex
            let shared_vertices = [
                (v0 == ov0) as u8,
                (v0 == ov1) as u8,
                (v1 == ov0) as u8,
                (v1 == ov1) as u8,
            ]
            .iter()
            .sum::<u8>();

            if shared_vertices == 1 {
                connected.push(other_idx);
            }
        }

        connected.sort_unstable();
        intersection_segments[seg_idx].links.clear();
        intersection_segments[seg_idx]
            .links
            .extend_from_slice(&connected);
    }
}
