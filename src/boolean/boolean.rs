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
    hash::Hash,
    ops::{Add, Div, Mul, Neg, Sub},
    time::Instant,
};

use crate::{
    geometry::{
        Aabb, AabbTree,
        plane::Plane,
        point::{Point, PointOps},
        segment::{Segment, SegmentOps},
        spatial_element::SpatialElement,
        tri_tri_intersect::{TriTriIntersectionResult, tri_tri_intersection},
        vector::{Vector, VectorOps},
    },
    impl_mesh,
    io::obj::write_obj,
    mesh::{
        basic_types::{Mesh, PointInMeshResult, VertexSource},
        intersection_segment::{IntersectionEndPoint, IntersectionSegment},
    },
    numeric::{cgar_f64::CgarF64, scalar::Scalar},
};

#[derive(Clone, Copy)]
pub struct EndPointHandle {
    segment_idx: usize,
    endpoint_idx: usize, // 0 or 1
}

#[derive(Clone, Copy, PartialEq)]
pub enum SplitType {
    Edge,
    Face,
}

pub struct Splits<T: Scalar, const N: usize> {
    pub splits: HashMap<Point<T, N>, (SplitType, usize, Vec<EndPointHandle>)>,
}

impl<T: Scalar, const N: usize> Splits<T, N> {
    pub fn new() -> Self {
        Self {
            splits: HashMap::new(),
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

            let mut boundary_faces = HashSet::new();
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

            let mut face_pairs: HashMap<usize, Vec<usize>> = HashMap::new();
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
        face: usize,
        coplanar: bool,
        skip_endpoint_b: bool,
    ) where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut endpoints_set = [false, false];
        let idx = container.len();
        container.push(IntersectionSegment::new(
            IntersectionEndPoint::new_default(),
            IntersectionEndPoint::new_default(),
            &segment,
            face,
            [usize::MAX, usize::MAX],
            coplanar,
        ));
        // let mut endpoints: [Option<IntersectionEndPoint<T, N>>; 2] = [None, None];
        let face = mesh.find_valid_face(face, &segment.a);

        let start_he = mesh.find_valid_half_edge(mesh.faces[face].half_edge, &segment.a);
        let mut next_he = start_he;
        loop {
            for i in 0..2 {
                if (i == 1 && skip_endpoint_b) || endpoints_set[i] {
                    continue;
                }
                if let Some(point_on_half_edge) = mesh.point_on_half_edge(next_he, &segment[i]) {
                    let mut found_he = usize::MAX;
                    let found_v_a = mesh.half_edges[next_he].vertex;
                    let found_v_b = mesh.half_edges[mesh.half_edges[next_he].twin].vertex;
                    let mut vertex_hint = usize::MAX;

                    if (&point_on_half_edge - &T::one()).is_zero() {
                        vertex_hint = found_v_a;
                    } else if point_on_half_edge.is_zero() {
                        vertex_hint = found_v_b;
                    } else {
                        found_he = next_he;
                    }

                    if found_he != usize::MAX {
                        container[idx][i].vertex_hint = Some([found_v_a, found_v_b]);
                        container[idx][i].half_edge_hint = Some(found_he);
                        container[idx][i].half_edge_u_hint = Some(point_on_half_edge.clone());

                        let v_b = &mesh.vertices[mesh.half_edges[found_he].vertex].position;
                        let v_a = &mesh.vertices
                            [mesh.half_edges[mesh.half_edges[found_he].twin].vertex]
                            .position;
                        let t = point_on_half_edge.clone();
                        let new_point = v_a + &(v_b - v_a).as_vector().scale(&t).0;
                        let entry = splits
                            .splits
                            .entry(new_point)
                            .or_insert_with(|| (SplitType::Edge, usize::MAX, Vec::new()));
                        entry.1 = found_he;
                        entry.2.push(EndPointHandle {
                            segment_idx: idx,
                            endpoint_idx: i,
                        });
                    } else if vertex_hint != usize::MAX {
                        container[idx][i].vertex_hint = Some([vertex_hint, usize::MAX]);
                    } else {
                        let barycentric_coords = mesh.barycentric_coords_on_face(face, &segment[i]);
                        container[idx][i].face_hint = Some(face);
                        container[idx][i].barycentric_hint = barycentric_coords.clone();
                        let entry = splits
                            .splits
                            .entry(segment[i].clone())
                            .or_insert_with(|| (SplitType::Face, usize::MAX, Vec::new()));

                        entry.1 = face;
                        entry.2.push(EndPointHandle {
                            segment_idx: idx,
                            endpoint_idx: i,
                        });
                    }

                    endpoints_set[i] = true;

                    if i == 1 {
                        // If we found the second endpoint, we can stop checking other half-edges
                        break;
                    }
                }
            }

            next_he = mesh.half_edges[next_he].next;
            if next_he == start_he {
                break;
            }
        }

        for i in 0..2 {
            if i == 1 && skip_endpoint_b || endpoints_set[i] {
                continue;
            }

            if let Some(bary) = mesh.barycentric_coords_on_face(face, &segment[i]) {
                // Point is on the face, but not on any half-edge

                container[idx][i].face_hint = Some(face);
                container[idx][i].barycentric_hint = Some(bary.clone());
                let entry = splits
                    .splits
                    .entry(segment[i].clone())
                    .or_insert_with(|| (SplitType::Face, usize::MAX, Vec::new()));

                entry.1 = face;
                entry.2.push(EndPointHandle {
                    segment_idx: idx,
                    endpoint_idx: i,
                });

                if i == 1 {
                    // If we found the second endpoint, we can stop checking other half-edges
                    break;
                }
            } else {
                panic!("Point not on face!");
            }
        }
    }

    fn boolean_split(
        mesh: &mut Mesh<T, N>,
        tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        splits: &mut Splits<T, N>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    ) {
        for (key, endpoint_tup) in &splits.splits {
            if endpoint_tup.0 == SplitType::Edge {
                let result = mesh
                    .split_edge(tree, endpoint_tup.1, &key)
                    .expect("Failed to split edge");

                for endpoint in &endpoint_tup.2 {
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .vertex_hint = Some([result.vertex, usize::MAX]);
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .half_edge_hint = None;
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .half_edge_u_hint = None;

                    if mesh.vertices[result.vertex].position
                        != intersection_segments[endpoint.segment_idx].segment
                            [endpoint.endpoint_idx]
                    {
                        panic!("Inconsistent A");
                    }
                }
            } else {
                let result = mesh
                    .split_face(tree, endpoint_tup.1, &key)
                    .expect("Failed to split face");

                for endpoint in &endpoint_tup.2 {
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .vertex_hint = Some([result.vertex, usize::MAX]);
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx].face_hint =
                        None;
                    intersection_segments[endpoint.segment_idx][endpoint.endpoint_idx]
                        .barycentric_hint = None;

                    if mesh.vertices[result.vertex].position
                        != intersection_segments[endpoint.segment_idx].segment
                            [endpoint.endpoint_idx]
                    {
                        panic!("Inconsistent B");
                    }
                }
            }
        }
    }

    pub fn boolean(&self, other: &Mesh<T, N>, op: BooleanOp) -> Mesh<T, N> {
        let mut a = self.clone();
        let mut b = other.clone();

        // 1. Collect ALL intersection segments
        let mut intersection_segments_a = Vec::new();
        let mut intersection_segments_b = Vec::new();
        let start = Instant::now();
        let mut tree_a = a.build_face_tree();
        let mut tree_b = b.build_face_tree();
        println!("Total AABB computation: {:.2?}", start.elapsed());

        let mut splits_a = Splits::new();
        let mut splits_b = Splits::new();

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
                    TriTriIntersectionResult::Proper(segment) => {
                        if segment.length2().is_positive() {
                            Self::create_intersection_segment(
                                &a,
                                &mut splits_a,
                                &mut intersection_segments_a,
                                &segment,
                                fa,
                                false,
                                false,
                            );
                            Self::create_intersection_segment(
                                &b,
                                &mut splits_b,
                                &mut intersection_segments_b,
                                &segment,
                                *fb,
                                false,
                                false,
                            );
                        }
                    }
                    TriTriIntersectionResult::Coplanar(segment) => {
                        if segment.length2().is_positive() {
                            Self::create_intersection_segment(
                                &a,
                                &mut splits_a,
                                &mut intersection_segments_a,
                                &segment,
                                fa,
                                true,
                                false,
                            );
                            Self::create_intersection_segment(
                                &b,
                                &mut splits_b,
                                &mut intersection_segments_b,
                                &segment,
                                *fb,
                                true,
                                false,
                            );
                        }
                    }
                    TriTriIntersectionResult::CoplanarPolygon(vs) => {
                        for segment in vs {
                            Self::create_intersection_segment(
                                &a,
                                &mut splits_a,
                                &mut intersection_segments_a,
                                &segment,
                                fa,
                                true,
                                false,
                            );
                            Self::create_intersection_segment(
                                &b,
                                &mut splits_b,
                                &mut intersection_segments_b,
                                &segment,
                                *fb,
                                true,
                                false,
                            );
                        }
                    }
                    _ => {}
                }
            }
        }

        // Remove duplicate segments from both lists.
        remove_duplicate_segments(&mut intersection_segments_a);
        remove_duplicate_segments(&mut intersection_segments_b);

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

        let mut intersection_by_edge_a = HashMap::new();
        let mut intersection_by_edge_b = HashMap::new();

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
        let mut vid_map = HashMap::new();

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

        let _ = write_obj(&result, "/mnt/v/cgar_meshes/a.obj");

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
        let mut coplanar_groups: HashMap<Plane<T, N>, Vec<IntersectionSegment<T, N>>> =
            HashMap::new();

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

        let mut vertex_to_segments: HashMap<usize, Vec<usize>> = HashMap::new();

        for (seg_idx, seg) in intersection_segments.iter().enumerate() {
            let [v0, v1] = seg.resulting_vertices_pair;
            vertex_to_segments.entry(v0).or_default().push(seg_idx);
            vertex_to_segments.entry(v1).or_default().push(seg_idx);
        }

        // === NON-COPLANAR SEGMENTS: Build double-linked structure ===
        for (seg_idx, seg) in intersection_segments.iter_mut().enumerate() {
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

            seg.links.extend_from_slice(&links);
        }

        // === COPLANAR SEGMENTS: Build links within each group ===
        // for group in coplanar_groups.values_mut() {
        //     let pairs: Vec<[usize; 2]> = group
        //         .iter()
        //         .map(|seg| seg.resulting_vertices_pair)
        //         .collect();

        //     for (i, seg) in group.iter_mut().enumerate() {
        //         let [v0, v1] = pairs[i];
        //         let mut links: Vec<usize> = pairs
        //             .iter()
        //             .enumerate()
        //             .filter_map(|(j, &[ov0, ov1])| {
        //                 if j != i && (ov0 == v0 || ov1 == v0 || ov0 == v1 || ov1 == v1) {
        //                     Some(j)
        //                 } else {
        //                     None
        //                 }
        //             })
        //             .collect();
        //         links.sort_unstable();
        //         seg.links.extend_from_slice(&links);
        //     }
        // }

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
        vertex_ab: &mut [usize; 2],
    ) -> bool {
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

        // We first check if a half_edge hint gives us an exact vertex.
        if segment_idx > 0 {
            // It may do now since topology is changing with every process.
            for i in 0..2 {
                if intersection_segments[segment_idx][i]
                    .half_edge_hint
                    .is_some()
                {
                    let mut vert_hint = usize::MAX;
                    let he = self.find_valid_half_edge(
                        intersection_segments[segment_idx][i]
                            .half_edge_hint
                            .unwrap(),
                        &intersection_segments[segment_idx].segment[i],
                    );
                    let twin = self.find_valid_half_edge(
                        self.half_edges[he].twin,
                        &intersection_segments[segment_idx].segment[i],
                    );

                    let mut updated = false;
                    if intersection_segments[segment_idx].segment[i]
                        == self.vertices[self.half_edges[he].vertex].position
                    {
                        vert_hint = self.half_edges[he].vertex;
                        updated = true;
                    } else if intersection_segments[segment_idx].segment[i]
                        == self.vertices[self.half_edges[twin].vertex].position
                    {
                        vert_hint = self.half_edges[twin].vertex;
                        updated = true;
                    }

                    if updated {
                        intersection_segments[segment_idx][i].half_edge_hint = None;
                        if intersection_segments[segment_idx][i].vertex_hint.is_none() {
                            intersection_segments[segment_idx][i].vertex_hint =
                                Some([usize::MAX, usize::MAX]);
                        }
                        intersection_segments[segment_idx][i].vertex_hint =
                            Some([vert_hint, usize::MAX]);
                    }
                }
            }
        }

        if Self::check_for_process_segment_early_exit(
            self,
            intersection_segments,
            segment_idx,
            &mut vertex_ab,
        ) {
            return;
        }

        let mut face_split = false;

        for i in 0..2 {
            if intersection_segments[segment_idx][i].face_hint.is_some() {
                if let Some(split_result) = self.split_face(
                    aabb_tree,
                    self.find_valid_face(
                        intersection_segments[segment_idx][i].face_hint.unwrap(),
                        &intersection_segments[segment_idx].segment[i],
                    ),
                    &intersection_segments[segment_idx].segment[i],
                ) {
                    // Let's update the segment's A endpoint.
                    intersection_segments[segment_idx][i].barycentric_hint = None;
                    intersection_segments[segment_idx][i].face_hint = None;
                    intersection_segments[segment_idx][i].vertex_hint =
                        Some([split_result.vertex, usize::MAX]);
                    face_split = true;
                    vertex_ab[i] = split_result.vertex;
                }
            }
        }

        if face_split
            && Self::check_for_process_segment_early_exit(
                self,
                intersection_segments,
                segment_idx,
                &mut vertex_ab,
            )
        {
            return;
        }

        fn get_face_from_vertex_and_direction<T: Scalar, const N: usize>(
            mesh: &Mesh<T, N>,
            vertex: usize,
            direction: &Vector<T, N>,
        ) -> usize
        where
            Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
            Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
            for<'a> &'a T: Sub<&'a T, Output = T>
                + Mul<&'a T, Output = T>
                + Add<&'a T, Output = T>
                + Div<&'a T, Output = T>
                + Neg<Output = T>,
        {
            for face in mesh.faces_around_vertex(vertex) {
                let verts = mesh.face_vertices(face);
                let mut it = verts.iter().copied().filter(|&i| i != vertex);

                let v1 = it.next().unwrap();
                let v2 = it.next().unwrap();

                let e1_v1 = mesh.vertices[vertex].position.clone();
                let e1_v2 = mesh.vertices[v1].position.clone();
                let e2_v1 = mesh.vertices[vertex].position.clone();
                let e2_v2 = mesh.vertices[v2].position.clone();

                let e1 = (e1_v2 - e1_v1).as_vector();
                let e2 = (e2_v2 - e2_v1).as_vector();

                let c1 = e1.cross(&direction);
                let c2 = direction.cross(&e2);
                let total = e1.cross(&e2);

                let is_between = if total.dot(&total) > T::zero() {
                    c1.dot(&total) >= T::zero() && c2.dot(&total) >= T::zero()
                } else {
                    direction.dot(&e1) >= T::zero()
                };

                // Face normal alignment test
                let normal = mesh.face_normal(face);
                let n_dot_d = normal.dot(&direction);
                let is_aligned = n_dot_d.is_zero();

                if is_between && is_aligned {
                    return face;
                }
            }

            usize::MAX
        }
        fn get_face<T: Scalar, const N: usize>(
            mesh: &Mesh<T, N>,
            segment: &IntersectionSegment<T, N>,
            endpoint: usize,
        ) -> usize
        where
            Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
            Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
            for<'a> &'a T: Sub<&'a T, Output = T>
                + Mul<&'a T, Output = T>
                + Add<&'a T, Output = T>
                + Div<&'a T, Output = T>
                + Neg<Output = T>,
        {
            // This function is a placeholder for the logic that would determine the face
            // based on the segment direction and the intersection point.
            // It should return the face index where the segment intersects.
            if let Some(half_edge) = segment[endpoint].half_edge_hint {
                let half_edge = mesh.find_valid_half_edge(half_edge, &segment.segment[endpoint]);

                mesh.half_edges[half_edge]
                    .face
                    .expect("Half-edge must have a face")
            } else if let Some(vertices) = segment[endpoint].vertex_hint {
                if vertices[0] != usize::MAX && vertices[1] == usize::MAX {
                    let direction = {
                        if endpoint == 0 {
                            segment.segment.direction()
                        } else {
                            -segment.segment.direction()
                        }
                    };
                    return get_face_from_vertex_and_direction(&mesh, vertices[0], &direction);
                } else {
                    panic!("Vertex hint must be set for the endpoint");
                }
            } else if let Some(face) = segment[endpoint].face_hint {
                return mesh.find_valid_face(face, &segment.segment[endpoint]);
            } else {
                panic!("Neither half-edge hint nor vertex hint is set for the endpoint");
            }
        }

        // Let's find out if both endpoints are on the same face or adjacent.
        let face_a = get_face(self, &intersection_segments[segment_idx], 0);
        let face_b = get_face(self, &intersection_segments[segment_idx], 1);

        if self.faces[face_a].removed
            || self.faces[face_b].removed
            || face_a == usize::MAX
            || face_b == usize::MAX
        {
            panic!("face removed");
        }

        if face_a != face_b {
            // For the sake of consistency, we find the endpoint B by using the segment direction.
            // We do that because we can't rely on endpoint B being on the same face as endpoint A.
            let starting_point = &intersection_segments[segment_idx].segment.a;

            // if let Some(half_edge_hint) = intersection_segments[segment_idx].a.half_edge_hint {
            //     let starting_half_edge = self.find_valid_half_edge(
            //         intersection_segments[segment_idx].a.half_edge_hint.unwrap(),
            //         starting_point,
            //     );
            // }

            let segment_direction = intersection_segments[segment_idx].segment.direction();
            if let Some((he, t, _u)) = self.get_first_half_edge_intersection_on_face(
                face_a,
                &starting_point,
                &segment_direction,
            ) {
                let v_a = &intersection_segments[segment_idx].segment[0];
                let new_point = v_a + &(segment_direction).scale(&t).0;
                // let new_point =
                //     &intersection_segments[segment_idx].segment[0] + &segment_direction.scale(&t).0;

                let new_segment =
                    Segment::new(&new_point, &intersection_segments[segment_idx].segment[1]);

                let updated_he = self.find_valid_half_edge(he, &new_point);
                let updated_he_twin =
                    self.find_valid_half_edge(self.half_edges[updated_he].twin, &new_point);

                let updated_face = self.half_edges[updated_he_twin]
                    .face
                    .expect("Half-edge must have a face");

                if let Some((_he_next, _t_next)) =
                    self.point_is_on_some_half_edge(updated_face, &new_point)
                {
                    Self::create_intersection_segment(
                        self,
                        edge_splits,
                        intersection_segments,
                        &new_segment,
                        updated_face,
                        intersection_segments[segment_idx].coplanar,
                        true,
                    );

                    let seg = intersection_segments[segment_idx].b.clone();
                    let len = intersection_segments.len() - 1;
                    let new_intersection = &mut intersection_segments[len];
                    new_intersection.b = seg;
                } else if let Some((he_next, _t_next, _u)) = self
                    .get_first_half_edge_intersection_on_face(
                        self.half_edges[updated_he_twin]
                            .face
                            .expect("Half-edge must have a face"),
                        &new_point,
                        &segment_direction,
                    )
                {
                    Self::create_intersection_segment(
                        self,
                        edge_splits,
                        intersection_segments,
                        &new_segment,
                        self.half_edges[he_next]
                            .face
                            .expect("Half-edge must have a face"),
                        intersection_segments[segment_idx].coplanar,
                        true,
                    );

                    let seg = intersection_segments[segment_idx].b.clone();
                    let len = intersection_segments.len() - 1;
                    let new_intersection = &mut intersection_segments[len];
                    new_intersection.b = seg;
                } else {
                    panic!(
                        "Failed to split segment at the intersection point: {:?}",
                        new_point
                    );
                }

                intersection_segments[segment_idx].segment[1] = new_point;
                intersection_segments[segment_idx].b.half_edge_hint = Some(updated_he);
                intersection_segments[segment_idx].b.vertex_hint = Some([
                    self.half_edges[updated_he].vertex,
                    self.half_edges[updated_he_twin].vertex,
                ]);

                vertex_ab[1] = usize::MAX;
            } else {
                let mut intersection = Mesh::new();
                intersection.add_vertex(intersection_segments[segment_idx].segment[0].clone());
                intersection.add_vertex(intersection_segments[segment_idx].segment[1].clone());
                let mut vertex = Mesh::new();
                let mds = intersection_segments[segment_idx].a.vertex_hint.unwrap();
                vertex.add_vertex(self.vertices[mds[0]].position.clone());

                let face_a_vertices = self.face_vertices(face_a);
                let mut triangle = Mesh::new();
                for v in face_a_vertices {
                    triangle.add_vertex(self.vertices[v].position.clone());
                }
                panic!(
                    "Failed to find a valid half-edge intersection on face {} for point {:?}",
                    face_a, starting_point
                );
            }
        }

        // Try again.
        let face_b = get_face(self, &intersection_segments[segment_idx], 1);

        if face_a == face_b {
            // Process the segment endpoints normally.
            for i in 0..2 {
                if vertex_ab[i] == usize::MAX {
                    if intersection_segments[segment_idx][i]
                        .half_edge_hint
                        .is_some()
                    {
                        let point = &intersection_segments[segment_idx].segment[i];
                        let half_edge = self.find_valid_half_edge(
                            intersection_segments[segment_idx][i]
                                .half_edge_hint
                                .unwrap(),
                            point,
                        );
                        // let u = point_position_on_segment(source_p, target_p, point).expect("Failed to compute point position on segment");
                        let split_result = self
                            .split_edge(
                                aabb_tree,
                                half_edge,
                                &intersection_segments[segment_idx].segment[i],
                            )
                            .expect("Failed to split edge for segment end");

                        vertex_ab[i] = split_result.vertex;

                        if intersection_segments[segment_idx].coplanar {
                            if let Some(new_edges) = self.half_edge_split_map.get(&half_edge) {
                                let new_edges_arr = [new_edges.0, new_edges.1];
                                for i in 0..2 {
                                    let mut other_vertex = usize::MAX;
                                    if self.half_edges[new_edges_arr[i]].vertex
                                        != split_result.vertex
                                    {
                                        other_vertex = self.half_edges[new_edges_arr[i]].vertex;
                                    } else if self.half_edges
                                        [self.half_edges[new_edges_arr[i]].twin]
                                        .vertex
                                        != split_result.vertex
                                    {
                                        other_vertex = self.half_edges
                                            [self.half_edges[new_edges_arr[i]].twin]
                                            .vertex;
                                    }

                                    let mut seg = IntersectionSegment::new(
                                        IntersectionEndPoint::new_default(),
                                        IntersectionEndPoint::new_default(),
                                        &Segment::new(
                                            &self.vertices[split_result.vertex].position,
                                            &self.vertices[other_vertex].position,
                                        ),
                                        intersection_segments[segment_idx].initial_face_reference,
                                        [split_result.vertex, other_vertex],
                                        true,
                                    );
                                    seg.split = false;
                                    intersection_segments.push(seg);
                                }
                            }
                        }
                    }
                }
            }

            if vertex_ab[0] == usize::MAX || vertex_ab[1] == usize::MAX {
                // If we still don't have both vertices, we can't proceed.
                intersection_segments[segment_idx].invalidated = true;
                return;
            }

            let he_ab = self.vertices_connection(vertex_ab[0], vertex_ab[1]);
            if he_ab != usize::MAX {
                intersection_segments[segment_idx].resulting_vertices_pair =
                    [vertex_ab[0], vertex_ab[1]];
            } else {
                panic!("Failed to connect segment endpoints");
            }
        } else {
            panic!("Faces are not adjacent or equal, but they should be. This is a bug.");
        }
    }

    fn get_seed_face(
        a: &Mesh<T, N>,
        b: &Mesh<T, N>,
        tree_b: &AabbTree<T, N, Point<T, N>, usize>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
        boundary_faces: &HashSet<usize>,
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
    use std::collections::HashMap;
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

    let mut seen: HashMap<u64, (usize, bool)> = HashMap::new();
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
    use std::collections::{HashMap, HashSet};

    // 1) adjacency and one seg index per undirected edge
    let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut edge_to_seg: HashMap<(usize, usize), usize> = HashMap::new();

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
    use std::collections::HashSet;

    if x_edges.is_empty() || y_edges.is_empty() {
        return Vec::new();
    }

    let tol = T::tolerance();
    let tol2 = &tol * &tol;

    // Collect unique X-vertices (stable order)
    let mut ordered_x_verts = Vec::new();
    let mut seen = HashSet::with_capacity(x_edges.len() * 2);
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

    let tree_y = AabbTree::<T, 3, _, _>::build(edge_boxes);

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

            // Segment range test with slack
            if t < &T::zero() - &tol || t > &T::one() + &tol {
                continue;
            }

            // Orthogonal distance test
            let closest = p0 + &(ab_vec.scale(&t)).0;
            let diff = (p - &closest).as_vector();
            if diff.dot(&diff) > tol2 {
                continue;
            }

            // Discard near-endpoint hits
            let d0 = (p - p0).as_vector().dot(&(p - p0).as_vector());
            if d0 <= tol2 {
                continue;
            }
            let d1 = (p - p1).as_vector().dot(&(p - p1).as_vector());
            if d1 <= tol2 {
                continue;
            }

            // Order edge indices and adjust u so it is measured from b_edge[0]
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
    intersections_edge_map: &mut HashMap<(usize, usize), (usize, IntersectionSegment<T, N>)>,
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
    intersections_edge_map: &mut HashMap<(usize, usize), (usize, IntersectionSegment<T, N>)>,
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
