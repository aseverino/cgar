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
    ops::{Add, Div, Mul, Neg, Sub},
    os::unix::process,
    result,
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
    mesh::{
        self,
        mesh::{
            IntersectionEndPoint, IntersectionSegment, Mesh, PointInMeshResult, VertexSource,
            point_position_on_segment,
        },
        vertex,
    },
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
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
{
    fn boolean(&self, other: &Self, op: BooleanOp) -> Self;
    fn create_segment(
        mesh: &Mesh<T, N>,
        s: &Segment<T, N>,
        face: usize,
        coplanar: bool,
    ) -> IntersectionSegment<T, N>;

    fn process_segment(
        &mut self,
        aabb_tree: &mut AabbTree<T, N, Point<T, N>, usize>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
        segment_idx: usize,
    );

    fn classify_faces_inside_intersection_loops(
        &self,
        other: &Mesh<T, N>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
        coplanar_intersections: &HashMap<Plane<T, N>, Vec<IntersectionSegment<T, N>>>,
        include_on_surface: bool,
    ) -> Vec<bool>;

    fn build_links(
        mesh: &Mesh<T, N>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    ) -> HashMap<Plane<T, N>, Vec<IntersectionSegment<T, N>>>;
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
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
{
    fn classify_faces_inside_intersection_loops(
        &self,
        other: &Mesh<T, N>,
        intersection_segments: &Vec<IntersectionSegment<T, N>>,
        coplanar_intersections: &HashMap<Plane<T, N>, Vec<IntersectionSegment<T, N>>>,
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

        let mut visited = vec![false; self.faces.len()];
        let mut inside = vec![false; self.faces.len()];

        // Non-coplanar intersections first.
        if !intersection_segments.is_empty() {
            let boundary_map = self.build_boundary_map(intersection_segments);

            let mut boundary_faces = HashSet::new();
            for &(a, b) in &boundary_map {
                boundary_faces.insert(a);
                boundary_faces.insert(b);
            }

            // 3) pick a seed face that lies inside B
            let (seed_intersection_idx, selected_face) = get_seed_face(
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
                    [self.find_valid_half_edge(self.half_edges[*he].twin, &seg.b.point)]
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

        let mut coplanar_group = 0;
        // Now do co-planar intersections.
        for (_plane, segments) in coplanar_intersections {
            println!(
                "Processing coplanar group {} with {} segments",
                coplanar_group,
                segments.len()
            );
            coplanar_group += 1;
            let boundary_map = self.build_boundary_map(segments);
            let mut boundary_faces = HashSet::new();
            for &(a, b) in &boundary_map {
                boundary_faces.insert(a);
                boundary_faces.insert(b);
            }

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

            // 4) iterative flood‐fill without crossing the boundary_map

            let mut face_pairs: HashMap<usize, Vec<usize>> = HashMap::new();
            for seg_idx in 0..intersection_segments.len() {
                // let seg = &intersection_segments[seg_idx];
                // let f0 = self
                //     .find_exact_valid_face(seg.resulting_faces[0], &seg.segment.a, None)
                //     .unwrap();
                // let f1 = self
                //     .find_exact_valid_face(seg.resulting_faces[1], &seg.segment.b, None)
                //     .unwrap();

                // face_pairs.entry(f0).or_default().push(f1);
                // face_pairs.entry(f1).or_default().push(f0);
            }
            let mut queue = VecDeque::new();

            // seed found as before
            visited[seed] = true;
            inside[seed] = true;
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

        inside
    }

    fn create_segment(
        mesh: &Mesh<T, N>,
        s: &Segment<T, N>,
        face: usize,
        coplanar: bool,
    ) -> IntersectionSegment<T, N>
    where
        Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
        Vector<T, N>: VectorOps<T, N>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let mut endpoints: [Option<IntersectionEndPoint<T, N>>; 2] = [None, None];
        println!("segment: {:?}", s);

        for he in mesh.face_half_edges(face) {
            for i in 0..2 {
                if endpoints[i].is_some() {
                    continue;
                }
                if let Some(point_on_half_edge) = mesh.point_on_half_edge(he, &s[i]) {
                    let mut found_he = usize::MAX;
                    let found_v_a = mesh.half_edges[he].vertex;
                    let found_v_b = mesh.half_edges[mesh.half_edges[he].twin].vertex;
                    let mut vertex_hint = usize::MAX;

                    if (&point_on_half_edge - &T::one()).is_zero() {
                        vertex_hint = found_v_a;
                    } else if point_on_half_edge.is_zero() {
                        vertex_hint = found_v_b;
                    } else {
                        found_he = he;
                    }

                    if found_he != usize::MAX {
                        endpoints[i] = Some(IntersectionEndPoint::new(
                            s[i].clone(),
                            Some([found_v_a, found_v_b]),
                            Some(found_he),
                            None,
                            None,
                        ));
                    } else if vertex_hint != usize::MAX {
                        endpoints[i] = Some(IntersectionEndPoint::new(
                            s[i].clone(),
                            Some([vertex_hint, usize::MAX]),
                            None,
                            None,
                            None,
                        ));
                    } else {
                        let barycentric_coords = mesh.barycentric_coords_on_face(face, &s[i]);
                        endpoints[i] = Some(IntersectionEndPoint::new(
                            s[i].clone(),
                            None,
                            None,
                            Some(face),
                            barycentric_coords,
                        ));
                    }
                }
            }
        }

        println!("endpoint A: {:?}", endpoints[0]);
        println!("endpoint B: {:?}", endpoints[1]);

        IntersectionSegment::new(
            endpoints[0].as_ref().unwrap(),
            endpoints[1].as_ref().unwrap(),
            &s,
            face,
            [usize::MAX, usize::MAX],
            coplanar,
        )
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
        let mut tree_a = a.build_face_tree();
        let mut tree_b = b.build_face_tree();
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
                            intersection_segments_a.push(Self::create_segment(&a, &s, fa, false));
                            intersection_segments_b.push(Self::create_segment(&b, &s, *fb, false));
                        }
                    }
                    TriTriIntersectionResult::Coplanar(s) => {
                        coplanar_num += 1;
                        if s.length().is_positive() {
                            let mut a = Self::create_segment(&a, &s, fa, false);
                            a.coplanar = true;
                            intersection_segments_a.push(a);
                            let mut b = Self::create_segment(&b, &s, *fb, false);
                            b.coplanar = true;
                            intersection_segments_b.push(b);
                        }
                    }
                    TriTriIntersectionResult::CoplanarPolygon(vs) => {
                        for s in vs {
                            coplanar_num += 1;
                            let mut a = Self::create_segment(&a, &s, fa, false);
                            a.coplanar = true;
                            intersection_segments_a.push(a);
                            let mut b = Self::create_segment(&b, &s, *fb, false);
                            b.coplanar = true;
                            intersection_segments_b.push(b);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Remove duplicate segments from both lists.
        remove_duplicate_segments(&mut intersection_segments_a);
        remove_duplicate_segments(&mut intersection_segments_b);

        // for seg in &intersection_segments_a {
        //     if !seg.coplanar {
        //         println!("Segment: {:?}", seg);
        //     }
        // }

        // let start = Instant::now();
        // Mesh::filter_degenerate_segments(&mut intersection_segments_a);
        // Mesh::filter_degenerate_segments(&mut intersection_segments_b);
        // println!("Point intersections filtered in {:.2?}", start.elapsed());

        println!("Processing segments A");
        let mut i = 0;
        while i < intersection_segments_a.len() {
            a.process_segment(&mut tree_a, &mut intersection_segments_a, i);
            let _ = write_obj(&a, format!("/mnt/v/cgar_meshes/new_a_{}.obj", i));
            i += 1;
        }

        println!("Processing segments B");
        i = 0;
        while i < intersection_segments_b.len() {
            b.process_segment(&mut tree_b, &mut intersection_segments_b, i);
            i += 1;
            let _ = write_obj(&b, format!("/mnt/v/cgar_meshes/new_b_{}.obj", i));
        }

        let _ = write_obj(&b, "/mnt/v/cgar_meshes/new.obj");

        intersection_segments_a.retain(|segment| !segment.invalidated);
        intersection_segments_b.retain(|segment| !segment.invalidated);

        for seg in &intersection_segments_a {
            // println!("Segment A: {:?}", seg);
        }

        for seg in &intersection_segments_b {
            // println!("Segment B: {:?}", seg);
        }

        println!("Intersection segments processed in {:.2?}", start.elapsed());

        let _ = write_obj(&a, "/mnt/v/cgar_meshes/socorro.obj");

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

        for (fa, inside) in a_classification.iter().enumerate() {
            if a.faces[fa].removed {
                continue;
            }
            println!("Face {}: inside = {}", fa, inside);

            if !inside {
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
                    false,
                );
                for (fb, inside) in b_classification.iter().enumerate() {
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
                    false,
                );
                for (fb, inside) in b_classification.iter().enumerate() {
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
                    false,
                );
                for (fb, inside) in b_classification.iter().enumerate() {
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
        // result.build_boundary_loops();
        result
    }

    /// Build clean links between "intersection_segments" in-place. Removing coplanars from the collection.
    /// Returns a separate vector of vectors of intersections that are coplanar and form graphs.
    /// The coplanar segments are grouped by a shared plane.
    fn build_links(
        mesh: &Mesh<T, N>,
        intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
    ) -> HashMap<Plane<T, N>, Vec<IntersectionSegment<T, N>>> {
        if intersection_segments.is_empty() {
            return HashMap::new();
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
                let plane_key = mesh.plane_from_face(seg.initial_face_reference);
                coplanar_groups.entry(plane_key).or_default().push(seg);
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
        for group in coplanar_groups.values_mut() {
            let pairs: Vec<[usize; 2]> = group
                .iter()
                .map(|seg| seg.resulting_vertices_pair)
                .collect();

            for (i, seg) in group.iter_mut().enumerate() {
                let [v0, v1] = pairs[i];
                let mut links: Vec<usize> = pairs
                    .iter()
                    .enumerate()
                    .filter_map(|(j, &[ov0, ov1])| {
                        if j != i && (ov0 == v0 || ov1 == v0 || ov0 == v1 || ov1 == v1) {
                            Some(j)
                        } else {
                            None
                        }
                    })
                    .collect();
                links.sort_unstable();
                seg.links.extend_from_slice(&links);
            }
        }

        coplanar_groups
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
            + Div<&'a T, Output = T>
            + Neg<Output = T>,
    {
        let mut vertex_ab = [usize::MAX, usize::MAX];
        println!("processing segment {}", segment_idx);
        self.validate_connectivity();

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
                        println!(
                            "Setting vertex based on half-edge for endpoint {}: {}",
                            i, vert_hint
                        );
                        intersection_segments[segment_idx][i].vertex_hint =
                            Some([vert_hint, usize::MAX]);
                    }
                }
            }
        }

        // We then check if endpoints of the segment are already represented by vertices.
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

        // // We also check if the previous segment ends (B) in a valid vertex. If so, we can update this segment's A vertex.
        // if vertex_ab[0] == usize::MAX && segment_idx > 0 {
        //     if intersection_segments[segment_idx - 1].resulting_vertices_pair[1] != usize::MAX {
        //         vertex_ab[0] = intersection_segments[segment_idx - 1].resulting_vertices_pair[1];
        //     }
        // }

        // If they are, let's see if they are connected for an early exit.
        if vertex_ab[0] != usize::MAX && vertex_ab[1] != usize::MAX {
            if vertex_ab[0] == vertex_ab[1] {
                // Both endpoints are the same vertex, so we invalidate and skip processing.
                intersection_segments[segment_idx].invalidated = true;
                return;
            } else {
                let he_ab = self.vertices_connection(vertex_ab[0], vertex_ab[1]);
                if he_ab != usize::MAX {
                    // Both endpoints are connected, so we can simply return.
                    intersection_segments[segment_idx].resulting_vertices_pair =
                        [vertex_ab[0], vertex_ab[1]];
                    return;
                }
            }
        }

        if intersection_segments[segment_idx].a.face_hint.is_some() {
            panic!("why here?");
            // This means everything happen inside the same face.
            // If the two vertices are not connected, we need to split the segment at a face border, starting from endpoint A.
            // But we won't split the segment here. We'll create a new intersection segment that starts at the border of the
            // face A and ends at the original endpoint B. We'll also update the original intersection segment to start at
            // endpoint A and end at the border of the face A.
            // if let Some((he, t, u)) = self.get_first_half_edge_intersection_on_face(
            //     self.find_exact_valid_face(
            //         faces[0],
            //         &intersection_segments[segment_idx].segment[0],
            //         Some(&segment_direction),
            //     )
            //     .expect("Failed to find valid face for segment end"),
            //     &intersection_segments[segment_idx].segment[0],
            //     &segment_direction,
            // ) {
            //     let new_point = &intersection_segments[segment_idx].segment[0]
            //         + &segment_direction.scale(&t).0;

            //     let new_intersection = IntersectionSegment::new_default(
            //         Segment::new(&new_point, &intersection_segments[segment_idx].segment[1]),
            //         faces[1],
            //     );

            //     intersection_segments.push(new_intersection);
            //     intersection_segments[segment_idx].segment[1] = new_point;

            //     intersection_segments.push(intersection_segments[segment_idx].clone());
            //     intersection_segments[segment_idx].invalidated = true;
            // }

            // TODO
        } else {
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
                    // let e1 = mesh.vertices[vertex]
                    //     .half_edge
                    //     .expect("Vertex must have a half-edge");
                    // let e2 = mesh.half_edges[mesh.half_edges[e1].prev].twin;

                    // let e1 = (&mesh.vertices[mesh.half_edges[e1].vertex].position
                    //     - &mesh.vertices[vertex].position)
                    //     .as_vector();
                    // let e2 = (&mesh.vertices[mesh.half_edges[e2].vertex].position
                    //     - &mesh.vertices[vertex].position)
                    //     .as_vector();

                    println!("vertex: {:?}", mesh.vertices[vertex].position);

                    let mut connected_edges_dirs: SmallVec<[Vector<T, N>; 2]> = SmallVec::new();
                    for he in mesh.face_half_edges(face) {
                        if mesh.half_edges[he].vertex == vertex {
                            let outgoing_half_edge = mesh.half_edges[he].twin;
                            let other_vertex = mesh.half_edges[outgoing_half_edge].vertex;
                            let edge_direction = (&mesh.vertices[other_vertex].position
                                - &mesh.vertices[vertex].position)
                                .as_vector()
                                .normalized();
                            connected_edges_dirs.push(edge_direction);
                        } else if mesh.half_edges[mesh.half_edges[he].twin].vertex == vertex {
                            let outgoing_half_edge = he;
                            let other_vertex = mesh.half_edges[outgoing_half_edge].vertex;
                            let edge_direction = (&mesh.vertices[other_vertex].position
                                - &mesh.vertices[vertex].position)
                                .as_vector()
                                .normalized();
                            connected_edges_dirs.push(edge_direction);
                        }
                        if connected_edges_dirs.len() == 2 {
                            break;
                        }
                    }

                    if connected_edges_dirs.len() < 2 {
                        continue;
                    }

                    // Geometric "between" wedge test
                    let e1 = &connected_edges_dirs[0];
                    let e2 = &connected_edges_dirs[1];
                    let d = direction.normalized();
                    println!("d: {:?}", d);

                    let c1 = e1.cross(&d);
                    let c2 = d.cross(&e2);
                    let total = e1.cross(&e2);

                    let is_between = if total.dot(&total) > T::zero() {
                        c1.dot(&total) >= T::zero() && c2.dot(&total) >= T::zero()
                    } else {
                        d.dot(&e1) >= T::zero()
                    };

                    // Face normal alignment test
                    let normal = mesh.face_normal(face);
                    println!("normal: {:?}", normal);
                    let n_dot_d = normal.normalized().dot(&d);
                    println!("n_dot_d: {:?}", n_dot_d);
                    let is_aligned = n_dot_d.is_zero();

                    println!("is_between: {}, is_aligned: {}", is_between, is_aligned);

                    if is_between && is_aligned {
                        println!("Found face (both): {}", face);
                        return face;
                    }
                }

                // None matched both criteria
                println!(
                    "No face found for vertex {} and direction {:?}",
                    vertex, direction
                );

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
                    println!("Using half-edge hint: {}", half_edge);
                    let half_edge =
                        mesh.find_valid_half_edge(half_edge, &segment.segment[endpoint]);

                    mesh.half_edges[half_edge]
                        .face
                        .expect("Half-edge must have a face")
                } else if let Some(vertices) = segment[endpoint].vertex_hint {
                    println!("Using vertex hint: {:?}", vertices);
                    if vertices[0] != usize::MAX && vertices[1] == usize::MAX {
                        let direction = {
                            if endpoint == 0 {
                                segment.segment.direction()
                            } else {
                                -segment.segment.direction()
                            }
                        };
                        return get_face_from_vertex_and_direction(&mesh, vertices[0], &direction);
                    // } else if vertices[1] != usize::MAX && vertices[0] == usize::MAX {
                    //     return get_face_from_vertex_and_direction(
                    //         &mesh,
                    //         vertices[1],
                    //         &-segment.segment.direction(),
                    //     );
                    } else {
                        println!(
                            "vertices[0] = {}, vertices[1] = {}",
                            vertices[0], vertices[1]
                        );
                        panic!("Vertex hint must be set for the endpoint");
                    }
                } else {
                    panic!("Neither half-edge hint nor vertex hint is set for the endpoint");
                }
            }

            println!("checking {}.................", segment_idx);
            println!("segment: {:?}", intersection_segments[segment_idx]);
            // Let's find out if both endpoints are on the same face or adjacent.
            let face_a = get_face(self, &intersection_segments[segment_idx], 0);
            let face_b = get_face(self, &intersection_segments[segment_idx], 1);
            self.validate_connectivity();

            if self.faces[face_a].removed
                || self.faces[face_b].removed
                || face_a == usize::MAX
                || face_b == usize::MAX
            {
                panic!("face removed");
            }

            println!(
                "Starting with faces face_a = {}, face_b = {}",
                face_a, face_b
            );

            if face_a != face_b {
                println!("Segments not on the same face or adjacent. Attempting to fix this.");
                // For the sake of consistency, we find the endpoint B by using the segment direction.
                // We do that because we can't rely on endpoint B being on the same face as endpoint A.
                let starting_point = &intersection_segments[segment_idx].a.point;

                // if let Some(half_edge_hint) = intersection_segments[segment_idx].a.half_edge_hint {
                //     let starting_half_edge = self.find_valid_half_edge(
                //         intersection_segments[segment_idx].a.half_edge_hint.unwrap(),
                //         starting_point,
                //     );
                // }

                println!("Face {} vertices:", face_a);
                for v in self.face_vertices(face_a) {
                    println!("Vertex {}: {:?}", v, self.vertices[v].position);
                }

                let segment_direction = intersection_segments[segment_idx].segment.direction();
                if let Some((he, t, _u)) = self.get_first_half_edge_intersection_on_face(
                    face_a,
                    &starting_point,
                    &segment_direction,
                ) {
                    println!("Creating new intersection to fill the gap.");
                    let new_point = &intersection_segments[segment_idx].segment[0]
                        + &segment_direction.scale(&t).0;

                    let new_segment =
                        Segment::new(&new_point, &intersection_segments[segment_idx].segment[1]);

                    let updated_he = self.find_valid_half_edge(he, &new_point);
                    let updated_he_twin =
                        self.find_valid_half_edge(self.half_edges[updated_he].twin, &new_point);

                    if let Some((he_next, t_next)) = self.point_is_on_some_half_edge(
                        self.half_edges[updated_he_twin]
                            .face
                            .expect("Half-edge must have a face"),
                        &new_point,
                    ) {
                        let new_intersection = Self::create_segment(
                            self,
                            &new_segment,
                            self.half_edges[he_next]
                                .face
                                .expect("Half-edge must have a face"),
                            intersection_segments[segment_idx].coplanar,
                        );

                        println!("New intersection segment added: {:?}", new_intersection);
                        intersection_segments.push(new_intersection);
                    } else if let Some((he_next, t_next, _u)) = self
                        .get_first_half_edge_intersection_on_face(
                            self.half_edges[updated_he_twin]
                                .face
                                .expect("Half-edge must have a face"),
                            &new_point,
                            &segment_direction,
                        )
                    {
                        let new_intersection = Self::create_segment(
                            self,
                            &new_segment,
                            self.half_edges[he_next]
                                .face
                                .expect("Half-edge must have a face"),
                            intersection_segments[segment_idx].coplanar,
                        );

                        println!("New intersection segment added: {:?}", new_intersection);
                        intersection_segments.push(new_intersection);
                    } else {
                        panic!(
                            "Failed to split segment at the intersection point: {:?}",
                            new_point
                        );
                    }

                    intersection_segments[segment_idx].segment[1] = new_point.clone();
                    intersection_segments[segment_idx].b.point = new_point;
                    intersection_segments[segment_idx].b.half_edge_hint = Some(updated_he);
                    intersection_segments[segment_idx].b.vertex_hint = Some([
                        self.half_edges[updated_he].vertex,
                        self.half_edges[updated_he_twin].vertex,
                    ]);

                    println!(
                        "Intersection segment updated: {:?}",
                        intersection_segments[segment_idx]
                    );
                    vertex_ab[1] = usize::MAX;
                } else {
                    panic!(
                        "Failed to find a valid half-edge intersection on face {} for point {:?}",
                        face_a, starting_point
                    );
                }
            }

            // Try again.
            let face_b = get_face(self, &intersection_segments[segment_idx], 1);
            println!("face_a = {}, face_b = {}", face_a, face_b);

            if face_a == face_b {
                println!("Saved?");
                println!("vertex_ab = {:?}", vertex_ab);
                // Process the segment endpoints normally.
                for i in 0..2 {
                    if vertex_ab[i] == usize::MAX {
                        if intersection_segments[segment_idx][i]
                            .half_edge_hint
                            .is_some()
                        {
                            let point = &intersection_segments[segment_idx][i].point;
                            // println!(
                            //     "half_edge_hint: {:?}",
                            //     self.vertices[self.half_edges[intersection_segments[segment_idx]
                            //         [i]
                            //         .half_edge_hint
                            //         .unwrap()]
                            //     .vertex]
                            //         .position
                            // );
                            let half_edge = self.find_valid_half_edge(
                                intersection_segments[segment_idx][i]
                                    .half_edge_hint
                                    .unwrap(),
                                point,
                            );
                            // println!(
                            //     "half_edge: {:?}",
                            //     self.vertices[self.half_edges[half_edge].vertex].position
                            // );
                            // let target_v = self.half_edges[half_edge].vertex;
                            // let source_v = self.half_edges[self.find_valid_half_edge(self.half_edges[half_edge].twin, point)].vertex;
                            // let target_p = &self.vertices[target_v].position;
                            // let source_p = &self.vertices[source_v].position;

                            // let u = point_position_on_segment(source_p, target_p, point).expect("Failed to compute point position on segment");
                            let split_result = self
                                .split_edge(
                                    aabb_tree,
                                    half_edge,
                                    &intersection_segments[segment_idx].segment[i],
                                )
                                .expect("Failed to split edge for segment end");

                            vertex_ab[i] = split_result.vertex;
                        }
                    }
                }

                if vertex_ab[0] == usize::MAX || vertex_ab[1] == usize::MAX {
                    // If we still don't have both vertices, we can't proceed.
                    intersection_segments[segment_idx].invalidated = true;
                    return;
                }

                println!("vertex_ab = {:?}", vertex_ab);

                let he_ab = self.vertices_connection(vertex_ab[0], vertex_ab[1]);
                if he_ab != usize::MAX {
                    // Both endpoints are connected, so we can simply update the intersection_segment and return.
                    // let face_1 = self.half_edges[he_ab]
                    //     .face
                    //     .expect("Half-edge must have a face");
                    // let face_2 = self.half_edges
                    //     [self.find_valid_half_edge(self.half_edges[he_ab].twin)]
                    // .face
                    // .expect("Half-edge must have a face");

                    intersection_segments[segment_idx].resulting_vertices_pair =
                        [vertex_ab[0], vertex_ab[1]];
                } else {
                    panic!("Failed to connect segment endpoints");
                }
            } else {
                panic!("Faces are not adjacent or equal, but they should be. This is a bug.");
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

// fn should_connect_segments<T: Scalar, const N: usize>(
//     seg1: &IntersectionSegment<T, N>,
//     seg2: &IntersectionSegment<T, N>,
//     shared_vertex: usize,
// ) -> bool {
//     // Don't connect coplanar segments here (handled separately)
//     if seg1.coplanar || seg2.coplanar {
//         return false;
//     }

//     // Check if segments share exactly one face (continuous boundary condition)
//     let faces1 = &seg1.resulting_faces;
//     let faces2 = &seg2.resulting_faces;

//     let shared_faces = faces1.iter().filter(|&&f| faces2.contains(&f)).count();

//     // Only connect if segments share exactly one face (forming a continuous boundary)
//     shared_faces == 1
// }

// fn build_coplanar_links<T: Scalar, const N: usize>(
//     intersection_segments: &mut [IntersectionSegment<T, N>],
//     segment_indices: &[usize],
// ) {
//     // For coplanar segments, build a more restricted connectivity
//     for &seg_idx in segment_indices {
//         let [v0, v1] = intersection_segments[seg_idx].resulting_vertices_pair;
//         let mut connected = Vec::new();

//         for &other_idx in segment_indices {
//             if other_idx == seg_idx {
//                 continue;
//             }

//             let [ov0, ov1] = intersection_segments[other_idx].resulting_vertices_pair;

//             // Only connect if they share exactly one vertex
//             let shared_vertices = [
//                 (v0 == ov0) as u8,
//                 (v0 == ov1) as u8,
//                 (v1 == ov0) as u8,
//                 (v1 == ov1) as u8,
//             ]
//             .iter()
//             .sum::<u8>();

//             if shared_vertices == 1 {
//                 connected.push(other_idx);
//             }
//         }

//         connected.sort_unstable();
//         intersection_segments[seg_idx].links.clear();
//         intersection_segments[seg_idx]
//             .links
//             .extend_from_slice(&connected);
//     }
// }

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
        // Hash as unordered pair: a ^ b
        let mut hasher_a = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut hasher_a);
        let ha = hasher_a.finish();

        let mut hasher_b = std::collections::hash_map::DefaultHasher::new();
        self.1.hash(&mut hasher_b);
        let hb = hasher_b.finish();

        (ha ^ hb).hash(state);
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
    let mut keep_flags = vec![false; segments.len()];

    for (i, seg) in segments.iter().enumerate() {
        let key = SegmentKey(&seg.segment.a, &seg.segment.b);

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        match seen.get(&hash) {
            Some(&(existing_i, existing_is_coplanar)) => {
                // If current one is coplanar and previous was not, prefer current
                if seg.coplanar && !existing_is_coplanar {
                    keep_flags[existing_i] = false;
                    keep_flags[i] = true;
                    seen.insert(hash, (i, true));
                } else {
                    // Otherwise, keep existing
                    // Current one will not be marked to keep
                }
            }
            None => {
                keep_flags[i] = true;
                seen.insert(hash, (i, seg.coplanar));
            }
        }
    }

    // Retain only marked segments
    let mut write = 0;
    for read in 0..segments.len() {
        if keep_flags[read] {
            if write != read {
                segments[write] = std::mem::take(&mut segments[read]);
            }
            write += 1;
        }
    }
    segments.truncate(write);
}

fn get_seed_face<T: Scalar, const N: usize>(
    a: &Mesh<T, N>,
    b: &Mesh<T, N>,
    tree_b: &AabbTree<T, 3, Point<T, 3>, usize>,
    intersection_segments: &Vec<IntersectionSegment<T, N>>,
    boundary_faces: &HashSet<usize>,
    include_on_surface: bool,
) -> (usize, usize)
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
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

            for (i, f) in faces.iter().enumerate() {
                let c = a.face_centroid(*f).0;
                let c3 = Point::<T, 3>::from_vals([c[0].clone(), c[1].clone(), c[2].clone()]);
                let point_in_mesh = b.point_in_mesh(&tree_b, &c3);
                if point_in_mesh == PointInMeshResult::Inside {
                    selected_face = i;
                    return true;
                } else if point_in_mesh == PointInMeshResult::OnSurface && include_on_surface {
                    selected_face = i;
                    return true;
                }
            }
            false
        })
        .expect("No seed face found inside B");

    (seed_idx, selected_face)
}
