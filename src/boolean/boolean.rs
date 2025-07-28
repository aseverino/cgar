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
    collections::{HashMap, VecDeque},
    ops::{Add, Div, Mul, Sub},
    time::Instant,
};

use crate::{
    geometry::{
        Aabb, AabbTree,
        point::{Point, PointOps},
        segment::{Segment, SegmentOps},
        spatial_element::SpatialElement,
        tri_tri_intersect::tri_tri_intersection,
        vector::{Vector, VectorOps},
    },
    mesh::mesh::{IntersectionSegment, Mesh, VertexSource},
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
        loop_segments: &[Vec<usize>],
    ) -> Vec<bool>;

    fn build_chains(intersection_segments: &Vec<IntersectionSegment<T, N>>) -> Vec<Vec<usize>>;
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
        let adj = self.build_face_adjacency_graph();
        let boundary_map = self.build_robust_boundary_map(intersection_segments, loop_segments);

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
        let seed = boundary_map
            .keys()
            .copied()
            .find(|&f| {
                f < self.faces.len() && !self.faces[f].removed && {
                    let c = self.face_centroid(f).0;
                    let c3 = Point::<T, 3>::from_vals([c[0].clone(), c[1].clone(), c[2].clone()]);
                    other.point_in_mesh(&tree_b, &c3)
                }
            })
            .unwrap_or(0);
        state[seed] = true;

        // 4) iterative flood‐fill without crossing the boundary_map
        let mut visited = vec![false; self.faces.len()];
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
                    if boundary_map.get(&curr) == Some(&nbr)
                        || boundary_map.get(&nbr) == Some(&curr)
                    {
                        continue;
                    }
                    visited[nbr] = true;
                    state[nbr] = true;
                    queue.push_back(nbr);
                }
            }
        }

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

                if let Some(s) = tri_tri_intersection(&pa, &pb) {
                    let segment_n = Segment::<T, N>::new(
                        &Point::<T, N>::from_vals(from_fn(|i| s.a[i].clone())),
                        &Point::<T, N>::from_vals(from_fn(|i| s.b[i].clone())),
                    );
                    if segment_n.length().is_positive() {
                        intersection_segments_a
                            .push(IntersectionSegment::new_default(segment_n.clone(), fa));
                        intersection_segments_b
                            .push(IntersectionSegment::new_default(segment_n, *fb));
                    }
                }
            }
        }
        println!("Intersection segments collected in {:.2?}", start.elapsed());

        let start = Instant::now();
        Mesh::filter_coplanar_intersections(&mut intersection_segments_a);
        Mesh::filter_coplanar_intersections(&mut intersection_segments_b);
        println!("Coplanar intersections filtered in {:.2?}", start.elapsed());

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
        let chains_a = Self::build_chains(&intersection_segments_a);
        let chains_b = Self::build_chains(&intersection_segments_b);
        println!("Chains built in {:.2?}", start.elapsed());

        // Classify A faces using topological method
        let start = Instant::now();
        let a_classifications =
            a.classify_faces_inside_intersection_loops(&b, &intersection_segments_a, &chains_a);
        println!(
            "A faces classified inside intersection loops in {:.2?}",
            start.elapsed()
        );

        // // remove_invalidated_faces(&mut a);

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

                let b_classifications = b.classify_faces_inside_intersection_loops(
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
                let b_classifications = b.classify_faces_inside_intersection_loops(
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
                let start = Instant::now();
                let b_classifications = b.classify_faces_inside_intersection_loops(
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
                println!(
                    "B faces classified inside intersection loops in {:.2?}",
                    start.elapsed()
                );
            }
        }

        result.remove_unused_vertices();
        result.remove_invalidated_faces();
        // result.build_boundary_loops();
        result
    }

    fn build_chains(intersection_segments: &Vec<IntersectionSegment<T, N>>) -> Vec<Vec<usize>> {
        let mut next_seg: HashMap<usize, usize> =
            HashMap::with_capacity(intersection_segments.len());
        for (i, seg) in intersection_segments.iter().enumerate() {
            next_seg.insert(seg.resulting_vertices_pair[0].clone(), i);
        }
        let mut visited = vec![false; intersection_segments.len()];
        let mut chains: Vec<Vec<usize>> = Vec::new();

        for i in 0..intersection_segments.len() {
            if visited[i] {
                continue;
            }
            // start a new loop
            let mut cycle = Vec::new();
            let mut cur = i;
            loop {
                if visited[cur] {
                    break;
                }
                visited[cur] = true;
                cycle.push(cur);
                let next_vertex = &intersection_segments[cur].resulting_vertices_pair[1];
                // follow the segment whose start == this end
                cur = match next_seg.get(next_vertex) {
                    Some(&j) => j,
                    None => break, // shouldn’t happen if data is consistent
                };
                if cur == i {
                    // we've closed the loop
                    break;
                }
            }
            chains.push(cycle);
        }

        chains
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
            self.find_valid_face(segment.original_face, &segment.segment.a)
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
            self.find_valid_face(segment.original_face, &segment.segment.b)
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
                    } else {
                        let he = self.half_edge_between(vertex_a, vertex_b);
                        if let Some(he) = he {
                            intersection_segments[segment_idx].resulting_faces[0] =
                                self.half_edges[he].face.unwrap_or(usize::MAX);
                            intersection_segments[segment_idx].resulting_faces[1] = self.half_edges
                                [self.half_edges[he].twin]
                                .face
                                .unwrap_or(usize::MAX);
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

        if vertex_a != usize::MAX
            && vertex_b != usize::MAX
            && !self.are_vertices_connected(vertex_a, vertex_b)
        {
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
