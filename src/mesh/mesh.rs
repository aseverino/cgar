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

use num_traits::ToPrimitive;
use std::hash::Hash;

use crate::{
    geometry::{
        Aabb, AabbTree, FromCoords, Point3, Vector3, point::PointOps,
        spatial_element::SpatialElement, tri_tri_intersect::tri_tri_intersection,
        vector::VectorOps,
    },
    numeric::{cgar_rational::CgarRational, scalar::Scalar},
    operations::{Abs, One, Pow, Sqrt, Zero},
};

use super::{face::Face, half_edge::HalfEdge, point_trait::PointTrait, vertex::Vertex};
use std::{
    collections::{HashMap, HashSet},
    ops::{Add, Div, Mul, Neg, Sub},
};

pub enum BooleanOp {
    Union,
    Intersection,
    Difference,
}

#[derive(Debug, Clone)]
pub struct Mesh<T: Scalar, const N: usize, P: SpatialElement<T, N>> {
    pub vertices: Vec<Vertex<T, N, P>>,
    pub half_edges: Vec<HalfEdge>,
    pub faces: Vec<Face>,

    pub edge_map: HashMap<(usize, usize), usize>,
}

impl<T: Scalar, const N: usize, P: SpatialElement<T, N>> Mesh<T, N, P> {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            half_edges: Vec::new(),
            faces: Vec::new(),
            edge_map: HashMap::new(),
        }
    }

    /// Compute the AABB of face `f`.
    pub fn face_aabb(&self, f: usize) -> Aabb<T, N, P>
    where
        T: Scalar,
        P: SpatialElement<T, 3> + FromCoords<T, 3>,
        for<'a> &'a T: Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Add<&'a T, Output = T>
            + Div<&'a T, Output = T>,
    {
        let he = self.faces[f].half_edge;
        let vs = self.face_half_edges(f);
        let p0 = &self.vertices[self.half_edges[vs[0]].vertex].position;
        let p1 = &self.vertices[self.half_edges[vs[1]].vertex].position;
        let p2 = &self.vertices[self.half_edges[vs[2]].vertex].position;
        Aabb::from_points(p0, p1).union(&Aabb::from_points(p1, p2))
    }

    pub fn add_vertex(&mut self, position: P) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(Vertex::new(position));
        idx
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
    pub fn face_centroid(&self, f: usize) -> Vec<f64> {
        let he_idxs = &self.face_half_edges(f);
        let dim = N;
        let mut sum = vec![0.0; dim];
        for he in he_idxs {
            let v_idx = self.half_edges[*he].vertex;
            let p = &self.vertices[v_idx].position;
            for i in 0..dim {
                // use to_f64, which returns Option<f64>
                sum[i] += p[i].to_f64().expect("cannot convert to f64");
            }
        }
        let n = he_idxs.len() as f64;
        sum.iter().map(|c| c / n).collect()
    }

    pub fn face_area(&self, f: usize) -> f64 {
        assert_eq!(N, 2);
        let vs = self.face_vertices(f);
        let p0 = &self.vertices[vs[0]].position;
        let p1 = &self.vertices[vs[1]].position;
        let p2 = &self.vertices[vs[2]].position;
        let x0 = p0[0].to_f64().unwrap();
        let y0 = p0[1].to_f64().unwrap();
        let x1 = p1[0].to_f64().unwrap();
        let y1 = p1[1].to_f64().unwrap();
        let x2 = p2[0].to_f64().unwrap();
        let y2 = p2[1].to_f64().unwrap();
        ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs() * 0.5
    }

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
            loop {
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

    /// Returns the 1-ring neighboring vertex indices of vertex `v`.
    pub fn one_ring_neighbors(&self, v: usize) -> Vec<usize> {
        self.outgoing_half_edges(v)
            .iter()
            .map(|&he_idx| self.half_edges[he_idx].vertex)
            .collect()
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

    /// Returns true if vertex `v` has any outgoing ghost edge (face == None).
    pub fn is_boundary_vertex(&self, v: usize) -> bool {
        self.outgoing_half_edges(v)
            .into_iter()
            .any(|he| self.half_edges[he].face.is_none())
    }

    /// Returns all vertex indices that lie on at least one boundary loop.
    pub fn boundary_vertices(&self) -> Vec<usize> {
        (0..self.vertices.len())
            .filter(|&v| self.is_boundary_vertex(v))
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
        let u = self.half_edges[he_c].vertex; // c→u
        let v = self.half_edges[he_a].vertex; // u→v
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

    /// Splits the interior edge `he` by inserting a new vertex at `pos`.
    /// The two adjacent triangles are each subdivided into two, yielding
    /// four faces in place of the original two.  Returns the new vertex index.
    pub fn split_edge_rebuild(&mut self, he: usize, pos: P) -> Result<usize, &'static str> {
        // 1) Pre-flight checks
        let he_twin = self.half_edges[he].twin;
        if he_twin == usize::MAX {
            return Err("cannot split a boundary edge");
        }
        let f0 = self.half_edges[he].face.ok_or("he has no face")?;
        let f1 = self.half_edges[he_twin].face.ok_or("twin has no face")?;

        // 2) Gather old vertex positions
        let mut old_positions: Vec<P> = self.vertices.iter().map(|v| v.position.clone()).collect();
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

    // pub fn boolean(&self, other: &Mesh<T, P>, op: BooleanOp) -> Mesh<T, P>
    // where
    //     T: Scalar,
    //     P: SpatialElement<T> + PointOps<T, Vector3<T>> + PointTrait<T> + FromCoords<T>,
    //     P::Vector: VectorOps<T, Vector3<T>> + From<Point3<T>> + From<(T, T, T)>,
    //     Vector3<T>: From<P::Vector>,
    //     for<'a> &'a T: Sub<&'a T, Output = T>
    //         + Mul<&'a T, Output = T>
    //         + Add<&'a T, Output = T>
    //         + Div<&'a T, Output = T>,
    // {
    //     // 1) Make working copies and ensure boundary‐loops are built
    //     let mut a = self.clone();
    //     let mut b = other.clone();
    //     a.build_boundary_loops();
    //     b.build_boundary_loops();

    //     // 2) Build an AABB‐tree over b’s faces
    //     let mut items_b = Vec::new();
    //     for fb in 0..b.faces.len() {
    //         let ab = b.face_aabb(fb);
    //         items_b.push((ab, fb));
    //     }
    //     let tree_b = AabbTree::build(items_b);

    //     // 3) Collect all intersection segments (fa, fb, p0, p1)
    //     let mut segments: Vec<(usize, usize, P, P)> = Vec::new();
    //     for fa in 0..a.faces.len() {
    //         let ab_a = a.face_aabb(fa);
    //         let mut cands = Vec::new();
    //         tree_b.query(&ab_a, &mut cands);

    //         let fv = a.face_vertices(fa);
    //         if fv.len() != 3 {
    //             panic!("Expected a triangle face");
    //         }
    //         let pa0 = a.vertices[fv[0]].position.clone();
    //         let pa1 = a.vertices[fv[1]].position.clone();
    //         let pa2 = a.vertices[fv[2]].position.clone();

    //         for &fb in &cands {
    //             let face_vs = b.face_vertices(*fb);
    //             let qb0 = b.vertices[face_vs[0]].position.clone();
    //             let qb1 = b.vertices[face_vs[1]].position.clone();
    //             let qb2 = b.vertices[face_vs[2]].position.clone();

    //             if let Some((i0, i1)) = tri_tri_intersection(&pa0, &pa1, &pa2, &qb0, &qb1, &qb2) {
    //                 segments.push((fa, *fb, i0, i1));
    //             }
    //         }
    //     }

    //     // 4) Split each mesh along those segments so they become conforming
    //     for (fa, _fb, p, q) in &segments {
    //         a.split_segment_on_face(*fa, p.clone(), q.clone());
    //     }
    //     for (_fa, fb, p, q) in &segments {
    //         b.split_segment_on_face(*fb, p.clone(), q.clone());
    //     }

    //     // 5) Classification: keep faces according to `op`
    //     let mut result = Mesh::new();
    //     // append all vertices/faces of a into result
    //     // (you’ll need a helper that bulk‐copies meshes, or just re‐add)
    //     // then for each face of b:
    //     //    compute its centroid
    //     //    let inside = result.point_in_mesh(&centroid)
    //     //    if matches(op, inside) then append that face

    //     // 6) Rebuild twins/boundaries and return
    //     result.build_boundary_loops();
    //     result
    // }

    // /// Split face `f` so that the segment `p0→p1` lies along its edges.
    // /// You can:
    // ///  - find which edges of `f` each point lies on (or if equal to a vertex),
    // ///  - call your in‐place `split_edge` on those edges,
    // ///  - repeat until both p0 & p1 are actual vertices of `a`.
    // fn split_segment_on_face(&mut self, f: usize, p0: P, p1: P)
    // where
    //     T: Scalar,
    //     for<'a> &'a T: Sub<&'a T, Output = T>
    //         + Add<&'a T, Output = T>
    //         + Mul<&'a T, Output = T>
    //         + Div<&'a T, Output = T>,
    //     P: PointTrait<T> + FromCoords<T> + SpatialElement<T>,
    // {
    //     // We’ll insert each point in turn.
    //     for point in [p0, p1].into_iter() {
    //         // 1) Walk the half-edge cycle around face `f`
    //         let start_he = self.faces[f].half_edge;
    //         let mut he = start_he;
    //         loop {
    //             let v_from = self.half_edges[he].prev; // he.prev → origin of this he
    //             let v_to = self.half_edges[he].vertex; // he.vertex → dest of this he
    //             let p_from = &self.vertices[v_from].position;
    //             let p_to = &self.vertices[v_to].position;

    //             // 2) If the point exactly matches an existing vertex, we’re done.
    //             if point == *p_from || point == *p_to {
    //                 break;
    //             }

    //             // 3) Otherwise, if it lies on this edge, split it here.
    //             if point_on_segment_3d(&point, p_from, p_to) {
    //                 // split_edge returns the index of the new vertex
    //                 let _new_vid = self.split_edge_rebuild(he, point.clone());
    //                 break;
    //             }

    //             // 4) Advance to the next boundary half‐edge of face `f`
    //             he = self.half_edges[he].next;
    //             if he == start_he {
    //                 // panic!(
    //                 //     "split_segment_on_face: point {:?} did not lie on face {} boundary",
    //                 //     point, f
    //                 // );
    //             }
    //         }
    //     }
    // }

    // /// Test if `point` lies inside this mesh (using ray‐cast + AABB‐tree).
    // fn point_in_mesh(&self, point: &P) -> bool {
    //     // TODO:
    //     // 1) Build/rerun an AABB‐tree on faces if you don’t have one.
    //     // 2) Cast a ray in any direction, count face crossings.
    //     // 3) Return `count % 2 == 1`.
    //     false
    // }
}

// /// Test whether `p` lies on the segment [a→b] in 3D.
// /// For floats, use an ε; for exact rationals you can do an exact check.
// fn point_on_segment_3d<P, T>(p: &P, a: &P, b: &P) -> bool
// where
//     T: Scalar,
//     P: SpatialElement<T> + PointTrait<T> + FromCoords<T>,
//     for<'a> &'a T: Sub<&'a T, Output = T>
//         + Add<&'a T, Output = T>
//         + Mul<&'a T, Output = T>
//         + Div<&'a T, Output = T>,
// {
//     // convert to 3D coordinates
//     let coords_p: Vec<T> = (0..P::dimensions()).map(|i| p.coord(i).clone()).collect();
//     let coords_a: Vec<T> = (0..P::dimensions()).map(|i| a.coord(i).clone()).collect();
//     let coords_b: Vec<T> = (0..P::dimensions()).map(|i| b.coord(i).clone()).collect();

//     // 1) Check collinearity: (p - a) × (b - a) == 0 vector
//     //    but in N dims, we check that (p - a) is a scalar multiple of (b - a)
//     let mut t_opt: Option<T> = None;
//     for i in 0..coords_p.len() {
//         let da = &coords_p[i] - &coords_a[i];
//         let db = &coords_b[i] - &coords_a[i];
//         if db != T::zero() {
//             // t = da / db must be the same for all coords where db != 0
//             let t = &da / &db;
//             if let Some(prev_t) = &t_opt {
//                 if (&t - &prev_t).abs() > T::from(1e-8) {
//                     return false;
//                 }
//             } else {
//                 t_opt = Some(t);
//             }
//         } else if da != T::zero() {
//             // b - a = 0 in this dim but p - a != 0 ⇒ off the line
//             return false;
//         }
//     }

//     // 2) Check 0 <= t <= 1 for the scalar factor
//     if let Some(t) = t_opt {
//         t >= T::zero() && t <= T::one()
//     } else {
//         // degenerate edge a==b; only true if p==a
//         coords_p == coords_a
//     }
// }
