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
    collections::{HashMap, HashSet},
    ops::{Add, Div, Mul, Neg, Sub},
    process::Output,
};

use smallvec::*;

use crate::{
    geometry::{
        Aabb, AabbTree,
        point::{Point, PointOps},
        spatial_element::SpatialElement,
        util::proj_along,
        vector::{Vector, VectorOps},
    },
    impl_mesh, kernel,
    mesh::{
        basic_types::*, face::Face, half_edge::HalfEdge, intersection_segment::IntersectionSegment,
        vertex::Vertex,
    },
    numeric::{cgar_f64::CgarF64, scalar::Scalar},
};

impl_mesh! {
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

    pub fn position_to_hash_key(&self, pos: &Point<T, N>) -> (i64, i64, i64) {
        // Cell size tied to merge threshold keeps hashing consistent with equality.
        let cell = T::point_merge_threshold().to_f64().unwrap_or(1e-5);
        let inv = if cell.is_finite() && cell > 0.0 { 1.0 / cell } else { 1.0e5 };

        #[inline(always)]
        fn floor_i64(x: f64) -> i64 {
            let xf = x.floor();
            if xf >= i64::MAX as f64 { i64::MAX }
            else if xf <= i64::MIN as f64 { i64::MIN }
            else { xf as i64 }
        }

        // Support any N; unused axes map to 0.
        let get = |i: usize| -> f64 {
            if i < N { pos[i].to_f64().unwrap_or(0.0) } else { 0.0 }
        };

        let x = floor_i64(get(0) * inv);
        let y = floor_i64(get(1) * inv);
        let z = floor_i64(get(2) * inv);
        (x, y, z)
    }

    pub fn build_boundary_map(
        &self,
        intersection_segments: &[IntersectionSegment<T, N>],
    ) -> HashSet<(usize, usize)> {
        fn ordered(a: usize, b: usize) -> (usize, usize) {
            if a < b { (a, b) } else { (b, a) }
        }

        let mut boundary_edges = HashSet::new();

        for (seg_idx, seg) in intersection_segments.iter().enumerate() {
            let he = self
                .edge_map
                .get(&(
                    seg.resulting_vertices_pair[0],
                    seg.resulting_vertices_pair[1],
                ))
                .expect(&format!(
                    "Edge map must contain the segment vertices pair. Segment {}",
                    seg_idx
                ));

            let face0 = self.half_edges[*he]
                .face
                .expect("Half-edge must have a face");
            let twin = self.find_valid_half_edge(self.half_edges[*he].twin, &seg.segment.a);
            let face1 = self.half_edges[twin]
                .face
                .expect("Half-edge must have a face");

            boundary_edges.insert(ordered(face0, face1));
        }

        boundary_edges
    }

    /// - Border half-edges have `face == None` and a valid interior twin `t = twin(b)`.
    /// - For each border `b = u->v`, we set:
    ///       b.next = the next border spoke at vertex v (found by rotating through interior faces)
    ///       and derive b_prev by setting `half_edges[b.next].prev = b`.
    pub fn build_boundary_loops(&mut self) {
        let m = self.half_edges.len();

        // 0) Collect all live border half-edges once
        let mut borders: Vec<usize> = Vec::new();
        borders.reserve(m);
        for i in 0..m {
            let e = &self.half_edges[i];
            if !e.removed && e.face.is_none() {
                borders.push(i);
            }
        }

        // 1) Compute next[b] by rotating around the head of `b` until we hit the next border spoke.
        //    We only traverse interior half-edges' prev/next; we do not read border prev/next here.
        let mut next_of = vec![usize::MAX; m];

        for &b in &borders {
            let t0 = self.half_edges[b].twin; // interior v->u (origin = head(b))
            if t0 >= m || self.half_edges[t0].removed || !self.face_ok(t0) {
                // Degenerate: keep a safe self-loop; will still derive prev from next below
                next_of[b] = b;
                continue;
            }

            // Walk around the head vertex via interior spokes until we encounter a border spoke.
            let mut t = t0;
            let mut steps = 0usize;
            let b_next = loop {
                // CCW around the origin (head of b): twin(prev(t))
                let prev_t = self.half_edges[t].prev;
                let cand   = self.half_edges[prev_t].twin; // this is a spoke leaving the same vertex

                if cand >= m || self.half_edges[cand].removed {
                    // Paranoia: bail to self-loop
                    break b;
                }
                if self.half_edges[cand].face.is_none() {
                    // Found the next BORDER spoke around the head vertex
                    break cand;
                }

                // Keep rotating around the same vertex via the interior side of `cand`
                t = self.half_edges[cand].twin;
                if t >= m || self.half_edges[t].removed || !self.face_ok(t) {
                    // If we lose a valid interior, bail to self-loop
                    break b;
                }

                steps += 1;
                if steps > m {
                    // Safety bound (should be <= valence at the vertex)
                    break b;
                }
            };

            next_of[b] = b_next;
        }

        // 2) Write next and derive prev from next to guarantee reciprocity
        for &b in &borders {
            let nb = next_of[b];
            if nb != usize::MAX {
                self.half_edges[b].next = nb;
            }
        }
        for &b in &borders {
            let nb = self.half_edges[b].next;
            if nb < m {
                self.half_edges[nb].prev = b;
            }
        }

        // 3) Optional sanity checks (enabled in debug builds)
        #[cfg(debug_assertions)]
        {
            for &b in &borders {
                let he = &self.half_edges[b];
                let n = he.next;
                let p = he.prev;
                assert!(n < m && p < m, "boundary next/prev out of range at {}", b);
                assert!(self.half_edges[n].face.is_none(), "b.next must be border at {}", b);
                assert!(self.half_edges[p].face.is_none(), "b.prev must be border at {}", b);
                assert_eq!(self.half_edges[n].prev, b, "boundary next->prev mismatch at {}", b);
                assert_eq!(self.half_edges[p].next, b, "boundary prev->next mismatch at {}", b);
            }
        }
    }

    pub fn build_face_adjacency_graph(&self) -> HashMap<usize, Vec<usize>> {
        let mut adjacency_graph = HashMap::new();

        for face_idx in 0..self.faces.len() {
            let mut adjacent_faces = Vec::new();

            // Use iterative approach with safety limits
            if let Some(half_edges) = self.get_face_half_edges_iterative(face_idx) {
                for he_idx in half_edges {
                    if he_idx < self.half_edges.len() {
                        let twin_idx = self.half_edges[he_idx].twin;

                        if twin_idx != usize::MAX && twin_idx < self.half_edges.len() {
                            if let Some(twin_face) = self.half_edges[twin_idx].face {
                                if twin_face != face_idx
                                    && twin_face < self.faces.len()
                                    && self.faces[twin_face].half_edge != usize::MAX
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

        adjacency_graph
    }

    /// Adds a triangle face given three vertex indices (in CCW order).
    /// Border (outside) half-edges have `face = None` instead of pointing to a ghost/null face.
    /// Returns the index of the newly created face.
    pub fn add_triangle(&mut self, v0: usize, v1: usize, v2: usize) -> usize {
        // CCW triangle directed edges
        let edge_vertices = [(v0, v1), (v1, v2), (v2, v0)];

        // Reserve a new real face slot NOW so the index stays stable
        let face_idx = self.faces.len();
        // Temporary placeholder; will set .half_edge after edges are wired
        self.faces.push(Face::new(0));

        // Build/reuse the 3 half-edges that bound this new face
        let mut edge_indices = [usize::MAX; 3];

        for (i, &(from, to)) in edge_vertices.iter().enumerate() {
            if let Some(&he_idx) = self.edge_map.get(&(from, to)) {
                // Directed edge already exists.
                let twin = self.half_edges[he_idx].twin;

                if self.half_edges[he_idx].face.is_none() {
                    // This direction is free: assign it to the new face.
                    self.half_edges[he_idx].face = Some(face_idx);
                    edge_indices[i] = he_idx;
                } else if self.half_edges[twin].face.is_none() {
                    // The opposite direction is free: assign the twin to the new face.
                    self.half_edges[twin].face = Some(face_idx);
                    edge_indices[i] = twin; // IMPORTANT: use the direction consistent with this face
                } else {
                    debug_assert!(
                        false,
                        "add_triangle: non-manifold edge ({},{}) — both directions already bound to faces",
                        from, to
                    );
                    // Fallback: keep mesh wired; pick he_idx to avoid UB.
                    edge_indices[i] = he_idx;
                }

                // Ensure twin linkage is symmetric
                let t = self.half_edges[edge_indices[i]].twin;
                self.half_edges[t].twin = edge_indices[i];
                self.half_edges[edge_indices[i]].twin = t;
            } else {
                // Create a brand-new interior half-edge for this triangle
                let he_idx = self.half_edges.len();
                let mut he = HalfEdge::new(to);
                he.face = Some(face_idx);
                self.half_edges.push(he);
                self.edge_map.insert((from, to), he_idx);
                edge_indices[i] = he_idx;

                // Create / hook twin (border) if missing
                if let Some(&rev_idx) = self.edge_map.get(&(to, from)) {
                    // Existing reverse edge becomes the twin
                    self.half_edges[he_idx].twin = rev_idx;
                    self.half_edges[rev_idx].twin = he_idx;
                } else {
                    // Create border half-edge (to -> from). It lives "outside", so face=None.
                    let border_idx = self.half_edges.len();
                    let mut bhe = HalfEdge::new(from);
                    bhe.twin = he_idx;
                    // temp self-loop; we will rewire below
                    bhe.next = border_idx;
                    bhe.prev = border_idx;
                    // face stays None (border)
                    self.half_edges.push(bhe);
                    self.edge_map.insert((to, from), border_idx);

                    self.half_edges[he_idx].twin = border_idx;
                }
            }
        }

        // Link the triangle ring (three interior half-edges)
        let e0 = edge_indices[0];
        let e1 = edge_indices[1];
        let e2 = edge_indices[2];
        self.half_edges[e0].next = e1;
        self.half_edges[e0].prev = e2;
        self.half_edges[e1].next = e2;
        self.half_edges[e1].prev = e0;
        self.half_edges[e2].next = e0;
        self.half_edges[e2].prev = e1;

        // --- Wire boundary next/prev locally for any border twin of these three edges ---
        // Walk until we hit a BORDER spoke (face=None), skipping interior spokes.
        let mut wire_border = |h_interior: usize, this: &mut Self| {
            let b = this.half_edges[h_interior].twin;
            if this.half_edges[b].removed || this.half_edges[b].face.is_some() {
                return; // not a border half-edge; nothing to wire
            }

            // Find b.next: rotate CCW around head(b) via interior edges until next border
            let mut t = h_interior; // interior with origin=head(b)
            let b_next = loop {
                let prev_t = this.half_edges[t].prev;            // ... w->head(b)
                let cand   = this.half_edges[prev_t].twin;       // head(b) -> w
                if this.half_edges[cand].face.is_none() && !this.half_edges[cand].removed {
                    break cand; // next border spoke
                }
                t = this.half_edges[cand].twin;                  // stay on interior around head(b)
                if t == h_interior { break b; }                  // safety: degenerate -> self
            };

            // Find b.prev: rotate CW around head(b) via interior edges until previous border
            let mut t = h_interior;
            let b_prev = loop {
                let next_t = this.half_edges[t].next;            // ... tail(b)->u
                let cand   = this.half_edges[next_t].twin;       // head(b) -> u
                if this.half_edges[cand].face.is_none() && !this.half_edges[cand].removed {
                    break cand; // previous border spoke
                }
                t = this.half_edges[cand].twin;
                if t == h_interior { break b; }                  // safety
            };

            // Write both directions to guarantee reciprocity
            this.half_edges[b].next = b_next;
            this.half_edges[b_next].prev = b;
            this.half_edges[b].prev = b_prev;
            this.half_edges[b_prev].next = b;
        };

        wire_border(e0, self);
        wire_border(e1, self);
        wire_border(e2, self);

        // Attach representative half-edges to vertices (don't overwrite if already set)
        self.vertices[v0].half_edge.get_or_insert(e0);
        self.vertices[v1].half_edge.get_or_insert(e1);
        self.vertices[v2].half_edge.get_or_insert(e2);

        // Finalize the face record
        self.faces[face_idx].half_edge = e0;

        face_idx
    }


    pub fn remove_invalidated_faces(&mut self) {
        // Filter out invalidated faces
        let mut new_faces = Vec::new();
        let mut face_mapping = vec![None; self.faces.len()];

        for (old_idx, face) in self.faces.iter().enumerate() {
            if !face.removed {
                let new_idx = new_faces.len();
                face_mapping[old_idx] = Some(new_idx);
                new_faces.push(face.clone());
            }
        }

        // Update half-edge face references
        for he in &mut self.half_edges {
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

        // Replace faces
        self.faces = new_faces;
    }

    /// Remove duplicate vertices and update all references
    /// Returns the number of vertices removed
    pub fn remove_duplicate_vertices(&mut self) -> usize {
        if self.vertices.is_empty() {
            return 0;
        }

        let tolerance = T::point_merge_threshold();
        let initial_count = self.vertices.len();

        // Build spatial hash for efficient duplicate detection
        let mut spatial_groups: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();

        for (vertex_idx, vertex) in self.vertices.iter().enumerate() {
            let hash_key = self.position_to_hash_key(&vertex.position);
            spatial_groups.entry(hash_key).or_default().push(vertex_idx);
        }

        // Find duplicates within each spatial group
        let mut vertex_mapping = (0..self.vertices.len()).collect::<Vec<_>>();
        let mut duplicates = HashSet::new();

        for group in spatial_groups.values() {
            if group.len() < 2 {
                continue;
            }

            // Check all pairs within the group
            for i in 0..group.len() {
                if duplicates.contains(&group[i]) {
                    continue;
                }

                for j in (i + 1)..group.len() {
                    if duplicates.contains(&group[j]) {
                        continue;
                    }

                    let pos_i = &self.vertices[group[i]].position;
                    let pos_j = &self.vertices[group[j]].position;

                    if kernel::are_equal(pos_i, pos_j) {
                        // Mark j as duplicate of i
                        vertex_mapping[group[j]] = group[i];
                        duplicates.insert(group[j]);
                    }
                }
            }
        }

        if duplicates.is_empty() {
            return 0;
        }

        // Create compacted vertex list and final mapping
        let mut new_vertices = Vec::new();
        let mut old_to_new = vec![usize::MAX; self.vertices.len()];

        for (old_idx, vertex) in self.vertices.iter().enumerate() {
            if !duplicates.contains(&old_idx) {
                let new_idx = new_vertices.len();
                old_to_new[old_idx] = new_idx;
                new_vertices.push(vertex.clone());
            }
        }

        // Update mapping for duplicates to point to their canonical vertex's new index
        for &duplicate_idx in &duplicates {
            let canonical_idx = vertex_mapping[duplicate_idx];
            old_to_new[duplicate_idx] = old_to_new[canonical_idx];
        }

        // Update all half-edge vertex references
        for half_edge in &mut self.half_edges {
            if half_edge.removed {
                continue;
            }

            let old_vertex = half_edge.vertex;
            if old_vertex < old_to_new.len() {
                if let Some(&new_vertex) = old_to_new.get(old_vertex) {
                    if new_vertex != usize::MAX {
                        half_edge.vertex = new_vertex;
                    }
                }
            }
        }

        // Update vertex half-edge references
        for vertex in &mut new_vertices {
            if let Some(he_idx) = vertex.half_edge {
                // Verify the half-edge still exists and is valid
                if he_idx >= self.half_edges.len() || self.half_edges[he_idx].removed {
                    vertex.half_edge = None;
                }
            }
        }

        // Rebuild edge map with new vertex indices
        let mut new_edge_map = HashMap::new();
        for (&(old_v1, old_v2), &he_idx) in &self.edge_map {
            if old_v1 < old_to_new.len() && old_v2 < old_to_new.len() {
                let new_v1 = old_to_new[old_v1];
                let new_v2 = old_to_new[old_v2];

                if new_v1 != usize::MAX && new_v2 != usize::MAX && new_v1 != new_v2 {
                    // Only keep edges that don't become self-loops
                    new_edge_map.insert((new_v1, new_v2), he_idx);
                }
            }
        }

        // Rebuild spatial hash with new vertex indices
        let mut new_spatial_hash: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (vertex_idx, vertex) in new_vertices.iter().enumerate() {
            let hash_key = self.position_to_hash_key(&vertex.position);
            new_spatial_hash
                .entry(hash_key)
                .or_default()
                .push(vertex_idx);
        }

        // Update mesh data structures
        self.vertices = new_vertices;
        self.edge_map = new_edge_map;
        self.vertex_spatial_hash = new_spatial_hash;

        // Remove any degenerate faces that may have been created
        // self.remove_degenerate_faces();

        initial_count - self.vertices.len()
    }

    pub fn remove_unused_vertices(&mut self) {
        // 1. Find used vertices
        let mut used = vec![false; self.vertices.len()];
        for face_idx in 0..self.faces.len() {
            // Skip removed faces
            if self.faces[face_idx].removed {
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
                new_vertices.push(vertex);
                old_to_new[old_idx] = Some(new_idx);
            }
        }

        // 3. Remap faces (skip invalidated ones)
        let mut new_mesh = Mesh::new();
        for v in &new_vertices {
            new_mesh.add_vertex(v.position.clone());
        }
        for face_idx in 0..self.faces.len() {
            // Skip removed faces
            if self.faces[face_idx].removed {
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

    pub fn build_face_tree(&self) -> AabbTree<CgarF64, N, Point<CgarF64, N>, usize> {
        let mut face_aabbs = Vec::with_capacity(self.faces.len());

        for (face_idx, face) in self.faces.iter().enumerate() {
            if face.removed {
                continue;
            }

            let hes = self.face_half_edges(face_idx);

            if hes[1] >= self.half_edges.len()
                || hes[2] >= self.half_edges.len()
                || self.half_edges[hes[2]].next != hes[0]
            {
                continue;
            }

            let v0_idx = self.half_edges[hes[0]].vertex;
            let v1_idx = self.half_edges[hes[1]].vertex;
            let v2_idx = self.half_edges[hes[2]].vertex;

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
            let mut min_coords = from_fn(|_| CgarF64::from(0));
            let mut max_coords = from_fn(|_| CgarF64::from(0));

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

                min_coords[i] = min_val.clone().into();
                max_coords[i] = max_val.clone().into();
            }

            let aabb =
                Aabb::from_points(&Point::from_vals(min_coords), &Point::from_vals(max_coords));

            face_aabbs.push((aabb, face_idx));
        }

        let tree = AabbTree::<CgarF64, N, Point<CgarF64, N>, usize>::build(face_aabbs);
        tree
    }

    /// Compute the AABB of face `f`.
    pub fn face_aabb(&self, f: usize) -> Aabb<CgarF64, N, Point<CgarF64, N>> {
        let face = &self.faces[f];
        if face.removed || face.half_edge == usize::MAX || self.half_edges[face.half_edge].removed {
            // Return degenerate AABB for invalid faces
            let origin = Point::<CgarF64, N>::from_vals(from_fn(|_| CgarF64::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        let hes = self.face_half_edges(f);

        // Safety checks
        if hes[1] >= self.half_edges.len() || hes[2] >= self.half_edges.len() {
            let origin = Point::<CgarF64, N>::from_vals(from_fn(|_| CgarF64::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        let v0_idx = self.half_edges[hes[0]].vertex;
        let v1_idx = self.half_edges[hes[1]].vertex;
        let v2_idx = self.half_edges[hes[2]].vertex;

        // Safety checks for vertex indices
        if v0_idx >= self.vertices.len()
            || v1_idx >= self.vertices.len()
            || v2_idx >= self.vertices.len()
        {
            let origin = Point::<CgarF64, N>::from_vals(from_fn(|_| CgarF64::from(0)));
            return Aabb::from_points(&origin, &origin);
        }

        let p0 = &self.vertices[v0_idx].position;
        let p1 = &self.vertices[v1_idx].position;
        let p2 = &self.vertices[v2_idx].position;

        // Compute AABB from the three vertices directly
        compute_triangle_aabb(p0, p1, p2)
    }

    fn get_or_insert_vertex(&mut self, pos: &Point<T, N>) -> (usize, bool) {
        // Center cell
        let (kx, ky, kz) = self.position_to_hash_key(pos);

        // 1) Check center cell first (fast path).
        if let Some(bucket) = self.vertex_spatial_hash.get(&(kx, ky, kz)) {
            for &vi in bucket {
                if kernel::are_equal(&self.vertices[vi].position, pos) {
                    return (vi, true);
                }
            }
        }

        // 2) Probe 26 neighboring cells to catch cross-cell near-equals.
        // Offsets are small fixed triplets; iterate deterministically.
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                for dz in -1i64..=1 {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    let key = (kx + dx, ky + dy, kz + dz);
        if let Some(bucket) = self.vertex_spatial_hash.get(&key) {
            for &vi in bucket {
                if kernel::are_equal(&self.vertices[vi].position, pos) {
                                return (vi, true);
                            }
                        }
                    }
                }
            }
        }

        // 3) Insert new vertex into the center cell.
        let idx = self.vertices.len();
        self.vertices.push(Vertex::new(pos.clone()));
        self.vertex_spatial_hash.entry((kx, ky, kz)).or_default().push(idx);
        (idx, false)
    }

    pub fn add_vertex(&mut self, position: Point<T, N>) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(Vertex::new(position));

        let key = self.position_to_hash_key(&self.vertices[idx].position);
        self.vertex_spatial_hash.entry(key).or_default().push(idx);

        idx
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
        let _u = self.half_edges[he_c].vertex; // c->u
        let _v = self.half_edges[he_a].vertex; // u->v
        let c = self.half_edges[he_b].vertex; // v->c
        let d = self.half_edges[he_e].vertex; // u->d

        // --- 4) reassign the two halves of the diagonal to c->d and d->c ---
        self.half_edges[he_a].vertex = d; // now u->d
        self.half_edges[he_d].vertex = c; // now v->c

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

    pub fn split_edge(
        &mut self,
        aabb_tree: &mut AabbTree<CgarF64, N, Point<CgarF64, N>, usize>,
        he: usize,
        pos: &Point<T, N>,
    ) -> Result<SplitResult, &'static str>
    {
        let he_ca = self.find_valid_half_edge(he, pos);
        let he_ab = self.half_edges[he_ca].next;
        let he_bc = self.half_edges[he_ab].next;
        let he_ac = self.half_edges[he_ca].twin;

        let a = self.half_edges[he_ca].vertex;
        let b = self.half_edges[he_ab].vertex;
        let c = self.half_edges[he_bc].vertex;

        let (w, reuse) = self.get_or_insert_vertex(pos);
        if reuse {
            println!("re-using");
            return Ok(SplitResult {
                kind: SplitResultKind::NoSplit,
                vertex: w,
                new_faces: [usize::MAX; 4]
            });
        }

        let original_face_1 = self.half_edges[he_ca].face.unwrap();
        let original_face_2 = self.half_edges[he_ac].face;

        if original_face_2.is_none() {
            // Open borders scenario
            let ex_he_ba = self.half_edges[he_ab].twin; // b->a
            let ex_he_cb = self.half_edges[he_bc].twin; // c->b

            // Mark old faces and half-edges removed
            self.faces[original_face_1].removed = true;
            self.half_edges[he_ca].removed = true;
            self.half_edges[he_ac].removed = true;

            // Remove old edge (a,c) / (c,a) from edge_map
            self.edge_map.remove(&(a, c));
            self.edge_map.remove(&(c, a));

            // Base indices
            let base_face_idx = self.faces.len(); // two new real faces
            let base_he_idx = self.half_edges.len(); // we will add 8 new half-edges:
            // interior: c->w (0), w->a (1), b->w (2), w->b (3)
            // border:   w->c (4), a->w (5), (two border twins of above)
            //           plus their self-loop next/prev initialization
            //           Actually we need exactly 6 new half-edges:
            //             interior: c->w, w->a, b->w, w->b (4)
            //             border:   w->c, a->w (2)  => total 6
            // (We do NOT add border twins for b-w; that is internal)

            // Create interior half-edges first (face assignment deferred)
            let he_cw = base_he_idx + 0; // c -> w (face 2 later)
            let he_wa = base_he_idx + 1; // w -> a (face 1 later)
            let he_bw = base_he_idx + 2; // b -> w (face 1)
            let he_wb = base_he_idx + 3; // w -> b (face 2)
            let bhe_wc = base_he_idx + 4; // w -> c (border null face)
            let bhe_aw = base_he_idx + 5; // a -> w (border null face)

            // Helper to push a blank half-edge
            let mut push_he = |to: usize| {
                let mut he = HalfEdge::new(to);
                he.face = None;
                self.half_edges.push(he);
            };

            // Interior edges
            push_he(w); // c->w
            push_he(a); // w->a
            push_he(w); // b->w
            push_he(b); // w->b
            // Border twins
            push_he(c); // w->c
            push_he(w); // a->w

            // Set origins implicitly via prev pointers later; ensure vertex field is 'to' (already)
            // Assign twins
            self.half_edges[he_cw].twin = bhe_wc;
            self.half_edges[bhe_wc].twin = he_cw;
            self.half_edges[he_wa].twin = bhe_aw;
            self.half_edges[bhe_aw].twin = he_wa;
            self.half_edges[he_bw].twin = he_wb;
            self.half_edges[he_wb].twin = he_bw;

            // Build two new real faces:
            // Face F1: (a,b,w)   cycle: he_ab (a->b), he_bw (b->w), he_wa (w->a)
            // Face F2: (w,b,c)   cycle: he_wb (w->b), he_bc (b->c), he_cw (c->w)

            // Insert faces
            self.faces.push(Face::new(he_ab)); // face index = base_face_idx (F1)
            self.faces.push(Face::new(he_wb)); // face index = base_face_idx+1 (F2)

            // Wire F1 half-edges
            self.half_edges[he_ab].face = Some(base_face_idx);
            self.half_edges[he_bw].face = Some(base_face_idx);
            self.half_edges[he_wa].face = Some(base_face_idx);

            self.half_edges[he_ab].next = he_bw;
            self.half_edges[he_ab].prev = he_wa;
            self.half_edges[he_bw].next = he_wa;
            self.half_edges[he_bw].prev = he_ab;
            self.half_edges[he_wa].next = he_ab;
            self.half_edges[he_wa].prev = he_bw;

            // Wire F2 half-edges
            self.half_edges[he_wb].face = Some(base_face_idx + 1);
            self.half_edges[he_bc].face = Some(base_face_idx + 1);
            self.half_edges[he_cw].face = Some(base_face_idx + 1);

            self.half_edges[he_wb].next = he_bc;
            self.half_edges[he_wb].prev = he_cw;
            self.half_edges[he_bc].next = he_cw;
            self.half_edges[he_bc].prev = he_wb;
            self.half_edges[he_cw].next = he_wb;
            self.half_edges[he_cw].prev = he_bc;

            // Re-hook external twins (keep adjacency) on edges a-b, b-c
            // (he_ab twin already points to ex_he_ba, he_bc twin to ex_he_cb)
            // Ensure those external twins still point back
            if ex_he_ba != usize::MAX {
                self.half_edges[ex_he_ba].twin = he_ab;
            }
            if ex_he_cb != usize::MAX {
                self.half_edges[ex_he_cb].twin = he_bc;
            }

            // Create two new null faces for border half-edges bhe_wc (w->c) and bhe_aw (a->w)
            self.half_edges[bhe_wc].face = None;
            // self-loop
            self.half_edges[bhe_wc].next = bhe_wc;
            self.half_edges[bhe_wc].prev = bhe_wc;

            self.half_edges[bhe_aw].face = None;
            self.half_edges[bhe_aw].next = bhe_aw;
            self.half_edges[bhe_aw].prev = bhe_aw;

            // Edge map updates
            self.edge_map.insert((c, w), he_cw);
            self.edge_map.insert((w, a), he_wa);
            self.edge_map.insert((b, w), he_bw);
            self.edge_map.insert((w, b), he_wb);
            self.edge_map.insert((w, c), bhe_wc);
            self.edge_map.insert((a, w), bhe_aw);

            // Update representative half-edges at vertices
            self.vertices[w].half_edge.get_or_insert(he_wb);
            self.vertices[a].half_edge.get_or_insert(he_ab);
            self.vertices[b].half_edge.get_or_insert(he_bc);
            self.vertices[c].half_edge.get_or_insert(he_cw);

            // Record half-edge split lineage
            self.half_edge_split_map.insert(he_ca, (he_cw, he_wa));
            self.half_edge_split_map.insert(he_ac, (bhe_aw, bhe_wc));

            // AABB tree updates (invalidate old real face; null face not in tree originally)
            aabb_tree.invalidate(&original_face_1);
            // Insert the two new real faces
            let f1_aabb = self.face_aabb(base_face_idx);
            let f2_aabb = self.face_aabb(base_face_idx + 1);
            aabb_tree.insert(f1_aabb, base_face_idx);
            aabb_tree.insert(f2_aabb, base_face_idx + 1);

            let split_result = SplitResult {
                kind: SplitResultKind::SplitEdge,
                vertex: w,
                new_faces: [base_face_idx, base_face_idx + 1, usize::MAX, usize::MAX],
            };

            return Ok(split_result);
        }

        let original_face_2 = original_face_2.unwrap();

        let he_cd = self.half_edges[he_ac].next;
        let he_da = self.half_edges[he_cd].next;

        let ex_he_ba = self.half_edges[he_ab].twin;
        let ex_he_cb = self.half_edges[he_bc].twin;
        let ex_he_dc = self.half_edges[he_cd].twin;
        let ex_he_ad = self.half_edges[he_da].twin;

        let d = self.half_edges[he_cd].vertex;

        // Create new vertex at split position

        // let mut new_face_results = Vec::new();

        // 1. Build the new faces (4 in total) and create their half-edges (no twins for now)
        let evs = [
            (w, b), // face wbc
            (c, w),
            (w, a), // face wab
            (b, w),
            (w, c), // face wcd
            (d, w),
            (w, d), // face wda
            (a, w),
        ];

        let existing_evs = [he_bc, he_ab, he_cd, he_da];

        let base_face_idx = self.faces.len();

        let base_half_edge_idx = self.half_edges.len();

        // let he_wb = base_half_edge_idx;
        let he_cw = base_half_edge_idx + 1;
        let he_wa = base_half_edge_idx + 2;
        // let he_bw = base_half_edge_idx + 3;
        let he_wc = base_half_edge_idx + 4;
        // let he_dw = base_half_edge_idx + 5;
        // let he_wd = base_half_edge_idx + 6;
        let he_aw = base_half_edge_idx + 7;

        let mut i = 0;
        for (from, to) in evs {
            let mut he = HalfEdge::new(to);
            he.face = Some(base_face_idx + i / 2);
            he.vertex = to;
            self.half_edges.push(he);
            self.edge_map.insert((from, to), base_half_edge_idx + i);
            i += 1;
        }

        for i in 0..4 {
            let base_he_idx = base_half_edge_idx + i * 2;
            let edge_indices = [base_he_idx, existing_evs[i], base_he_idx + 1];
            self.half_edges[edge_indices[0]].next = edge_indices[1];
            self.half_edges[edge_indices[0]].prev = edge_indices[2];
            self.half_edges[edge_indices[1]].next = edge_indices[2];
            self.half_edges[edge_indices[1]].prev = edge_indices[0];
            self.half_edges[edge_indices[2]].next = edge_indices[0];
            self.half_edges[edge_indices[2]].prev = edge_indices[1];

            self.faces.push(Face::new(edge_indices[0]));
            self.faces[base_face_idx + i].half_edge = edge_indices[0];
        }

        let twins = [
            (0, 3), // w -> b | b -> w
            (1, 4), // c -> w | w -> c
            (2, 7), // w -> a | a -> w
            (5, 6), // d -> w | w -> d
        ];

        for i in 0..4 {
            self.half_edges[base_half_edge_idx + twins[i].0].twin = base_half_edge_idx + twins[i].1;
            self.half_edges[base_half_edge_idx + twins[i].1].twin = base_half_edge_idx + twins[i].0;
        }

        // Update vertices half-edges
        self.vertices[w].half_edge = Some(base_half_edge_idx); // w -> b
        self.vertices[a].half_edge = Some(he_ab); // a -> b
        self.vertices[b].half_edge = Some(he_bc); // b -> c
        self.vertices[c].half_edge = Some(he_cd); // c -> d
        self.vertices[d].half_edge = Some(he_da); // d -> a

        // Remove old edges from edge map
        self.edge_map.remove(&(c, a));
        self.edge_map.remove(&(a, c));

        // External twins (unchanged neighboring faces)
        self.half_edges[ex_he_ba].twin = he_ab; // external b->a | a->b
        self.half_edges[he_ab].twin = ex_he_ba;

        self.half_edges[ex_he_cb].twin = he_bc; // external c->b | b->c
        self.half_edges[he_bc].twin = ex_he_cb;

        self.half_edges[ex_he_dc].twin = he_cd; // external d->c | c->d
        self.half_edges[he_cd].twin = ex_he_dc;

        self.half_edges[ex_he_ad].twin = he_da; // external a->d | d->a
        self.half_edges[he_da].twin = ex_he_ad;

        let triangle_wbc = Triangle {
            face_idx: base_face_idx,
            vertices: [w, b, c],
        };
        let triangle_wab = Triangle {
            face_idx: base_face_idx + 1,
            vertices: [w, a, b],
        };
        let triangle_wcd = Triangle {
            face_idx: base_face_idx + 2,
            vertices: [w, c, d],
        };
        let triangle_wda = Triangle {
            face_idx: base_face_idx + 3,
            vertices: [w, d, a],
        };

        // Mark old faces as removed
        self.faces[original_face_1].removed = true;
        self.faces[original_face_2].removed = true;

        // Mark old half-edges as removed
        self.half_edges[he_ca].removed = true;
        self.half_edges[he_ac].removed = true;

        self.half_edges[he_bc].face = Some(base_face_idx);
        self.half_edges[he_ab].face = Some(base_face_idx + 1);
        self.half_edges[he_cd].face = Some(base_face_idx + 2);
        self.half_edges[he_da].face = Some(base_face_idx + 3);

        // Connect the new half-edges to their respective vertices
        self.half_edges[he_bc].vertex = c; // b -> c
        self.half_edges[he_ab].vertex = b; // a -> b
        self.half_edges[he_cd].vertex = d; // c -> d
        self.half_edges[he_da].vertex = a; // d -> a

        self.half_edge_split_map.insert(he_ca, (he_cw, he_wa));
        self.half_edge_split_map.insert(he_ac, (he_aw, he_wc));

        let face_split = FaceSplitMap {
            face: original_face_1,
            new_faces: smallvec![triangle_wbc, triangle_wab],
        };
        self.face_split_map.insert(original_face_1, face_split);

        let face_split = FaceSplitMap {
            face: original_face_2,
            new_faces: smallvec![triangle_wcd, triangle_wda],
        };
        self.face_split_map.insert(original_face_2, face_split);

        let split_result = SplitResult {
            kind: SplitResultKind::SplitEdge,
            vertex: w,
            new_faces: [
                base_face_idx,
                base_face_idx + 1,
                base_face_idx + 2,
                base_face_idx + 3,
            ],
        };

        for &new_face_idx in &split_result.new_faces {
            let face_aabb = self.face_aabb(new_face_idx);
            aabb_tree.insert(face_aabb, new_face_idx);
        }

        aabb_tree.invalidate(&original_face_1);
        aabb_tree.invalidate(&original_face_2);

        Ok(split_result)
    }

    pub fn split_face(
        &mut self,
        aabb_tree: &mut AabbTree<CgarF64, N, Point<CgarF64, N>, usize>,
        face: usize,
        p: &Point<T, N>,
    ) -> Option<SplitResult>
    {
        let face = self.find_valid_face(face, p);
        if self.faces[face].removed {
            panic!("Cannot find or insert vertex on a removed face");
        }

        let he_ab = self.find_valid_half_edge(self.faces[face].half_edge, &p);
        let he_bc = self.half_edges[he_ab].next;
        let he_ca = self.half_edges[he_bc].next;

        let w = self.add_vertex(p.clone());

        let a = self.half_edges[he_ca].vertex;
        let b = self.half_edges[he_ab].vertex;
        let c = self.half_edges[he_bc].vertex;

        let subface_1_verts = [a, w, c];
        let subface_2_verts = [c, w, b];
        let subface_3_verts = [b, w, a];

        let subface_1_idx = self.faces.len();
        let subface_2_idx = self.faces.len() + 1;
        let subface_3_idx = self.faces.len() + 2;

        // Create subface 1
        let base_he_idx_1 = self.half_edges.len();
        let edge_vertices_1 = [
            (subface_1_verts[0], subface_1_verts[1]), // a -> w
            (subface_1_verts[1], subface_1_verts[2]), // w -> c
                                                      // c -> a
        ];

        // Create half-edges for face 1
        for (_i, &(_from, to)) in edge_vertices_1.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(subface_1_idx);
            self.half_edges.push(he);
        }

        // Link face 1 half-edges
        let edge_indices_1 = [base_he_idx_1, base_he_idx_1 + 1, he_ca];
        self.half_edges[edge_indices_1[0]].next = edge_indices_1[1];
        self.half_edges[edge_indices_1[0]].prev = edge_indices_1[2];
        self.half_edges[edge_indices_1[1]].next = edge_indices_1[2];
        self.half_edges[edge_indices_1[1]].prev = edge_indices_1[0];
        self.half_edges[edge_indices_1[2]].next = edge_indices_1[0];
        self.half_edges[edge_indices_1[2]].prev = edge_indices_1[1];

        // Create face 2
        let base_he_idx_2 = self.half_edges.len();
        let edge_vertices_2 = [
            (subface_2_verts[0], subface_2_verts[1]), // c -> w
            (subface_2_verts[1], subface_2_verts[2]), // w -> b
                                                      // b -> c
        ];

        // Create half-edges for face 2
        for (_i, &(_from, to)) in edge_vertices_2.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(subface_2_idx);
            self.half_edges.push(he);
        }

        // Link face 2 half-edges
        let edge_indices_2 = [base_he_idx_2, base_he_idx_2 + 1, he_bc];
        self.half_edges[edge_indices_2[0]].next = edge_indices_2[1];
        self.half_edges[edge_indices_2[0]].prev = edge_indices_2[2];
        self.half_edges[edge_indices_2[1]].next = edge_indices_2[2];
        self.half_edges[edge_indices_2[1]].prev = edge_indices_2[0];
        self.half_edges[edge_indices_2[2]].next = edge_indices_2[0];
        self.half_edges[edge_indices_2[2]].prev = edge_indices_2[1];

        // Create face 3
        let base_he_idx_3 = self.half_edges.len();
        let edge_vertices_3 = [
            (subface_3_verts[0], subface_3_verts[1]), // b -> w
            (subface_3_verts[1], subface_3_verts[2]), // w -> a
                                                      // a -> b
        ];

        // Create half-edges for face 3
        for (_i, &(_from, to)) in edge_vertices_3.iter().enumerate() {
            let mut he = HalfEdge::new(to);
            he.face = Some(subface_3_idx);
            self.half_edges.push(he);
        }

        // Link face 3 half-edges
        let edge_indices_3 = [base_he_idx_3, base_he_idx_3 + 1, he_ab];
        self.half_edges[edge_indices_3[0]].next = edge_indices_3[1];
        self.half_edges[edge_indices_3[0]].prev = edge_indices_3[2];
        self.half_edges[edge_indices_3[1]].next = edge_indices_3[2];
        self.half_edges[edge_indices_3[1]].prev = edge_indices_3[0];
        self.half_edges[edge_indices_3[2]].next = edge_indices_3[0];
        self.half_edges[edge_indices_3[2]].prev = edge_indices_3[1];

        self.faces.push(Face::new(edge_indices_1[0]));
        self.faces.push(Face::new(edge_indices_2[0]));
        self.faces.push(Face::new(edge_indices_3[0]));

        self.faces[face].removed = true;

        // self.half_edges[base_he_idx_1].face = Some(subface_1_idx);
        // self.half_edges[base_he_idx_1 + 1].face = Some(subface_1_idx);
        self.half_edges[he_ca].face = Some(subface_1_idx);

        // self.half_edges[base_he_idx_2].face = Some(subface_2_idx);
        // self.half_edges[base_he_idx_2 + 1].face = Some(subface_2_idx);
        self.half_edges[he_bc].face = Some(subface_2_idx);

        // self.half_edges[base_he_idx_3].face = Some(subface_3_idx);
        // self.half_edges[base_he_idx_3 + 1].face = Some(subface_3_idx);
        self.half_edges[he_ab].face = Some(subface_3_idx);

        self.vertices[w].half_edge = Some(base_he_idx_2 + 1); // w -> b
        self.vertices[a].half_edge = Some(he_ab);
        self.vertices[b].half_edge = Some(he_bc);
        self.vertices[c].half_edge = Some(he_ca);

        self.half_edges[base_he_idx_1].vertex = w; // a -> w
        self.half_edges[base_he_idx_1 + 1].vertex = c; // w -> c

        self.half_edges[base_he_idx_2].vertex = w; // c -> w
        self.half_edges[base_he_idx_2 + 1].vertex = b; // w -> b

        self.half_edges[base_he_idx_3].vertex = w; // b -> w
        self.half_edges[base_he_idx_3 + 1].vertex = a; // w -> a

        self.faces[subface_1_idx].half_edge = base_he_idx_1;
        self.faces[subface_2_idx].half_edge = base_he_idx_2;
        self.faces[subface_3_idx].half_edge = base_he_idx_3;

        // internal twins
        // w -> a and a -> w, respectively
        self.half_edges[base_he_idx_3 + 1].twin = base_he_idx_1;
        self.half_edges[base_he_idx_1].twin = base_he_idx_3 + 1;

        // w -> c and c -> w, respectively
        self.half_edges[base_he_idx_1 + 1].twin = base_he_idx_2;
        self.half_edges[base_he_idx_2].twin = base_he_idx_1 + 1;

        // w -> b and b -> w, respectively
        self.half_edges[base_he_idx_2 + 1].twin = base_he_idx_3;
        self.half_edges[base_he_idx_3].twin = base_he_idx_2 + 1;

        self.edge_map.insert((a, w), base_he_idx_1);
        self.edge_map.insert((w, c), base_he_idx_1 + 1);

        self.edge_map.insert((c, w), base_he_idx_2);
        self.edge_map.insert((w, b), base_he_idx_2 + 1);

        self.edge_map.insert((b, w), base_he_idx_3);
        self.edge_map.insert((w, a), base_he_idx_3 + 1);

        let triangle_awc = Triangle {
            face_idx: subface_1_idx,
            vertices: [a, w, c],
        };
        let triangle_cwb = Triangle {
            face_idx: subface_2_idx,
            vertices: [c, w, b],
        };
        let triangle_bwa = Triangle {
            face_idx: subface_3_idx,
            vertices: [b, w, a],
        };

        let face_split = FaceSplitMap {
            face: face,
            new_faces: smallvec![triangle_awc, triangle_cwb, triangle_bwa],
        };
        self.face_split_map.insert(face, face_split);

        let split_result = SplitResult {
            kind: SplitResultKind::SplitFace,
            vertex: w,
            new_faces: [subface_1_idx, subface_2_idx, subface_3_idx, usize::MAX],
        };

        for i in 0..3 {
            let face_aabb = self.face_aabb(split_result.new_faces[i]);
            aabb_tree.insert(face_aabb, split_result.new_faces[i]);
        }

        aabb_tree.invalidate(&face);

        Some(split_result)
    }

    /// Tangential (Taubin/Desbrun style) smoothing.
    /// - `sweeps`: number of Jacobi sweeps
    /// - `step`:   blend factor in (0,1], e.g. 0.2
    /// Returns true if any vertex moved.
    pub fn smooth_tangential(&mut self, v: usize, alpha: T) -> bool
    where
        T: Scalar + PartialOrd + Clone,
        for<'a> &'a T:
            core::ops::Add<&'a T, Output = T> +
            core::ops::Sub<&'a T, Output = T> +
            core::ops::Mul<&'a T, Output = T> +
            core::ops::Div<&'a T, Output = T>,
    {
        // Only defined for 3D meshes
        if N != 3 { return false; }

        // Vertex existence and outgoing half-edge
        if v >= self.vertices.len() || self.vertices[v].removed {
            return false;
        }

        // Optional: skip boundary vertices (keeps silhouette stable)
        // We treat "boundary" as having any incident half-edge without a face.
        let is_boundary = {
            let mut boundary = false;
            for he in self.vertex_ring_ccw(v).halfedges_ccw {
                if self.half_edges[he].face.is_none() {
                    boundary = true;
                    break;
                }
            }
            boundary
        };
        if is_boundary {
            return false;
        }

        // Gather the ordered 1-ring as half-edges originating at v, then build the neighbor index cycle.
        // Assumes `vertex_ring(v)` yields a CCW ring of half-edges around v where each `he` has:
        //   - to-vertex = half_edges[he].vertex = b
        //   - next(he) has to-vertex = c (forming triangle (a, b, c))
        // Duplicates may appear on degenerate/border cases; we will guard against them.
        let ring_hes = self.vertex_ring_ccw(v);
        if ring_hes.neighbors_ccw.is_empty() { return false; }

        // Current position a
        let a = self.vertices[v].position.clone();

        // Accumulate:
        // - area-weighted normal n = sum over faces ( (b-a) x (c-a) )
        // - unnormalized Laplacian direction d = sum over neighbors (b - a)
        let mut n0 = T::zero();
        let mut n1 = T::zero();
        let mut n2 = T::zero();

        let mut d0 = T::zero();
        let mut d1 = T::zero();
        let mut d2 = T::zero();

        let mut valence = 0usize;

        for &he in ring_hes.halfedges_ccw.iter() {
            // Skip invalid or removed half-edges
            if he >= self.half_edges.len() || self.half_edges[he].removed {
                continue;
            }

            // Neighbor b = head of he
            let b_idx = self.half_edges[he].vertex;
            if b_idx >= self.vertices.len() || self.vertices[b_idx].removed {
                continue;
            }
            let b = &self.vertices[b_idx].position;

            // Add Laplacian term (b - a)
            {
                let bx0 = &b[0] - &a[0];
                let by1 = &b[1] - &a[1];
                let bz2 = &b[2] - &a[2];
                d0 = &d0 + &bx0;
                d1 = &d1 + &by1;
                d2 = &d2 + &bz2;
                valence += 1;
            }

            // If this half-edge has a face, we can accumulate a face normal (b - a) x (c - a)
            if let Some(_) = self.half_edges[he].face {
                let he_next = self.half_edges[he].next;
                if he_next >= self.half_edges.len() || self.half_edges[he_next].removed {
                    continue;
                }
                let c_idx = self.half_edges[he_next].vertex;
                if c_idx >= self.vertices.len() || self.vertices[c_idx].removed {
                    continue;
                }
                let c = &self.vertices[c_idx].position;

                // u = b - a, v = c - a
                let ux = &b[0] - &a[0];
                let uy = &b[1] - &a[1];
                let uz = &b[2] - &a[2];
                let vx = &c[0] - &a[0];
                let vy = &c[1] - &a[1];
                let vz = &c[2] - &a[2];

                // u x v
                let cx = &(&uy * &vz) - &(&uz * &vy);
                let cy = &(&uz * &vx) - &(&ux * &vz);
                let cz = &(&ux * &vy) - &(&uy * &vx);

                n0 = &n0 + &cx;
                n1 = &n1 + &cy;
                n2 = &n2 + &cz;
            }
        }

        if valence < 2 {
            return false; // nothing to do or cannot form faces reliably
        }

        // Normalize Laplacian by valence: d /= valence
        let inv_val = T::one() / T::from(valence as f64);
        d0 = &d0 * &inv_val;
        d1 = &d1 * &inv_val;
        d2 = &d2 * &inv_val;

        // Project d onto the tangent plane at a using accumulated normal n:
        // d_tan = d - n * (dot(d, n) / dot(n, n))
        let nn = &n0 * &n0 + &n1 * &n1 + &n2 * &n2;
        let mut t0 = d0.clone();
        let mut t1 = d1.clone();
        let mut t2 = d2.clone();

        if nn > T::zero() {
            let dn = &d0 * &n0 + &d1 * &n1 + &d2 * &n2;
            let k = dn / nn;
            t0 = &t0 - &(&n0 * &k);
            t1 = &t1 - &(&n1 * &k);
            t2 = &t2 - &(&n2 * &k);
        }

        // Proposed new position a' = a + alpha * d_tan
        let axp = &a[0] + &(&alpha * &t0);
        let ayp = &a[1] + &(&alpha * &t1);
        let azp = &a[2] + &(&alpha * &t2);

        // Degeneracy (robust): reject if any incident triangle (a', b, c) is degenerate
        // Use the same ring traversal to get (b, c) per face around v.
        // Also optionally prevent flips: (n_old · n_new) <= 0 → reject.
        let mut reject = false;

        for &he in ring_hes.halfedges_ccw.iter() {
            if he >= self.half_edges.len() || self.half_edges[he].removed {
                continue;
            }
            // Only check real faces
            let Some(_) = self.half_edges[he].face else { continue; };

            let b_idx = self.half_edges[he].vertex;
            let he_next = self.half_edges[he].next;
            if he_next >= self.half_edges.len() || self.half_edges[he_next].removed {
                continue;
            }
            let c_idx = self.half_edges[he_next].vertex;

            if b_idx >= self.vertices.len() || c_idx >= self.vertices.len() {
                continue;
            }
            let b = &self.vertices[b_idx].position;
            let c = &self.vertices[c_idx].position;
            let arr = [&axp, &ayp, &azp];

            // Robust degenerate test using kernel
            if kernel::triangle_is_degenerate::<T, N>(
                // construct a temporary Point<T,N> from (axp, ayp, azp) in-place style
                &Point::<T, N>::from_vals(from_fn(|i| arr[i].clone())),
                b,
                c,
            ) {
                reject = true;
                break;
            }

            // n_old = (b - a) x (c - a), n_new = (b - a') x (c - a')
            let bax = &b[0] - &a[0];
            let bay = &b[1] - &a[1];
            let baz = &b[2] - &a[2];
            let cax = &c[0] - &a[0];
            let cay = &c[1] - &a[1];
            let caz = &c[2] - &a[2];
            let nold_x = &(&bay * &caz) - &(&baz * &cay);
            let nold_y = &(&baz * &cax) - &(&bax * &caz);
            let nold_z = &(&bax * &cay) - &(&bay * &cax);

            let bapx = &b[0] - &axp;
            let bapy = &b[1] - &ayp;
            let bapz = &b[2] - &azp;
            let capx = &c[0] - &axp;
            let capy = &c[1] - &ayp;
            let capz = &c[2] - &azp;
            let nnew_x = &(&bapy * &capz) - &(&bapz * &capy);
            let nnew_y = &(&bapz * &capx) - &(&bapx * &capz);
            let nnew_z = &(&bapx * &capy) - &(&bapy * &capx);

            let dot_old_new = &nold_x * &nnew_x + &nold_y * &nnew_y + &nold_z * &nnew_z;
            if dot_old_new <= T::zero() {
                reject = true;
                break;
            }
        }

        if reject {
            return false;
        }

        // Commit new position
        let mut a_new = a;
        a_new[0] = axp;
        a_new[1] = ayp;
        a_new[2] = azp;
        self.vertices[v].position = a_new;

        true
    }
}

pub fn compute_triangle_aabb<T: Scalar, const N: usize>(
    p0: &Point<T, N>,
    p1: &Point<T, N>,
    p2: &Point<T, N>,
) -> Aabb<CgarF64, N, Point<CgarF64, N>>
where
    T: Into<CgarF64>,
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

    Aabb::from_points(
        &Point::from_vals(min_coords.into()),
        &Point::from_vals(max_coords.into()),
    )
}
