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

use std::ops::{Add, Div, Mul, Neg, Sub};

use ahash::AHashSet;
use smallvec::SmallVec;

use crate::{
    geometry::{point::*, vector::*},
    impl_mesh,
    mesh::{basic_types::*, half_edge::HalfEdge},
    numeric::scalar::Scalar,
};

/// What the “begin” phase returns if everything is okay.
/// You’ll feed this into the actual commit step.
#[derive(Debug)]
pub struct CollapsePlan<T: Scalar, const N: usize> {
    pub v_keep: usize,
    pub v_gone: usize,
    pub p_star: Point<T, N>,
}

#[derive(Debug)]
pub enum CollapseReject {
    NotAdjacent,
    BorderForbidden,
    LinkCondition,  // common neighbors != {a,b} (or {a} on border)
    DuplicateEdges, // shared neighbors beyond {a,b}
    TwoGon,         // a == b on interior edge
    DegenerateFace, // area ~ 0 after placement
    NormalFlip,     // orientation flips after placement
    InternalError,  // should not happen
}

pub struct CollapseOpts<T> {
    /// Triangles with area^2 <= area_eps2 are considered degenerate.
    pub area_eps2: T,
    /// If true, do not allow collapsing a border edge.
    pub forbid_border: bool,
    /// If true, reject when any surviving triangle flips orientation at placement p*.
    pub forbid_normal_flip: bool,
}

impl<T: Scalar> Default for CollapseOpts<T>
where
    T: Scalar,
    for<'a> &'a T: Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Add<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    fn default() -> Self {
        let tol = T::tolerance();
        Self {
            area_eps2: &tol * &tol,
            forbid_border: false,
            forbid_normal_flip: true,
        }
    }
}

pub trait Placement<T: Scalar, const N: usize> {
    fn place(&self, mesh: &Mesh<T, N>, v0: usize, v1: usize) -> Point<T, N>;
}

pub struct Midpoint;
impl<T: Scalar, const N: usize> Placement<T, N> for Midpoint
where
    Vector<T, N>: VectorOps<T, N>,
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    for<'a> &'a T: core::ops::Add<&'a T, Output = T>
        + core::ops::Sub<&'a T, Output = T>
        + core::ops::Mul<&'a T, Output = T>
        + core::ops::Div<&'a T, Output = T>,
{
    fn place(&self, mesh: &Mesh<T, N>, v0: usize, v1: usize) -> Point<T, N> {
        let p0 = mesh.vertices[v0].position.clone();
        let p1 = mesh.vertices[v1].position.clone();
        (&p0 + &p1).as_vector().scale(&T::from_num_den(1, 2)).0
    }
}

impl_mesh! {
    #[inline]
    fn rings_adjacency_ok(&self, pr: &PairRing, v0: usize, v1: usize) -> bool {
        let he01 = match self.half_edge_between(v0, v1) { Some(h) => h, None => return false };
        let he10 = match self.half_edge_between(v1, v0) { Some(h) => h, None => return false };

        let i0 = match pr.idx_v1_in_ring0 { Some(i) => i, None => return false };
        let i1 = match pr.idx_v0_in_ring1 { Some(i) => i, None => return false };

        // Indices must be in range
        if i0 >= pr.ring0.halfedges_ccw.len() || i1 >= pr.ring1.halfedges_ccw.len() { return false; }

        // The half-edge at those indices must be the exact adjacency
        if pr.ring0.halfedges_ccw[i0] != he01 { return false; }
        if pr.ring1.halfedges_ccw[i1] != he10 { return false; }

        // And neighbors at those indices must match the opposite vertex
        if pr.ring0.neighbors_ccw[i0] != v1 { return false; }
        if pr.ring1.neighbors_ccw[i1] != v0 { return false; }

        true
    }

    #[inline]
    fn neighbor_sets_excluding_endpoints(
        &self,
        pr: &PairRing,
        v0: usize,
        v1: usize,
    ) -> (AHashSet<usize>, AHashSet<usize>) {

        let set0: AHashSet<_> = pr.ring0.neighbors_ccw
            .iter().copied()
            .filter(|&x| x != v1 && x != v0) // exclude the edge endpoint and (defensively) the center
            .collect();

        let set1: AHashSet<_> = pr.ring1.neighbors_ccw
            .iter().copied()
            .filter(|&x| x != v0 && x != v1)
            .collect();

        (set0, set1)
    }

    #[inline]
    fn opposites_count(&self, pr: &PairRing) -> usize {
        (pr.opposite_a.is_some() as usize) + (pr.opposite_b.is_some() as usize)
    }

    /// (A) Triangle link condition, **robust** to buggy pr.common_neighbors / pr.is_border_edge.
    /// interior: intersection(N(v0)\{v1}, N(v1)\{v0}) == {a,b}
    /// border:   ... == {a}
    pub fn check_link_condition_triangle(&self, v0: usize, v1: usize) -> bool {
        let Some(pr) = self.ring_pair(v0, v1) else { return false; };
        if !self.rings_adjacency_ok(&pr, v0, v1) { return false; }

        // Build the intersection directly from the rings
        let (set0, set1) = self.neighbor_sets_excluding_endpoints(&pr, v0, v1);
        let common: AHashSet<_> = set0.intersection(&set1).copied().collect();

        // Expected set from the two incident faces (ignoring missing ones)
        let mut expected = AHashSet::new();
        if let Some(a) = pr.opposite_a { expected.insert(a); }
        if let Some(b) = pr.opposite_b { expected.insert(b); }

        match self.opposites_count(&pr) {
            2 => common == expected,           // interior: must be {a,b}
            1 => common == expected,           // border:   must be {a}
            _ => false,                        // 0 or >2 ⇒ invalid configuration
        }
    }

    /// (B1) Duplicate-edge creation check, computed from raw rings.
    pub fn would_create_duplicate_edges(&self, v0: usize, v1: usize) -> bool {
        let Some(pr) = self.ring_pair(v0, v1) else { return true; };
        if !self.rings_adjacency_ok(&pr, v0, v1) { return true; }

        let (set0, set1) = self.neighbor_sets_excluding_endpoints(&pr, v0, v1);
        let mut inter: AHashSet<_> = set0.intersection(&set1).copied().collect();

        // Duplicates if there are shared neighbors **beyond the face opposites**
        if let Some(a) = pr.opposite_a { inter.remove(&a); }
        if let Some(b) = pr.opposite_b { inter.remove(&b); }

        !inter.is_empty()
    }

    /// (B2) 2‑gon creation check; infer border from opposites.
    pub fn would_create_2gons(&self, v0: usize, v1: usize) -> bool {
        let Some(pr) = self.ring_pair(v0, v1) else { return true; };
        if !self.rings_adjacency_ok(&pr, v0, v1) { return true; }

        match (pr.opposite_a, pr.opposite_b) {
            (Some(a), Some(b)) => a == b, // interior: two faces wedge to same third vertex → 2‑gon
            (Some(_), None) | (None, Some(_)) => false, // border: cannot create 2‑gon
            _ => true, // neither face present ⇒ invalid to collapse
        }
    }

    /// Full check (topology + geometry) with **border inferred from opposites**.
    pub fn verify_collapse_prereqs(
        &self,
        v0: usize,
        v1: usize,
        placement: &impl Placement<T, N>,
        opts: &CollapseOpts<T>,
    ) -> Result<Point<T, N>, CollapseReject>
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        let Some(pr) = self.ring_pair(v0, v1) else {
            return Err(CollapseReject::NotAdjacent);
        };
        if !self.rings_adjacency_ok(&pr, v0, v1) {
            return Err(CollapseReject::NotAdjacent);
        }

        // Border policy
        let opp_count = self.opposites_count(&pr);
        if opts.forbid_border && opp_count == 1 {
            return Err(CollapseReject::BorderForbidden);
        }

        // Topology guards
        if !self.check_link_condition_triangle(v0, v1) {
            return Err(CollapseReject::LinkCondition);
        }
        if self.would_create_duplicate_edges(v0, v1) {
            return Err(CollapseReject::DuplicateEdges);
        }
        if self.would_create_2gons(v0, v1) {
            return Err(CollapseReject::TwoGon);
        }

        // Geometry at placement p*
        let p_star = placement.place(self, v0, v1);

        let survivors = self.surviving_faces_after_collapse(v0, v1);
        for &f in &survivors {
            let a2x4 = self.tri_area2_after_move_face(f, v0, v1, &p_star);
            let eps = T::tolerance();
            if a2x4 <= &(&eps * &eps) * &(&eps * &eps) {
                return Err(CollapseReject::DegenerateFace);
            }
        }

        Ok(p_star)
    }

    fn surviving_faces_after_collapse(&self, v_keep: usize, v_gone: usize) -> Vec<usize> {
        let s0 = self.incident_faces(v_keep);
        let s1 = self.incident_faces(v_gone);
        let mut out = Vec::new();
        for &f in s0.union(&s1) {
            let [a, b, c] = self.face_vertices(f);
            let has_keep = a == v_keep || b == v_keep || c == v_keep;
            let has_gone = a == v_gone || b == v_gone || c == v_gone;
            if has_keep && has_gone { continue; } // this face disappears
            out.push(f);
        }
        out
    }

    fn tri_area2_after_move_face(
        &self,
        f: usize,
        v_keep: usize,
        v_gone: usize,
        p_star: &Point<T, N>,
    ) -> T
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        let [i, j, k] = self.face_vertices(f);

        let pos = |idx: usize| -> &Point<T, N> {
            if idx == v_keep || idx == v_gone { p_star } else { &self.vertices[idx].position }
        };
        let pa = pos(i);
        let pb = pos(j);
        let pc = pos(k);

        let ab = (pb - pa).as_vector();
        let ac = (pc - pa).as_vector();
        let n = ab.cross(&ac);
        n.dot(&n) // (2*area)^2
    }

    /// Begin collapse from a half-edge handle `he_ab`.
    /// Convention here: keep `a`, delete `b`.
    pub fn collapse_edge_begin_he(
        &self,
        he_ab: usize,
        placement: &impl Placement<T, N>,
        opts: &CollapseOpts<T>,
    ) -> Result<CollapsePlan<T, N>, CollapseReject> where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        // --- sanity on the half-edge
        if he_ab >= self.half_edges.len() || self.half_edges[he_ab].removed {
            return Err(CollapseReject::NotAdjacent);
        }
        let a = self.source(he_ab);
        let b = self.target(he_ab);

        // Optional: if you want to always keep the vertex with larger valence (or smaller),
        // choose here and swap (a,b) accordingly before verifying.

        // --- run the gate (topology + geometry)
        let p_star = self.verify_collapse_prereqs(a, b, placement, opts)?;

        Ok(CollapsePlan {
            v_keep: a,
            v_gone: b,
            p_star,
        })
    }

    /// Same, but explicit choice of kept and gone vertices
    pub fn collapse_edge_begin_vertices(
        &self,
        v_keep: usize,
        v_gone: usize,
        placement: &impl Placement<T, N>,
        opts: &CollapseOpts<T>,
    ) -> Result<CollapsePlan<T, N>, CollapseReject> where Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        // Quick adjacency guard (reuses ring_pair)
        if self.ring_pair(v_keep, v_gone).is_none() {
            return Err(CollapseReject::NotAdjacent);
        }

        let p_star = self.verify_collapse_prereqs(v_keep, v_gone, placement, opts)?;
        Ok(CollapsePlan {
            v_keep,
            v_gone,
            p_star,
        })
    }

    /// Utility: compute the "from" vertex of a half-edge using prev->to.
    #[inline]
    fn he_from(&self, he: usize) -> usize {
        self.half_edges[self.half_edges[he].prev].vertex
    }

    /// Utility: walk a face cycle once, applying `mutator` to each half-edge id.
    fn for_each_he_in_face<F: FnMut(&mut HalfEdge)>(&mut self, f: usize, mut mutator: F) {
        let start = self.faces[f].half_edge;
        let mut h = start;
        let mut first = true;
        let guard = 0;
        while first || h != start {
            first = false;
            mutator(&mut self.half_edges[h]);
            h = self.half_edges[h].next;
            if guard > 100 {
                panic!("error on for_each_he_in_face: too many iterations");
            }
        }
    }

    #[inline]
    fn edge_map_insert_bidir(&mut self, src: usize, dst: usize, h: usize) {
        let t = self.half_edges[h].twin;
        self.edge_map.insert((src, dst), h);
        self.edge_map.insert((dst, src), t);
    }

    pub fn collapse_edge_commit(&mut self, plan: CollapsePlan<T, N>) -> Result<(), CollapseReject> {
        // self.validate_connectivity();
        let u = plan.v_keep;
        let v = plan.v_gone;
        if u == v { return Err(CollapseReject::InternalError); }

        // ---------- 1. Locate collapsing directed half-edge pair (u->v , v->u) ----------
        let he_uv = match self.edge_map.get(&(u, v)) {
            Some(&h) => h,
            None => return Err(CollapseReject::InternalError),
        };
        let he_vu = self.half_edges[he_uv].twin;
        if he_vu == usize::MAX || self.half_edges[he_vu].removed {
            return Err(CollapseReject::InternalError);
        }

        // ---------- 2. Cache local 2-face wedge structure ----------
        // Left face (u,v,a)
        let he_va = self.half_edges[he_uv].next; // v->a
        let he_au = self.half_edges[he_uv].prev; // a->u
        // Right face (v,u,b)
        let he_ub = self.half_edges[he_vu].next; // u->b
        let he_bv = self.half_edges[he_vu].prev; // b->v

        let f_left  = self.half_edges[he_uv].face;
        let f_right = self.half_edges[he_vu].face;
        let left_exists  = f_left.is_some()  && !self.faces[f_left.unwrap()].removed;
        let right_exists = f_right.is_some() && !self.faces[f_right.unwrap()].removed;

        // Splice anchors (only valid if respective face exists)
        let splice_l_from = if left_exists  { self.half_edges[he_au].prev } else { usize::MAX };
        let splice_l_to   = if left_exists  { self.half_edges[he_va].next } else { usize::MAX };
        let splice_r_from = if right_exists { self.half_edges[he_bv].prev } else { usize::MAX };
        let splice_r_to   = if right_exists { self.half_edges[he_ub].next } else { usize::MAX };

        // Half-edges to remove (the two faces' 6 interior half-edges)
        let remove_set = [he_uv, he_vu, he_va, he_au, he_ub, he_bv];

        // ---------- 2b. Collect all half-edges incident to u or v (for retargeting) ----------
        let affected_u = self.collect_incident_half_edges(u);
        let affected_v = self.collect_incident_half_edges(v);
        let mut all_affected = affected_u;
        all_affected.extend_from_slice(&affected_v);

        // ---------- 3. Collect neighbor set S = (N(u) ∪ N(v)) \ {u,v} BEFORE mutation ----------
        // We only rebuild twins / edge_map for edges incident to u and S afterwards.
        let mut neighbor_flag = AHashSet::new();

        self.collect_vertex_source_neighbors(u, &mut neighbor_flag);
        self.collect_vertex_source_neighbors(v, &mut neighbor_flag);
        self.collect_vertex_target_neighbors(u, &mut neighbor_flag);
        self.collect_vertex_target_neighbors(v, &mut neighbor_flag);

        neighbor_flag.remove(&u);
        neighbor_flag.remove(&v);

        // ---------- 4. Remove all edge_map entries for edges incident to u or v (we rebuild later) ----------
        self.clear_edge_map_incident(u);
        self.clear_edge_map_incident(v);

        // ---------- 5. Retarget all surviving half-edges whose target is v -> u (excluding removals) ----------
        println!("Before retargeting - checking for duplicates:");
        let mut edge_counts = std::collections::HashMap::new();
        for hid in &all_affected {
            let he = &self.half_edges[*hid];
            if he.removed { continue; }
            if self.in_remove_set(*hid, &remove_set) { continue; }
            let src = self.he_from(*hid);
            let dst = he.vertex;
            *edge_counts.entry((src, dst)).or_insert(0) += 1;
        }
        for ((src, dst), count) in &edge_counts {
            if *count > 1 {
                println!("Before retargeting: {} half-edges from {} to {}", count, src, dst);
            }
        }

        for hid in affected_v {
            let he = &self.half_edges[hid];
            if he.removed { continue; }
            if self.in_remove_set(hid, &remove_set) { continue; }
            if he.vertex == v {
                self.half_edges[hid].vertex = u;
            }
        }

        println!("After retargeting - checking for duplicates:");
        let mut edge_counts = std::collections::HashMap::new();
        for hid in &all_affected {
            let he = &self.half_edges[*hid];
            if he.removed { continue; }
            if self.in_remove_set(*hid, &remove_set) { continue; }
            let src = self.he_from(*hid);
            let dst = he.vertex;
            *edge_counts.entry((src, dst)).or_insert(0) += 1;
        }
        for ((src, dst), count) in &edge_counts {
            if *count > 1 {
                println!("After retargeting: {} half-edges from {} to {}", count, src, dst);
            }
        }

        // ---------- 6. Splice out the two faces locally (bridge the gaps) ----------
        if left_exists {
            self.half_edges[splice_l_from].next = splice_l_to;
            self.half_edges[splice_l_to].prev = splice_l_from;
        }
        if right_exists {
            self.half_edges[splice_r_from].next = splice_r_to;
            self.half_edges[splice_r_to].prev = splice_r_from;
        }

        // ---------- 7. Mark faces removed ----------
        if left_exists  { self.faces[f_left.unwrap()].removed = true; }
        if right_exists { self.faces[f_right.unwrap()].removed = true; }

        // ---------- 8. Detach & remove the 6 interior half-edges ----------
        for &h in &remove_set {
            let twin = self.half_edges[h].twin;
            if twin != usize::MAX && !self.half_edges[twin].removed {
                // Twin becomes border for now
                self.half_edges[twin].twin = usize::MAX;
            }
            self.half_edges[h].removed = true;
            self.half_edges[h].twin = usize::MAX;
            self.half_edges[h].next = h;
            self.half_edges[h].prev = h;
            self.half_edges[h].face = None;
        }

        // ---------- 9. Retire v, move position into u ----------
        self.vertices[u].position = plan.p_star;
        self.vertices[v].removed = true;
        self.vertices[v].half_edge = None;

        // ---------- 10. Rebuild twin pairs ONLY for edges (u,w) with w in neighbor_flag ----------
        // Build temporary buckets keyed by unordered pair (min,max)
        use ahash::AHashMap;
        struct PairDir { d0: Vec<usize>, d1: Vec<usize> } // Allow multiple half-edges per direction
        let mut buckets: AHashMap<(usize,usize), PairDir> = AHashMap::new();

        // Collect all orphaned half-edges that lost their twins in step 8
        // for hid in 0..self.half_edges.len() {
        //     let he = &self.half_edges[hid];
        //     if he.removed || he.twin != usize::MAX {
        //         continue;
        //     }

        //     let src = self.he_from(hid);
        //     let tgt = he.vertex;
        //     let key = if src < tgt { (src, tgt) } else { (tgt, src) };

        //     buckets.entry(key).or_insert_with(|| PairDir { d0: Vec::new(), d1: Vec::new() });

        //     if src < tgt {
        //         buckets.get_mut(&key).unwrap().d0.push(hid);
        //     } else {
        //         buckets.get_mut(&key).unwrap().d1.push(hid);
        //     }
        // }

        // Collect candidate half-edges where one endpoint is u and the other is in neighbor_flag
        // We also include half-edges whose source is neighbor and target u (post-retarget).
        for hid in 0..self.half_edges.len() {
            let he = &self.half_edges[hid];
            if he.removed { continue; }
            let src = self.he_from(hid);
            let dst = he.vertex;
            if src == dst { continue; }
            let involved = (src == u && neighbor_flag.contains(&dst)) ||
                           (dst == u && neighbor_flag.contains(&src));
            if !involved { continue; }

            let (a,b) = if src < dst {(src,dst)} else {(dst,src)};
            let entry = buckets.entry((a,b)).or_insert(PairDir { d0: Vec::new(), d1: Vec::new() });

            // Assign directional slots: first direction stored in d0, opposite in d1.
            if src == a && dst == b {
                entry.d0.push(hid);
            } else {
                entry.d1.push(hid);
            }
        }

        // Debug: Report any duplicates
        for ((a,b), pd) in &buckets {
            if pd.d0.len() > 1 {
                println!("Multiple half-edges {} -> {}: {:?}", a, b, pd.d0);
            }
            if pd.d1.len() > 1 {
                println!("Multiple half-edges {} -> {}: {:?}", b, a, pd.d1);
            }
        }

        // Clear any stale twins among candidates before setting new ones
        for (_, pd) in buckets.iter() {
            for &h in &pd.d0 {
                self.half_edges[h].twin = usize::MAX;
            }
            for &h in &pd.d1 {
                self.half_edges[h].twin = usize::MAX;
            }
        }

        // Assign twins & rebuild edge_map - only use first half-edge of each direction
        for ((a,b), pd) in buckets {
            let h_ab = pd.d0.first().copied();
            let h_ba = pd.d1.first().copied();

            match (h_ab, h_ba) {
                (Some(h_ab), Some(h_ba)) => {
                    self.half_edges[h_ab].twin = h_ba;
                    self.half_edges[h_ba].twin = h_ab;
                    self.edge_map_insert_bidir(a, b, h_ab);
                }
                (Some(h_ab), None) => {
                    // Border (only a->b)
                    let src = self.he_from(h_ab); let dst = self.half_edges[h_ab].vertex;
                    self.edge_map.insert((src,dst), h_ab);
                }
                (None, Some(h_ba)) => {
                    // Border (only b->a)
                    let src = self.he_from(h_ba); let dst = self.half_edges[h_ba].vertex;
                    self.edge_map.insert((src,dst), h_ba);
                }
                _ => {}
            }

            // Remove any extra half-edges
            for &h in pd.d0.iter().skip(1) {
                self.half_edges[h].removed = true;
            }
            for &h in pd.d1.iter().skip(1) {
                self.half_edges[h].removed = true;
            }
        }

        if self.vertices[u].half_edge.map(|h| self.half_edges[h].removed || self.he_from(h) != u).unwrap_or(true) {
            self.vertices[u].half_edge = None;
            for (hid, he) in self.half_edges.iter().enumerate() {
                if !he.removed && self.he_from(hid) == u {
                    self.vertices[u].half_edge = Some(hid);
                    break;
                }
            }
        }

        for he in &self.half_edges {
            if he.removed { continue; }
            assert_ne!(he.twin, usize::MAX, "half-edge with invalid twin");
        }

        #[cfg(debug_assertions)]
        {
            for (i, he) in self.half_edges.iter().enumerate() {
                if he.removed { continue; }
                assert_ne!(he.vertex, v, "half-edge still targets removed vertex");
                let n = he.next; let p = he.prev;
                assert!(n < self.half_edges.len() && p < self.half_edges.len());
                assert_eq!(self.half_edges[n].prev, i);
                assert_eq!(self.half_edges[p].next, i);
                if he.twin != usize::MAX {
                    assert_eq!(self.half_edges[he.twin].twin, i);
                }
            }
        }

        // ---------- 11. Fix ALL vertex half_edge pointers, not just u ----------
        for vid in 0..self.vertices.len() {
            if self.vertices[vid].removed { continue; }

            // Check if current half_edge pointer is valid
            let mut needs_update = false;
            if let Some(h) = self.vertices[vid].half_edge {
                if h >= self.half_edges.len() ||
                   self.half_edges[h].removed ||
                   self.he_from(h) != vid {
                    needs_update = true;
                }
            } else {
                needs_update = true;
            }

            if needs_update {
                self.vertices[vid].half_edge = None;

                // Find a valid half_edge
                for (hid, he) in self.half_edges.iter().enumerate() {
                    if !he.removed && self.he_from(hid) == vid {
                        self.vertices[vid].half_edge = Some(hid);
                        break;
                    }
                }
            }
        }

        // ---------- 12. Comprehensive edge_map rebuild ----------
        // The current approach might miss some edges. Let's rebuild everything.
        self.edge_map.clear();
        for (hid, he) in self.half_edges.iter().enumerate() {
            if he.removed { continue; }
            let src = self.he_from(hid);
            let dst = he.vertex;

            // Only insert if this direction isn't already in the map
            if !self.edge_map.contains_key(&(src, dst)) {
                self.edge_map.insert((src, dst), hid);
            }
        }

        // // ---------- 13. Additional validation ----------
        // println!("Post-collapse validation:");

        // // Check for orphaned half-edges
        // let mut orphan_count = 0;
        // for (hid, he) in self.half_edges.iter().enumerate() {
        //     if he.removed { continue; }
        //     let src = self.he_from(hid);
        //     let dst = he.vertex;

        //     if self.vertices[src].removed || self.vertices[dst].removed {
        //         println!("Orphaned half-edge {}: {} -> {} (vertices removed)", hid, src, dst);
        //         orphan_count += 1;
        //     }
        // }

        // if orphan_count > 0 {
        //     return Err(CollapseReject::InternalError);
        // }

        // // Check for duplicate edges in edge_map
        // let mut reverse_map = std::collections::HashMap::new();
        // for (&(src, dst), &hid) in &self.edge_map {
        //     if let Some(&existing) = reverse_map.get(&hid) {
        //         println!("Half-edge {} maps to multiple directions: {:?} and ({}, {})",
        //                  hid, existing, src, dst);
        //         return Err(CollapseReject::InternalError);
        //     }
        //     reverse_map.insert(hid, (src, dst));
        // }

        // self.validate_connectivity();
        // println!("valid!");

        Ok(())
    }

    #[inline(always)]
    fn in_remove_set(&self, h: usize, set: &[usize;6]) -> bool {
        for &x in set { if x == h { return true; } }
        false
    }

    #[inline(always)]
    fn collect_vertex_source_neighbors(&self, v: usize, out: &mut AHashSet<usize>) {
        // Traverse all half-edges whose source is v (source = prev.target)
        for (hid, he) in self.half_edges.iter().enumerate() {
            if he.removed { continue; }
            if self.he_from(hid) == v {
                let dst = he.vertex;
                if dst != v { out.insert(dst); }
            }
        }
    }

    #[inline(always)]
    fn collect_vertex_target_neighbors(&self, v: usize, out: &mut AHashSet<usize>) {
        for he in &self.half_edges {
            if he.removed { continue; }
            if he.vertex == v {
                // source = prev.target
                // we don't know prev if removed? ensure prev not removed
                let src = self.half_edges[he.prev].vertex;
                if src != v { out.insert(src); }
            }
        }
    }

    #[inline(always)]
    fn clear_edge_map_incident(&mut self, v: usize) {
        // Collect keys first to avoid borrowing issues
        let mut to_remove = SmallVec::<[(usize,usize);64]>::new();
        for (&(a,b), _) in self.edge_map.iter() {
            if a == v || b == v {
                to_remove.push((a,b));
            }
        }
        for (a,b) in to_remove {
            let _ = self.edge_map.remove(&(a,b));
        }
    }

    pub fn collapse_edge(
        &mut self,
        vertex_to_keep: usize,
        vertex_to_remove: usize,
    ) -> Result<(), CollapseReject>
    where
        Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    {
        let opts = CollapseOpts::default();
        let placement = Midpoint;
        println!("vertex to keep: {}, vertex to remove: {}", vertex_to_keep, vertex_to_remove);
        let plan = self.collapse_edge_begin_vertices(vertex_to_keep, vertex_to_remove, &placement, &opts);

        if let Ok(plan) = plan {
            return self.collapse_edge_commit(plan).map_err(|_| CollapseReject::NotAdjacent);
        } else {
            panic!("Edge collapse failed to begin, {:?}", plan.err());
        }
    }

    fn collect_incident_half_edges(&self, vertex: usize) -> SmallVec<[usize; 16]> {
        let mut incident = SmallVec::new();

        // Outgoing edges (where vertex is source)
        if let Some(start_he) = self.vertices[vertex].half_edge {
            let mut current = start_he;
            loop {
                incident.push(current);
                let twin = self.half_edges[current].twin;
                if twin == usize::MAX { break; }
                current = self.half_edges[twin].next;
                if current == start_he { break; }
            }
        }

        // Incoming edges (where vertex is target)
        if let Some(start_he) = self.vertices[vertex].half_edge {
            let mut current = start_he;
            loop {
                let twin = self.half_edges[current].twin;
                if twin != usize::MAX {
                    incident.push(twin);
                }
                if twin == usize::MAX { break; }
                current = self.half_edges[twin].next;
                if current == start_he { break; }
            }
        }

        incident
    }
}
