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

// ===== Placement policy (keep it simple; you can swap in QEM later) =====
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
    ) -> (std::collections::HashSet<usize>, std::collections::HashSet<usize>) {
        use std::collections::HashSet;

        let set0: HashSet<_> = pr.ring0.neighbors_ccw
            .iter().copied()
            .filter(|&x| x != v1 && x != v0) // exclude the edge endpoint and (defensively) the center
            .collect();

        let set1: HashSet<_> = pr.ring1.neighbors_ccw
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
        use std::collections::HashSet;
        let (set0, set1) = self.neighbor_sets_excluding_endpoints(&pr, v0, v1);
        let common: HashSet<_> = set0.intersection(&set1).copied().collect();

        // Expected set from the two incident faces (ignoring missing ones)
        let mut expected = HashSet::new();
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

        use std::collections::HashSet;
        let (set0, set1) = self.neighbor_sets_excluding_endpoints(&pr, v0, v1);
        let mut inter: HashSet<_> = set0.intersection(&set1).copied().collect();

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
        while first || h != start {
            first = false;
            mutator(&mut self.half_edges[h]);
            h = self.half_edges[h].next;
        }
    }

    /// Commit the actual contraction of `v_gone` into `v_keep`.
    /// PRE: your "begin" step succeeded and supplied a valid plan (p_star, etc.).
    pub fn collapse_edge_commit(
        &mut self,
        plan: CollapsePlan<T, N>,
    ) -> Result<(), &'static str> {
        let v_keep = plan.v_keep;
        let v_gone = plan.v_gone;

        // 0) Place the merged vertex
        self.vertices[v_keep].position = plan.p_star;

        // 1) Find the two directed half-edges of the edge (v_keep <-> v_gone)
        let pr = self
            .ring_pair(v_keep, v_gone)
            .ok_or("collapse_commit: endpoints not adjacent anymore")?;

        let i0 = pr
            .idx_v1_in_ring0
            .ok_or("collapse_commit: missing k->g half-edge")?;
        let i1 = pr
            .idx_v0_in_ring1
            .ok_or("collapse_commit: missing g->k half-edge")?;

        let h_k_g = pr.ring0.halfedges_ccw[i0]; // k -> g
        let h_g_k = pr.ring1.halfedges_ccw[i1]; // g -> k (twin)

        // Faces incident to the collapsing edge (might be None on border)
        let f_k_g = self.half_edges[h_k_g].face;
        let f_g_k = self.half_edges[h_g_k].face;

        // The three half-edges around each triangle (if present)
        let (hkg_next, hkg_prev) = (self.half_edges[h_k_g].next, self.half_edges[h_k_g].prev);
        let (hgk_next, hgk_prev) = (self.half_edges[h_g_k].next, self.half_edges[h_g_k].prev);

        // Sanity: twins should be each other
        debug_assert_eq!(self.half_edges[h_k_g].twin, h_g_k);
        debug_assert_eq!(self.half_edges[h_g_k].twin, h_k_g);

        // 2) Convert incident faces to boundary (clear their face index).
        //    We do NOT delete those half-edges; they will become border cycles.
        if let Some(f0) = f_k_g {
            if !self.faces[f0].removed {
                self.for_each_he_in_face(f0, |h| h.face = None);
                self.faces[f0].removed = true;
            }
        }
        if let Some(f1) = f_g_k {
            if !self.faces[f1].removed {
                self.for_each_he_in_face(f1, |h| h.face = None);
                self.faces[f1].removed = true;
            }
        }

        // 3) Retarget *all* half-edges that end at v_gone so they end at v_keep.
        //    This is the key trick: it implicitly changes the "from" of their `next`
        //    half-edges so outgoing spokes from v_gone now emanate from v_keep.
        {
            for (hid, he) in self.half_edges.iter_mut().enumerate() {
                if he.removed { continue; }
                if he.vertex == v_gone {
                    he.vertex = v_keep;
                    // Optional: remember one as a future seed for v_keep
                    if self.vertices[v_keep].half_edge.is_none() {
                        self.vertices[v_keep].half_edge = Some(hid);
                    }
                }
            }
        }

        // 4) Remove the collapsing directed edge pair itself.
        self.half_edges[h_k_g].removed = true;
        self.half_edges[h_g_k].removed = true;

        // 5) Stitch the two gaps around v_keep so next/prev cycles remain valid.
        //
        //    For triangle (k->g, g->a, a->k) and (g->k, k->b, b->g):
        //    After removing k->g and g->k, we want the ring to go:
        //      ... a->k  ->  k->b ...
        //    and the (formerly) g-ring wedge to go:
        //      ... b->k  ->  k->a ...
        //
        //    The four "stitch" half-edges are:
        //      left  side around k:  a->k  (h_k_g.prev)   and  k->b (h_g_k.next)
        //      right side around k:  b->g  (h_g_k.prev)   and  g->a (h_k_g.next)
        //    After step (3), the ones that touched `g` now point to `k`.
        //
        let left_a_k = hkg_prev;   // a -> k
        let left_k_b = hgk_next;   // k -> b

        let right_b_k = hgk_prev;  // b -> g  (now b -> k after step 3)
        let right_k_a = hkg_next;  // g -> a  (now k -> a after step 3)

        // Patch left seam: a->k -> k->b
        self.half_edges[left_a_k].next = left_k_b;
        self.half_edges[left_k_b].prev = left_a_k;

        // Patch right seam: b->k -> k->a
        self.half_edges[right_b_k].next = right_k_a;
        self.half_edges[right_k_a].prev = right_b_k;

        // 6) Re-seed v_keep.half_edge if needed (choose any outgoing from k).
        if self.vertices[v_keep].half_edge.is_none() || {
            let he = self.vertices[v_keep].half_edge.unwrap();
            self.half_edges[he].removed || self.he_from(he) != v_keep
        } {
            // Find an outgoing from k by rotating CCW once.
            if let Some(mut he0) = pr.ring0.halfedges_ccw.into_iter().find(|&h| {
                !self.half_edges[h].removed && self.he_from(h) == v_keep
            }) {
                // If that was the deleted one, step to the stitched neighbor.
                if self.half_edges[he0].removed {
                    let tw = self.half_edges[he0].twin;
                    he0 = self.half_edges[tw].next;
                }
                self.vertices[v_keep].half_edge = Some(he0);
            } else {
                // Fallback scan (rare)
                for (hid, he) in self.half_edges.iter().enumerate() {
                    if !he.removed && self.he_from(hid) == v_keep {
                        self.vertices[v_keep].half_edge = Some(hid);
                        break;
                    }
                }
            }
        }

        // 7) Kill v_gone (no outgoing half-edge anymore).
        self.vertices[v_gone].half_edge = None;
        // If you track a "removed" flag on vertices, set it here.
        // self.vertices[v_gone].removed = true;

        // 8) Optional: make twins along the two seams consistent.
        //    If your mesh always maintains a twin for every half-edge, you may need to
        //    re-pair boundary half-edges that just became opposites. A simple local
        //    re-twinning at the two seams (left_a_k <-> right_k_a, right_b_k <-> left_k_b)
        //    *may* be correct depending on your global invariant. If you already have
        //    a "fix_twin_locally(from,to)" utility, call it here.

        // 9) (Recommended) Local connectivity validation (debug builds)
        // self.validate_connectivity();

        Ok(())
    }
}
