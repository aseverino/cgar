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

use crate::{
    geometry::{
        point::{Point, PointOps},
        vector::{Vector, VectorOps},
    },
    impl_mesh,
    mesh::{basic_types::Mesh, face::Face, half_edge::HalfEdge},
    numeric::scalar::Scalar,
};

#[inline(always)]
fn explain_edge_block<TS: Scalar, const M: usize>(
    this: &Mesh<TS, M>,
    from: usize,
    to: usize,
) -> String {
    match this.edge_map.get(&(from, to)) {
        None => "free (absent)".to_string(),
        Some(&h) => {
            if h >= this.half_edges.len() {
                return format!("edge_map→{} (OOB)", h);
            }
            let he = &this.half_edges[h];
            let kind = if he.removed {
                "removed"
            } else if he.face.is_none() {
                "BORDER"
            } else {
                "INTERIOR"
            };
            let f = he.face.map(|x| x as isize).unwrap_or(-1);
            let t = he.twin as isize;
            let f_str = format!("{} h={} twin={} face={}", kind, h, t, f);
            if let Some(f) = he.face {
                if f < this.faces.len() && !this.faces[f].removed {
                    let he0 = he;
                    let he1 = &this.half_edges[he.next];
                    let he2 = &this.half_edges[he1.next];
                    let v0 = this.half_edges[he.twin].vertex;
                    let v1 = he.vertex;
                    let v2 = he1.vertex;
                    let p0 = &this.vertices[v0].position;
                    let p1 = &this.vertices[v1].position;
                    let p2 = &this.vertices[v2].position;
                    f_str
                        + &format!(
                            " verts=({},{},{}) pos=({:?},{:?},{:?})",
                            v0, v1, v2, p0, p1, p2
                        )
                        .to_string()
                } else {
                    f_str.to_string()
                }
            } else {
                f_str.to_string()
            }
        }
    }
}

impl_mesh! {
    /// Lightweight owner-safe map removal for (tail=head(twin), head=vertex)
    #[inline(always)]
    fn remove_map_entry_if_owner(&mut self, h: usize) {
        if h >= self.half_edges.len() { return; }
        let he = &self.half_edges[h];
        let t = he.twin;
        if t >= self.half_edges.len() { return; }
        let u = self.half_edges[t].vertex; // tail(origin)
        let v = he.vertex;                 // head
        if let Some(&owner) = self.edge_map.get(&(u, v)) {
            if owner == h { self.edge_map.remove(&(u, v)); }
        }
    }

    /// Collect all border components reachable from `starts` in ONE pass.
    /// This is your same algorithm, just seeded globally with a single `visited`.
    pub fn weld_border_components_from(&mut self, starts: &[usize]) {
        if starts.is_empty() { return; }
        let mut visited = vec![false; self.half_edges.len()];
        for &b in starts {
            if b != usize::MAX
                && b < self.half_edges.len()
                && !self.half_edges[b].removed
                && self.half_edges[b].face.is_none()
                && !visited[b]
            {
                self.rebuild_border_component_from(b, &mut visited);
            }
        }
    }

    /// Phase 3 of your old `remove_triangle`, but batched over many vertices.
    pub fn fix_vertices_outgoing_for(&mut self, verts: &[usize]) {
        #[inline(always)]
        fn is_live<TS: Scalar, const M: usize>(m: &Mesh<TS,M>, h: usize) -> bool {
            h < m.half_edges.len() && !m.half_edges[h].removed
        }
        #[inline(always)]
        fn is_outgoing_from<TS: Scalar, const M: usize>(m: &Mesh<TS,M>, h: usize, v: usize) -> bool {
            is_live(m,h) && m.half_edges[m.half_edges[h].prev].vertex == v
        }
        #[inline(always)]
        fn normalize_to_outgoing<TS: Scalar, const M: usize>(
            m: &Mesh<TS,M>, h: usize, v: usize
        ) -> Option<usize> {
            if !is_live(m,h) { return None; }
            if is_outgoing_from(m,h,v) { return Some(h); }
            if m.half_edges[h].vertex == v {
                let t = m.half_edges[h].twin;
                if t < m.half_edges.len() && is_live(m,t) && is_outgoing_from(m,t,v) {
                    return Some(t);
                }
            }
            None
        }
        fn find_any_outgoing<TS: Scalar, const M: usize>(
            m: &Mesh<TS,M>, v: usize
        ) -> Option<usize> {
            // Local walk from any spoke touching v (cheap global scan; this runs only for touched verts)
            for h in 0..m.half_edges.len() {
                if !is_live(m,h) { continue; }
                if m.half_edges[h].vertex == v || m.half_edges[m.half_edges[h].twin].vertex == v {
                    if let Some(hh) = normalize_to_outgoing(m,h,v) { return Some(hh); }
                }
            }
            None
        }

        // De-dup input
        use ahash::AHashSet;
        let mut uniq: AHashSet<usize> = AHashSet::with_capacity(verts.len());
        for &v in verts { uniq.insert(v); }

        for v in uniq {
            let needs = match self.vertices[v].half_edge {
                Some(h) => !is_outgoing_from(self, h, v),
                None => true,
            };
            if needs {
                self.vertices[v].half_edge = find_any_outgoing(self, v);
            }
        }
    }

    /// NEW: remove many triangles but **defer** border stitching.
    /// Returns (starts_for_weld, affected_vertices).
    pub fn remove_triangles_deferred(&mut self, faces: &[usize]) -> (Vec<usize>, Vec<usize>) {
        #[inline(always)]
        fn is_live<TS: Scalar, const M: usize>(m: &Mesh<TS,M>, h: usize) -> bool {
            h < m.half_edges.len() && !m.half_edges[h].removed
        }
        #[inline(always)]
        fn key_pair<TS: Scalar, const M: usize>(m: &Mesh<TS, M>, h: usize) -> (usize, usize) {
            let tail = m.half_edges[m.half_edges[h].twin].vertex;
            let head = m.half_edges[h].vertex;
            (tail, head)
        }

        let mut starts: Vec<usize> = Vec::with_capacity(faces.len() * 6);
        let mut affected_vs: Vec<usize> = Vec::with_capacity(faces.len() * 3);

        for &f in faces {
            if f >= self.faces.len() || self.faces[f].removed { continue; }

            // spokes of the face
            let he0 = self.faces[f].half_edge;
            let he1 = self.half_edges[he0].next;
            let he2 = self.half_edges[he1].next;
            let hes = [he0, he1, he2];

            // capture tail vertices (origins) BEFORE mutation
            affected_vs.push(self.half_edges[self.half_edges[he0].twin].vertex);
            affected_vs.push(self.half_edges[self.half_edges[he1].twin].vertex);
            affected_vs.push(self.half_edges[self.half_edges[he2].twin].vertex);

            self.faces[f].removed = true;

            for he in hes {
                if !is_live(self, he) { continue; }
                let twin = self.half_edges[he].twin;

                let opp_alive = if let Some(adj_f) = self.half_edges[twin].face {
                    !self.faces[adj_f].removed
                } else {
                    false
                };

                if opp_alive {
                    // Case A: convert this INTERIOR→BORDER, make it a temp self-loop.
                    self.half_edges[he].face = None;
                    self.half_edges[he].next = he;
                    self.half_edges[he].prev = he;
                    self.half_edges[he].removed = false;

                    // keep its (u,v) map as-is
                    starts.push(he); // seed for welding later
                } else {
                    // Case B: delete BOTH spokes (no surviving opposite interior)
                    if is_live(self, twin) {
                        // If the opposite was border, DO NOT splice now. Just kill and drop map.
                        self.remove_map_entry_if_owner(twin);
                        self.half_edges[twin].removed = true;

                        // Its neighbors will need welding later. Capture them as seeds if they exist.
                        let bp = self.half_edges[twin].prev;
                        let bn = self.half_edges[twin].next;
                        if bp < self.half_edges.len() { starts.push(bp); }
                        if bn < self.half_edges.len() { starts.push(bn); }
                    }

                    self.remove_map_entry_if_owner(he);
                    self.half_edges[he].removed = true;
                }
            }

            // Keep face handle pointing to any of its spokes; it's removed anyway.
            self.faces[f].half_edge = he0;
        }

        // Debug: no removed spoke should still own its map entry
        #[cfg(debug_assertions)]
        {
            for h in 0..self.half_edges.len() {
                if self.half_edges[h].removed {
                    let (u,v) = key_pair(self, h);
                    if let Some(&owner) = self.edge_map.get(&(u,v)) {
                        debug_assert_ne!(owner, h, "removed spoke {} still owns edge_map ({},{})", h, u, v);
                    }
                }
            }
        }

        (starts, affected_vs)
    }

    /// rotate around the **head** of interior half-edge `k`:
    /// next interior ending at that head is `prev(twin(k))`.
    #[inline(always)]
    fn rotate_around_head(
        &self, k: usize
    ) -> usize {
        let tk = self.half_edges[k].twin;
        debug_assert!(tk != usize::MAX);
        self.half_edges[tk].prev
    }
    /// Given a BORDER `b`, return the **next BORDER** in CCW order around the *outside face*.
    /// We start from an interior with head==head(b) and rotate until we hit a border spoke.
    #[inline(always)]
    fn find_next_border(
        &self, b: usize
    ) -> usize {
        debug_assert!(self.half_edges[b].face.is_none());
        let t = self.half_edges[b].twin; // interior across b
        debug_assert!(t != usize::MAX && self.half_edges[t].face.is_some());

        let mut k = self.half_edges[t].prev; // interior with HEAD == head(b)
        let start_k = k;
        let limit = self.half_edges.len().saturating_add(64);
        let mut steps = 0usize;

        loop {
            let cand = self.half_edges[k].twin; // spoke leaving head(b)
            if cand != usize::MAX &&
                !self.half_edges[cand].removed &&
                self.half_edges[cand].face.is_none() &&
                cand != b {
                return cand; // next border around outside
            }
            k = self.rotate_around_head(k);
            steps += 1;
            if k == start_k || steps > limit { break; }
        }
        usize::MAX // would indicate a crack; atomic add should avoid this
    }

    /// Rebuild the **single border component** that contains `start_b` (two-phase: collect then write).
    #[inline(always)]
    fn rebuild_border_component_from(
        &mut self, start_b: usize, visited: &mut Vec<bool>
    ) {
        if start_b == usize::MAX { return; }
        if self.half_edges[start_b].removed || self.half_edges[start_b].face.is_some() { return; }

        // Phase 1: collect the ring without writing anything
        let mut ring: Vec<usize> = Vec::with_capacity(16);
        let mut cur = start_b;
        let limit = self.half_edges.len().saturating_add(64);
        let mut steps = 0usize;

        while !visited[cur] {
            visited[cur] = true;
            ring.push(cur);
            let nb = self.find_next_border(cur);
            assert!(nb != usize::MAX,
                "atomic add_triangle: border crack at vertex {} (no next border for {})",
                self.half_edges[cur].vertex, cur);
            cur = nb;
            steps += 1;
            if cur == start_b || steps > limit { break; }
        }

        // Phase 2: write next/prev for the ring (reciprocal by construction)
        let n = ring.len();
        if n == 0 { return; }
        for i in 0..n {
            let b  = ring[i];
            let nb = ring[(i + 1) % n];
            let pb = ring[(i + n - 1) % n];
            self.half_edges[b].next = nb;
            self.half_edges[b].prev = pb;
        }
    }

    /// Exact inverse of `add_triangle` for a single triangular face `f`.
    /// Mirrors the three per-edge cases of `ensure_dir`:
    ///   A) If opposite is INTERIOR → convert this spoke to BORDER, then wire its border component(s).
    ///   B) If opposite is BORDER (or already dead) → delete BOTH spokes and splice the border ring.
    /// Map policy:
    ///   - Never insert/re-key here. We only **remove** directed entries for spokes we actually delete.
    ///   - Entries for spokes we convert (interior→border) remain as-is.
    /// Twin policy:
    ///   - Twins stay reciprocal. We never break twin links; if both spokes die, both are flagged `removed`.
    pub fn remove_triangle(&mut self, f: usize)
    {
        if self.faces[f].removed {
            return;
        }

        #[inline(always)]
        fn is_live<TS: Scalar, const M: usize>(m: &Mesh<TS, M>, h: usize) -> bool {
            h < m.half_edges.len() && !m.half_edges[h].removed
        }

        // Stable key for (tail, head) that does NOT depend on prev/next:
        #[inline(always)]
        fn key_pair<TS: Scalar, const M: usize>(m: &Mesh<TS, M>, h: usize) -> (usize, usize) {
            let tail = m.half_edges[m.half_edges[h].twin].vertex; // origin of h
            let head = m.half_edges[h].vertex; // head of h
            (tail, head)
        }

        #[inline(always)]
        fn remove_map_entry_if_owner<TS: Scalar, const M: usize>(m: &mut Mesh<TS, M>, h: usize) {
            if !is_live(m, h) {
                // Even if not live, we may want to clear the entry if it still points to this id.
                let (u, v) = key_pair(m, h);
                if let Some(&owner) = m.edge_map.get(&(u, v)) {
                    if owner == h {
                        m.edge_map.remove(&(u, v));
                    }
                }
                return;
            }
            let (u, v) = key_pair(m, h);
            if let Some(&owner) = m.edge_map.get(&(u, v)) {
                if owner == h {
                    m.edge_map.remove(&(u, v));
                }
            }
        }

        // Splice a BORDER spoke out of its ring and drop its map entry.
        #[inline(always)]
        fn splice_out_border<TS: Scalar, const M: usize>(m: &mut Mesh<TS, M>, b: usize) {
            debug_assert!(is_live(m, b));
            debug_assert!(m.half_edges[b].face.is_none());

            let bp = m.half_edges[b].prev;
            let bn = m.half_edges[b].next;

            // Stitch neighbors around the outside
            m.half_edges[bp].next = bn;
            m.half_edges[bn].prev = bp;

            // Remove only THIS spoke's map entry, keyed by its (tail=twin.vertex, head=vertex)
            remove_map_entry_if_owner(m, b);

            // Kill the spoke
            m.half_edges[b].removed = true;
        }

        // ---------- fetch the three interior spokes (triangle ring) ----------
        let he0 = self.faces[f].half_edge;
        let he1 = self.half_edges[he0].next;
        let he2 = self.half_edges[he1].next;
        let hes = [he0, he1, he2];

        // Capture the triangle's three vertices (origins of those spokes) BEFORE any mutation.
        // Using twin.vertex guarantees stability even if `prev` is temporarily self-looped later.
        let tri_tail_vs = [
            self.half_edges[self.half_edges[he0].twin].vertex,
            self.half_edges[self.half_edges[he1].twin].vertex,
            self.half_edges[self.half_edges[he2].twin].vertex,
        ];

        // Mark the face as removed up front
        self.faces[f].removed = true;

        // Collect spokes that become NEW borders (to wire after the loop): (border_spoke, interior_twin)
        let mut new_borders: [(usize, usize); 3] = [(usize::MAX, usize::MAX); 3];
        let mut nb_count = 0usize;

        // ---------- Phase 1: per-edge inverse of `ensure_dir` ----------
        for he in hes {
            if !is_live(self, he) {
                continue;
            }

            let twin = self.half_edges[he].twin;
            debug_assert!(twin < self.half_edges.len(), "Malformed mesh: missing twin");

            // Opposite side alive?
            let opp_alive = if let Some(adj_f) = self.half_edges[twin].face {
                !self.faces[adj_f].removed
            } else {
                false
            };

            if opp_alive {
                // Case A (inverse of "existing BORDER promoted to INTERIOR" in add_triangle):
                // Convert this INTERIOR to a BORDER on this side; keep its key (u,v) the same.
                self.half_edges[he].face = None;

                // Temporary self-loops; we'll wire the component afterward.
                self.half_edges[he].next = he;
                self.half_edges[he].prev = he;
                self.half_edges[he].removed = false;

                // Map: leave (u,v) → he as-is. (This spoke already owned that key.)
                // Twin stays interior and keeps its (v,u) key.

                if nb_count < 3 {
                    new_borders[nb_count] = (he, twin);
                    nb_count += 1;
                }
            } else {
                // Case B (inverse of "created NEW border twin" in add_triangle):
                // There is no surviving interior on the opposite side → delete BOTH spokes.
                // If the opposite was actually a BORDER spoke, splice it out of the ring.
                if is_live(self, twin) && self.half_edges[twin].face.is_none() {
                    splice_out_border(self, twin);
                } else if is_live(self, twin) {
                    // Opposite might be a dead/degenerate leftover; just drop its map entry and kill it.
                    remove_map_entry_if_owner(self, twin);
                    self.half_edges[twin].removed = true;
                }

                // Drop this interior spoke's map entry, then kill it.
                remove_map_entry_if_owner(self, he);
                self.half_edges[he].removed = true;
            }
        }

        // ---------- Phase 2: wire only the border components that changed ----------
        if nb_count > 0 {
            let mut visited = vec![false; self.half_edges.len()];
            for i in 0..nb_count {
                let (b, _) = new_borders[i];
                if b != usize::MAX
                    && is_live(self, b)
                    && self.half_edges[b].face.is_none()
                    && !visited[b]
                {
                    // Rebuild the entire border component that contains `b`,
                    // exactly like `add_triangle` does for newly-created border twins.
                    self.rebuild_border_component_from(b, &mut visited);
                }
            }
        }

        // ---------- Phase 3: fix vertex.half_edge at the three triangle vertices ----------
        // Invariant: vertex.half_edge must be OUTGOING from that vertex,
        // i.e., half_edges[prev(h)].vertex == v.
        #[inline(always)]
        fn is_outgoing_from<TS: Scalar, const M: usize>(m: &Mesh<TS, M>, h: usize, v: usize) -> bool {
            is_live(m, h) && m.half_edges[m.half_edges[h].prev].vertex == v
        }
        #[inline(always)]
        fn normalize_to_outgoing<TS: Scalar, const M: usize>(
            m: &Mesh<TS, M>,
            h: usize,
            v: usize,
        ) -> Option<usize> {
            if !is_live(m, h) {
                return None;
            }
            if is_outgoing_from(m, h, v) {
                return Some(h);
            }
            if m.half_edges[h].vertex == v {
                let t = m.half_edges[h].twin;
                if t < m.half_edges.len() && is_live(m, t) && is_outgoing_from(m, t, v) {
                    return Some(t);
                }
            }
            None
        }
        fn find_any_outgoing<TS: Scalar, const M: usize>(
            m: &Mesh<TS, M>,
            v: usize,
            seeds: &[usize],
        ) -> Option<usize> {
            // quick normalize from seeds
            for &s in seeds {
                if let Some(h) = normalize_to_outgoing(m, s, v) {
                    return Some(h);
                }
            }
            // bounded local walk around seeds
            for &s in seeds {
                if !is_live(m, s) {
                    continue;
                }
                let mut t = s;
                for _ in 0..24 {
                    let nxt = m.half_edges[t].next;
                    if nxt >= m.half_edges.len() || !is_live(m, nxt) {
                        break;
                    }
                    let cand = m.half_edges[nxt].twin;
                    if cand >= m.half_edges.len() || !is_live(m, cand) {
                        break;
                    }
                    if let Some(h) = normalize_to_outgoing(m, cand, v) {
                        return Some(h);
                    }
                    t = cand;
                }
            }
            // rare fallback
            for h in 0..m.half_edges.len() {
                if is_outgoing_from(m, h, v) {
                    return Some(h);
                }
            }
            None
        }

        // Seed set: the 3 spokes, their twins, plus any new borders & their twins.
        let mut seeds: Vec<usize> = Vec::with_capacity(12);
        seeds.extend_from_slice(&hes);
        for &h in &hes {
            let t = self.half_edges[h].twin;
            if t < self.half_edges.len() {
                seeds.push(t);
            }
        }
        for i in 0..nb_count {
            let (b, t) = new_borders[i];
            if b != usize::MAX {
                seeds.push(b);
            }
            if t != usize::MAX && t < self.half_edges.len() {
                seeds.push(t);
            }
        }

        for &v in &tri_tail_vs {
            let needs = match self.vertices[v].half_edge {
                Some(h) => !is_outgoing_from(self, h, v),
                None => true,
            };
            if needs {
                self.vertices[v].half_edge = find_any_outgoing(self, v, &seeds);
            }
        }

        // Face handle can remain pointing to its old half-edge; it's removed anyway.
        self.faces[f].half_edge = he0;
    }

    /// Atomic add of a CCW triangle (v0, v1, v2).
    /// On return, the mesh has **no cracks**: border half-edges form proper cycles, no self-loops.
    pub fn add_triangle(&mut self, v0: usize, v1: usize, v2: usize) -> usize {
        #[inline(always)]
        fn dir_is_free<TS: Scalar, const M: usize>(this: &Mesh<TS, M>, from: usize, to: usize) -> bool {
            if let Some(&h) = this.edge_map.get(&(from, to)) {
                !this.half_edges[h].removed && this.half_edges[h].face.is_none()
            } else {
                true
            }
        }

        /// Remove BORDER half-edge `b` from its border ring by stitching neighbors.
        #[inline(always)]
        fn unlink_border<TS: Scalar, const M: usize>(this: &mut Mesh<TS, M>, b: usize) {
            debug_assert!(b != usize::MAX);
            debug_assert!(this.half_edges[b].face.is_none() && !this.half_edges[b].removed);
            let p = this.half_edges[b].prev;
            let n = this.half_edges[b].next;
            if p != b && n != b {
                // neighbors are border spokes as well
                this.half_edges[p].next = n;
                this.half_edges[n].prev = p;
            }
            // caller will overwrite b.{next,prev} when repurposing
        }

        /// Ensure (from->to) exists as **INTERIOR** for `face_idx`.
        /// If it existed as BORDER, unlink then promote.
        /// Returns (interior_idx, twin_idx, created_new_border_twin).
        #[inline(always)]
        fn ensure_dir<TS: Scalar, const M: usize>(
            this: &mut Mesh<TS, M>,
            face_idx: usize,
            from: usize,
            to: usize,
        ) -> (usize, usize, bool, Option<(usize, usize)>) {
            if let Some(&he) = this.edge_map.get(&(from, to)) {
                // Check if this edge is already interior (manifold violation)
                if this.half_edges[he].face.is_some() {
                    panic!("Non-manifold mesh: edge ({},{}) already has a face", from, to);
                }

                // Edge exists as border; capture neighbors BEFORE unlink
                debug_assert!(this.half_edges[he].face.is_none());
                let p = this.half_edges[he].prev;
                let n = this.half_edges[he].next;

                unlink_border::<TS, M>(this, he);
                this.half_edges[he].face = Some(face_idx); // promote to interior

                let t = this.half_edges[he].twin;
                if t != usize::MAX {
                    this.half_edges[t].twin = he;
                }
                (he, t, false, Some((p, n)))
            } else {
                // create interior
                let he = this.half_edges.len();
                let mut h = HalfEdge::new(to);
                h.face = Some(face_idx);
                this.half_edges.push(h);
                this.edge_map.insert((from, to), he);

                // link/create twin
                if let Some(&rev) = this.edge_map.get(&(to, from)) {
                    this.half_edges[he].twin = rev;
                    this.half_edges[rev].twin = he;
                    (he, rev, false, None)
                } else {
                    // NEW border twin
                    let b = this.half_edges.len();
                    let mut bh = HalfEdge::new(from);
                    bh.twin = he;
                    bh.next = b; // temporary
                    bh.prev = b;
                    this.half_edges.push(bh);
                    this.edge_map.insert((to, from), b);
                    this.half_edges[he].twin = b;
                    (he, b, true, None)
                }
            }
        }

        // ---------- choose feasible orientation (auto-flip if needed) ----------
        let ccw_ok = dir_is_free::<T, N>(self, v0, v1)
            && dir_is_free::<T, N>(self, v1, v2)
            && dir_is_free::<T, N>(self, v2, v0);
        let cw_ok = dir_is_free::<T, N>(self, v0, v2)
            && dir_is_free::<T, N>(self, v2, v1)
            && dir_is_free::<T, N>(self, v1, v0);

        let (edges, verts) = if ccw_ok {
            ([(v0, v1), (v1, v2), (v2, v0)], [v0, v1, v2])
        } else if cw_ok {
            ([(v0, v2), (v2, v1), (v1, v0)], [v0, v2, v1]) // auto-flip to feasible directed cycle
        } else {
            eprintln!(
        "add_triangle blocked for tri=({},{},{}):\n\
         CCW edges: ({},{}): {}\n\
         \t        ({},{}): {}\n\
         \t        ({},{}): {}\n\
         CW  edges: ({},{}): {}\n\
         \t        ({},{}): {}\n\
         \t        ({},{}): {}\n",
        v0, v1, v2,
        v0, v1, explain_edge_block(self, v0, v1),
        v1, v2, explain_edge_block(self, v1, v2),
        v2, v0, explain_edge_block(self, v2, v0),
        v0, v2, explain_edge_block(self, v0, v2),
        v2, v1, explain_edge_block(self, v2, v1),
        v1, v0, explain_edge_block(self, v1, v0),
    );
            //panic!("add_triangle: neither CCW nor CW directed sides are free (non-manifold/winding).");
            return usize::MAX;
        };

        // Fail fast if any selected directed edge already has a face on this side.
        for &(from, to) in &edges {
            if let Some(&h) = self.edge_map.get(&(from, to)) {
                assert!(
                    self.half_edges[h].face.is_none(),
                    "selected orientation would reuse directed edge ({},{}) already bound to a face",
                    from,
                    to
                );
            }
        }

        // ---------- create face & three directed edges ----------
        let face_idx = self.faces.len();
        self.faces.push(Face::new(0));

        let (e0, t0, n0, p0n0) = ensure_dir::<T, N>(self, face_idx, edges[0].0, edges[0].1);
        let (e1, t1, n1, p1n1) = ensure_dir::<T, N>(self, face_idx, edges[1].0, edges[1].1);
        let (e2, t2, n2, p2n2) = ensure_dir::<T, N>(self, face_idx, edges[2].0, edges[2].1);

        // Interior CCW ring (of chosen orientation)
        self.half_edges[e0].next = e1;
        self.half_edges[e1].prev = e0;
        self.half_edges[e1].next = e2;
        self.half_edges[e2].prev = e1;
        self.half_edges[e2].next = e0;
        self.half_edges[e0].prev = e2;

        // Build worklist of border components that **changed**.
        let mut starts: Vec<usize> = Vec::with_capacity(6);

        // (A) Newly created border twins (same as before)
        if n0 {
            starts.push(self.half_edges[e0].twin);
        }
        if n1 {
            starts.push(self.half_edges[e1].twin);
        }
        if n2 {
            starts.push(self.half_edges[e2].twin);
        }

        // (B) Neighbors of any border we promoted to interior
        if let Some((p, n)) = p0n0 {
            starts.push(p);
            starts.push(n);
        }
        if let Some((p, n)) = p1n1 {
            starts.push(p);
            starts.push(n);
        }
        if let Some((p, n)) = p2n2 {
            starts.push(p);
            starts.push(n);
        }

        // Weld only those components (dedupe by visited[] across the whole mesh)
        if !starts.is_empty() {
            let mut visited = vec![false; self.half_edges.len()];
            for &b in &starts {
                if b != usize::MAX
                    && b < self.half_edges.len()
                    && !self.half_edges[b].removed
                    && self.half_edges[b].face.is_none()
                    && !visited[b]
                {
                    self.rebuild_border_component_from(b, &mut visited);
                }
            }
        }

        // ---------- finalize ----------
        self.faces[face_idx].half_edge = e0;
        self.vertices[verts[0]].half_edge.get_or_insert(e0);
        self.vertices[verts[1]].half_edge.get_or_insert(e1);
        self.vertices[verts[2]].half_edge.get_or_insert(e2);

        // Strong local postcondition on touched borders: no self-loops among starts
        #[cfg(debug_assertions)]
        for &b in &starts {
            if b != usize::MAX && b < self.half_edges.len() {
                if !self.half_edges[b].removed && self.half_edges[b].face.is_none() {
                    debug_assert!(
                        self.half_edges[b].next != b && self.half_edges[b].prev != b,
                        "border self-loop at he {}",
                        b
                    );
                }
            }
        }

        face_idx
    }

    pub fn add_triangles_deferred(&mut self, triangles: &[(usize, usize, usize)]) -> Vec<usize> {
        let mut new_borders = Vec::with_capacity(triangles.len() * 3);
        let mut affected_vertices = Vec::with_capacity(triangles.len() * 3);
        let mut face_indices = Vec::with_capacity(triangles.len());

        for &(v0, v1, v2) in triangles {
            let face_idx = self.add_triangle_no_weld(v0, v1, v2, &mut new_borders, &mut affected_vertices);
            if face_idx != usize::MAX {
                face_indices.push(face_idx);
            }
        }

        // Single border rebuild pass for all affected components
        if !new_borders.is_empty() {
            self.weld_border_components_from(&new_borders);
        }

        // Single vertex fix pass
        if !affected_vertices.is_empty() {
            self.fix_vertices_outgoing_for(&affected_vertices);
        }

        face_indices
    }

    /// Core triangle addition without border rebuilding
    fn add_triangle_no_weld(
        &mut self,
        v0: usize,
        v1: usize,
        v2: usize,
        new_borders: &mut Vec<usize>,
        affected_vertices: &mut Vec<usize>
    ) -> usize {
        let ccw_ok = self.dir_is_free(v0, v1) && self.dir_is_free(v1, v2) && self.dir_is_free(v2, v0);
        let cw_ok = self.dir_is_free(v0, v2) && self.dir_is_free(v2, v1) && self.dir_is_free(v1, v0);

        let (edges, verts) = if ccw_ok {
            ([(v0, v1), (v1, v2), (v2, v0)], [v0, v1, v2])
        } else if cw_ok {
            ([(v0, v2), (v2, v1), (v1, v0)], [v0, v2, v1])
        } else {
            return usize::MAX;
        };

        let face_idx = self.faces.len();
        self.faces.push(Face::new(0));

        let (e0, _, n0, p0n0) = self.ensure_dir_no_weld(face_idx, edges[0].0, edges[0].1);
        let (e1, _, n1, p1n1) = self.ensure_dir_no_weld(face_idx, edges[1].0, edges[1].1);
        let (e2, _, n2, p2n2) = self.ensure_dir_no_weld(face_idx, edges[2].0, edges[2].1);

        // Interior ring
        self.half_edges[e0].next = e1;
        self.half_edges[e1].prev = e0;
        self.half_edges[e1].next = e2;
        self.half_edges[e2].prev = e1;
        self.half_edges[e2].next = e0;
        self.half_edges[e0].prev = e2;

        // Collect border starts for later welding
        if n0 { new_borders.push(self.half_edges[e0].twin); }
        if n1 { new_borders.push(self.half_edges[e1].twin); }
        if n2 { new_borders.push(self.half_edges[e2].twin); }

        if let Some((p, n)) = p0n0 { new_borders.extend_from_slice(&[p, n]); }
        if let Some((p, n)) = p1n1 { new_borders.extend_from_slice(&[p, n]); }
        if let Some((p, n)) = p2n2 { new_borders.extend_from_slice(&[p, n]); }

        // Collect affected vertices
        affected_vertices.extend_from_slice(&verts);

        self.faces[face_idx].half_edge = e0;
        face_idx
    }

    /// Lightweight dir_is_free check
    #[inline(always)]
    fn dir_is_free(&self, from: usize, to: usize) -> bool {
        if let Some(&h) = self.edge_map.get(&(from, to)) {
            !self.half_edges[h].removed && self.half_edges[h].face.is_none()
        } else {
            true
        }
    }

    /// ensure_dir without border component rebuilding
    #[inline(always)]
    fn ensure_dir_no_weld(
        &mut self,
        face_idx: usize,
        from: usize,
        to: usize,
    ) -> (usize, usize, bool, Option<(usize, usize)>) {
        if let Some(&he) = self.edge_map.get(&(from, to)) {
            let p = self.half_edges[he].prev;
            let n = self.half_edges[he].next;

            // Unlink from border ring
            if p != he && n != he {
                self.half_edges[p].next = n;
                self.half_edges[n].prev = p;
            }

            self.half_edges[he].face = Some(face_idx);
            let t = self.half_edges[he].twin;
            (he, t, false, Some((p, n)))
        } else {
            let he = self.half_edges.len();
            let mut h = HalfEdge::new(to);
            h.face = Some(face_idx);
            self.half_edges.push(h);
            self.edge_map.insert((from, to), he);

            if let Some(&rev) = self.edge_map.get(&(to, from)) {
                self.half_edges[he].twin = rev;
                self.half_edges[rev].twin = he;
                (he, rev, false, None)
            } else {
                let b = self.half_edges.len();
                let mut bh = HalfEdge::new(from);
                bh.twin = he;
                bh.next = b;  // temporary self-loop
                bh.prev = b;
                self.half_edges.push(bh);
                self.edge_map.insert((to, from), b);
                self.half_edges[he].twin = b;
                (he, b, true, None)
            }
        }
    }
}
