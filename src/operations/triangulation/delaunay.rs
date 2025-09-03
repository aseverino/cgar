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

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

use ahash::{AHashMap, AHashSet};
use smallvec::SmallVec;

use crate::boolean::batching::{FaceJobUV, on_edge_with_t, print_face_job_and_dt};
use crate::geometry::Point2;
use crate::geometry::point::Point;
use crate::geometry::spatial_element::SpatialElement;
use crate::kernel::point_in_or_on_triangle;
use crate::kernel::predicates::{TrianglePoint, bbox, centroid2, incircle, orient2d};
use crate::mesh::basic_types::Mesh;
use crate::numeric::scalar::Scalar;

pub const SQRT_3: f64 = 1.7320508075688772;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Edge(usize, usize);

impl Edge {
    #[inline]
    fn new(a: usize, b: usize) -> Self {
        if a < b { Edge(a, b) } else { Edge(b, a) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Triangle(pub usize, pub usize, pub usize);

impl Triangle {
    #[inline]
    pub fn as_sorted_indices(&self) -> (usize, usize, usize) {
        let mut v = [self.0, self.1, self.2];
        v.sort_unstable();
        (v[0], v[1], v[2])
    }
}

#[derive(Default)]
struct Adj {
    // undirected edge -> up to 2 triangle indices
    edge2tris: AHashMap<Edge, SmallVec<[usize; 2]>>,
    // vertex -> incident triangle indices (no order needed for the walk)
    vert2tris: Vec<SmallVec<[usize; 8]>>,
}

impl Adj {
    fn rebuild(points_len: usize, tris: &[Triangle]) -> Self {
        let mut s = Adj {
            edge2tris: AHashMap::with_capacity_and_hasher(tris.len() * 3, Default::default()),
            vert2tris: vec![SmallVec::new(); points_len],
        };
        for (ti, t) in tris.iter().enumerate() {
            s.add_tri(ti, *t);
        }
        s
    }
    #[inline]
    fn add_tri(&mut self, ti: usize, t: Triangle) {
        let e01 = Edge::new(t.0, t.1);
        let e12 = Edge::new(t.1, t.2);
        let e20 = Edge::new(t.2, t.0);
        self.edge2tris.entry(e01).or_default().push(ti);
        self.edge2tris.entry(e12).or_default().push(ti);
        self.edge2tris.entry(e20).or_default().push(ti);
        self.vert2tris[t.0].push(ti);
        self.vert2tris[t.1].push(ti);
        self.vert2tris[t.2].push(ti);
    }
    #[inline]
    fn remove_tri(&mut self, ti: usize, t: Triangle) {
        for e in [
            Edge::new(t.0, t.1),
            Edge::new(t.1, t.2),
            Edge::new(t.2, t.0),
        ] {
            if let Some(v) = self.edge2tris.get_mut(&e) {
                if let Some(pos) = v.iter().position(|&x| x == ti) {
                    v.swap_remove(pos);
                }
                if v.is_empty() {
                    self.edge2tris.remove(&e);
                }
            }
        }
        for v in [t.0, t.1, t.2] {
            let lst = &mut self.vert2tris[v];
            if let Some(pos) = lst.iter().position(|&x| x == ti) {
                lst.swap_remove(pos);
            }
        }
    }
    #[inline]
    fn replace_tri(&mut self, ti: usize, old_t: Triangle, new_t: Triangle) {
        self.remove_tri(ti, old_t);
        self.add_tri(ti, new_t);
    }
}

#[derive(Clone, Debug)]
pub struct Delaunay<T: Scalar> {
    pub points: Vec<Point2<T>>,
    pub triangles: Vec<Triangle>, // indices into points
}

impl<T: Scalar> Delaunay<T>
where
    Point<T, 2>:
        crate::geometry::point::PointOps<T, 2, Vector = crate::geometry::vector::Vector<T, 2>>,
    crate::geometry::vector::Vector<T, 2>: crate::geometry::vector::VectorOps<T, 2>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    fn finalize_cdt(&mut self, constrained: &std::collections::HashSet<Edge>) {
        use ahash::{AHashMap as FastMap, AHashSet as FastSet};
        use smallvec::SmallVec;

        // 1) First pass: dedupe, drop degens/collinear, enforce CCW, drop "long-edge" tris
        let mut uniq: Vec<Triangle> = Vec::with_capacity(self.triangles.len());
        let mut seen: FastSet<(usize, usize, usize)> = FastSet::default();

        for t in self.triangles.drain(..) {
            // range + degenerate
            if t.0 >= self.points.len() || t.1 >= self.points.len() || t.2 >= self.points.len() {
                continue;
            }
            if t.0 == t.1 || t.1 == t.2 || t.0 == t.2 {
                continue;
            }

            // long-edge guard: if any edge passes through another vertex, drop
            if edge_has_mid_vertex(t.0, t.1, &self.points)
                || edge_has_mid_vertex(t.1, t.2, &self.points)
                || edge_has_mid_vertex(t.2, t.0, &self.points)
            {
                continue;
            }

            // dedupe (undirected)
            let mut key = [t.0, t.1, t.2];
            key.sort_unstable();
            let key_t = (key[0], key[1], key[2]);
            if !seen.insert(key_t) {
                continue;
            }

            // orient CCW; drop collinear
            let o = orient2d(&self.points[t.0], &self.points[t.1], &self.points[t.2]);
            if o.is_zero() {
                continue;
            }
            if o.is_negative() {
                uniq.push(Triangle(t.0, t.2, t.1));
            } else {
                uniq.push(t);
            }
        }
        self.triangles = uniq;

        // 2) Edge→tris map on current set
        let mut edge2tris: FastMap<Edge, SmallVec<[usize; 4]>> = FastMap::default();
        for (ti, t) in self.triangles.iter().enumerate() {
            for (u, v) in tri_edges(*t) {
                edge2tris.entry(Edge::new(u, v)).or_default().push(ti);
            }
        }

        // Helpers (use functions for determinism and speed; avoid cross-capturing closures)
        #[inline]
        fn bucket<T: Scalar>(
            e: Edge,
            triangles: &Vec<Triangle>,
            points: &Vec<Point2<T>>,
            edge2tris: &FastMap<Edge, SmallVec<[usize; 4]>>,
        ) -> (
            SmallVec<[usize; 4]>,
            SmallVec<[usize; 4]>,
            SmallVec<[usize; 4]>,
        )
        where
            for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
                + std::ops::Mul<&'a T, Output = T>
                + std::ops::Add<&'a T, Output = T>
                + std::ops::Div<&'a T, Output = T>
                + std::ops::Neg<Output = T>,
        {
            let mut pos = SmallVec::new();
            let mut neg = SmallVec::new();
            let mut zer = SmallVec::new();
            if let Some(list) = edge2tris.get(&e) {
                for &ti in list {
                    if let Some(k) = third_vertex(triangles[ti], e.0, e.1) {
                        let s = orient2d(&points[e.0], &points[e.1], &points[k]);
                        if s.is_positive() {
                            pos.push(ti);
                        } else if s.is_negative() {
                            neg.push(ti);
                        } else {
                            zer.push(ti);
                        }
                    }
                }
            }
            (pos, neg, zer)
        }

        #[inline]
        fn count_constrained_other_edges(
            tri: Triangle,
            e: Edge,
            constrained: &std::collections::HashSet<Edge>,
        ) -> u8 {
            let mut c = 0u8;
            for (u, v) in tri_edges(tri) {
                let ee = Edge::new(u, v);
                if ee != e && constrained.contains(&ee) {
                    c += 1;
                }
            }
            c
        }

        #[inline]
        fn apex_abs<T: Scalar>(e: Edge, tri: Triangle, points: &Vec<Point2<T>>) -> T
        where
            for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
                + std::ops::Mul<&'a T, Output = T>
                + std::ops::Add<&'a T, Output = T>
                + std::ops::Div<&'a T, Output = T>
                + std::ops::Neg<Output = T>,
        {
            let k = third_vertex(tri, e.0, e.1).unwrap();
            let s = orient2d(&points[e.0], &points[e.1], &points[k]);
            if s.is_negative() { -s } else { s }
        }

        #[inline]
        fn better_for_e<T: Scalar>(
            a_ti: usize,
            b_ti: usize,
            e: Edge,
            triangles: &Vec<Triangle>,
            points: &Vec<Point2<T>>,
            constrained: &std::collections::HashSet<Edge>,
        ) -> bool
        where
            for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
                + std::ops::Mul<&'a T, Output = T>
                + std::ops::Add<&'a T, Output = T>
                + std::ops::Div<&'a T, Output = T>
                + std::ops::Neg<Output = T>,
        {
            let ta = triangles[a_ti];
            let tb = triangles[b_ti];
            // 1) more constrained-adjacent edges (prefer faces glued to constraints)
            let ca = count_constrained_other_edges(ta, e, constrained);
            let cb = count_constrained_other_edges(tb, e, constrained);
            if ca != cb {
                return ca > cb;
            }
            // 2) larger distance from edge (more stable)
            let da = apex_abs(e, ta, points);
            let db = apex_abs(e, tb, points);
            match da.partial_cmp(&db) {
                Some(std::cmp::Ordering::Greater) => true,
                Some(std::cmp::Ordering::Less) => false,
                _ => {
                    // 3) deterministic tiebreak: smallest lex apex index
                    let wa = third_vertex(ta, e.0, e.1).unwrap();
                    let wb = third_vertex(tb, e.0, e.1).unwrap();
                    wa < wb
                }
            }
        }

        // 3) First pass: mark triangles selected by constrained edges
        let mut selected_by_constrained = vec![false; self.triangles.len()];
        for &e in constrained {
            let (mut pos, mut neg, _zer) = bucket(e, &self.triangles, &self.points, &edge2tris);

            // POS side: select best
            if !pos.is_empty() {
                let mut best = pos[0];
                for &ti in pos.iter().skip(1) {
                    if better_for_e(ti, best, e, &self.triangles, &self.points, constrained) {
                        best = ti;
                    }
                }
                selected_by_constrained[best] = true;
            }

            // NEG side: select best
            if !neg.is_empty() {
                let mut best = neg[0];
                for &ti in neg.iter().skip(1) {
                    if better_for_e(ti, best, e, &self.triangles, &self.points, constrained) {
                        best = ti;
                    }
                }
                selected_by_constrained[best] = true;
            }
        }

        // 4) Second pass: prune ALL edges (constrained + unconstrained) using priority ranking
        let mut keep = vec![true; self.triangles.len()];
        for (&e, _) in edge2tris.iter() {
            let (mut pos, mut neg, zer) = bucket(e, &self.triangles, &self.points, &edge2tris);

            // Drop zero-area triangles
            for ti in zer {
                keep[ti] = false;
            }

            // POS side: keep best, prefer constrained-selected triangles
            if pos.len() > 1 {
                let mut best = pos[0];
                for &ti in pos.iter().skip(1) {
                    if selected_by_constrained[ti] && !selected_by_constrained[best] {
                        best = ti; // prefer constrained-selected
                    } else if selected_by_constrained[best] && !selected_by_constrained[ti] {
                        // keep current best
                    } else {
                        // both same priority level, use geometric ranking
                        if better_for_e(ti, best, e, &self.triangles, &self.points, constrained) {
                            best = ti;
                        }
                    }
                }
                for &ti in &pos {
                    if ti != best {
                        keep[ti] = false;
                    }
                }
            }

            // NEG side: same logic
            if neg.len() > 1 {
                let mut best = neg[0];
                for &ti in neg.iter().skip(1) {
                    if selected_by_constrained[ti] && !selected_by_constrained[best] {
                        best = ti;
                    } else if selected_by_constrained[best] && !selected_by_constrained[ti] {
                        // keep current best
                    } else {
                        if better_for_e(ti, best, e, &self.triangles, &self.points, constrained) {
                            best = ti;
                        }
                    }
                }
                for &ti in &neg {
                    if ti != best {
                        keep[ti] = false;
                    }
                }
            }
        }

        // 5) Compact
        self.triangles = self
            .triangles
            .drain(..)
            .enumerate()
            .filter_map(|(i, t)| if keep[i] { Some(t) } else { None })
            .collect();
    }

    fn triangulate_constrained_loops(&mut self, constrained: &[Edge], adj: &mut Adj)
    where
        T: PartialOrd,
    {
        // Find simple 4-cycles in the constrained graph and triangulate them
        let mut cn = std::collections::HashMap::new();
        for &Edge(u, v) in constrained {
            cn.entry(u).or_insert_with(Vec::new).push(v);
            cn.entry(v).or_insert_with(Vec::new).push(u);
        }

        // Find all 4-cycles (quads)
        let mut processed_edges = std::collections::HashSet::new();
        let mut processed_quads = std::collections::HashSet::new();
        for &Edge(a, b) in constrained {
            let edge_key = (a.min(b), a.max(b));
            if processed_edges.contains(&edge_key) {
                continue;
            }
            processed_edges.insert(edge_key);

            // Try to complete quad: a-b-c-d-a
            if let (Some(neighbors_b), Some(neighbors_a)) = (cn.get(&b), cn.get(&a)) {
                for &c in neighbors_b {
                    if c == a {
                        continue;
                    }
                    if let Some(neighbors_c) = cn.get(&c) {
                        for &d in neighbors_c {
                            if d == b || d == a {
                                continue;
                            }
                            if neighbors_a.contains(&d) {
                                // Found quad a-b-c-d-a
                                let quad = [a, b, c, d];
                                let key = {
                                    let mut sorted = quad;
                                    sorted.sort();
                                    (sorted[0], sorted[1], sorted[2], sorted[3])
                                };
                                if processed_quads.insert(key) {
                                    self.triangulate_simple_quad(quad, adj);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn triangulate_simple_quad(&mut self, quad: [usize; 4], adj: &mut Adj) {
        let [a, b, c, d] = quad;

        // Add diagonal (a,c) - splits quad into triangles (a,b,c) and (a,c,d)
        if !self.triangle_exists(a, b, c) {
            let s1 = orient2d(&self.points[a], &self.points[b], &self.points[c]);
            if !s1.is_zero() {
                let t1 = if s1.is_positive() {
                    Triangle(a, b, c)
                } else {
                    Triangle(a, c, b)
                };
                let ti = self.triangles.len();
                self.triangles.push(t1);
                adj.add_tri(ti, t1);
            }
        }

        if !self.triangle_exists(a, c, d) {
            let s2 = orient2d(&self.points[a], &self.points[c], &self.points[d]);
            if !s2.is_zero() {
                let t2 = if s2.is_positive() {
                    Triangle(a, c, d)
                } else {
                    Triangle(a, d, c)
                };
                let ti = self.triangles.len();
                self.triangles.push(t2);
                adj.add_tri(ti, t2);
            }
        }
    }

    fn finalize_cdt_unused(&mut self, constrained: &std::collections::HashSet<Edge>) {
        use ahash::{AHashMap as FastMap, AHashSet as FastSet};
        use smallvec::SmallVec;

        // 1) First pass: dedupe, drop degens/collinear, enforce CCW, drop "long-edge" tris
        let mut uniq: Vec<Triangle> = Vec::with_capacity(self.triangles.len());
        let mut seen: FastSet<(usize, usize, usize)> = FastSet::default();

        for t in self.triangles.drain(..) {
            // range + degenerate
            if t.0 >= self.points.len() || t.1 >= self.points.len() || t.2 >= self.points.len() {
                continue;
            }
            if t.0 == t.1 || t.1 == t.2 || t.0 == t.2 {
                continue;
            }

            // long-edge guard: if any edge passes through another vertex, drop
            if edge_has_mid_vertex(t.0, t.1, &self.points)
                || edge_has_mid_vertex(t.1, t.2, &self.points)
                || edge_has_mid_vertex(t.2, t.0, &self.points)
            {
                continue;
            }

            // dedupe (undirected)
            let mut key = [t.0, t.1, t.2];
            key.sort_unstable();
            let key_t = (key[0], key[1], key[2]);
            if !seen.insert(key_t) {
                continue;
            }

            // orient CCW; drop collinear
            let o = orient2d(&self.points[t.0], &self.points[t.1], &self.points[t.2]);
            if o.is_zero() {
                continue;
            }
            if o.is_negative() {
                uniq.push(Triangle(t.0, t.2, t.1));
            } else {
                uniq.push(t);
            }
        }
        self.triangles = uniq;

        // 2) Edge→tris map on current set
        let mut edge2tris: FastMap<Edge, SmallVec<[usize; 4]>> = FastMap::default();
        for (ti, t) in self.triangles.iter().enumerate() {
            for (u, v) in tri_edges(*t) {
                edge2tris.entry(Edge::new(u, v)).or_default().push(ti);
            }
        }

        // Helpers
        let mut keep = vec![true; self.triangles.len()];
        let mut locked = vec![false; self.triangles.len()];

        let bucket = |e: Edge,
                      tris: &Vec<Triangle>,
                      pts: &Vec<Point2<T>>|
         -> (
            SmallVec<[usize; 4]>,
            SmallVec<[usize; 4]>,
            SmallVec<[usize; 4]>,
        ) {
            let mut pos = SmallVec::new();
            let mut neg = SmallVec::new();
            let mut zer = SmallVec::new();
            if let Some(list) = edge2tris.get(&e) {
                for &ti in list {
                    if let Some(k) = third_vertex(tris[ti], e.0, e.1) {
                        let s = orient2d(&pts[e.0], &pts[e.1], &pts[k]);
                        if s.is_positive() {
                            pos.push(ti);
                        } else if s.is_negative() {
                            neg.push(ti);
                        } else {
                            zer.push(ti);
                        }
                    }
                }
            }
            (pos, neg, zer)
        };

        // 3) Classify constrained edges: interior vs boundary (by having tris on both sides)
        let mut interior_constrained: FastSet<Edge> = FastSet::default();
        for &e in constrained {
            let (pos, neg, _zer) = bucket(e, &self.triangles, &self.points);
            if !pos.is_empty() && !neg.is_empty() {
                interior_constrained.insert(e);
            }
        }

        // helpers for ranking triangles adjacent to an edge e
        let shares_interior_constrained_adjacent = |ti: usize, e: Edge| -> bool {
            let t = self.triangles[ti];
            for (u, v) in tri_edges(t) {
                let ee = Edge::new(u, v);
                if ee == e {
                    continue;
                }
                if interior_constrained.contains(&ee)
                    && (u == e.0 || u == e.1 || v == e.0 || v == e.1)
                {
                    return true;
                }
            }
            false
        };

        let apex_distance_metric = |ti: usize, e: Edge| -> T {
            // smaller is better; uses |orient2d(i,j,k)|
            let k = third_vertex(self.triangles[ti], e.0, e.1).unwrap();
            let s = orient2d(&self.points[e.0], &self.points[e.1], &self.points[k]);
            if s.is_negative() { -s } else { s }
        };

        let better_for_e = |a_ti: usize, b_ti: usize, e: Edge| -> bool {
            // true if a_ti is preferred over b_ti for edge e
            let a_pref = shares_interior_constrained_adjacent(a_ti, e);
            let b_pref = shares_interior_constrained_adjacent(b_ti, e);
            if a_pref != b_pref {
                return a_pref;
            }
            let da = apex_distance_metric(a_ti, e);
            let db = apex_distance_metric(b_ti, e);
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Less)
                .is_lt()
        };

        // 3) Lock exactly one keeper per *side* of every constrained edge, using the ranking above
        for &e in constrained {
            let (mut pos, mut neg, zer) = bucket(e, &self.triangles, &self.points);
            for ti in zer {
                keep[ti] = false;
            } // drop collinear on constrained edges too

            // pick best on each side (if any)
            let pick_best = |list: &mut smallvec::SmallVec<[usize; 4]>| -> Option<usize> {
                if list.is_empty() {
                    return None;
                }
                let mut best = list[0];
                for &ti in list.iter().skip(1) {
                    if better_for_e(ti, best, e) {
                        best = ti;
                    }
                }
                Some(best)
            };

            if let Some(best_pos) = pick_best(&mut pos) {
                keep[best_pos] = true;
                locked[best_pos] = true;
            }
            if let Some(best_neg) = pick_best(&mut neg) {
                keep[best_neg] = true;
                locked[best_neg] = true;
            }
        }

        // 4) For every edge: enforce ≤1 per side, honoring locks; use the same ranking if needed
        for &e in constrained {
            let (mut pos, mut neg, zer) = bucket(e, &self.triangles, &self.points);
            for ti in zer {
                keep[ti] = false;
            }

            // POS side
            let mut kept_any = pos.iter().any(|&ti| locked[ti]);
            if !kept_any && !pos.is_empty() {
                // choose best by ranking
                let mut best = pos[0];
                for &ti in pos.iter().skip(1) {
                    if better_for_e(ti, best, e) {
                        best = ti;
                    }
                }
                // drop all except best
                for &ti in &pos {
                    if ti != best {
                        keep[ti] = false;
                    }
                }
                kept_any = true;
            } else if kept_any {
                // we already have a locked one; drop the rest
                for &ti in &pos {
                    if !locked[ti] {
                        keep[ti] = false;
                    }
                }
            }

            // NEG side
            let mut kept_any = neg.iter().any(|&ti| locked[ti]);
            if !kept_any && !neg.is_empty() {
                let mut best = neg[0];
                for &ti in neg.iter().skip(1) {
                    if better_for_e(ti, best, e) {
                        best = ti;
                    }
                }
                for &ti in &neg {
                    if ti != best {
                        keep[ti] = false;
                    }
                }
            } else if kept_any {
                for &ti in &neg {
                    if !locked[ti] {
                        keep[ti] = false;
                    }
                }
            }
        }

        // compact at the end (unchanged)
        self.triangles = self
            .triangles
            .drain(..)
            .enumerate()
            .filter_map(|(i, t)| if keep[i] { Some(t) } else { None })
            .collect();

        // compact_one_per_side(&self.points, &mut self.triangles);
    }

    /// Build Delaunay triangulation of `pts`. Duplicates are ignored.
    pub fn build(pts: &[Point2<T>]) -> Self {
        let mut points = pts.to_vec();
        if points.len() < 3 {
            return Self {
                points,
                triangles: Vec::new(),
            };
        }

        // Create super-triangle that contains all points
        let (minx, miny, maxx, maxy) = bbox(&points);
        let dx = &maxx - &minx;
        let dy = &maxy - &miny;
        let delta = dx.max(dy);
        let cx = &(minx + maxx) * &T::from_num_den(1, 2);
        let cy = &(miny + maxy) * &T::from_num_den(1, 2);

        let r = T::from(64) * delta + T::one();
        let sqrt_3 = T::from(SQRT_3);
        let p_super0 =
            Point2::<T>::from_vals([T::from(cx.clone()), T::from(&(&cy + &T::from(2.0)) * &r)]);
        let p_super1 = Point2::<T>::from_vals([T::from(&cx - &(&sqrt_3 * &r)), T::from(&cy - &r)]);
        let p_super2 = Point2::<T>::from_vals([T::from(&cx + &(&sqrt_3 * &r)), T::from(&cy - &r)]);

        let s0 = points.len();
        let s1 = s0 + 1;
        let s2 = s0 + 2;

        points.push(p_super0);
        points.push(p_super1);
        points.push(p_super2);

        // Initialize with super-triangle
        let mut triangles = vec![Triangle(s0, s1, s2)];

        // Insert each point using Bowyer-Watson
        for pid in 0..s0 {
            Self::bowyer_watson_insert_point(pid, &points, &mut triangles);
        }

        // Remove super-triangles
        triangles.retain(|t| t.0 < s0 && t.1 < s0 && t.2 < s0);
        points.truncate(s0);

        Self { points, triangles }
    }

    /// Insert a single point using Bowyer-Watson algorithm
    fn bowyer_watson_insert_point(pid: usize, points: &[Point2<T>], triangles: &mut Vec<Triangle>) {
        let p = &points[pid];

        // Find triangles whose circumcircle contains p
        let mut bad_triangles = Vec::new();
        for (i, &t) in triangles.iter().enumerate() {
            if Self::point_in_circumcircle(p, t, points) {
                bad_triangles.push(i);
            }
        }

        if bad_triangles.is_empty() {
            return; // Point already well-positioned
        }

        // Find cavity boundary
        let mut edge_count = std::collections::HashMap::new();
        for &i in &bad_triangles {
            let t = triangles[i];
            for edge in [
                Edge::new(t.0, t.1),
                Edge::new(t.1, t.2),
                Edge::new(t.2, t.0),
            ] {
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        // Boundary edges appear exactly once
        let boundary_edges: Vec<Edge> = edge_count
            .into_iter()
            .filter_map(|(edge, count)| if count == 1 { Some(edge) } else { None })
            .collect();

        // Remove bad triangles (in reverse order to preserve indices)
        bad_triangles.sort_unstable();
        for &i in bad_triangles.iter().rev() {
            triangles.swap_remove(i);
        }

        // Create new triangles by connecting boundary edges to new point
        for edge in boundary_edges {
            let new_triangle = if orient2d(&points[edge.0], &points[edge.1], p).is_positive() {
                Triangle(edge.0, edge.1, pid)
            } else {
                Triangle(edge.0, pid, edge.1)
            };
            triangles.push(new_triangle);
        }
    }

    /// Test if point is inside circumcircle of triangle
    fn point_in_circumcircle(p: &Point2<T>, t: Triangle, points: &[Point2<T>]) -> bool {
        let (a, b, c) = (t.0, t.1, t.2);

        // Ensure CCW orientation for consistent incircle test
        let (aa, bb, cc) = if orient2d(&points[a], &points[b], &points[c]).is_positive() {
            (a, b, c)
        } else {
            (a, c, b)
        };

        incircle(&points[aa], &points[bb], &points[cc], p).is_positive()
    }

    /// Complete Bowyer-Watson with constraint handling
    pub fn build_with_constraints_bowyer_watson<const N: usize>(
        pts: &[Point2<T>],
        constraints_in: &[[usize; 2]],
        _mesh: &Mesh<T, N>,
        _job: &FaceJobUV<T>,
    ) -> Self {
        // 1) Build unconstrained Delaunay
        let mut dt = Self::build(pts);

        // 2) Process constraints
        let mut constraints: Vec<Edge> = Vec::new();
        let mut seen = HashSet::new();

        for &[a, b] in constraints_in {
            if a >= dt.points.len() || b >= dt.points.len() || a == b {
                continue;
            }

            // Split constraint at collinear vertices
            for ab in split_constraint_chain(&dt.points, a, b) {
                let e = Edge::new(ab[0], ab[1]);
                if seen.insert(e) {
                    constraints.push(e);
                }
            }
        }

        let mut constrained = HashSet::<Edge>::new();
        let mut adj = Adj::rebuild(dt.points.len(), &dt.triangles);

        // 3) Insert constraints
        for e in &constraints {
            if dt.edge_exists(e.0, e.1) {
                constrained.insert(*e);
                continue;
            }
            match dt.insert_constraint_walk(e.0, e.1, &mut adj, &constrained) {
                Ok(()) => {
                    constrained.insert(*e);
                }
                Err(err) => {
                    eprintln!("constraint {:?} failed: {}", e, err);
                }
            }
        }

        // 4) Complete constrained triangulation
        // dt.complete_bowyer_watson_cdt(&constrained, &mut adj);

        dt
    }

    /// Complete the CDT using Bowyer-Watson principles
    fn complete_bowyer_watson_cdt(&mut self, constrained: &HashSet<Edge>, adj: &mut Adj)
    where
        T: PartialOrd,
    {
        // 1) Insert interior vertices using Bowyer-Watson
        self.bowyer_watson_insert_interior_vertices(constrained, adj);

        // 2) Triangulate constrained quads
        self.triangulate_constrained_quads(constrained, adj);

        // 3) Final cleanup
        self.finalize_bowyer_watson_cdt(constrained);
    }

    /// Insert interior vertices (not on constraints) using Bowyer-Watson
    fn bowyer_watson_insert_interior_vertices(
        &mut self,
        constrained: &HashSet<Edge>,
        adj: &mut Adj,
    ) {
        // Find vertices not on any constraint
        let mut constrained_verts = AHashSet::default();
        for &Edge(u, v) in constrained {
            constrained_verts.insert(u);
            constrained_verts.insert(v);
        }

        let interior_verts: Vec<usize> = (0..self.points.len())
            .filter(|&v| !constrained_verts.contains(&v))
            .collect();

        // Insert each interior vertex
        for &vid in &interior_verts {
            self.bowyer_watson_insert_vertex_constrained(vid, constrained, adj);
        }
    }

    /// Insert a vertex using Bowyer-Watson while respecting constraints
    fn bowyer_watson_insert_vertex_constrained(
        &mut self,
        vid: usize,
        constrained: &HashSet<Edge>,
        adj: &mut Adj,
    ) {
        let p = self.points[vid].clone();

        // Find bad triangles (circumcircle contains p, not already containing vid)
        let mut bad_triangles = Vec::new();
        for (ti, &t) in self.triangles.iter().enumerate() {
            if has_vertex(t, vid) {
                continue;
            }
            if Self::point_in_circumcircle(&p, t, &self.points) {
                bad_triangles.push(ti);
            }
        }

        if bad_triangles.is_empty() {
            return;
        }

        // Find cavity boundary (respecting constraints)
        let mut edge_count = AHashMap::default();
        for &ti in &bad_triangles {
            let t = self.triangles[ti];
            for edge in [
                Edge::new(t.0, t.1),
                Edge::new(t.1, t.2),
                Edge::new(t.2, t.0),
            ] {
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        // Boundary edges appear once AND are not constrained
        let boundary_edges: Vec<Edge> = edge_count
            .into_iter()
            .filter_map(|(edge, count)| {
                if count == 1 && !constrained.contains(&edge) {
                    Some(edge)
                } else {
                    None
                }
            })
            .collect();

        if boundary_edges.is_empty() {
            return; // Cannot create cavity due to constraints
        }

        // Remove bad triangles
        bad_triangles.sort_unstable();
        for &ti in bad_triangles.iter().rev() {
            let old_t = self.triangles[ti];
            adj.remove_tri(ti, old_t);
            self.triangles.swap_remove(ti);
        }

        // Update adjacency after removals
        self.rebuild_adjacency_after_removals(&bad_triangles, adj);

        // Create new triangles
        for edge in boundary_edges {
            // Skip if would create mid-edge vertex
            if edge_has_mid_vertex(edge.0, vid, &self.points)
                || edge_has_mid_vertex(edge.1, vid, &self.points)
            {
                continue;
            }

            let new_triangle =
                if orient2d(&self.points[edge.0], &self.points[edge.1], &p).is_positive() {
                    Triangle(edge.0, edge.1, vid)
                } else {
                    Triangle(edge.0, vid, edge.1)
                };

            let ti = self.triangles.len();
            self.triangles.push(new_triangle);
            adj.add_tri(ti, new_triangle);
        }
    }

    /// Rebuild adjacency after triangle removals
    fn rebuild_adjacency_after_removals(&mut self, removed_indices: &[usize], adj: &mut Adj) {
        // After swap_remove, triangle indices have shifted
        for &removed_ti in removed_indices {
            if removed_ti < self.triangles.len() {
                let moved_triangle = self.triangles[removed_ti];
                adj.add_tri(removed_ti, moved_triangle);
            }
        }
    }

    /// Triangulate constrained quads deterministically
    fn triangulate_constrained_quads(&mut self, constrained: &HashSet<Edge>, adj: &mut Adj) {
        // Build constrained adjacency graph
        let mut constrained_adj = AHashMap::<usize, AHashSet<usize>>::default();
        for &Edge(u, v) in constrained {
            constrained_adj.entry(u).or_default().insert(v);
            constrained_adj.entry(v).or_default().insert(u);
        }

        // Find 4-cycles (quads)
        let mut processed_quads = AHashSet::default();

        for &Edge(a, b) in constrained {
            if let Some(neighbors_b) = constrained_adj.get(&b) {
                for &c in neighbors_b {
                    if c == a {
                        continue;
                    }

                    if let Some(neighbors_c) = constrained_adj.get(&c) {
                        for &d in neighbors_c {
                            if d == b || d == a {
                                continue;
                            }

                            // Check if d connects back to a (completing quad)
                            if constrained_adj.get(&d).map_or(false, |ns| ns.contains(&a)) {
                                // Found quad a-b-c-d-a
                                let mut quad = [a, b, c, d];
                                quad.sort_unstable();
                                let quad_key = (quad[0], quad[1], quad[2], quad[3]);

                                if processed_quads.insert(quad_key) {
                                    self.triangulate_quad_with_diagonal([a, b, c, d], adj);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Triangulate a quad by adding diagonal
    fn triangulate_quad_with_diagonal(&mut self, quad: [usize; 4], adj: &mut Adj) {
        let [a, b, c, d] = quad;

        // Add diagonal (a,c) creating triangles (a,b,c) and (a,c,d)
        if !self.triangle_exists(a, b, c) {
            let s1 = orient2d(&self.points[a], &self.points[b], &self.points[c]);
            if !s1.is_zero() {
                let t1 = if s1.is_positive() {
                    Triangle(a, b, c)
                } else {
                    Triangle(a, c, b)
                };

                // Check for mid-edge vertices
                if !self.triangle_has_mid_edge_vertex(t1) {
                    let ti = self.triangles.len();
                    self.triangles.push(t1);
                    adj.add_tri(ti, t1);
                }
            }
        }

        if !self.triangle_exists(a, c, d) {
            let s2 = orient2d(&self.points[a], &self.points[c], &self.points[d]);
            if !s2.is_zero() {
                let t2 = if s2.is_positive() {
                    Triangle(a, c, d)
                } else {
                    Triangle(a, d, c)
                };

                // Check for mid-edge vertices
                if !self.triangle_has_mid_edge_vertex(t2) {
                    let ti = self.triangles.len();
                    self.triangles.push(t2);
                    adj.add_tri(ti, t2);
                }
            }
        }
    }

    /// Check if triangle has any edge passing through another vertex
    fn triangle_has_mid_edge_vertex(&self, t: Triangle) -> bool {
        edge_has_mid_vertex(t.0, t.1, &self.points)
            || edge_has_mid_vertex(t.1, t.2, &self.points)
            || edge_has_mid_vertex(t.2, t.0, &self.points)
    }

    /// Final cleanup for Bowyer-Watson CDT
    fn finalize_bowyer_watson_cdt(&mut self, constrained: &HashSet<Edge>) {
        // Remove degenerate and duplicate triangles
        let mut valid_triangles = Vec::new();
        let mut seen = AHashSet::default();

        for &t in &self.triangles {
            // Skip degenerate
            let area = orient2d(&self.points[t.0], &self.points[t.1], &self.points[t.2]);
            if area.is_zero() {
                continue;
            }

            // Skip if has mid-edge vertex
            if self.triangle_has_mid_edge_vertex(t) {
                continue;
            }

            // Deduplicate
            let mut key = [t.0, t.1, t.2];
            key.sort_unstable();
            if !seen.insert((key[0], key[1], key[2])) {
                continue;
            }

            // Ensure CCW orientation
            if area.is_positive() {
                valid_triangles.push(t);
            } else {
                valid_triangles.push(Triangle(t.0, t.2, t.1));
            }
        }

        self.triangles = valid_triangles;

        // Enforce one triangle per side for constrained edges
        self.enforce_one_per_side_constrained(constrained);
    }

    /// Enforce exactly one triangle per side of constrained edges
    fn enforce_one_per_side_constrained(&mut self, constrained: &HashSet<Edge>) {
        use smallvec::SmallVec;

        let mut keep = vec![true; self.triangles.len()];

        // Only process constrained edges
        for &edge in constrained {
            let mut edge_triangles = SmallVec::<[usize; 4]>::new();

            // Find triangles that contain this specific constrained edge
            for (ti, &t) in self.triangles.iter().enumerate() {
                for (u, v) in tri_edges(t) {
                    if Edge::new(u, v) == edge {
                        edge_triangles.push(ti);
                        break;
                    }
                }
            }

            if edge_triangles.len() <= 2 {
                continue; // No conflict on this edge
            }

            // Separate by side
            let (mut pos_tris, mut neg_tris) = (SmallVec::new(), SmallVec::new());
            for &ti in &edge_triangles {
                if let Some(apex) = third_vertex(self.triangles[ti], edge.0, edge.1) {
                    let s = orient2d(
                        &self.points[edge.0],
                        &self.points[edge.1],
                        &self.points[apex],
                    );
                    if s.is_positive() {
                        pos_tris.push(ti);
                    } else if s.is_negative() {
                        neg_tris.push(ti);
                    } else {
                        keep[ti] = false; // collinear
                    }
                }
            }

            // Keep best on each side
            self.keep_best_triangles_on_side(&mut pos_tris, edge, &mut keep);
            self.keep_best_triangles_on_side(&mut neg_tris, edge, &mut keep);
        }

        // Compact
        self.triangles = self
            .triangles
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| if keep[i] { Some(t) } else { None })
            .collect();
    }

    /// Keep the best triangle on one side of an edge
    fn keep_best_triangles_on_side(
        &self,
        tris: &mut SmallVec<[usize; 4]>,
        edge: Edge,
        keep: &mut [bool],
    ) {
        if tris.len() <= 1 {
            return;
        }

        // Find triangle with maximum distance from edge
        let mut best_ti = tris[0];
        let mut best_dist = T::zero();

        for &ti in tris.iter() {
            if let Some(apex) = third_vertex(self.triangles[ti], edge.0, edge.1) {
                let dist = orient2d(
                    &self.points[edge.0],
                    &self.points[edge.1],
                    &self.points[apex],
                )
                .abs();
                if dist > best_dist {
                    best_dist = dist;
                    best_ti = ti;
                }
            }
        }

        // Mark all others for removal
        for &ti in tris.iter() {
            if ti != best_ti {
                keep[ti] = false;
            }
        }
    }

    fn edge_exists(&self, a: usize, b: usize) -> bool
    where
        T: PartialOrd,
    {
        let e = Edge::new(a, b);

        // edge must be present in some triangle…
        let mut present = false;
        for t in &self.triangles {
            let tri = [t.0, t.1, t.2];
            if Edge::new(tri[0], tri[1]) == e
                || Edge::new(tri[1], tri[2]) == e
                || Edge::new(tri[2], tri[0]) == e
            {
                present = true;
                break;
            }
        }
        if !present {
            return false;
        }

        true
    }

    pub fn build_with_constraints_walk<const N: usize>(
        pts: &[Point2<T>],
        constraints_in: &[[usize; 2]],
        mesh: &Mesh<T, N>,
        job: &FaceJobUV<T>,
    ) -> Self {
        let mut dt = Self::build(pts); // now indices match exactly

        // Remap, filter, and split constraints *in dt-space*
        let mut constraints: Vec<Edge> = Vec::new();
        let mut seen = HashSet::new();

        for &[a, b] in constraints_in {
            if a >= dt.points.len() || b >= dt.points.len() || a == b {
                continue;
            }

            // Split long edge at collinear interior vertices
            for ab in split_constraint_chain(&dt.points, a, b) {
                let e = Edge::new(ab[0], ab[1]);
                if seen.insert(e) {
                    constraints.push(e);
                }
            }
        }

        let mut constrained = HashSet::<Edge>::new();
        let mut adj = Adj::rebuild(dt.points.len(), &dt.triangles);

        for e in &constraints {
            if dt.edge_exists(e.0, e.1) {
                constrained.insert(*e);
                continue;
            }
            match dt.insert_constraint_walk(e.0, e.1, &mut adj, &constrained) {
                Ok(()) => {
                    constrained.insert(*e);
                }
                Err(err) => {
                    eprintln!("constraint {:?} failed: {}", e, err);
                    println!("pts: {:?}", pts);
                    println!("dt:  {:?}", dt);
                }
            }
        }

        // dt.triangulate_constrained_loops(&constraints, &mut adj);
        // dt.complete_constrained_edges(&constraints, &mut adj);
        // dt.complete_constrained_cycles(&constrained, &mut adj);

        println!("TEST 1");
        // println!("{:?}", constrained);
        print_face_job_and_dt(&mesh, &job, &dt);
        // dt.legalize_all_with_adj(&mut adj, &constrained);
        // println!("TEST 2");
        // print_face_job_and_dt(&mesh, &job, &dt);
        // dt.finalize_cdt(&constrained);
        // println!("TEST 3");
        // print_face_job_and_dt(&mesh, &job, &dt);
        // println!("TEST 4");
        // print_face_job_and_dt(&mesh, &job, &dt);
        // dt.stitch_single_triangle_gaps(&mut adj, &constrained);
        // println!("TEST 5");
        // print_face_job_and_dt(&mesh, &job, &dt);
        dt
    }

    /// Triangulate each simple loop made entirely of constrained edges (handles quads and general n-gons).
    /// Adds unconstrained diagonals as needed. Deterministic and local (ear clipping).

    #[inline]
    fn complete_constrained_edges(&mut self, edges: &[Edge], adj: &mut Adj)
    where
        T: PartialOrd,
    {
        for &e in edges {
            // Only try to complete for edges that exist
            if self.edge_exists(e.0, e.1) {
                // No recorded apexes in this pass; we’ll pick a best common neighbor deterministically
                self.add_missing_triangle_for_edge(e.0, e.1, None, None, adj);
            }
        }
    }

    fn complete_constrained_cycles(
        &mut self,
        constrained: &std::collections::HashSet<Edge>,
        adj: &mut Adj,
    ) where
        T: PartialOrd,
    {
        use ahash::{AHashMap as FastMap, AHashSet as FastSet};

        // Build constrained neighbor lists
        let mut cn: FastMap<usize, FastSet<usize>> = FastMap::default();
        cn.reserve(self.points.len());
        for &Edge(u, v) in constrained {
            cn.entry(u).or_default().insert(v);
            cn.entry(v).or_default().insert(u);
        }

        // For each constrained edge (u,v), find common constrained neighbors w
        // and add Triangle(u,v,w) if missing.
        for &Edge(u, v) in constrained {
            let Some(nu) = cn.get(&u) else {
                continue;
            };
            let Some(nv) = cn.get(&v) else {
                continue;
            };

            // Iterate intersection without allocations
            for &w in nu {
                if w == v || !nv.contains(&w) {
                    continue;
                }
                if self.triangle_exists(u, v, w) {
                    continue;
                }

                // Non-degenerate and CCW
                let s = orient2d(&self.points[u], &self.points[v], &self.points[w]);
                if s.is_zero() {
                    continue;
                }
                let t_new = if s.is_positive() {
                    Triangle(u, v, w)
                } else {
                    Triangle(u, w, v)
                };

                let ti = self.triangles.len();
                self.triangles.push(t_new);
                adj.add_tri(ti, t_new);
            }
        }
    }

    fn triangle_exists(&self, a: usize, b: usize, c: usize) -> bool {
        let mut k = [a, b, c];
        k.sort_unstable();
        self.triangles.iter().any(|t| {
            let mut s = [t.0, t.1, t.2];
            s.sort_unstable();
            s == k
        })
    }

    fn neighbors_of(&self, v: usize) -> AHashSet<usize> {
        let mut set = AHashSet::default();
        for t in &self.triangles {
            if t.0 == v {
                set.insert(t.1);
                set.insert(t.2);
            }
            if t.1 == v {
                set.insert(t.0);
                set.insert(t.2);
            }
            if t.2 == v {
                set.insert(t.0);
                set.insert(t.1);
            }
        }
        set.remove(&v);
        set
    }

    fn stitch_single_triangle_gaps(&mut self, adj: &mut Adj, constrained: &HashSet<Edge>) {
        // collect targets first; we’ll mutate self.triangles afterwards
        let mut to_add: Vec<(usize, usize, usize)> = Vec::new();

        for &e in constrained {
            // skip if already present or geometrically invalid
            if self.edge_exists(e.0, e.1) {
                continue;
            }
            if edge_has_mid_vertex(e.0, e.1, &self.points) {
                continue;
            }

            let nu = self.neighbors_of(e.0);
            let nv = self.neighbors_of(e.1);

            // find a common neighbor; in “one triangle hole” cases there will be one (or two)
            for &w in nu.intersection(&nv) {
                if w == e.0 || w == e.1 {
                    continue;
                }
                if self.triangle_exists(e.0, e.1, w) {
                    continue;
                }

                // make sure triangle is non-degenerate and oriented
                let o = orient2d(&self.points[e.0], &self.points[e.1], &self.points[w]);
                if o.is_zero() {
                    continue;
                }

                let (a, b, c) = if o.is_positive() {
                    (e.0, e.1, w)
                } else {
                    (e.0, w, e.1)
                };
                to_add.push((a, b, c));
                break; // one is enough per missing edge
            }
        }

        // apply
        for (a, b, c) in to_add {
            let t = Triangle(a, b, c);
            let ti = self.triangles.len();
            self.triangles.push(t);
            adj.add_tri(ti, t);
        }
    }

    fn insert_constraint_walk(
        &mut self,
        mut a: usize,
        mut b: usize,
        adj: &mut Adj,
        constrained: &HashSet<Edge>,
    ) -> Result<(), &'static str> {
        if self.edge_exists(a, b) {
            return Ok(());
        }

        // Choose a starting triangle that actually faces toward the segment (a,b).
        // Prefer a’s incident triangles; if none useful, swap endpoints and try b.
        let pick_start = |end: usize, other: usize| -> Option<usize> {
            if let Some(inc) = adj.vert2tris.get(end) {
                for &ti in inc.iter() {
                    if exit_edge_of_triangle(&self.points, adj, &self.triangles, ti, end, other)
                        .is_some()
                    {
                        return Some(ti);
                    }
                }
                if let Some(&ti) = inc.get(0) {
                    return Some(ti);
                }
            }
            None
        };
        let mut current_tri = if let Some(ti) = pick_start(a, b) {
            ti
        } else if let Some(ti) = pick_start(b, a) {
            std::mem::swap(&mut a, &mut b);
            ti
        } else {
            return Err("no incident triangle for either endpoint");
        };

        // Track the best (latest) apex seen on each side of (a,b).
        let mut last_pos_apex: Option<usize> = None;
        let mut last_neg_apex: Option<usize> = None;

        // Visit-state: avoid cycles
        let mut visited = ahash::AHashSet::<(usize, Edge)>::default();
        let mut steps = 0usize;
        let max_steps = 4 * self.triangles.len().max(10);

        while !self.edge_exists(a, b) {
            steps += 1;
            if steps > max_steps {
                return Err("insert_constraint_walk: too many steps");
            }

            // Decide the edge we must cross from current_tri toward segment (a,b)
            let (cross_e, nei) = match exit_edge_of_triangle(
                &self.points,
                adj,
                &self.triangles,
                current_tri,
                a,
                b,
            ) {
                Some(x) => x,
                None => return Err("walker stuck: no exit edge"),
            };

            // Record apex on the side we are leaving across cross_e
            if let Some(w) = third_vertex(self.triangles[current_tri], cross_e.0, cross_e.1) {
                let s = orient2d(&self.points[a], &self.points[b], &self.points[w]);
                if s.is_positive() {
                    last_pos_apex = Some(w);
                } else if s.is_negative() {
                    last_neg_apex = Some(w);
                }
            }

            if !visited.insert((current_tri, cross_e)) {
                return Err("walker cycle detected");
            }

            // If we already have the target edge, stop walking; we’ll add missing side(s) below.
            if (cross_e.0 == a && cross_e.1 == b) || (cross_e.0 == b && cross_e.1 == a) {
                break;
            }

            // Interior vs boundary
            match adj.edge2tris.get(&cross_e).map(|v| v.len()).unwrap_or(0) {
                2 => {
                    // Interior: try to flip if allowed, otherwise step across.
                    if !constrained.contains(&cross_e) {
                        if let Some(new_e) = self.flip_shared_edge_with_adj(
                            cross_e,
                            adj.edge2tris[&cross_e][0],
                            adj.edge2tris[&cross_e][1],
                            constrained,
                            adj,
                        ) {
                            // If this flip created (a,b), stop walking.
                            if (new_e.0 == a && new_e.1 == b) || (new_e.0 == b && new_e.1 == a) {
                                break;
                            }
                            // Re-evaluate from the same triangle index for stability.
                            continue;
                        }
                    }
                    // No flip (or constrained): step to neighbor if available.
                    if let Some(next_ti) = nei {
                        current_tri = next_ti;
                    } else {
                        return Err("no neighbor across interior edge");
                    }
                }
                _ => {
                    // Boundary: split triangles to introduce progress for (a,b), then continue walking.
                    self.handle_boundary_crossing(current_tri, cross_e, a, b, adj)?;
                    visited.clear(); // topology changed; clear cycle state
                }
            }
        }

        // At this point, (a,b) should exist. If it has only one incident triangle, add the missing one
        // deterministically using the last recorded apex on the opposite side; fall back to best common neighbor.
        self.add_missing_triangle_for_edge(a, b, last_pos_apex, last_neg_apex, adj);

        Ok(())
    }

    /// After (a,b) exists, if it has only one incident triangle, add the missing one.
    /// Prefer the recorded apexes from the walk; otherwise choose the best common neighbor on the opposite side.
    fn add_missing_triangle_for_edge(
        &mut self,
        a: usize,
        b: usize,
        last_pos_apex: Option<usize>,
        last_neg_apex: Option<usize>,
        adj: &mut Adj,
    ) where
        T: PartialOrd,
    {
        let e = Edge::new(a, b);
        let list = match adj.edge2tris.get(&e) {
            Some(v) => v.clone(),
            None => return,
        };
        if list.len() >= 2 {
            return; // already has both sides
        }
        if list.is_empty() {
            return; // unexpected; nothing to do
        }

        // Existing side sign
        let ti_exist = list[0];
        if ti_exist >= self.triangles.len() {
            return;
        }
        let w_exist = match third_vertex(self.triangles[ti_exist], a, b) {
            Some(w) => w,
            None => return,
        };
        let s_exist = orient2d(&self.points[a], &self.points[b], &self.points[w_exist]);
        if s_exist.is_zero() {
            return;
        }

        // Target side is the opposite one
        let want_pos = s_exist.is_negative();

        // 1) Prefer recorded apex from the walk on the wanted side
        let mut candidate = if want_pos {
            last_pos_apex
        } else {
            last_neg_apex
        };

        // 2) Fallback: search common neighbors of a and b on the wanted side, pick max |area|
        if candidate.is_none() {
            let na = self.neighbors_of(a);
            let nb = self.neighbors_of(b);
            let mut best_w: Option<usize> = None;
            let mut best_score: Option<T> = None;
            for &w in na.intersection(&nb) {
                if w == a || w == b || w == w_exist {
                    continue;
                }
                if self.triangle_exists(a, b, w) {
                    continue;
                }
                // avoid creating long edges
                if edge_has_mid_vertex(a, w, &self.points)
                    || edge_has_mid_vertex(b, w, &self.points)
                {
                    continue;
                }
                let s = orient2d(&self.points[a], &self.points[b], &self.points[w]);
                if s.is_zero() || (want_pos != s.is_positive()) {
                    continue;
                }
                let score = if s.is_negative() { -s } else { s }; // |area|
                if best_score.as_ref().map_or(true, |bs| score > *bs) {
                    best_score = Some(score);
                    best_w = Some(w);
                }
            }
            candidate = best_w;
        }

        let Some(w) = candidate else {
            return;
        };

        // Create triangle (a,b,w) with CCW orientation
        let t_new = if orient2d(&self.points[a], &self.points[b], &self.points[w]).is_positive() {
            Triangle(a, b, w)
        } else {
            Triangle(a, w, b)
        };

        let ti = self.triangles.len();
        self.triangles.push(t_new);
        adj.add_tri(ti, t_new);
    }

    fn find_crossing_edge(&self, tri_idx: usize, a: usize, b: usize) -> Result<Edge, &'static str> {
        let tri = self.triangles[tri_idx];
        let pa = &self.points[a];
        let pb = &self.points[b];

        // Test each edge of the triangle
        for (u, v) in [(tri.0, tri.1), (tri.1, tri.2), (tri.2, tri.0)] {
            let pu = &self.points[u];
            let pv = &self.points[v];

            // Check if segment (a,b) crosses edge (u,v)
            let o1 = orient2d(pa, pb, pu);
            let o2 = orient2d(pa, pb, pv);
            let o3 = orient2d(pu, pv, pa);
            let o4 = orient2d(pu, pv, pb);

            // Proper intersection or touching
            let crosses = (&o1 * &o2).is_negative() && (&o3 * &o4).is_negative();
            let touches = (o1.is_zero() && (&o3 * &o4).is_negative())
                || (o2.is_zero() && (&o3 * &o4).is_negative())
                || (o3.is_zero() && (&o1 * &o2).is_negative())
                || (o4.is_zero() && (&o1 * &o2).is_negative());

            if crosses || touches {
                return Ok(Edge::new(u, v));
            }
        }

        Err("no crossing edge found")
    }

    fn handle_boundary_crossing(
        &mut self,
        tri_idx: usize,
        boundary_edge: Edge,
        a: usize,
        b: usize,
        adj: &mut Adj,
    ) -> Result<(), &'static str> {
        // We are crossing a boundary edge (u,v) from triangle tri_idx. Replace it with TWO triangles
        // that introduce the constrained edge between the endpoint on the boundary (anchor) and the other endpoint.
        let tri = self.triangles[tri_idx];
        let (u, v) = (boundary_edge.0, boundary_edge.1);
        let w = third_vertex(tri, u, v).ok_or("invalid boundary edge")?;

        // Determine which endpoint of (a,b) lies on this boundary edge; that must be the anchor.
        let anchor = if u == a || v == a {
            a
        } else if u == b || v == b {
            b
        } else {
            return Err("hit unrelated boundary while inserting constraint");
        };
        let other = if anchor == a { b } else { a };

        // First triangle introduces (anchor, other) with w
        let t0 =
            if orient2d(&self.points[anchor], &self.points[other], &self.points[w]).is_positive() {
                Triangle(anchor, other, w)
            } else {
                Triangle(anchor, w, other)
            };

        // Second triangle closes the fan on the boundary edge using the vertex of (u,v) not equal to anchor
        let x = if u == anchor { v } else { u };
        let t1 = if orient2d(&self.points[other], &self.points[x], &self.points[w]).is_positive() {
            Triangle(other, x, w)
        } else {
            Triangle(other, w, x)
        };

        // Commit both triangles
        let old = self.triangles[tri_idx];
        self.triangles[tri_idx] = t0;
        let t1_idx = self.triangles.len();
        self.triangles.push(t1);

        adj.replace_tri(tri_idx, old, t0);
        adj.add_tri(t1_idx, t1);

        Ok(())
    }

    fn insert_constraint_walk_2(
        &mut self,
        mut a: usize,
        mut b: usize,
        adj: &mut Adj,
        constrained: &HashSet<Edge>,
    ) -> Result<(), &'static str> {
        // Already present?
        if self.edge_exists(a, b) {
            return Ok(());
        }

        // Pick a starting triangle from a (or b, swapping endpoints)
        let mut start_ti = adj.vert2tris.get(a).and_then(|v| v.get(0)).copied();
        if start_ti.is_none() {
            if let Some(&ti_b) = adj.vert2tris.get(b).and_then(|v| v.get(0)) {
                std::mem::swap(&mut a, &mut b);
                start_ti = Some(ti_b);
            }
        }
        let Some(mut ti) = start_ti else {
            return Err("no incident triangle for either endpoint");
        };

        // Anti-backtrack state
        let mut last_ti: Option<usize> = None;
        let mut last_edge: Option<Edge> = None;

        // Cheap cycle detector
        use ahash::AHashSet as FastSet;
        let mut visited: FastSet<(usize, Edge)> = FastSet::default();

        // Proportional cap
        let mut steps = 0usize;
        let step_cap = 4 * self.triangles.len().max(1);

        while !self.edge_exists(a, b) {
            steps += 1;
            if steps > step_cap {
                return Err("insert_constraint_walk: step cap");
            }

            // Decide exit edge from current triangle toward (a,b)
            let (mut e_cross, mut nei) =
                match exit_edge_of_triangle(&self.points, adj, &self.triangles, ti, a, b) {
                    Some(x) => x,
                    None => return Err("walker stuck: no exit edge"),
                };

            // Anti-backtrack: if we would go straight back over the same edge into the previous triangle,
            // switch to the *other* edge incident to `a` in the current triangle.
            if let (Some(prev_ti), Some(prev_e)) = (last_ti, last_edge) {
                if nei == Some(prev_ti) && e_cross == prev_e {
                    if has_vertex(self.triangles[ti], a) {
                        let t = self.triangles[ti];
                        for (u, v) in tri_edges(t) {
                            let e2 = Edge::new(u, v);
                            if (u == a || v == a) && e2 != e_cross {
                                e_cross = e2;
                                nei = adj
                                    .edge2tris
                                    .get(&e_cross)
                                    .and_then(|vv| vv.iter().copied().find(|&x| x != ti));
                                break;
                            }
                        }
                    }
                }
            }

            // Record after anti-backtrack adjustment
            if !visited.insert((ti, e_cross)) {
                return Err("walker cycle detected");
            }

            // If the chosen exit edge is exactly (a,b), we're done
            if (e_cross.0 == a && e_cross.1 == b) || (e_cross.0 == b && e_cross.1 == a) {
                break;
            }

            // Interior or boundary?
            match adj.edge2tris.get(&e_cross) {
                // Interior: two incident triangles
                Some(v) if v.len() == 2 => {
                    let (t0, t1) = (v[0], v[1]);

                    // If not constrained, try to flip; otherwise just step to neighbor.
                    if !constrained.contains(&e_cross) {
                        if let Some(_new_e) =
                            self.flip_shared_edge_with_adj(e_cross, t0, t1, constrained, adj)
                        {
                            // After a successful flip, continue from the triangle that now contains `a`
                            let next_ti = if has_vertex(self.triangles[t0], a) {
                                t0
                            } else {
                                t1
                            };
                            last_ti = Some(ti);
                            last_edge = Some(e_cross);
                            ti = next_ti;
                        } else {
                            // Flip blocked (would create constrained or long edge, or degenerate): step across if possible
                            if let Some(n) = nei {
                                last_ti = Some(ti);
                                last_edge = Some(e_cross);
                                ti = n;
                            } else {
                                return Err("cannot flip or step");
                            }
                        }
                    } else {
                        // Crossed edge is constrained: only step to neighbor
                        if let Some(n) = nei {
                            last_ti = Some(ti);
                            last_edge = Some(e_cross);
                            ti = n;
                        } else {
                            return Err("constrained boundary with no neighbor");
                        }
                    }
                }

                // Boundary (0 or 1 incident triangle in map)
                _ => {
                    // We are crossing a boundary edge (u,v). Split the current triangle to create
                    // two triangles that include the constrained edge (anchor, other).
                    let t = self.triangles[ti];
                    let (u, v) = (e_cross.0, e_cross.1);
                    let w = match third_vertex(t, u, v) {
                        Some(w) => w,
                        None => return Err("boundary edge has no third vertex in current tri"),
                    };

                    // Which endpoint of (a,b) lies on this boundary edge?
                    let (anchor, other) = if u == a || v == a {
                        (a, b)
                    } else if u == b || v == b {
                        (b, a)
                    } else {
                        // Trying to leave mesh at an unrelated boundary: PSLG/classification mismatch.
                        return Err("hit boundary while inserting constraint");
                    };

                    // Make two triangles that introduce (anchor,other)
                    let t0_new =
                        if orient2d(&self.points[anchor], &self.points[other], &self.points[w])
                            .is_positive()
                        {
                            Triangle(anchor, other, w)
                        } else {
                            Triangle(anchor, w, other)
                        };

                    // The second triangle uses the other endpoint of the boundary edge
                    let x = if u == anchor { v } else { u };
                    let t1_new = if orient2d(&self.points[other], &self.points[x], &self.points[w])
                        .is_positive()
                    {
                        Triangle(other, x, w)
                    } else {
                        Triangle(other, w, x)
                    };

                    // Commit the split
                    let t_old = self.triangles[ti];
                    self.triangles[ti] = t0_new;
                    let t1_idx = self.triangles.len();
                    self.triangles.push(t1_new);
                    adj.replace_tri(ti, t_old, t0_new);
                    adj.add_tri(t1_idx, t1_new);

                    // After this split, (a,b) now exists (or will pass the existence check on next loop)
                }
            }
        }

        Ok(())
    }

    fn flip_shared_edge_with_adj(
        &mut self,
        e: Edge,
        t0_idx: usize,
        t1_idx: usize,
        constrained: &HashSet<Edge>,
        adj: &mut Adj,
    ) -> Option<Edge>
    where
        T: PartialOrd,
    {
        if t0_idx >= self.triangles.len() || t1_idx >= self.triangles.len() {
            return None;
        }

        let t0_old = self.triangles[t0_idx];
        let t1_old = self.triangles[t1_idx];
        let (u, v) = (e.0, e.1);

        let x = other_across(t0_old, e)?;
        let y = other_across(t1_old, e)?;
        if x == y {
            return None;
        }

        // Don’t create a constrained edge implicitly.
        let new_e = Edge::new(x, y);
        if constrained.contains(&new_e) {
            return None;
        }

        // Do not create an edge that passes through another vertex
        if edge_has_mid_vertex(x, y, &self.points) {
            return None;
        }

        // Build new triangles (preserve CCW)
        let a0 = orient2d(&self.points[x], &self.points[u], &self.points[y]);
        let a1 = orient2d(&self.points[x], &self.points[y], &self.points[v]);
        let t0_new = if a0.is_positive_or_zero() {
            Triangle(x, u, y)
        } else {
            Triangle(x, y, u)
        };
        let t1_new = if a1.is_positive_or_zero() {
            Triangle(x, y, v)
        } else {
            Triangle(x, v, y)
        };

        self.triangles[t0_idx] = t0_new;
        self.triangles[t1_idx] = t1_new;
        adj.replace_tri(t0_idx, t0_old, t0_new);
        adj.replace_tri(t1_idx, t1_old, t1_new);
        Some(new_e)
    }

    fn legalize_all_with_adj(&mut self, adj: &mut Adj, constrained: &HashSet<Edge>) {
        // One light pass; for stricter legality you can loop until no flips
        for ti in 0..self.triangles.len() {
            let t = self.triangles[ti];
            for (u, v) in tri_edges(t) {
                let e = Edge::new(u, v);
                if constrained.contains(&e) {
                    continue;
                }
                // Find neighbor across e
                if let Some(neis) = adj.edge2tris.get(&e) {
                    if neis.len() != 2 {
                        continue;
                    }
                    let (t0, t1) = (neis[0], neis[1]);
                    let a = third_vertex(self.triangles[t0], u, v).unwrap();
                    let b = third_vertex(self.triangles[t1], u, v).unwrap();
                    // Ensure (u,v,a) is CCW for incircle sign
                    let u_ = u;
                    let mut v_ = v;
                    let mut a_ = a;
                    let (aa, bb, cc) = if orient2d(
                        &self.points[u],
                        &self.points[v],
                        &self.points[a],
                    )
                    .is_positive()
                    {
                        (u, v, a) // already CCW
                    } else {
                        (v, u, a) // flip edge direction to make (u,v,a) CCW
                    };
                    let inc = incircle(
                        &self.points[aa],
                        &self.points[bb],
                        &self.points[cc],
                        &self.points[b],
                    );
                    if inc.is_positive() {
                        let _ = self.flip_shared_edge_with_adj(
                            Edge::new(u, v),
                            t0,
                            t1,
                            constrained,
                            adj,
                        );
                    }
                }
            }
        }
    }
}

#[inline]
fn third_vertex(t: Triangle, u: usize, v: usize) -> Option<usize> {
    let vs = [t.0, t.1, t.2];
    let mut count = 0;
    let mut other = None;
    for &w in &vs {
        if w != u && w != v {
            other = Some(w);
        } else {
            count += 1;
        }
    }
    if count == 2 { other } else { None }
}

#[inline]
fn other_across(t: Triangle, e: Edge) -> Option<usize> {
    let vs = [t.0, t.1, t.2];
    vs.into_iter().find(|&w| w != e.0 && w != e.1)
}

#[inline]
fn tri_edges(t: Triangle) -> [(usize, usize); 3] {
    [(t.0, t.1), (t.1, t.2), (t.2, t.0)]
}

#[inline]
fn tri_as_array(t: Triangle) -> [usize; 3] {
    [t.0, t.1, t.2]
}

#[inline]
fn has_vertex(t: Triangle, v: usize) -> bool {
    t.0 == v || t.1 == v || t.2 == v
}

/// Given current triangle index `ti`, decide the **one** edge we exit across,
/// when walking from a -> b. Uses exact orient2d.
/// Returns (crossed undirected edge, neighbor triangle index if interior).
fn exit_edge_of_triangle<T: Scalar>(
    points: &[Point2<T>],
    adj: &Adj,
    tris: &[Triangle],
    ti: usize,
    a: usize,
    b: usize,
) -> Option<(Edge, Option<usize>)>
where
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let t = tris[ti];
    let vs = tri_as_array(t);
    let pa = &points[a];
    let pb = &points[b];

    // Case A: a is a vertex of this triangle (common at start)
    if has_vertex(t, a) {
        // let the other two vertices be x,y (not equal to a).
        let mut other = [0usize; 2];
        let mut k = 0;
        for &v in &vs {
            if v != a {
                other[k] = v;
                k += 1;
            }
        }
        let x = other[0];
        let y = other[1];
        // If segment (a,b) exits the triangle, it exits across the edge between the two "other" vertices
        // iff orient(a,b,x) and orient(a,b,y) have opposite signs.
        let s1 = orient2d(pa, pb, &points[x]);
        let s2 = orient2d(pa, pb, &points[y]);

        // Touching a vertex (one sign == 0): step through the edge adjacent to that vertex that is crossed.
        if s1.is_zero() && s2.is_zero() {
            // (a,b) is collinear with both -> (a,b) lies exactly along the side through a.
            // Walk out through the edge that contains a and heads towards b.
            for (u, v) in tri_edges(t) {
                if u == a || v == a {
                    // choose the directed edge (a,*) whose midpoint is most "forward" towards b
                    let mid = Point2::<T>::from_vals([
                        (&points[u][0] + &points[v][0]) * T::from(0.5),
                        (&points[u][1] + &points[v][1]) * T::from(0.5),
                    ]);
                    let o = orient2d(pa, pb, &mid);
                    if !o.is_zero() {
                        let e = Edge::new(u, v);
                        let nei = adj
                            .edge2tris
                            .get(&e)
                            .and_then(|vv| vv.iter().copied().find(|&x| x != ti));
                        return Some((e, nei));
                    }
                }
            }
            return None;
        } else if s1.is_zero() || s2.is_zero() {
            // Through vertex: go through the edge opposite the *other* vertex
            let vtx = if s1.is_zero() { x } else { y };
            // Opposite edge is the edge that does not contain vtx
            let mut e_opt = None;
            for (u, v) in tri_edges(t) {
                if u != vtx && v != vtx {
                    e_opt = Some(Edge::new(u, v));
                    break;
                }
            }
            if let Some(e) = e_opt {
                let nei = adj
                    .edge2tris
                    .get(&e)
                    .and_then(|vv| vv.iter().copied().find(|&x| x != ti));
                return Some((e, nei));
            }
            return None;
        } else if (s1.is_positive()) != (s2.is_positive()) {
            // Different signs -> exit across the edge (x,y) (the one NOT incident to a)
            let e = Edge::new(x, y);
            let nei = adj
                .edge2tris
                .get(&e)
                .and_then(|vv| vv.iter().copied().find(|&x| x != ti));
            return Some((e, nei));
        } else {
            // Both on same side: we do not leave through the opposite edge; try an edge incident to a
            // Pick the incident edge whose midpoint lies most "forward" wrt (a->b)
            let mut best: Option<(Edge, T)> = None;
            for (u, v) in tri_edges(t) {
                if u == a || v == a {
                    let mid = Point2::<T>::from_vals([
                        (&points[u][0] + &points[v][0]) * T::from(0.5),
                        (&points[u][1] + &points[v][1]) * T::from(0.5),
                    ]);
                    // Compare by projection on direction (a->b)
                    let dx = &pb[0] - &pa[0];
                    let dy = &pb[1] - &pa[1];
                    let sx = &mid[0] - &pa[0];
                    let sy = &mid[1] - &pa[1];
                    let proj = dx * sx + dy * sy;
                    let cand = (Edge::new(u, v), proj.clone());
                    if best.as_ref().map_or(true, |(_, p)| proj > *p) {
                        best = Some(cand);
                    }
                }
            }
            if let Some((e, _)) = best {
                let nei = adj
                    .edge2tris
                    .get(&e)
                    .and_then(|vv| vv.iter().copied().find(|&x| x != ti));
                return Some((e, nei));
            }
            return None;
        }
    }

    // Case B: general triangle — test the three edges for proper crossing or interior touch
    for (u, v) in tri_edges(t) {
        let pu = &points[u];
        let pv = &points[v];
        let o1 = orient2d(pa, pb, pu);
        let o2 = orient2d(pa, pb, pv);
        let o3 = orient2d(pu, pv, pa);
        let o4 = orient2d(pu, pv, pb);
        let proper = (&o1 * &o2).is_negative() && (&o3 * &o4).is_negative();
        let touch_inside = o1.is_zero() && (&o3 * &o4).is_negative()
            || (o2.is_zero() && (&o3 * &o4).is_negative())
            || (o3.is_zero() && (&o1 * &o2).is_negative())
            || (o4.is_zero() && (&o1 * &o2).is_negative());
        if proper || touch_inside {
            let e = Edge::new(u, v);
            let nei = adj
                .edge2tris
                .get(&e)
                .and_then(|vv| vv.iter().copied().find(|&x| x != ti));
            return Some((e, nei));
        }
    }
    None
}

#[inline]
fn on_segment<T: Scalar + PartialOrd>(a: &Point2<T>, b: &Point2<T>, p: &Point2<T>) -> bool
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let o = orient2d(a, b, p);
    if !o.is_zero() {
        return false;
    }
    let (minx, maxx) = if a[0] <= b[0] {
        (a[0].clone(), b[0].clone())
    } else {
        (b[0].clone(), a[0].clone())
    };
    let (miny, maxy) = if a[1] <= b[1] {
        (a[1].clone(), b[1].clone())
    } else {
        (b[1].clone(), a[1].clone())
    };
    p[0] >= minx && p[0] <= maxx && p[1] >= miny && p[1] <= maxy
}

#[inline]
fn edge_has_mid_vertex<T: Scalar + PartialOrd>(i: usize, j: usize, pts: &[Point2<T>]) -> bool
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    for k in 0..pts.len() {
        if k == i || k == j {
            continue;
        }
        if on_segment(&pts[i], &pts[j], &pts[k]) {
            return true;
        }
    }
    false
}

#[inline]
fn param_t<T: Scalar>(a: &Point2<T>, b: &Point2<T>, p: &Point2<T>) -> T
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>,
{
    let ab0 = &b[0] - &a[0];
    let ab1 = &b[1] - &a[1];
    let ap0 = &p[0] - &a[0];
    let ap1 = &p[1] - &a[1];
    &ap0 * &ab0 + &ap1 * &ab1
}

fn split_constraint_chain<T: Scalar + PartialOrd>(
    pts: &[Point2<T>],
    a: usize,
    b: usize,
) -> Vec<[usize; 2]>
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    if a == b {
        return Vec::new();
    }
    let pa = &pts[a];
    let pb = &pts[b];

    // collect strict interior collinear points on segment (exclude endpoints)
    let mut mids: Vec<(T, usize)> = Vec::new();
    for k in 0..pts.len() {
        if k == a || k == b {
            continue;
        }
        if on_segment(pa, pb, &pts[k]) {
            // param monotone along AB; no need to normalize
            let t = param_t(pa, pb, &pts[k]);
            mids.push((t, k));
        }
    }
    if mids.is_empty() {
        return vec![[a, b]];
    }

    // order along AB and emit a chain a->m1->m2->...->b
    mids.sort_by(|(ti, _), (tj, _)| ti.partial_cmp(tj).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = Vec::with_capacity(mids.len() + 1);
    let mut prev = a;
    for &(_, m) in &mids {
        if prev != m {
            out.push([prev, m]);
        }
        prev = m;
    }
    if prev != b {
        out.push([prev, b]);
    }
    out
}

#[inline(always)]
fn build_edge2tris(dt_tris: &[Triangle]) -> ahash::AHashMap<Edge, smallvec::SmallVec<[usize; 2]>> {
    use ahash::AHashMap;
    use smallvec::SmallVec;
    let mut e2t: AHashMap<Edge, SmallVec<[usize; 2]>> = AHashMap::with_capacity(dt_tris.len() * 3);
    for (ti, t) in dt_tris.iter().enumerate() {
        for (u, v) in [(t.0, t.1), (t.1, t.2), (t.2, t.0)] {
            let e = Edge::new(u, v); // undirected
            e2t.entry(e).or_default().push(ti);
        }
    }
    e2t
}

pub fn triangles_inside_for_job<T: Scalar + PartialOrd>(
    pts: &[Point2<T>],
    dt_tris: &[Triangle],
    _job_segments: &[[usize; 2]],
) -> ahash::AHashSet<usize>
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    use ahash::{AHashMap as FastMap, AHashSet as FastSet};
    use std::collections::VecDeque;

    // ---------- helpers (minimal; reuse your existing predicates) ----------
    #[inline]
    fn to_f(p: &Point2<impl Scalar>) -> (f64, f64) {
        (p[0].to_f64().unwrap(), p[1].to_f64().unwrap())
    }
    #[inline]
    fn on_edge_with_t<TS: Scalar>(a: &Point2<TS>, b: &Point2<TS>, p: &Point2<TS>) -> Option<f64>
    where
        for<'x> &'x TS: std::ops::Sub<&'x TS, Output = TS>
            + std::ops::Mul<&'x TS, Output = TS>
            + std::ops::Add<&'x TS, Output = TS>,
    {
        let (ax, ay) = to_f(a);
        let (bx, by) = to_f(b);
        let (px, py) = to_f(p);
        let ux = bx - ax;
        let uy = by - ay;
        let vx = px - ax;
        let vy = py - ay;
        let cross = ux * vy - uy * vx;
        let eps = 1e-12 * (ux.abs() + uy.abs()).max(1.0);
        if cross.abs() > eps {
            return None;
        }
        let dot = ux * vx + uy * vy;
        let len2 = ux * ux + uy * uy;
        if len2 <= 0.0 {
            return None;
        }
        let t = dot / len2;
        if (-1e-12..=1.0 + 1e-12).contains(&t) {
            Some(t.clamp(0.0, 1.0))
        } else {
            None
        }
    }
    #[inline]
    fn centroid<TS: Scalar>(a: &Point2<TS>, b: &Point2<TS>, c: &Point2<TS>) -> (f64, f64)
    where
        for<'x> &'x TS: std::ops::Add<&'x TS, Output = TS> + std::ops::Div<&'x TS, Output = TS>,
    {
        let three = TS::from(3);
        let x = &(&(&a[0] + &b[0]) + &c[0]) / &three;
        let y = &(&(&a[1] + &b[1]) + &c[1]) / &three;
        (x.to_f64().unwrap(), y.to_f64().unwrap())
    }
    #[inline]
    fn point_in_polygon_even_odd(
        ring: &[usize],
        pts: &[Point2<impl Scalar>],
        q: (f64, f64),
    ) -> bool {
        let (qx, qy) = q;
        let mut inside = false;
        let n = ring.len();
        if n < 3 {
            return false;
        }
        for i in 0..n {
            let i0 = ring[i];
            let i1 = ring[(i + 1) % n];
            let (x0, y0) = to_f(&pts[i0]);
            let (x1, y1) = to_f(&pts[i1]);

            // boundary-inclusive
            let ux = x1 - x0;
            let uy = y1 - y0;
            let vx = qx - x0;
            let vy = qy - y0;
            let cross = ux * vy - uy * vx;
            if cross.abs() <= 1e-14 * (ux.abs() + uy.abs()).max(1.0) {
                let dot = ux * vx + uy * vy;
                let len2 = ux * ux + uy * uy;
                if dot >= -1e-14 && dot <= len2 + 1e-14 {
                    return true;
                }
            }

            if ((y0 > qy) != (y1 > qy)) && (qx < (x1 - x0) * (qy - y0) / (y1 - y0 + 1e-300) + x0) {
                inside = !inside;
            }
        }
        inside
    }
    #[inline]
    fn segs_properly_intersect<TS: Scalar>(
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        pts: &[Point2<TS>],
    ) -> bool
    where
        for<'x> &'x TS: std::ops::Sub<&'x TS, Output = TS>
            + std::ops::Mul<&'x TS, Output = TS>
            + std::ops::Add<&'x TS, Output = TS>
            + std::ops::Div<&'x TS, Output = TS>
            + std::ops::Neg<Output = TS>,
    {
        let pa = &pts[a];
        let pb = &pts[b];
        let pc = &pts[c];
        let pd = &pts[d];
        let o1 = orient2d(pa, pb, pc);
        let o2 = orient2d(pa, pb, pd);
        let o3 = orient2d(pc, pd, pa);
        let o4 = orient2d(pc, pd, pb);
        (&o1 * &o2).is_negative() && (&o3 * &o4).is_negative()
    }
    #[inline]
    fn tri_crosses_ring<TS: Scalar>(t: Triangle, ring: &[usize], pts: &[Point2<TS>]) -> bool
    where
        for<'x> &'x TS: std::ops::Sub<&'x TS, Output = TS>
            + std::ops::Mul<&'x TS, Output = TS>
            + std::ops::Add<&'x TS, Output = TS>
            + std::ops::Div<&'x TS, Output = TS>
            + std::ops::Neg<Output = TS>,
    {
        for &(i, j) in &[(t.0, t.1), (t.1, t.2), (t.2, t.0)] {
            // skip if (i,j) lies on a boundary side
            let on01 = on_edge_with_t(&pts[0], &pts[1], &pts[i]).is_some()
                && on_edge_with_t(&pts[0], &pts[1], &pts[j]).is_some();
            let on12 = on_edge_with_t(&pts[1], &pts[2], &pts[i]).is_some()
                && on_edge_with_t(&pts[1], &pts[2], &pts[j]).is_some();
            let on20 = on_edge_with_t(&pts[2], &pts[0], &pts[i]).is_some()
                && on_edge_with_t(&pts[2], &pts[0], &pts[j]).is_some();
            if on01 || on12 || on20 {
                continue;
            }
            for k in 0..ring.len() {
                let r0 = ring[k];
                let r1 = ring[(k + 1) % ring.len()];
                if r0 == i || r0 == j || r1 == i || r1 == j {
                    continue;
                }
                if segs_properly_intersect(i, j, r0, r1, pts) {
                    return true;
                }
            }
        }
        false
    }
    #[inline]
    fn tri_area_abs(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
        ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs() * 0.5
    }

    // ---------- 1) build the boundary ring (0..1, 1..2, 2..0) ----------
    if pts.len() < 3 {
        return FastSet::default();
    }
    let c01 = {
        let mut v = Vec::new();
        let a = &pts[0];
        let b = &pts[1];
        for (k, p) in pts.iter().enumerate() {
            if on_edge_with_t(a, b, p).is_some() {
                v.push(k);
            }
        }
        v.sort_unstable_by(|&i, &j| {
            let ti = on_edge_with_t(a, b, &pts[i]).unwrap();
            let tj = on_edge_with_t(a, b, &pts[j]).unwrap();
            ti.partial_cmp(&tj).unwrap()
        });
        v
    };
    let c12 = {
        let mut v = Vec::new();
        let a = &pts[1];
        let b = &pts[2];
        for (k, p) in pts.iter().enumerate() {
            if on_edge_with_t(a, b, p).is_some() {
                v.push(k);
            }
        }
        v.sort_unstable_by(|&i, &j| {
            let ti = on_edge_with_t(a, b, &pts[i]).unwrap();
            let tj = on_edge_with_t(a, b, &pts[j]).unwrap();
            ti.partial_cmp(&tj).unwrap()
        });
        v
    };
    let c20 = {
        let mut v = Vec::new();
        let a = &pts[2];
        let b = &pts[0];
        for (k, p) in pts.iter().enumerate() {
            if on_edge_with_t(a, b, p).is_some() {
                v.push(k);
            }
        }
        v.sort_unstable_by(|&i, &j| {
            let ti = on_edge_with_t(a, b, &pts[i]).unwrap();
            let tj = on_edge_with_t(a, b, &pts[j]).unwrap();
            ti.partial_cmp(&tj).unwrap()
        });
        v
    };

    let mut ring: Vec<usize> =
        Vec::with_capacity(c01.len() + c12.len() + c20.len().saturating_sub(3));
    ring.extend(c01.iter().copied());
    ring.extend(c12.iter().copied().skip(1));
    ring.extend(c20.iter().copied().skip(1));
    if ring.last() == ring.first() {
        ring.pop();
    }

    // orientation of ring (CCW => interior is left side)
    let ring_is_ccw = {
        let mut sum = 0.0;
        for i in 0..ring.len() {
            let (x0, y0) = to_f(&pts[ring[i]]);
            let (x1, y1) = to_f(&pts[ring[(i + 1) % ring.len()]]);
            sum += x0 * y1 - y0 * x1;
        }
        sum > 0.0
    };
    let wanted_on_directed_ring: i8 = if ring_is_ccw { 1 } else { -1 };

    // ---------- 2) choose candidate triangles ----------
    let mut candidate_set: FastSet<usize> = FastSet::default();
    for (ti, t) in dt_tris.iter().enumerate() {
        let (a, b, c) = (t.0, t.1, t.2);
        if a >= pts.len() || b >= pts.len() || c >= pts.len() {
            continue;
        }
        // positive area
        if tri_area_abs(to_f(&pts[a]), to_f(&pts[b]), to_f(&pts[c])) <= 1e-18 {
            continue;
        }
        // centroid inside and triangle doesn't cross the ring
        let q = centroid(&pts[a], &pts[b], &pts[c]);
        if !point_in_polygon_even_odd(&ring, pts, q) {
            continue;
        }
        if tri_crosses_ring(*t, &ring, pts) {
            continue;
        }
        candidate_set.insert(ti);
    }

    // ---------- 3) build edge -> candidate triangles with side (canonical) ----------
    let mut e2cand: FastMap<Edge, Vec<(usize, i8, T)>> = FastMap::default();
    for (ti, t) in dt_tris.iter().enumerate() {
        if !candidate_set.contains(&ti) {
            continue;
        }
        for (i, j, k) in [(t.0, t.1, t.2), (t.1, t.2, t.0), (t.2, t.0, t.1)] {
            let mut a = i;
            let mut b = j;
            let mut s = orient2d(&pts[i], &pts[j], &pts[k]); // side wrt directed i->j
            // store side wrt canonical (min,max)
            if b < a {
                std::mem::swap(&mut a, &mut b);
                s = -s;
            }
            let side = if s.is_positive() {
                1
            } else if s.is_negative() {
                -1
            } else {
                0
            };
            if side == 0 {
                continue;
            }
            e2cand
                .entry(Edge::new(a, b))
                .or_default()
                .push((ti, side, s.abs()));
        }
    }

    // canonicalizer for (u,v): returns (Edge(min,max), sign) where sign = +1 if (u,v)==(min,max) else -1
    #[inline]
    fn canon(u: usize, v: usize) -> (Edge, i8) {
        if u < v {
            (Edge(u, v), 1)
        } else {
            (Edge(v, u), -1)
        }
    }

    // ---------- 4) flood-fill from ring edges, honoring side convention ----------
    let mut accepted: FastSet<usize> = FastSet::default();
    let mut seen_edge_side: FastSet<(Edge, i8)> = FastSet::default();
    let mut q = VecDeque::<(usize, usize, i8)>::new();

    for i in 0..ring.len() {
        let u = ring[i];
        let v = ring[(i + 1) % ring.len()];
        q.push_back((u, v, wanted_on_directed_ring));
    }

    while let Some((u, v, wanted_dir)) = q.pop_front() {
        // convert wanted side to canonical edge orientation
        let (e, dir) = canon(u, v);
        let wanted_canon = wanted_dir * dir;

        if !seen_edge_side.insert((e, wanted_canon)) {
            continue;
        }

        let Some(list) = e2cand.get(&e) else {
            continue;
        };

        // choose the furthest triangle on that side (max |orient|)
        let mut best: Option<(usize, T)> = None;
        for &(ti, side, ref score) in list {
            if side != wanted_canon || accepted.contains(&ti) {
                continue;
            }
            if best.as_ref().map_or(true, |&(_, ref s)| *score > *s) {
                best = Some((ti, score.clone()));
            }
        }
        let Some((ti, _)) = best else {
            continue;
        };

        // accept it
        accepted.insert(ti);

        // push its other two edges with **directed** wanted side = orient(i->j, k)
        let t = dt_tris[ti];
        for (i, j, k) in [(t.0, t.1, t.2), (t.1, t.2, t.0), (t.2, t.0, t.1)] {
            // skip the edge we used to reach this tri
            let (emin, _) = canon(i, j);
            if emin == e {
                continue;
            }
            let s = orient2d(&pts[i], &pts[j], &pts[k]);
            let side_dir = if s.is_positive() {
                1
            } else if s.is_negative() {
                -1
            } else {
                0
            };
            if side_dir != 0 {
                q.push_back((i, j, side_dir));
            }
        }
    }

    accepted
}

// call this at the end of build(), right before returning
fn compact_one_per_side<T: Scalar>(points: &[Point2<T>], triangles: &mut Vec<Triangle>)
where
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    use ahash::AHashMap as FastMap;
    use smallvec::SmallVec;

    // edge -> incident triangles
    let mut e2t: FastMap<Edge, SmallVec<[usize; 8]>> = FastMap::default();
    for (ti, t) in triangles.iter().enumerate() {
        for (u, v) in [(t.0, t.1), (t.1, t.2), (t.2, t.0)] {
            e2t.entry(Edge::new(u, v)).or_default().push(ti);
        }
    }

    let mut keep = vec![true; triangles.len()];

    let side = |e: Edge, ti: usize| -> i8 {
        let t = triangles[ti];
        let k = [t.0, t.1, t.2]
            .into_iter()
            .find(|&w| w != e.0 && w != e.1)
            .unwrap();
        let s = orient2d(&points[e.0], &points[e.1], &points[k]);
        if s.is_positive() {
            1
        } else if s.is_negative() {
            -1
        } else {
            0
        }
    };

    for (e, idxs) in e2t {
        let mut pos = SmallVec::<[usize; 8]>::new();
        let mut neg = SmallVec::<[usize; 8]>::new();

        for ti in idxs {
            match side(e, ti) {
                1 => pos.push(ti),
                -1 => neg.push(ti),
                _ => keep[ti] = false, // collinear on the edge -> drop
            }
        }

        // If more than one on a side, keep the one “furthest” from the edge
        let mut keep_best = |lst: &mut SmallVec<[usize; 8]>| {
            if lst.len() <= 1 {
                return;
            }
            let best = *lst
                .iter()
                .max_by(|&&a, &&b| {
                    let ka = [triangles[a].0, triangles[a].1, triangles[a].2]
                        .into_iter()
                        .find(|&w| w != e.0 && w != e.1)
                        .unwrap();
                    let kb = [triangles[b].0, triangles[b].1, triangles[b].2]
                        .into_iter()
                        .find(|&w| w != e.0 && w != e.1)
                        .unwrap();
                    orient2d(&points[e.0], &points[e.1], &points[ka])
                        .abs()
                        .partial_cmp(&orient2d(&points[e.0], &points[e.1], &points[kb]).abs())
                        .unwrap()
                })
                .unwrap();
            for &ti in lst.iter() {
                if ti != best {
                    keep[ti] = false;
                }
            }
        };

        keep_best(&mut pos);
        keep_best(&mut neg);
    }

    *triangles = triangles
        .iter()
        .cloned()
        .zip(keep.into_iter())
        .filter_map(|(t, k)| if k { Some(t) } else { None })
        .collect();
}

#[inline]
fn segs_properly_intersect<TS: Scalar>(
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    pts: &[Point2<TS>],
) -> bool
where
    for<'x> &'x TS: std::ops::Sub<&'x TS, Output = TS>
        + std::ops::Mul<&'x TS, Output = TS>
        + std::ops::Add<&'x TS, Output = TS>
        + std::ops::Div<&'x TS, Output = TS>
        + std::ops::Neg<Output = TS>,
{
    // standard robust test with boundary treated as non-proper
    let pa = &pts[a];
    let pb = &pts[b];
    let pc = &pts[c];
    let pd = &pts[d];
    let o1 = orient2d(pa, pb, pc);
    let o2 = orient2d(pa, pb, pd);
    let o3 = orient2d(pc, pd, pa);
    let o4 = orient2d(pc, pd, pb);

    // proper intersection only (all strict, not touching/collinear)
    (&o1 * &o2).is_negative() && (&o3 * &o4).is_negative()
}

#[inline]
fn tri_crosses_ring<TS: Scalar>(t: Triangle, ring: &[usize], pts: &[Point2<TS>]) -> bool
where
    for<'x> &'x TS: std::ops::Sub<&'x TS, Output = TS>
        + std::ops::Mul<&'x TS, Output = TS>
        + std::ops::Add<&'x TS, Output = TS>
        + std::ops::Div<&'x TS, Output = TS>
        + std::ops::Neg<Output = TS>,
{
    let tri_edges = [(t.0, t.1), (t.1, t.2), (t.2, t.0)];
    for &(i, j) in &tri_edges {
        // skip if this edge lies on a boundary edge
        let on01 = on_edge_with_t(&pts[0], &pts[1], &pts[i]).0
            && on_edge_with_t(&pts[0], &pts[1], &pts[j]).0;
        let on12 = on_edge_with_t(&pts[1], &pts[2], &pts[i]).0
            && on_edge_with_t(&pts[1], &pts[2], &pts[j]).0;
        let on20 = on_edge_with_t(&pts[2], &pts[0], &pts[i]).0
            && on_edge_with_t(&pts[2], &pts[0], &pts[j]).0;
        if on01 || on12 || on20 {
            continue;
        }

        // test vs every ring segment
        for k in 0..ring.len() {
            let r0 = ring[k];
            let r1 = ring[(k + 1) % ring.len()];
            // if the ring segment shares endpoints with (i,j), skip
            if (r0 == i || r0 == j || r1 == i || r1 == j) {
                continue;
            }
            if segs_properly_intersect(i, j, r0, r1, pts) {
                println!("Triangle crosses ring edge ({}, {})", r0, r1);
                return true; // crosses boundary → reject triangle
            }
        }
    }
    false
}
