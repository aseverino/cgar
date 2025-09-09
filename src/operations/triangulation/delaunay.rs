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

use ahash::{AHashMap, AHashSet};
use smallvec::SmallVec;

use crate::geometry::Point2;
use crate::geometry::point::Point;
use crate::geometry::spatial_element::SpatialElement;
use crate::geometry::util::EPS;
use crate::kernel::predicates::{bbox, bbox_approx, incircle, orient2d};
use crate::mesh::basic_types::{Edge, Mesh, Triangle};
use crate::mesh_processing::batching::FaceJobUV;
use crate::numeric::scalar::Scalar;

pub const SQRT_3: f64 = 1.7320508075688772;

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
    pub fn build_batch_with_constraints_bowyer_watson<const N: usize>(
        jobs: &[FaceJobUV<T>],
        _mesh: &Mesh<T, N>,
    ) -> Vec<Self> {
        if jobs.is_empty() {
            return Vec::new();
        }

        // Pre-allocate results
        let mut results = Vec::with_capacity(jobs.len());

        // Batch super-triangle computation using single bbox pass
        let mut super_triangles = Vec::with_capacity(jobs.len());
        for job in jobs {
            if job.points_uv.len() < 3 {
                results.push(Self {
                    points: job.points_uv.clone(),
                    triangles: Vec::new(),
                });
                super_triangles.push(None);
                continue;
            }

            let (minx, miny, maxx, maxy) = bbox_approx(&job.points_uv);
            let dx = maxx - minx;
            let dy = maxy - miny;
            let delta = dx.max(dy);
            let cx = (minx + maxx) * 0.5;
            let cy = (miny + maxy) * 0.5;
            let r = 64.0 * delta + 1.0;

            super_triangles.push(Some((cx, cy, r)));
        }

        // Process each job with pre-computed super-triangle
        for (job_idx, job) in jobs.iter().enumerate() {
            if job.points_uv.len() < 3 {
                continue;
            }

            let Some((cx, cy, r)) = super_triangles[job_idx] else {
                continue;
            };

            let mut points = job.points_uv.clone();
            let sqrt_3 = SQRT_3;
            let s0 = points.len();
            let s1 = s0 + 1;
            let s2 = s0 + 2;

            points.extend([
                Point2::<T>::from_vals([T::from(cx), T::from(cy + 2.0 * r)]),
                Point2::<T>::from_vals([T::from(cx - sqrt_3 * r), T::from(cy - r)]),
                Point2::<T>::from_vals([T::from(cx + sqrt_3 * r), T::from(cy - r)]),
            ]);

            let mut triangles = vec![Triangle(s0, s1, s2)];

            // Bowyer-Watson insertion
            for pid in 0..s0 {
                Self::bowyer_watson_insert_point(pid, &points, &mut triangles);
            }

            // Remove super-triangles and finalize
            triangles.retain(|t| t.0 < s0 && t.1 < s0 && t.2 < s0);
            points.truncate(s0);

            let mut dt = Self { points, triangles };

            // Batch constraint processing
            Self::process_constraints_batch(&mut dt, &job.segments);

            results.push(dt);
        }

        results
    }

    fn process_constraints_batch(dt: &mut Self, constraints_in: &[[usize; 2]]) {
        let mut constraints = Vec::new();
        let mut seen = AHashSet::new();

        // Batch constraint splitting
        for &[a, b] in constraints_in {
            if a >= dt.points.len() || b >= dt.points.len() || a == b {
                continue;
            }
            for ab in split_constraint_chain(&dt.points, a, b) {
                let e = Edge::new(ab[0], ab[1]);
                if seen.insert(e) {
                    constraints.push(e);
                }
            }
        }

        let mut constrained = AHashSet::<Edge>::new();
        let mut adj = Adj::rebuild(dt.points.len(), &dt.triangles);

        // Process constraints in batches of similar length/complexity
        constraints.sort_by_key(|e| {
            let dx = dt.points[e.a()][0].ball_center_f64() - dt.points[e.b()][0].ball_center_f64();
            let dy = dt.points[e.a()][1].ball_center_f64() - dt.points[e.b()][1].ball_center_f64();
            ((dx * dx + dy * dy) * 1000.0) as i32
        });

        for e in &constraints {
            if dt.edge_exists(*e) {
                constrained.insert(*e);
                continue;
            }
            if dt
                .insert_constraint_walk(e.a(), e.b(), &mut adj, &constrained)
                .is_ok()
            {
                constrained.insert(*e);
            }
        }
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
        let (minx, miny, maxx, maxy) = bbox_approx(&points);
        let dx = &maxx - &minx;
        let dy = &maxy - &miny;
        let delta = dx.max(dy);
        let cx = &(minx + maxx) * 0.5;
        let cy = &(miny + maxy) * 0.5;

        let r = 64.0 * delta + 1.0;
        let sqrt_3 = SQRT_3;
        let p_super0 = Point2::<T>::from_vals([T::from(cx.clone()), T::from((cy + 2.0) * r)]);
        let p_super1 = Point2::<T>::from_vals([T::from(&cx - (sqrt_3 * r)), T::from(cy - r)]);
        let p_super2 = Point2::<T>::from_vals([T::from(&cx + (sqrt_3 * r)), T::from(cy - r)]);

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
            if Self::point_in_circumcircle_fast(p, t, points) {
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
            let new_triangle = if orient2d(&points[edge.a()], &points[edge.b()], p).is_positive() {
                Triangle(edge.a(), edge.b(), pid)
            } else {
                Triangle(edge.a(), pid, edge.b())
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

    fn point_in_circumcircle_fast(p: &Point2<T>, t: Triangle, points: &[Point2<T>]) -> bool {
        if let Some((cx, cy, r2)) =
            circumcircle_approx_fast(&points[t.0], &points[t.1], &points[t.2])
        {
            let px = p[0].ball_center_f64();
            let py = p[1].ball_center_f64();

            if [px, py, cx, cy, r2].iter().all(|x| x.is_finite()) {
                let dx = px - cx;
                let dy = py - cy;
                let dist2 = dx * dx + dy * dy;

                let tolerance = EPS * r2.max(1.0);

                // Clearly outside circumcircle
                if dist2 > r2 + tolerance {
                    return false;
                }
                // Clearly inside circumcircle
                if dist2 < r2 - tolerance {
                    return true;
                }
            }
        }

        // Uncertain cases: fallback to exact
        Self::point_in_circumcircle(p, t, points)
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
        let mut seen = AHashSet::new();

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

        let mut constrained = AHashSet::<Edge>::new();
        let mut adj = Adj::rebuild(dt.points.len(), &dt.triangles);

        // 3) Insert constraints
        for e in &constraints {
            if dt.edge_exists(*e) {
                constrained.insert(*e);
                continue;
            }
            match dt.insert_constraint_walk(e.a(), e.b(), &mut adj, &constrained) {
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

    fn edge_exists(&self, edge: Edge) -> bool
    where
        T: PartialOrd,
    {
        // edge must be present in some triangle…
        let mut present = false;
        for t in &self.triangles {
            let tri = [t.0, t.1, t.2];
            if Edge::new(tri[0], tri[1]) == edge
                || Edge::new(tri[1], tri[2]) == edge
                || Edge::new(tri[2], tri[0]) == edge
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

    fn insert_constraint_walk(
        &mut self,
        mut a: usize,
        mut b: usize,
        adj: &mut Adj,
        constrained: &AHashSet<Edge>,
    ) -> Result<(), &'static str> {
        if self.edge_exists(Edge::new(a, b)) {
            return Ok(());
        }

        // Choose a starting triangle that actually faces toward the segment (a,b).
        // Prefer a's incident triangles; if none useful, swap endpoints and try b.
        let pick_start = |end: usize, other: usize| -> Option<usize> {
            if let Some(inc) = adj.vert2tris.get(end) {
                for &ti in inc.iter() {
                    if exit_edge_of_triangle_fast(
                        &self.points,
                        adj,
                        &self.triangles,
                        ti,
                        end,
                        other,
                    )
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

        while !self.edge_exists(Edge::new(a, b)) {
            steps += 1;
            if steps > max_steps {
                return Err("insert_constraint_walk: too many steps");
            }

            // Decide the edge we must cross from current_tri toward segment (a,b)
            let (cross_e, nei) = match exit_edge_of_triangle_fast(
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
            if let Some(w) = third_vertex(self.triangles[current_tri], cross_e.a(), cross_e.b()) {
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

            // If we already have the target edge, stop walking; we'll add missing side(s) below.
            if (cross_e.a() == a && cross_e.b() == b) || (cross_e.a() == b && cross_e.b() == a) {
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
                            if (new_e.a() == a && new_e.b() == b)
                                || (new_e.a() == b && new_e.b() == a)
                            {
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
        let (u, v) = (boundary_edge.a(), boundary_edge.b());
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

    fn flip_shared_edge_with_adj(
        &mut self,
        e: Edge,
        t0_idx: usize,
        t1_idx: usize,
        constrained: &AHashSet<Edge>,
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
        let (u, v) = (e.a(), e.b());

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
    vs.into_iter().find(|&w| w != e.a() && w != e.b())
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

fn exit_edge_of_triangle_fast<T: Scalar>(
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
    // Fast path using ball centers
    if let (Some(ax), Some(ay), Some(bx), Some(by)) = (
        points[a][0].as_f64_fast(),
        points[a][1].as_f64_fast(),
        points[b][0].as_f64_fast(),
        points[b][1].as_f64_fast(),
    ) {
        // Use fast f64 computation for edge selection
        return exit_edge_fast_f64(points, adj, tris, ti, a, b, ax, ay, bx, by);
    }

    // Fallback to exact
    exit_edge_of_triangle(points, adj, tris, ti, a, b)
}

fn exit_edge_fast_f64<T: Scalar>(
    points: &[Point2<T>],
    adj: &Adj,
    tris: &[Triangle],
    ti: usize,
    a: usize,
    b: usize,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
) -> Option<(Edge, Option<usize>)> {
    let t = tris[ti];
    let vs = [t.0, t.1, t.2];

    // Fast f64 orientation tests
    for (u, v) in [(t.0, t.1), (t.1, t.2), (t.2, t.0)] {
        let ux = points[u][0].ball_center_f64();
        let uy = points[u][1].ball_center_f64();
        let vx = points[v][0].ball_center_f64();
        let vy = points[v][1].ball_center_f64();

        // Fast crossing test
        let o1 = (bx - ax) * (uy - ay) - (by - ay) * (ux - ax);
        let o2 = (bx - ax) * (vy - ay) - (by - ay) * (vx - ax);
        let o3 = (vx - ux) * (ay - uy) - (vy - uy) * (ax - ux);
        let o4 = (vx - ux) * (by - uy) - (vy - uy) * (bx - ux);

        let eps = EPS * ((bx - ax).abs() + (by - ay).abs()).max(1.0);

        if (o1 * o2 < -eps * eps) && (o3 * o4 < -eps * eps) {
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
        let eps = EPS * (ux.abs() + uy.abs()).max(1.0);
        if cross.abs() > eps {
            return None;
        }
        let dot = ux * vx + uy * vy;
        let len2 = ux * ux + uy * uy;
        if len2 <= 0.0 {
            return None;
        }
        let t = dot / len2;
        if (-EPS..=1.0 + EPS).contains(&t) {
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
            if cross.abs() <= EPS * (ux.abs() + uy.abs()).max(1.0) {
                let dot = ux * vx + uy * vy;
                let len2 = ux * ux + uy * uy;
                if dot >= -EPS && dot <= len2 + EPS {
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
        let edge = Edge::new(u, v);
        if u < v { (edge, 1) } else { (edge, -1) }
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

fn circumcircle_approx_fast<T: Scalar>(
    a: &Point2<T>,
    b: &Point2<T>,
    c: &Point2<T>,
) -> Option<(f64, f64, f64)> {
    let ax = a[0].ball_center_f64();
    let ay = a[1].ball_center_f64();
    let bx = b[0].ball_center_f64();
    let by = b[1].ball_center_f64();
    let cx = c[0].ball_center_f64();
    let cy = c[1].ball_center_f64();

    if ![ax, ay, bx, by, cx, cy].iter().all(|x| x.is_finite()) {
        return None;
    }

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-15 {
        return None;
    }

    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;

    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;

    let dx = ax - ux;
    let dy = ay - uy;
    let r2 = dx * dx + dy * dy;

    Some((ux, uy, r2))
}
