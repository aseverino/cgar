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
        vector::{Cross3, Vector, VectorOps},
    },
    impl_mesh,
    numeric::scalar::Scalar,
};

#[derive(Debug, Clone)]
pub struct RemeshOptions<T: Scalar> {
    pub target_edge_length: T,
    pub min_edge_length: T,
    pub max_edge_length: T,
    pub max_iterations: usize,
    pub convergence_threshold: T,
}

impl<T: Scalar> RemeshOptions<T>
where
    for<'a> &'a T: std::ops::Mul<&'a T, Output = T> + std::ops::Div<&'a T, Output = T>,
{
    pub fn new(target_length: T) -> Self {
        let min_len = &target_length * &T::from_num_den(4, 5);
        let max_len = &target_length * &T::from_num_den(4, 3);

        Self {
            target_edge_length: target_length,
            min_edge_length: min_len,
            max_edge_length: max_len,
            max_iterations: 10,
            convergence_threshold: T::from_num_den(1, 20),
        }
    }
}

#[derive(Debug)]
pub enum RemeshError {
    InvalidOptions,
    ConvergenceFailure,
}

impl_mesh! {
    pub fn compute_average_edge_length(&self) -> f64 {
        let mut total_length_sq = 0.0;
        let mut count = 0;

        for he_idx in 0..self.half_edges.len() {
            let he = &self.half_edges[he_idx];
            if he.removed || he_idx >= he.twin {
                continue;
            }

            let v0 = self.source(he_idx);
            let v1 = he.vertex;

            if self.vertices[v0].removed || self.vertices[v1].removed {
                continue;
            }

            let p0 = &self.vertices[v0].position;
            let p1 = &self.vertices[v1].position;
            let edge_vec = p1 - p0;
            let length_sq = edge_vec.as_vector().norm2();
            total_length_sq = &total_length_sq + &length_sq.to_f64().unwrap();
            count += 1;
        }

        if count == 0 {
            1.0
        } else {
            let avg_length_sq = &total_length_sq / count as f64;
            avg_length_sq.sqrt()
        }
    }

    pub fn smooth_vertices(&mut self, iterations: usize, max_edge_length: &T) -> usize {
        let max_sq = max_edge_length * max_edge_length;
        let mut moved_count = 0;

        for _ in 0..iterations {
            for v in 0..self.vertices.len() {
                if self.vertices[v].removed { continue; }

                let alpha = T::from(0.5);

                // Check if move would violate length constraints
                if self.smooth_tangential(v, alpha, &max_sq) {
                    moved_count += 1;
                }
            }
        }

        moved_count
    }

    fn vertex_valence(&self, vid: usize) -> usize {
        let mut count = 0;
        if let Some(start) = self.vertices[vid].half_edge {
            let mut current = start;
            loop {
                if !self.half_edges[current].removed {
                    count += 1;
                }
                let twin = self.half_edges[current].twin;
                if twin == usize::MAX { break; }
                current = self.half_edges[twin].next;
                if current == start { break; }
            }
        }
        count
    }

    /// Compute RMS edge length over unique edges (he < twin).
    fn rms_edge_length_f64(&self) -> f64 {
        let mut acc = 0.0;
        let mut cnt = 0u64;
        for i in 0..self.half_edges.len() {
            let he = &self.half_edges[i];
            if he.removed || he.twin == usize::MAX || i >= he.twin { continue; }
            let v0 = self.source(i);
            let v1 = he.vertex;
            if self.vertices[v0].removed || self.vertices[v1].removed { continue; }
            let p0 = &self.vertices[v0].position;
            let p1 = &self.vertices[v1].position;
            let d  = (p1 - p0).as_vector();
            let l2 = d.norm2();
            if let Some(l2f) = l2.to_f64() {
                acc += l2f;
                cnt += 1;
            }
        }
        if cnt == 0 { return 1.0; }
        (acc / cnt as f64).sqrt()
    }

    /// Fast (law-of-cosines) max(cos(angle)) for a triangle (i.e. cos of its smallest angle).
    #[inline]
    fn tri_min_angle_cos_f64(&self, a: usize, b: usize, c: usize) -> Option<f64> {
        if a==b || b==c || c==a { return None; }
        let pa = &self.vertices[a].position;
        let pb = &self.vertices[b].position;
        let pc = &self.vertices[c].position;

        let ab = (pb - pa).as_vector(); let ab2 = ab.norm2().to_f64()?;
        let bc = (pc - pb).as_vector(); let bc2 = bc.norm2().to_f64()?;
        let ca = (pa - pc).as_vector(); let ca2 = ca.norm2().to_f64()?;

        if ab2 == 0.0 || bc2 == 0.0 || ca2 == 0.0 { return None; }

        // Opposite side squares
        let a2 = bc2;
        let b2 = ca2;
        let c2 = ab2;

        // cos at A (vertex a) is (b2 + c2 - a2)/(2 sqrt(b2 c2))
        #[inline(always)]
        fn cos_from(a2: f64, b2: f64, c2: f64) -> f64 {
            let den = 2.0 * (b2 * c2).sqrt();
            if den == 0.0 { return 1.0; }
            let mut v = (b2 + c2 - a2) / den;
            if v > 1.0 { v = 1.0; } else if v < -1.0 { v = -1.0; }
            v
        }

        let cos_a = cos_from(a2, b2, c2);
        let cos_b = cos_from(b2, c2, a2);
        let cos_c = cos_from(c2, a2, b2);

        // Smallest angle → largest cosine
        Some(cos_a.max(cos_b).max(cos_c))
    }

    /// Quality-based flip predicate:
    /// Accept if flip strictly reduces max(cos(min angle)) of the pair of triangles
    /// (i.e. increases the smallest angle across the local configuration),
    /// and does not introduce degeneracy.
    fn should_flip_edge(&self, he_idx: usize) -> bool
    where
        Vector<T,N>: VectorOps<T, N> + Cross3<T>,
    {
        if he_idx >= self.half_edges.len() { return false; }
        let he = &self.half_edges[he_idx];
        if he.removed { return false; }
        let twin = he.twin;
        if twin == usize::MAX || twin >= self.half_edges.len() { return false; }
        if self.half_edges[twin].removed { return false; }
        if he.face.is_none() || self.half_edges[twin].face.is_none() { return false; }

        // Vertices
        let v0 = self.source(he_idx);
        let v1 = he.vertex;
        let v2 = self.half_edges[self.half_edges[he_idx].next].vertex;
        let v3 = self.half_edges[self.half_edges[twin].next].vertex;

        if v0==v1 || v1==v2 || v2==v0 || v0==v3 || v1==v3 || v2==v3 { return false; }
        if v2 == v3 { return false; } // would collapse

        // Old triangles: (v0,v1,v2) and (v1,v0,v3)
        let old1 = self.tri_min_angle_cos_f64(v0,v1,v2).unwrap();
        let old2 = self.tri_min_angle_cos_f64(v1,v0,v3).unwrap();
        let old_min = old1.max(old2); // since each returns cos(minAngle of tri), worst small angle

        // New triangles after flip: (v0,v2,v3) and (v1,v3,v2)
        let new1 = self.tri_min_angle_cos_f64(v0,v2,v3).unwrap();
        let new2 = self.tri_min_angle_cos_f64(v1,v3,v2).unwrap();
        let new_min = new1.max(new2);

        // Require a strict improvement with a small epsilon
        if new_min + 1e-12 >= old_min { return false; }

        // Degeneracy guard: ensure cross products non-zero (area)
        let pa = &self.vertices[v0].position;
        let pb = &self.vertices[v1].position;
        let pc = &self.vertices[v2].position;
        let pd = &self.vertices[v3].position;
        let area_safe = |p0: &Point::<T, N>, p1: &Point::<T, N>, p2: &Point::<T, N>| {
            let e0 = (p1 - p0).as_vector();
            let e1 = (p2 - p0).as_vector();
            let n  = e0.cross(&e1);
            let z  = n.norm2();
            z > T::zero()
        };
        area_safe(&pa,&pb,&pc) &&
        area_safe(&pb,&pa,&pd) &&
        area_safe(&pa,&pc,&pd) &&
        area_safe(&pb,&pd,&pc)
    }

    /// Flip edges using quality-based predicate plus lightweight valence improvement.
    fn flip_edges(&mut self) -> usize
    where
        Vector<T,N>: VectorOps<T, N> + Cross3<T>
    {
        let mut list = Vec::new();
        list.reserve(self.half_edges.len()/3);
        for i in 0..self.half_edges.len() {
            let he = &self.half_edges[i];
            if he.removed || he.twin == usize::MAX || i >= he.twin { continue; }
            if he.face.is_none() || self.half_edges[he.twin].face.is_none() { continue; }
            list.push(i);
        }

        let mut flips = 0usize;
        for he_idx in list {
            if he_idx >= self.half_edges.len() { continue; }
            if self.half_edges[he_idx].removed { continue; }
            let twin = self.half_edges[he_idx].twin;
            if twin == usize::MAX || twin >= self.half_edges.len() { continue; }
            if self.half_edges[twin].removed { continue; }

            let v0 = self.source(he_idx);
            let v1 = self.half_edges[he_idx].vertex;
            let v2 = self.half_edges[self.half_edges[he_idx].next].vertex;
            let v3 = self.half_edges[self.half_edges[twin].next].vertex;

            // Simple valence targets (interior)
            // let before =
            //     self.vertex_valence(v0) as i32 +
            //     self.vertex_valence(v1) as i32 +
            //     self.vertex_valence(v2) as i32 +
            //     self.vertex_valence(v3) as i32;

            if !self.should_flip_edge(he_idx) { continue; }
            if self.flip_edge(he_idx).is_err() { continue; }

            // Optional mild valence monotonicity: accept always since flip done;
            // (Could revert if worsens, but revert cost > gain here.)
            let _after =
                self.vertex_valence(v0) as i32 +
                self.vertex_valence(v1) as i32 +
                self.vertex_valence(v2) as i32 +
                self.vertex_valence(v3) as i32;

            flips += 1;
        }
        flips
    }

    /// Split edges longer than split_len with per-vertex cooldown (single pass).
    fn split_long_edges(
        &mut self,
        split_len: &T,
        iter: i32,
        v_last_op: &mut [i32],
    ) -> usize
    where
        for<'a> &'a T: std::ops::Mul<&'a T, Output = T> + std::ops::Div<&'a T, Output = T>,
    {
        let thresh2 = split_len * split_len;
        let mut candidates = Vec::new();
        candidates.reserve(self.half_edges.len()/4);

        for h in 0..self.half_edges.len() {
            let he = &self.half_edges[h];
            if he.removed || he.twin == usize::MAX || h >= he.twin { continue; }
            let v0 = self.source(h);
            let v1 = he.vertex;
            if self.vertices[v0].removed || self.vertices[v1].removed { continue; }
            if v_last_op[v0] == iter || v_last_op[v1] == iter { continue; } // cooldown inside iteration

            let p0 = &self.vertices[v0].position;
            let p1 = &self.vertices[v1].position;
            let d  = (p1 - p0).as_vector();
            if d.norm2() > thresh2 {
                candidates.push(h);
            }
        }

        if candidates.is_empty() { return 0; }
        let mut aabb = self.build_face_tree();

        let mut splits = 0usize;
        for h in candidates {
            if h >= self.half_edges.len() { continue; }
            if self.half_edges[h].removed { continue; }
            let v0 = self.source(h);
            let v1 = self.half_edges[h].vertex;
            if self.vertices[v0].removed || self.vertices[v1].removed { continue; }
            if v_last_op[v0] == iter || v_last_op[v1] == iter { continue; }

            let mid = {
                let p0 = &self.vertices[v0].position;
                let p1 = &self.vertices[v1].position;
                (&(*p0) + &(*p1)).as_vector().scale(&T::from_num_den(1,2)).0
            };

            if self.split_edge(&mut aabb, h, &mid, false).is_ok() {
                splits += 1;
                v_last_op[v0] = iter;
                v_last_op[v1] = iter;
            }
        }
        splits
    }

    /// Collapse edges shorter than collapse_len with per-vertex cooldown and deduplication.
    fn collapse_short_edges(
        &mut self,
        collapse_len: &T,
        iter: i32,
        v_last_op: &mut [i32],
    ) -> usize
    where
        Vector<T,N>: VectorOps<T, N> + Cross3<T>,
        for<'a> &'a T: std::ops::Mul<&'a T, Output = T>
    {
        let thresh2 = collapse_len * collapse_len;
        use ahash::AHashSet;
        let mut set: AHashSet<(usize,usize)> = AHashSet::with_capacity(self.half_edges.len()/8);

        for h in 0..self.half_edges.len() {
            let he = &self.half_edges[h];
            if he.removed { continue; }
            let twin = he.twin;
            if twin == usize::MAX { continue; }
            if h >= twin { continue; }
            let v0 = self.source(h);
            let v1 = he.vertex;
            if self.vertices[v0].removed || self.vertices[v1].removed { continue; }
            if v_last_op[v0] == iter || v_last_op[v1] == iter { continue; }

            let p0 = &self.vertices[v0].position;
            let p1 = &self.vertices[v1].position;
            let d  = (p1 - p0).as_vector();
            if d.norm2() < thresh2 {
                let (a,b) = if v0 < v1 {(v0,v1)} else {(v1,v0)};
                set.insert((a,b));
            }
        }

        let mut collapsed = 0usize;
        for (a,b) in set {
            if self.vertices[a].removed || self.vertices[b].removed { continue; }
            if v_last_op[a] == iter || v_last_op[b] == iter { continue; }

            // Keep a (arbitrary); if rejected, skip
            if self.collapse_edge(a,b).is_ok() {
                collapsed += 1;
                v_last_op[a] = iter;
                v_last_op[b] = iter;
            }
        }
        collapsed
    }

    /// Improved isotropic remesh with hysteresis and quality-based flips.
    pub fn isotropic_remesh(&mut self, options: &RemeshOptions<T>) -> Result<(), RemeshError>
    where
        Vector<T,N>: VectorOps<T, N> + Cross3<T>,
        for<'a> &'a T:
            std::ops::Mul<&'a T, Output = T> +
            std::ops::Div<&'a T, Output = T> +
            std::ops::Add<&'a T, Output = T>,
        T: PartialOrd + Clone + From<f64>,
    {
        // Derive initial target if zero / degenerate
        let rms = self.rms_edge_length_f64();
        let target_len = if let Some(tl) = options.target_edge_length.to_f64() {
            if tl > 0.0 { tl } else { rms }
        } else { rms };

        // Hysteresis thresholds (can be tuned)
        let split_factor   = 1.45; // > target ⇒ split
        let collapse_factor= 0.75; // < target ⇒ collapse

        let mut v_last_op: Vec<i32> = vec![-1; self.vertices.len()];

        for iter in 0..options.max_iterations {
            // Reallocate cooldown array if vertex count grew (start of iteration)
            if v_last_op.len() < self.vertices.len() {
                v_last_op.resize(self.vertices.len(), -1);
            }

            let split_len    = T::from(target_len * split_factor);
            let collapse_len = T::from(target_len * collapse_factor);

            let splits    = self.split_long_edges(&split_len, iter as i32, &mut v_last_op);

            // ensure v_last_op covers vertices created by splits before collapses
            if v_last_op.len() < self.vertices.len() {
                v_last_op.resize(self.vertices.len(), -1);
            }

            let collapses = self.collapse_short_edges(&collapse_len, iter as i32, &mut v_last_op);
            let flips     = self.flip_edges();
            let _smoothed  = self.smooth_vertices(2, &split_len); // 2 sweeps; conservative

            // Convergence heuristic: no structural change and low activity
            if splits == 0 && collapses == 0 && flips < self.half_edges.len()/200 {
                break;
            }
        }
        Ok(())
    }
}
