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
        aabb::Aabb,
        spatial_element::SpatialElement,
        util::{f64_next_down, f64_next_up},
    },
    numeric::{cgar_f64::CgarF64, cgar_rational::CgarRational, scalar::Scalar},
};
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

/// A simple (unbalanced) AABB‐tree of generic data `D`.
pub enum AabbTree<T: Scalar, const N: usize, P: SpatialElement<T, N>, D> {
    Leaf {
        aabb: Aabb<T, N, P>, // exact
        data: Arc<D>,
        valid: bool,
        amin: [f64; N], // outward-safe approx
        amax: [f64; N],
    },
    Node {
        aabb: Aabb<T, N, P>, // exact (built once; feel free to make this Option if you want)
        left: Box<AabbTree<T, N, P, D>>,
        right: Box<AabbTree<T, N, P, D>>,
        valid_count: usize,
        total_count: usize,
        amin: [f64; N], // outward-safe approx union
        amax: [f64; N],
    },
}

impl<T: Scalar, const N: usize, P: SpatialElement<T, N>, D> AabbTree<T, N, P, D>
where
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    #[inline(always)]
    fn approx_bounds(&self) -> ([f64; N], [f64; N]) {
        match self {
            AabbTree::Leaf { amin, amax, .. } => (*amin, *amax),
            AabbTree::Node { amin, amax, .. } => (*amin, *amax),
        }
    }

    /// Build an AABB‐tree over `(aabb, data)` pairs via recursive median split.
    /// Partitioning decisions use approximate centers/extents only (fast),
    /// while node AABBs are exact unions.
    pub fn build(mut items: Vec<(Aabb<T, N, P>, D)>) -> Self
    where
        T: Scalar,
    {
        if items.is_empty() {
            panic!("Cannot build tree from empty items");
        }
        // Recursively partition by median of approximate center along longest approximate axis.
        Self::build_median(&mut items)
    }

    pub fn build_with_lookup(items: Vec<(Aabb<T, N, P>, D)>) -> (Self, Vec<Aabb<T, N, P>>)
    where
        D: Copy,
        T: From<CgarRational>,
    {
        let mut aabb_lookup = vec![Aabb::default(); items.len()];
        for (i, (aabb, _)) in items.iter().enumerate() {
            aabb_lookup[i] = aabb.clone();
        }
        let tree = Self::build(items);
        (tree, aabb_lookup)
    }

    fn build_median(items: &mut Vec<(Aabb<T, N, P>, D)>) -> Self
    where
        T: Scalar,
    {
        let n = items.len();
        if n == 1 {
            let (aabb, data) = items.pop().unwrap();
            let (amin, amax) = approx_from_exact(&aabb);
            return AabbTree::Leaf {
                aabb,
                data: Arc::new(data),
                valid: true,
                amin,
                amax,
            };
        }

        let axis = choose_axis_approx::<T, N, P, D>(items);
        let mid = n / 2;

        // Use pdqsort-based select_nth with optimized comparison
        items.select_nth_unstable_by(mid, |a, b| compare_centers(a, b, axis));

        let mut right_items = items.split_off(mid);
        let left_child = Box::new(Self::build_median(items));
        let right_child = Box::new(Self::build_median(&mut right_items));

        // Try approximate union first
        let node_aabb = left_child.aabb().union(right_child.aabb());

        let total_items = left_child.size() + right_child.size();
        let (lmn, lmx) = left_child.approx_bounds();
        let (rmn, rmx) = right_child.approx_bounds();
        let (amin, amax) = approx_union(&lmn, &lmx, &rmn, &rmx);

        AabbTree::Node {
            aabb: node_aabb,
            left: left_child,
            right: right_child,
            valid_count: total_items,
            total_count: total_items,
            amin,
            amax,
        }
    }

    fn build_binary_tree(mut items: Vec<(Aabb<T, N, P>, D)>) -> Self
    where
        T: Scalar + From<CgarRational>,
    {
        if items.len() == 1 {
            let (aabb, data) = items.pop().unwrap();
            let (amin, amax) = approx_from_exact(&aabb);
            return AabbTree::Leaf {
                aabb,
                data: Arc::new(data),
                valid: true,
                amin,
                amax,
            };
        }

        let mid = items.len() / 2;
        let right_items = items.split_off(mid);

        let left_child = Box::new(Self::build_binary_tree(items));
        let right_child = Box::new(Self::build_binary_tree(right_items));

        let node_aabb = left_child.aabb().union(right_child.aabb());
        let total_items = left_child.size() + right_child.size();

        let (lmn, lmx) = left_child.approx_bounds();
        let (rmn, rmx) = right_child.approx_bounds();
        let (amin, amax) = approx_union(&lmn, &lmx, &rmn, &rmx);

        AabbTree::Node {
            aabb: node_aabb,
            left: left_child,
            right: right_child,
            valid_count: total_items,
            total_count: total_items,
            amin,
            amax,
        }
    }

    /// Get AABB of this node/leaf
    fn aabb(&self) -> &Aabb<T, N, P> {
        match self {
            AabbTree::Leaf { aabb, .. } => aabb,
            AabbTree::Node { aabb, .. } => aabb,
        }
    }

    /// Collect all `&D` whose AABB intersects `query`.
    pub fn query<'a>(&'a self, query: &Aabb<T, N, P>, out: &mut Vec<&'a D>) {
        let (qmn, qmx) = approx_from_exact(query);
        self.query_impl(&qmn, &qmx, query, out);
    }

    fn query_impl<'a>(
        &'a self,
        qmn: &[f64; N],
        qmx: &[f64; N],
        q_exact: &Aabb<T, N, P>,
        out: &mut Vec<&'a D>,
    ) {
        match self {
            AabbTree::Leaf {
                aabb,
                data,
                valid: _valid,
                amin,
                amax,
            } => {
                // First: cheap approximate test
                if !intersects_approx::<N>(amin, amax, qmn, qmx) {
                    return;
                }
                // Second: try approximate AABB test if possible
                if let Some(definitely_no_intersect) = aabb.intersects_approx(q_exact) {
                    if !definitely_no_intersect {
                        return;
                    }
                }

                // Third: exact test only if needed
                if aabb.intersects(q_exact) {
                    out.push(data);
                }
            }
            AabbTree::Node {
                left,
                right,
                amin,
                amax,
                ..
            } => {
                if !intersects_approx::<N>(amin, amax, qmn, qmx) {
                    return;
                }
                left.query_impl(qmn, qmx, q_exact, out);
                right.query_impl(qmn, qmx, q_exact, out);
            }
        }
    }

    /// Mark entries as invalid (O(log n))
    pub fn invalidate(&mut self, target: &D) -> usize
    where
        D: PartialEq,
    {
        match self {
            AabbTree::Leaf { data, valid, .. } => {
                if *valid && data.as_ref() == target {
                    *valid = false;
                    1
                } else {
                    0
                }
            }
            AabbTree::Node {
                left,
                right,
                valid_count,
                ..
            } => {
                let rm = left.invalidate(target) + right.invalidate(target);
                if rm > 0 {
                    *valid_count -= rm;
                }
                rm
            }
        }
    }

    pub fn insert(&mut self, new_aabb: Aabb<T, N, P>, new_data: D) {
        match self {
            AabbTree::Leaf {
                aabb,
                data,
                valid,
                amin,
                amax,
            } => {
                let left_leaf = AabbTree::Leaf {
                    aabb: aabb.clone(),
                    data: data.clone(),
                    valid: *valid,
                    amin: *amin,
                    amax: *amax,
                };
                let (nmn, nmx) = approx_from_exact(&new_aabb);
                let right_leaf = AabbTree::Leaf {
                    aabb: new_aabb.clone(),
                    data: Arc::new(new_data),
                    valid: true,
                    amin: nmn,
                    amax: nmx,
                };

                let axis = longest_axis_from_cached::<N>(amin, amax);
                let c_left = 0.5 * (amin[axis] + amax[axis]);
                let c_right = approx_center_axis::<T, N, P>(&new_aabb, axis);

                let (left_child, right_child) = if c_right < c_left {
                    (Box::new(right_leaf), Box::new(left_leaf))
                } else {
                    (Box::new(left_leaf), Box::new(right_leaf))
                };

                let node_aabb = left_child.aabb().union(right_child.aabb()); // exact once here
                let total_items = left_child.size() + right_child.size();
                let (lmn, lmx) = left_child.approx_bounds();
                let (rmn, rmx) = right_child.approx_bounds();
                let (amin_u, amax_u) = approx_union(&lmn, &lmx, &rmn, &rmx);

                *self = AabbTree::Node {
                    aabb: node_aabb,
                    left: left_child,
                    right: right_child,
                    valid_count: total_items,
                    total_count: total_items,
                    amin: amin_u,
                    amax: amax_u,
                };
            }
            AabbTree::Node {
                aabb,
                left,
                right,
                valid_count,
                total_count,
                amin,
                amax,
            } => {
                *valid_count += 1;
                *total_count += 1;

                // Decide side purely from approx (no exact math)
                let (lmn, lmx) = left.approx_bounds();
                let (rmn, rmx) = right.approx_bounds();
                let (nmn, nmx) = approx_from_exact(&new_aabb);

                let left_cost = approx_union_sum_extents::<N>(&lmn, &lmx, &nmn, &nmx)
                    - approx_sum_extents::<N>(&lmn, &lmx);
                let right_cost = approx_union_sum_extents::<N>(&rmn, &rmx, &nmn, &nmx)
                    - approx_sum_extents::<N>(&rmn, &rmx);

                // Update approx bounds at this node (cheap)
                let (amin_u, amax_u) = approx_union(amin, amax, &nmn, &nmx);
                *amin = amin_u;
                *amax = amax_u;

                // Optional: comment this out if inserts are frequent and you prefer periodic refit:
                *aabb = aabb.union(&new_aabb); // exact

                if left_cost <= right_cost {
                    left.insert(new_aabb, new_data);
                } else {
                    right.insert(new_aabb, new_data);
                }
            }
        }
    }

    pub fn query_valid<'a>(&'a self, query: &Aabb<T, N, P>, out: &mut Vec<&'a D>) {
        let (qmn, qmx) = approx_from_exact(query); // once
        self.query_valid_impl(&qmn, &qmx, query, out);
    }

    fn query_valid_impl<'a>(
        &'a self,
        qmn: &[f64; N],
        qmx: &[f64; N],
        q_exact: &Aabb<T, N, P>,
        out: &mut Vec<&'a D>,
    ) {
        match self {
            AabbTree::Leaf {
                aabb,
                data,
                valid,
                amin,
                amax,
            } => {
                if !*valid {
                    return;
                }
                if !intersects_approx::<N>(amin, amax, qmn, qmx) {
                    return;
                }
                if aabb.intersects(q_exact) {
                    // exact only here
                    out.push(data);
                }
            }
            AabbTree::Node {
                left,
                right,
                valid_count,
                amin,
                amax,
                ..
            } => {
                if *valid_count == 0 {
                    return;
                }
                if !intersects_approx::<N>(amin, amax, qmn, qmx) {
                    return;
                }
                left.query_valid_impl(qmn, qmx, q_exact, out);
                right.query_valid_impl(qmn, qmx, q_exact, out);
            }
        }
    }

    /// Check if tree needs rebuilding (O(1))
    pub fn needs_rebuild(&self) -> bool {
        match self {
            AabbTree::Leaf { valid, .. } => !valid,
            AabbTree::Node {
                valid_count,
                total_count,
                ..
            } => {
                (*valid_count as f64 / *total_count as f64) < 0.5 // 50% threshold
            }
        }
    }

    /// Get tree size for balancing decisions (O(1))
    fn size(&self) -> usize {
        match self {
            AabbTree::Leaf { .. } => 1,
            AabbTree::Node { total_count, .. } => *total_count,
        }
    }

    /// Compact tree by removing invalid entries (O(n))
    pub fn compact(&mut self) -> Option<Self>
    where
        T: From<CgarRational>,
        D: Clone,
    {
        let mut valid_items = Vec::new();
        self.collect_valid(&mut valid_items);

        if valid_items.is_empty() {
            return None;
        }

        Some(Self::build(valid_items))
    }

    /// Collect all valid (aabb, data) pairs
    fn collect_valid(&self, out: &mut Vec<(Aabb<T, N, P>, D)>)
    where
        D: Clone,
    {
        match self {
            AabbTree::Leaf {
                aabb, data, valid, ..
            } => {
                if *valid {
                    out.push((aabb.clone(), data.as_ref().clone()));
                }
            }
            AabbTree::Node { left, right, .. } => {
                left.collect_valid(out);
                right.collect_valid(out);
            }
        }
    }

    pub fn get_aabb(&self, data_index: usize) -> Option<&Aabb<T, N, P>>
    where
        D: PartialEq<usize>,
    {
        self.find_aabb_by_data(data_index)
    }

    fn find_aabb_by_data(&self, target: usize) -> Option<&Aabb<T, N, P>>
    where
        D: PartialEq<usize>,
    {
        match self {
            AabbTree::Leaf { aabb, data, .. } => {
                if data.as_ref() == &target {
                    Some(aabb)
                } else {
                    None
                }
            }
            AabbTree::Node { left, right, .. } => left
                .find_aabb_by_data(target)
                .or_else(|| right.find_aabb_by_data(target)),
        }
    }
}

#[inline(always)]
fn as_f64<T: Scalar>(x: &T) -> f64 {
    let v: CgarF64 = x.ref_into();
    if v.0.is_nan() {
        0.0
    } else if v.0.is_infinite() {
        v.0.signum() * 1.0e308 // conservative huge finite
    } else {
        v.0
    }
}

#[inline(always)]
fn approx_center_axis<T: Scalar, const N: usize, P: SpatialElement<T, N>>(
    aabb: &Aabb<T, N, P>,
    axis: usize,
) -> f64 {
    // center ~= (min + max) * 0.5 using f64 only; never builds LazyExact nodes
    let mn = as_f64(&aabb.min[axis]);
    let mx = as_f64(&aabb.max[axis]);
    0.5 * (mn + mx)
}

#[inline(always)]
fn approx_sum_extents<const N: usize>(mins: &[f64; N], maxs: &[f64; N]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..N {
        s += (maxs[i] - mins[i]).abs();
    }
    s
}

#[inline(always)]
fn approx_union_sum_extents<const N: usize>(
    a_mins: &[f64; N],
    a_maxs: &[f64; N],
    b_mins: &[f64; N],
    b_maxs: &[f64; N],
) -> f64 {
    let mut s = 0.0f64;
    for i in 0..N {
        let mn = a_mins[i].min(b_mins[i]);
        let mx = a_maxs[i].max(b_maxs[i]);
        s += (mx - mn).abs();
    }
    s
}

#[inline(always)]
fn choose_axis_approx<T: Scalar, const N: usize, P: SpatialElement<T, N>, D>(
    items: &[(Aabb<T, N, P>, D)],
) -> usize {
    // Compute global approx bounds and pick axis with largest extent
    let mut gmin = [f64::INFINITY; N];
    let mut gmax = [f64::NEG_INFINITY; N];
    for (aabb, _) in items {
        for i in 0..N {
            let mn = as_f64(&aabb.min[i]);
            let mx = as_f64(&aabb.max[i]);
            if mn < gmin[i] {
                gmin[i] = mn;
            }
            if mx > gmax[i] {
                gmax[i] = mx;
            }
        }
    }
    let mut axis = 0usize;
    let mut best = gmax[0] - gmin[0];
    for i in 1..N {
        let e = gmax[i] - gmin[i];
        if e > best {
            best = e;
            axis = i;
        }
    }
    axis
}

#[inline(always)]
fn approx_union<const N: usize>(
    a_mn: &[f64; N],
    a_mx: &[f64; N],
    b_mn: &[f64; N],
    b_mx: &[f64; N],
) -> ([f64; N], [f64; N]) {
    let mut mn = [0.0; N];
    let mut mx = [0.0; N];
    for i in 0..N {
        mn[i] = a_mn[i].min(b_mn[i]);
        mx[i] = a_mx[i].max(b_mx[i]);
    }
    (mn, mx)
}

#[inline(always)]
fn intersects_approx<const N: usize>(
    a_mn: &[f64; N],
    a_mx: &[f64; N],
    b_mn: &[f64; N],
    b_mx: &[f64; N],
) -> bool {
    for i in 0..N {
        if a_mx[i] < b_mn[i] || b_mx[i] < a_mn[i] {
            return false;
        }
    }
    true
}

#[inline(always)]
fn longest_axis_from_cached<const N: usize>(mins: &[f64; N], maxs: &[f64; N]) -> usize {
    let mut axis = 0usize;
    let mut best = maxs[0] - mins[0];
    for i in 1..N {
        let e = maxs[i] - mins[i];
        if e > best {
            best = e;
            axis = i;
        }
    }
    axis
}

#[inline(always)]
fn approx_from_exact<T: Scalar, const N: usize, P: SpatialElement<T, N>>(
    a: &Aabb<T, N, P>,
) -> ([f64; N], [f64; N]) {
    let mut mn = [0.0; N];
    let mut mx = [0.0; N];
    for i in 0..N {
        // outward-safe: step one ULP past the rounded extremes
        let lo = as_f64(&a.min[i]);
        let hi = as_f64(&a.max[i]);
        mn[i] = f64_next_down(lo);
        mx[i] = f64_next_up(hi);
    }
    (mn, mx)
}

#[inline(always)]
fn approx_center_axis_fast<T: Scalar, const N: usize, P: SpatialElement<T, N>>(
    aabb: &Aabb<T, N, P>,
    axis: usize,
) -> f64 {
    // Try fast path first
    if let Some(mn) = aabb.min[axis].as_f64_fast() {
        if let Some(mx) = aabb.max[axis].as_f64_fast() {
            return 0.5 * (mn + mx);
        }
    }

    // Fallback to current approximation method
    approx_center_axis(aabb, axis)
}

#[inline(always)]
fn compare_centers<T: Scalar, const N: usize, P: SpatialElement<T, N>, D>(
    a: &(Aabb<T, N, P>, D),
    b: &(Aabb<T, N, P>, D),
    axis: usize,
) -> std::cmp::Ordering {
    let ca = approx_center_axis_fast(&a.0, axis);
    let cb = approx_center_axis_fast(&b.0, axis);
    ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
}
