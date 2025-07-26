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
    geometry::{aabb::Aabb, spatial_element::SpatialElement},
    numeric::{cgar_rational::CgarRational, scalar::Scalar},
};
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

/// A simple (unbalanced) AABB‐tree of generic data `D`.
pub enum AabbTree<T: Scalar, const N: usize, P: SpatialElement<T, N>, D> {
    Leaf {
        aabb: Aabb<T, N, P>,
        data: Arc<D>,
        valid: bool, // Track validity without structural changes
    },
    Node {
        aabb: Aabb<T, N, P>,
        left: Box<AabbTree<T, N, P, D>>,
        right: Box<AabbTree<T, N, P, D>>,
        valid_count: usize, // Count of valid children
        total_count: usize, // Total children
    },
}

impl<T: Scalar, const N: usize, P: SpatialElement<T, N>, D> AabbTree<T, N, P, D>
where
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    /// Build an AABB‐tree over `(aabb, data)` pairs via recursive median split.
    pub fn build(mut items: Vec<(Aabb<T, N, P>, D)>) -> Self
    where
        T: Scalar + From<CgarRational>,
    {
        if items.is_empty() {
            panic!("Cannot build tree from empty items");
        }

        // Single-pass global bounds computation + axis selection
        let mut global_min = items[0].0.min().coords().clone();
        let mut global_max = items[0].0.max().coords().clone();

        for (aabb, _) in &items[1..] {
            let min_coords = aabb.min().coords();
            let max_coords = aabb.max().coords();
            for i in 0..N {
                if min_coords[i] < global_min[i] {
                    global_min[i] = min_coords[i].clone();
                }
                if max_coords[i] > global_max[i] {
                    global_max[i] = max_coords[i].clone();
                }
            }
        }

        // Find longest axis without creating AABB object
        let longest_axis = (0..N)
            .max_by(|&i, &j| {
                (&global_max[i] - &global_min[i])
                    .partial_cmp(&(&global_max[j] - &global_min[j]))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap_or(0);

        // Sort by precomputed centers to avoid repeated computation
        items.sort_unstable_by(|(a1, _), (a2, _)| {
            a1.center(longest_axis)
                .partial_cmp(&a2.center(longest_axis))
                .unwrap_or(Ordering::Equal)
        });

        Self::build_binary_tree(&mut items[..])
    }

    /// Build balanced binary tree using simple median split
    fn build_binary_tree(items: &mut [(Aabb<T, N, P>, D)]) -> Self
    where
        T: Scalar + From<CgarRational>,
    {
        if items.len() == 1 {
            let (aabb, data) = std::mem::replace(&mut items[0], unsafe { std::mem::zeroed() });
            return AabbTree::Leaf {
                aabb,
                data: Arc::new(data),
                valid: true,
            };
        }

        // Simple median split - no recomputation needed since items are pre-sorted
        let mid = items.len() / 2;
        let (left_items, right_items) = items.split_at_mut(mid);

        // Build children first
        let left_child = Box::new(Self::build_binary_tree(left_items));
        let right_child = Box::new(Self::build_binary_tree(right_items));

        // Compute node AABB from children (only 1 union operation)
        let node_aabb = left_child.aabb().union(right_child.aabb());
        let total_items = left_items.len() + right_items.len();

        AabbTree::Node {
            aabb: node_aabb,
            left: left_child,
            right: right_child,
            valid_count: total_items,
            total_count: total_items,
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
        match self {
            AabbTree::Leaf { aabb, data, valid } => {
                if *valid && aabb.intersects(query) {
                    // Check validity
                    out.push(data);
                }
            }
            AabbTree::Node {
                aabb,
                left,
                right,
                valid_count,
                ..
            } => {
                if *valid_count > 0 && aabb.intersects(query) {
                    // Check valid_count
                    left.query(query, out);
                    right.query(query, out);
                }
            }
        }
    }

    /// Mark entries as invalid (O(log n))
    pub fn invalidate(&mut self, target: &D) -> bool
    where
        for<'a> &'a T: Add<&'a T, Output = T>
            + Sub<&'a T, Output = T>
            + Mul<&'a T, Output = T>
            + Div<&'a T, Output = T>,
        D: PartialEq,
    {
        match self {
            AabbTree::Leaf { data, valid, .. } => {
                if data.as_ref() == target && *valid {
                    *valid = false;
                    return true;
                }
                false
            }
            AabbTree::Node {
                left,
                right,
                valid_count,
                ..
            } => {
                let left_removed = left.invalidate(target);
                let right_removed = right.invalidate(target);

                if left_removed || right_removed {
                    *valid_count -= 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Add new entries to existing tree (O(log n))
    pub fn insert(&mut self, new_aabb: Aabb<T, N, P>, new_data: D) {
        match self {
            AabbTree::Leaf { aabb, data, valid } => {
                // Take ownership safely
                let old_aabb = aabb.clone();
                let old_data = data.clone();
                let old_valid = *valid;

                let new_leaf = AabbTree::Leaf {
                    aabb: new_aabb.clone(),
                    data: Arc::new(new_data),
                    valid: true,
                };

                let old_leaf = AabbTree::Leaf {
                    aabb: old_aabb.clone(),
                    data: old_data,
                    valid: old_valid,
                };

                let combined_aabb = old_aabb.union(&new_aabb);

                *self = AabbTree::Node {
                    aabb: combined_aabb,
                    left: Box::new(old_leaf),
                    right: Box::new(new_leaf),
                    valid_count: if old_valid { 2 } else { 1 },
                    total_count: 2,
                };
            }
            AabbTree::Node {
                aabb,
                left,
                right,
                valid_count,
                total_count,
            } => {
                *aabb = aabb.union(&new_aabb);
                *valid_count += 1;
                *total_count += 1;

                if left.size() <= right.size() {
                    left.insert(new_aabb, new_data);
                } else {
                    right.insert(new_aabb, new_data);
                }
            }
        }
    }

    /// Query with automatic invalid filtering (O(log n))
    pub fn query_valid<'a>(&'a self, query: &Aabb<T, N, P>, out: &mut Vec<&'a D>) {
        match self {
            AabbTree::Leaf { aabb, data, valid } => {
                if *valid && aabb.intersects(query) {
                    out.push(data);
                }
            }
            AabbTree::Node {
                aabb,
                left,
                right,
                valid_count,
                ..
            } => {
                if *valid_count > 0 && aabb.intersects(query) {
                    left.query_valid(query, out);
                    right.query_valid(query, out);
                }
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
            AabbTree::Leaf { aabb, data, valid } => {
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
}
