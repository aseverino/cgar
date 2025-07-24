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
    },
    Node {
        aabb: Aabb<T, N, P>,
        left: Box<AabbTree<T, N, P, D>>,
        right: Box<AabbTree<T, N, P, D>>,
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

        AabbTree::Node {
            aabb: node_aabb,
            left: left_child,
            right: right_child,
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
            AabbTree::Leaf { aabb, data } => {
                if aabb.intersects(query) {
                    out.push(data);
                }
            }
            AabbTree::Node { aabb, left, right } => {
                if aabb.intersects(query) {
                    left.query(query, out);
                    right.query(query, out);
                }
            }
        }
    }
}
