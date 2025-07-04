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

use crate::{geometry::aabb::Aabb, operations::Abs};
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

/// A simple (unbalanced) AABB‐tree of generic data `D`.
pub enum AabbTree<T, P, D>
where
    P: crate::mesh::point_trait::PointTrait<T>,
{
    Leaf {
        aabb: Aabb<T, P>,
        data: Arc<D>,
    },
    Node {
        aabb: Aabb<T, P>,
        left: Box<AabbTree<T, P, D>>,
        right: Box<AabbTree<T, P, D>>,
    },
}

impl<T, P, D> AabbTree<T, P, D>
where
    T: PartialOrd + Clone + Abs + From<f64>,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
    P: crate::mesh::point_trait::PointTrait<T> + crate::geometry::aabb::FromCoords<T>,
{
    /// Build an AABB‐tree over `(aabb, data)` pairs via recursive median split.
    pub fn build(mut items: Vec<(Aabb<T, P>, D)>) -> Self {
        fn recurse<T, P, D>(mut items: Vec<(Aabb<T, P>, D)>) -> AabbTree<T, P, D>
        where
            T: PartialOrd + Clone + Abs + From<f64>,
            for<'a> &'a T: Add<&'a T, Output = T>
                + Sub<&'a T, Output = T>
                + Mul<&'a T, Output = T>
                + Div<&'a T, Output = T>,
            P: crate::mesh::point_trait::PointTrait<T> + crate::geometry::aabb::FromCoords<T>,
        {
            // compute bounding box of all items
            let mut node_aabb = items[0].0.clone();
            for (aabb, _) in &items[1..] {
                node_aabb = node_aabb.union(aabb);
            }

            if items.len() == 1 {
                let (a, d) = items.pop().unwrap();
                return AabbTree::Leaf {
                    aabb: a,
                    data: Arc::new(d),
                };
            }

            // split along longest axis at median
            let axis = node_aabb.longest_axis();
            items.sort_by(|(a1, _), (a2, _)| {
                a1.center(axis)
                    .partial_cmp(&a2.center(axis))
                    .unwrap_or(Ordering::Equal)
            });
            let mid = items.len() / 2;
            let right = items.split_off(mid);
            let left = items;

            AabbTree::Node {
                aabb: node_aabb,
                left: Box::new(recurse(left)),
                right: Box::new(recurse(right)),
            }
        }

        recurse(items)
    }

    /// Collect all `&D` whose AABB intersects `query`.
    pub fn query<'a>(&'a self, query: &Aabb<T, P>, out: &mut Vec<&'a D>) {
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
