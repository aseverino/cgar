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

use cgar::geometry::aabb::Aabb;
use cgar::geometry::aabb_tree::AabbTree;
use cgar::geometry::{FromCoords, Point2};
use cgar::mesh::point_trait::PointTrait;
use cgar::numeric::cgar_f64::CgarF64;

#[test]
fn test_aabb_from_union_and_intersection_2() {
    // Test from_points
    let p1 = Point2 {
        x: CgarF64(0.0),
        y: CgarF64(1.0),
    };
    let p2 = Point2 {
        x: CgarF64(2.0),
        y: CgarF64(-1.0),
    };
    let a = Aabb::from_points(&p1, &p2);

    assert_eq!(a.min.x, CgarF64(0.0));
    assert_eq!(a.min.y, CgarF64(-1.0));
    assert_eq!(a.max.x, CgarF64(2.0));
    assert_eq!(a.max.y, CgarF64(1.0));

    // Test union
    let b = Aabb::from_points(
        &Point2 {
            x: CgarF64(0.5),
            y: CgarF64(-0.5),
        },
        &Point2 {
            x: CgarF64(2.0),
            y: CgarF64(0.5),
        },
    );
    let u = a.union(&b);
    assert_eq!(u.min.x, CgarF64(0.0));
    assert_eq!(u.min.y, CgarF64(-1.0));
    assert_eq!(u.max.x, CgarF64(2.0));
    assert_eq!(u.max.y, CgarF64(1.0));

    // Test intersects
    let c = Aabb::from_points(
        &Point2 {
            x: CgarF64(2.5),
            y: CgarF64(2.5),
        },
        &Point2 {
            x: CgarF64(3.0),
            y: CgarF64(3.0),
        },
    );
    assert!(a.intersects(&b));
    assert!(!a.intersects(&c));
}

#[test]
fn test_aabb_tree_build_and_query_2() {
    // Build three AABBs labeled 1,2,3
    let items = vec![
        (
            Aabb::from_points(
                &Point2 {
                    x: CgarF64(0.0),
                    y: CgarF64(0.0),
                },
                &Point2 {
                    x: CgarF64(1.0),
                    y: CgarF64(1.0),
                },
            ),
            1,
        ),
        (
            Aabb::from_points(
                &Point2 {
                    x: CgarF64(1.5),
                    y: CgarF64(1.5),
                },
                &Point2 {
                    x: CgarF64(2.5),
                    y: CgarF64(2.5),
                },
            ),
            2,
        ),
        (
            Aabb::from_points(
                &Point2 {
                    x: CgarF64(-1.0),
                    y: CgarF64(-1.0),
                },
                &Point2 {
                    x: CgarF64(-0.5),
                    y: CgarF64(-0.5),
                },
            ),
            3,
        ),
    ];

    // Build the tree
    let tree = AabbTree::build(items.clone());

    // Query for boxes overlapping the region [0.5,0.5]–[2.0,2.0]
    let mut hits = Vec::new();
    let query = Aabb::from_points(
        &Point2 {
            x: CgarF64(0.5),
            y: CgarF64(0.5),
        },
        &Point2 {
            x: CgarF64(2.0),
            y: CgarF64(2.0),
        },
    );
    tree.query(&query, &mut hits);

    // We expect to hit items 1 and 2
    let mut result: Vec<_> = hits.into_iter().cloned().collect();
    result.sort();
    assert_eq!(result, vec![1, 2]);
}

#[derive(Clone, Debug)]
struct TestPoint3(CgarF64, CgarF64, CgarF64);

impl TestPoint3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        TestPoint3(x.into(), y.into(), z.into())
    }
}

impl PointTrait<CgarF64> for TestPoint3 {
    fn dimensions() -> usize {
        3
    }
    fn coord(&self, axis: usize) -> CgarF64 {
        match axis {
            0 => self.0.clone(),
            1 => self.1.clone(),
            2 => self.2.clone(),
            _ => panic!("Invalid axis"),
        }
    }
}

impl FromCoords<CgarF64> for TestPoint3 {
    fn from_coords(min_coords: Vec<CgarF64>, max_coords: Vec<CgarF64>) -> Aabb<CgarF64, Self> {
        let min = TestPoint3(
            min_coords[0].clone(),
            min_coords[1].clone(),
            min_coords[2].clone(),
        );
        let max = TestPoint3(
            max_coords[0].clone(),
            max_coords[1].clone(),
            max_coords[2].clone(),
        );
        Aabb::new(min, max)
    }
}

#[test]
fn test_aabb_3d_from_union_and_intersection() {
    // from_points on diagonal corners
    let p1 = TestPoint3::new(0.0, 1.0, -1.0);
    let p2 = TestPoint3::new(2.0, -1.0, 3.0);
    let a = Aabb::from_points(&p1, &p2);

    assert_eq!(a.min.0, CgarF64(0.0));
    assert_eq!(a.min.1, CgarF64(-1.0));
    assert_eq!(a.min.2, CgarF64(-1.0));
    assert_eq!(a.max.0, CgarF64(2.0));
    assert_eq!(a.max.1, CgarF64(1.0));
    assert_eq!(a.max.2, CgarF64(3.0));

    // union with a smaller box inside
    let b = Aabb::from_points(
        &TestPoint3::new(0.5, -0.5, 0.0),
        &TestPoint3::new(1.5, 0.5, 2.0),
    );
    let u = a.union(&b);
    assert_eq!(u.min.0, CgarF64(0.0));
    assert_eq!(u.min.1, CgarF64(-1.0));
    assert_eq!(u.min.2, CgarF64(-1.0));
    assert_eq!(u.max.0, CgarF64(2.0));
    assert_eq!(u.max.1, CgarF64(1.0));
    assert_eq!(u.max.2, CgarF64(3.0));

    // disjoint box
    let c = Aabb::from_points(
        &TestPoint3::new(3.0, 3.0, 3.0),
        &TestPoint3::new(4.0, 4.0, 4.0),
    );
    assert!(a.intersects(&b));
    assert!(!a.intersects(&c));
}

#[test]
fn test_aabb_tree_build_and_query_3d() {
    // Build three 3D AABBs labeled 1,2,3
    let items = vec![
        (
            Aabb::from_points(
                &TestPoint3::new(0.0, 0.0, 0.0),
                &TestPoint3::new(1.0, 1.0, 1.0),
            ),
            1,
        ),
        (
            Aabb::from_points(
                &TestPoint3::new(1.5, 1.5, 1.5),
                &TestPoint3::new(2.5, 2.5, 2.5),
            ),
            2,
        ),
        (
            Aabb::from_points(
                &TestPoint3::new(-1.0, -1.0, -1.0),
                &TestPoint3::new(-0.5, -0.5, -0.5),
            ),
            3,
        ),
    ];

    // Build the tree
    let tree = AabbTree::build(items.clone());

    // Query for boxes overlapping [0.5,0.5,0.5]–[2.0,2.0,2.0]
    let mut hits = Vec::new();
    let query = Aabb::from_points(
        &TestPoint3::new(0.5, 0.5, 0.5),
        &TestPoint3::new(2.0, 2.0, 2.0),
    );
    tree.query(&query, &mut hits);

    // We expect to hit items 1 and 2
    let mut result: Vec<_> = hits.into_iter().cloned().collect();
    result.sort();
    assert_eq!(result, vec![1, 2]);
}
