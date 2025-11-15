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
use std::time::Instant;

use ahash::AHashSet;
use cgar::geometry::spatial_element::SpatialElement;
use cgar::geometry::{Point2, Point3};
use cgar::io::obj::{read_obj, write_obj};
use cgar::mesh::basic_types::Mesh;
use cgar::mesh_processing::boolean::BooleanOp;
use cgar::numeric::cgar_f64::CgarF64;
use cgar::numeric::cgar_rational::CgarRational;
use cgar::numeric::lazy_exact::LazyExact;
use cgar::numeric::scalar::Scalar;

#[test]
fn test_add_vertices_and_triangle_2() {
    let mut mesh = Mesh::<CgarF64, 2>::new();

    let v0 = mesh.add_vertex(Point2::from_vals([0.0, 0.0]));
    let v1 = mesh.add_vertex(Point2::from_vals([1.0, 0.0]));
    let v2 = mesh.add_vertex(Point2::from_vals([0.0, 1.0]));

    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.vertices[v0].position, Point2::from_vals([0.0, 0.0]));
    assert_eq!(mesh.vertices[v1].position, Point2::from_vals([1.0, 0.0]));
    assert_eq!(mesh.vertices[v2].position, Point2::from_vals([0.0, 1.0]));

    let face_idx = mesh.add_triangle(v0, v1, v2);
    assert_eq!(mesh.faces.len(), 1);
    assert_eq!(mesh.half_edges.len(), 3);

    let face = &mesh.faces[face_idx];
    let he0 = &mesh.half_edges[face.half_edge];
    let he1 = &mesh.half_edges[he0.next];
    let he2 = &mesh.half_edges[he1.next];

    // Check cycle
    assert_eq!(he0.next, face.half_edge + 1);
    assert_eq!(he1.next, face.half_edge + 2);
    assert_eq!(he2.next, face.half_edge); // closes the cycle

    assert_eq!(he0.prev, face.half_edge + 2);
    assert_eq!(he1.prev, face.half_edge + 0);
    assert_eq!(he2.prev, face.half_edge + 1);
}

#[test]
fn test_add_vertices_and_triangle_2_rational() {
    let mut mesh = Mesh::<CgarRational, 2>::new();

    let v0 = mesh.add_vertex(Point2::from_vals([0, 0]));
    let v1 = mesh.add_vertex(Point2::from_vals([1, 0]));
    let v2 = mesh.add_vertex(Point2::from_vals([0, 1]));

    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.vertices[v0].position, Point2::from_vals([0, 0]));
    assert_eq!(mesh.vertices[v1].position, Point2::from_vals([1, 0]));
    assert_eq!(mesh.vertices[v2].position, Point2::from_vals([0, 1]));

    let face_idx = mesh.add_triangle(v0, v1, v2);
    assert_eq!(mesh.faces.len(), 1);
    assert_eq!(mesh.half_edges.len(), 3);

    let face = &mesh.faces[face_idx];
    let he0 = &mesh.half_edges[face.half_edge];
    let he1 = &mesh.half_edges[he0.next];
    let he2 = &mesh.half_edges[he1.next];

    // Check cycle
    assert_eq!(he0.next, face.half_edge + 1);
    assert_eq!(he1.next, face.half_edge + 2);
    assert_eq!(he2.next, face.half_edge); // closes the cycle

    assert_eq!(he0.prev, face.half_edge + 2);
    assert_eq!(he1.prev, face.half_edge + 0);
    assert_eq!(he2.prev, face.half_edge + 1);
}

#[test]
fn test_connected_two_triangles_2() {
    let mut mesh = Mesh::<CgarF64, 2>::new();

    let v0 = mesh.add_vertex(Point2::from_vals([0.0, 0.0]));
    let v1 = mesh.add_vertex(Point2::from_vals([1.0, 0.0]));
    let v2 = mesh.add_vertex(Point2::from_vals([0.0, 1.0]));
    let v3 = mesh.add_vertex(Point2::from_vals([1.0, 1.0]));

    let f0 = mesh.add_triangle(v0, v1, v2);
    let f1 = mesh.add_triangle(v1, v3, v2);

    assert_eq!(mesh.vertices.len(), 4);
    assert_eq!(mesh.faces.len(), 2);
    assert_eq!(mesh.half_edges.len(), 6);

    mesh.build_boundary_loops();

    let he0_idx = mesh.faces[f0].half_edge;
    let he1_idx = mesh.faces[f1].half_edge;

    let he0 = &mesh.half_edges[he0_idx];
    let he1 = &mesh.half_edges[he0.next];
    let he2 = &mesh.half_edges[he1.next];

    assert_eq!(he0.vertex, v1);
    assert_eq!(he1.vertex, v2);
    assert_eq!(he2.vertex, v0);

    let forward = mesh.edge_map.get(&(v1, v2)).unwrap();
    let backward = mesh.edge_map.get(&(v2, v1)).unwrap();

    let he_fwd = &mesh.half_edges[*forward];
    let he_bwd = &mesh.half_edges[*backward];

    assert_eq!(he_fwd.twin, *backward);
    assert_eq!(he_bwd.twin, *forward);

    let ring_v1 = mesh.one_ring_neighbors(v1);
    assert_eq!(ring_v1.len(), 3);
    assert!(ring_v1.contains(&v0));
    assert!(ring_v1.contains(&v3));
}

#[test]
fn test_connected_two_triangles_2_rational() {
    let mut mesh = Mesh::<CgarRational, 2>::new();

    let v0 = mesh.add_vertex(Point2::from_vals([0, 0]));
    let v1 = mesh.add_vertex(Point2::from_vals([1, 0]));
    let v2 = mesh.add_vertex(Point2::from_vals([0, 1]));
    let v3 = mesh.add_vertex(Point2::from_vals([1, 1]));

    let f0 = mesh.add_triangle(v0, v1, v2);
    let f1 = mesh.add_triangle(v1, v3, v2);

    assert_eq!(mesh.vertices.len(), 4);
    assert_eq!(mesh.faces.len(), 2);
    assert_eq!(mesh.half_edges.len(), 6);

    mesh.build_boundary_loops();

    let forward = mesh.edge_map.get(&(v1, v2)).unwrap();
    let backward = mesh.edge_map.get(&(v2, v1)).unwrap();

    let he_fwd = &mesh.half_edges[*forward];
    let he_bwd = &mesh.half_edges[*backward];

    assert_eq!(he_fwd.twin, *backward);
    assert_eq!(he_bwd.twin, *forward);

    let ring_v1 = mesh.one_ring_neighbors(v1);
    assert_eq!(ring_v1.len(), 3);
    assert!(ring_v1.contains(&v0));
    assert!(ring_v1.contains(&v3));
}

#[test]
fn test_build_boundary_loops_2() {
    let mut mesh = Mesh::<CgarF64, 2>::new();

    let v0 = mesh.add_vertex(Point2::from_vals([0.0, 0.0]));
    let v1 = mesh.add_vertex(Point2::from_vals([1.0, 0.0]));
    let v2 = mesh.add_vertex(Point2::from_vals([0.0, 1.0]));
    let v3 = mesh.add_vertex(Point2::from_vals([1.0, 1.0]));

    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v1, v3, v2);

    mesh.build_boundary_loops();

    assert_eq!(mesh.half_edges.len(), 10);

    let ring = mesh.outgoing_half_edges(v1);
    assert_eq!(ring.len(), 3);

    let ring = mesh.outgoing_half_edges(v0);
    assert_eq!(ring.len(), 2);

    let ring = mesh.outgoing_half_edges(v3);
    assert_eq!(ring.len(), 2);
}

#[test]
fn test_edge_flip_2() {
    let mut mesh = Mesh::<CgarF64, 2>::new();

    let v0 = mesh.add_vertex(Point2::from_vals([0.0, 0.0]));
    let v1 = mesh.add_vertex(Point2::from_vals([1.0, 0.0]));
    let v2 = mesh.add_vertex(Point2::from_vals([0.0, 1.0]));
    let v3 = mesh.add_vertex(Point2::from_vals([1.0, 1.0]));

    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v1, v3, v2);
    mesh.build_boundary_loops();

    let he_shared = *mesh.edge_map.get(&(v1, v2)).unwrap();
    mesh.flip_edge(he_shared).expect("flip must succeed");

    let f0_vs = mesh.face_vertices(0);
    let set0: AHashSet<_> = f0_vs.into_iter().collect();
    assert_eq!(set0, [v0, v2, v3].into_iter().collect());

    let f1_vs = mesh.face_vertices(1);
    let set1: AHashSet<_> = f1_vs.into_iter().collect();
    assert_eq!(set1, [v0, v3, v1].into_iter().collect());
}

#[test]
fn test_face_area_and_centroid_2d() {
    let mut mesh = Mesh::<CgarF64, 2>::new();

    let v0 = mesh.add_vertex(Point2::from_vals([0.0, 0.0]));
    let v1 = mesh.add_vertex(Point2::from_vals([1.0, 0.0]));
    let v2 = mesh.add_vertex(Point2::from_vals([0.0, 1.0]));
    mesh.add_triangle(v0, v1, v2);

    let cent = mesh.face_centroid(0);
    assert!((cent[0].0 - (1.0 / 3.0)).abs() < 1e-12);
    assert!((cent[1].0 - (1.0 / 3.0)).abs() < 1e-12);

    let area = mesh.face_area(0);
    assert!((area.0 - 0.5).abs() < 1e-12);
}

fn make_square(origin: [f64; 2], size: f64) -> Mesh<CgarF64, 2> {
    let mut m = Mesh::new();
    let [ox, oy] = origin;

    let v0 = m.add_vertex(Point2::from_vals([ox, oy]));
    let v1 = m.add_vertex(Point2::from_vals([ox + size, oy]));
    let v2 = m.add_vertex(Point2::from_vals([ox + size, oy + size]));
    let v3 = m.add_vertex(Point2::from_vals([ox, oy + size]));

    m.add_triangle(v0, v1, v2);
    m.add_triangle(v0, v2, v3);

    m
}

fn make_square_exact(origin: [f64; 2], size: f64) -> Mesh<CgarRational, 2> {
    let mut m = Mesh::new();
    let [ox, oy] = origin;

    let v0 = m.add_vertex(Point2::from_vals([ox, oy]));
    let v1 = m.add_vertex(Point2::from_vals([ox + size, oy]));
    let v2 = m.add_vertex(Point2::from_vals([ox + size, oy + size]));
    let v3 = m.add_vertex(Point2::from_vals([ox, oy + size]));

    m.add_triangle(v0, v1, v2);
    m.add_triangle(v0, v2, v3);

    m
}

#[test]
fn union_boolean_2d() {
    let mut a = make_square([0.0, 0.0], 1.0);
    let mut b = make_square([0.5, 0.5], 1.0);

    let result = a.corefine_and_boolean(&mut b, BooleanOp::Union);

    result.validate_connectivity();
    assert!(result.faces.len() > 0);
}

#[test]
fn intersection_boolean_2d() {
    let mut a = make_square([0.0, 0.0], 1.0);
    let mut b = make_square([0.5, 0.5], 1.0);

    let result = a.corefine_and_boolean(&mut b, BooleanOp::Intersection);

    result.validate_connectivity();
    assert!(result.faces.len() > 0);
}

#[test]
fn difference_boolean_2d() {
    let mut a = make_square([0.0, 0.0], 1.0);
    let mut b = make_square([0.5, 0.5], 1.0);

    let result = a.corefine_and_boolean(&mut b, BooleanOp::Difference);

    result.validate_connectivity();
    assert!(result.faces.len() > 0);
}

#[test]
fn union_boolean_2d_exact() {
    let mut a = make_square_exact([0.0, 0.0], 1.0);
    let mut b = make_square_exact([0.5, 0.5], 1.0);

    let result = a.corefine_and_boolean(&mut b, BooleanOp::Union);

    result.validate_connectivity();
    assert!(result.faces.len() > 0);
}

#[test]
fn intersection_boolean_2d_exact() {
    let mut a = make_square_exact([0.0, 0.0], 1.0);
    let mut b = make_square_exact([0.5, 0.5], 1.0);

    let result = a.corefine_and_boolean(&mut b, BooleanOp::Intersection);

    result.validate_connectivity();
    assert!(result.faces.len() > 0);
}

#[test]
fn difference_boolean_2d_exact() {
    let mut a = make_square_exact([0.0, 0.0], 1.0);
    let mut b = make_square_exact([0.5, 0.5], 1.0);

    // Convert to 3D for OBJ export (add z=0)
    let a_3d = to_3d(&a);
    let b_3d = to_3d(&b);

    write_obj(&a_3d, "/mnt/v/cgar_meshes/square_a.obj").unwrap();
    write_obj(&b_3d, "/mnt/v/cgar_meshes/square_b.obj").unwrap();

    let result = a.corefine_and_boolean(&mut b, BooleanOp::Difference);

    let result_3d = to_3d(&result);
    write_obj(&result_3d, "/mnt/v/cgar_meshes/difference_result.obj").unwrap();

    result.validate_connectivity();
    assert!(result.faces.len() > 0);
}

fn to_3d<T: Scalar>(mesh_2d: &Mesh<T, 2>) -> Mesh<T, 3>
where
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>
        + Neg<Output = T>,
{
    let mut mesh_3d = Mesh::<T, 3>::new();
    let mut vertex_map = vec![0; mesh_2d.vertices.len()];

    for (i, v) in mesh_2d.vertices.iter().enumerate() {
        vertex_map[i] = mesh_3d.add_vertex(Point3::<T>::from_vals([
            v.position[0].clone(),
            v.position[1].clone(),
            T::zero(),
        ]));
    }

    for face in &mesh_2d.faces {
        if face.removed {
            continue;
        }
        let verts = mesh_2d.face_vertices(
            mesh_2d
                .faces
                .iter()
                .position(|f| std::ptr::eq(f, face))
                .unwrap(),
        );
        mesh_3d.add_triangle(
            vertex_map[verts[0]],
            vertex_map[verts[1]],
            vertex_map[verts[2]],
        );
    }

    mesh_3d
}
