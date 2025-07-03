use cgar::geometry::point::Point2;
use cgar::mesh::{mesh::Mesh, point_trait::PointTrait};
use cgar::numeric::cgar_rational::{self, CgarRational};
use rug::Rational;

#[derive(Clone, Debug, PartialEq)]
struct TestPoint2F64(f64, f64);

impl PointTrait<f64> for TestPoint2F64 {
    fn dimensions() -> usize {
        2
    }
    fn coord(&self, axis: usize) -> f64 {
        match axis {
            0 => self.0,
            1 => self.1,
            _ => panic!("invalid axis"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TestPoint2Rational(CgarRational, CgarRational);

impl PointTrait<CgarRational> for TestPoint2Rational {
    fn dimensions() -> usize {
        2
    }
    fn coord(&self, axis: usize) -> CgarRational {
        match axis {
            0 => self.0.clone(),
            1 => self.1.clone(),
            _ => panic!("invalid axis"),
        }
    }
}

#[test]
fn test_add_vertices_and_triangle() {
    let mut mesh = Mesh::<f64, TestPoint2F64>::new();

    let v0 = mesh.add_vertex(TestPoint2F64(0.0, 0.0));
    let v1 = mesh.add_vertex(TestPoint2F64(1.0, 0.0));
    let v2 = mesh.add_vertex(TestPoint2F64(0.0, 1.0));

    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.vertices[v0].position, TestPoint2F64(0.0, 0.0));
    assert_eq!(mesh.vertices[v1].position, TestPoint2F64(1.0, 0.0));
    assert_eq!(mesh.vertices[v2].position, TestPoint2F64(0.0, 1.0));

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
fn test_add_vertices_and_triangle_rational() {
    let mut mesh = Mesh::<CgarRational, TestPoint2Rational>::new();

    let v0 = mesh.add_vertex(TestPoint2Rational(
        CgarRational::from(0),
        CgarRational::from(0),
    ));
    let v1 = mesh.add_vertex(TestPoint2Rational(
        CgarRational::from(1),
        CgarRational::from(0),
    ));
    let v2 = mesh.add_vertex(TestPoint2Rational(
        CgarRational::from(0),
        CgarRational::from(1),
    ));

    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(
        mesh.vertices[v0].position,
        TestPoint2Rational(CgarRational::from(0), CgarRational::from(0))
    );
    assert_eq!(
        mesh.vertices[v1].position,
        TestPoint2Rational(CgarRational::from(1), CgarRational::from(0))
    );
    assert_eq!(
        mesh.vertices[v2].position,
        TestPoint2Rational(CgarRational::from(0), CgarRational::from(1))
    );

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
fn test_connected_two_triangles() {
    let mut mesh = Mesh::<f64, TestPoint2F64>::new();

    let v0 = mesh.add_vertex(TestPoint2F64(0.0, 0.0));
    let v1 = mesh.add_vertex(TestPoint2F64(1.0, 0.0));
    let v2 = mesh.add_vertex(TestPoint2F64(0.0, 1.0));
    let v3 = mesh.add_vertex(TestPoint2F64(1.0, 1.0));

    let f0 = mesh.add_triangle(v0, v1, v2); // Lower-left triangle
    let f1 = mesh.add_triangle(v1, v3, v2); // Upper-right triangle

    assert_eq!(mesh.vertices.len(), 4);
    assert_eq!(mesh.faces.len(), 2);
    assert_eq!(mesh.half_edges.len(), 6); // 3 per triangle (no twin connection yet)

    mesh.build_boundary_loops();

    // Let's check connectivity for f0 and f1
    let he0_idx = mesh.faces[f0].half_edge;
    let he1_idx = mesh.faces[f1].half_edge;

    let he0 = &mesh.half_edges[he0_idx];
    let he1 = &mesh.half_edges[he0.next];
    let he2 = &mesh.half_edges[he1.next];

    assert_eq!(he0.vertex, v1);
    assert_eq!(he1.vertex, v2);
    assert_eq!(he2.vertex, v0);

    // Get the two half-edges between v1 and v2 in opposite directions
    let forward = mesh.edge_map.get(&(v1, v2)).unwrap();
    let backward = mesh.edge_map.get(&(v2, v1)).unwrap();

    let he_fwd = &mesh.half_edges[*forward];
    let he_bwd = &mesh.half_edges[*backward];

    assert_eq!(he_fwd.twin, *backward);
    assert_eq!(he_bwd.twin, *forward);

    let g0 = &mesh.half_edges[he0_idx];
    let g1 = &mesh.half_edges[he1_idx];
    let g2 = &mesh.half_edges[g1.next];
    let g3 = &mesh.half_edges[g2.next];

    assert_eq!(g1.vertex, v3);
    assert_eq!(g2.vertex, v2);
    assert_eq!(g3.vertex, v1);

    // Check if shared edge v1 -> v2 exists twice (opposite direction)
    let edge_count = mesh
        .half_edges
        .iter()
        .filter(|he| he.vertex == v2 && mesh.half_edges[he.prev].vertex == v1)
        .count();
    assert_eq!(edge_count, 1, "v1 → v2 should appear once");

    let twin_edge_count = mesh
        .half_edges
        .iter()
        .filter(|he| he.vertex == v1 && mesh.half_edges[he.prev].vertex == v2)
        .count();
    assert_eq!(twin_edge_count, 1, "v2 → v1 should appear once");

    let ring_v1 = mesh.one_ring_neighbors(v1);
    assert_eq!(ring_v1.len(), 3);
    assert!(ring_v1.contains(&v0));
    assert!(ring_v1.contains(&v3));
}

#[test]
fn test_connected_two_triangles_rational() {
    let mut mesh = Mesh::<CgarRational, TestPoint2Rational>::new();

    let v0 = mesh.add_vertex(TestPoint2Rational(
        CgarRational::from(0),
        CgarRational::from(0),
    ));
    let v1 = mesh.add_vertex(TestPoint2Rational(
        CgarRational::from(1),
        CgarRational::from(0),
    ));
    let v2 = mesh.add_vertex(TestPoint2Rational(
        CgarRational::from(0),
        CgarRational::from(1),
    ));
    let v3 = mesh.add_vertex(TestPoint2Rational(
        CgarRational::from(1),
        CgarRational::from(1),
    ));

    let f0 = mesh.add_triangle(v0, v1, v2); // Lower-left triangle
    let f1 = mesh.add_triangle(v1, v3, v2); // Upper-right triangle

    assert_eq!(mesh.vertices.len(), 4);
    assert_eq!(mesh.faces.len(), 2);
    assert_eq!(mesh.half_edges.len(), 6); // 3 per triangle (no twin connection yet)

    mesh.build_boundary_loops();

    // Let's check connectivity for f0 and f1
    let he0_idx = mesh.faces[f0].half_edge;
    let he1_idx = mesh.faces[f1].half_edge;

    let he0 = &mesh.half_edges[he0_idx];
    let he1 = &mesh.half_edges[he0.next];
    let he2 = &mesh.half_edges[he1.next];

    assert_eq!(he0.vertex, v1);
    assert_eq!(he1.vertex, v2);
    assert_eq!(he2.vertex, v0);

    // Get the two half-edges between v1 and v2 in opposite directions
    let forward = mesh.edge_map.get(&(v1, v2)).unwrap();
    let backward = mesh.edge_map.get(&(v2, v1)).unwrap();

    let he_fwd = &mesh.half_edges[*forward];
    let he_bwd = &mesh.half_edges[*backward];

    assert_eq!(he_fwd.twin, *backward);
    assert_eq!(he_bwd.twin, *forward);

    let g0 = &mesh.half_edges[he0_idx];
    let g1 = &mesh.half_edges[he1_idx];
    let g2 = &mesh.half_edges[g1.next];
    let g3 = &mesh.half_edges[g2.next];

    assert_eq!(g1.vertex, v3);
    assert_eq!(g2.vertex, v2);
    assert_eq!(g3.vertex, v1);

    // Check if shared edge v1 -> v2 exists twice (opposite direction)
    let edge_count = mesh
        .half_edges
        .iter()
        .filter(|he| he.vertex == v2 && mesh.half_edges[he.prev].vertex == v1)
        .count();
    assert_eq!(edge_count, 1, "v1 → v2 should appear once");

    let twin_edge_count = mesh
        .half_edges
        .iter()
        .filter(|he| he.vertex == v1 && mesh.half_edges[he.prev].vertex == v2)
        .count();
    assert_eq!(twin_edge_count, 1, "v2 → v1 should appear once");

    let ring_v1 = mesh.one_ring_neighbors(v1);

    assert_eq!(ring_v1.len(), 3);
    assert!(ring_v1.contains(&v0));
    assert!(ring_v1.contains(&v3));
}

#[test]
fn test_build_boundary_loops() {
    let mut mesh = Mesh::<f64, TestPoint2F64>::new();

    let v0 = mesh.add_vertex(TestPoint2F64(0.0, 0.0));
    let v1 = mesh.add_vertex(TestPoint2F64(1.0, 0.0));
    let v2 = mesh.add_vertex(TestPoint2F64(0.0, 1.0));
    let v3 = mesh.add_vertex(TestPoint2F64(1.0, 1.0));

    let f0 = mesh.add_triangle(v0, v1, v2); // Lower-left triangle
    let f1 = mesh.add_triangle(v1, v3, v2); // Upper-right triangle

    // now build the holes
    mesh.build_boundary_loops();

    // we had 6 “real” half-edges, and 4 boundary edges → 4 ghosts
    assert_eq!(mesh.half_edges.len(), 10);

    // and outgoing_half_edges_strict(v1) no longer panics…
    let ring = mesh.outgoing_half_edges(v1);
    assert_eq!(ring.len(), 3);

    let ring = mesh.outgoing_half_edges(v0);
    assert_eq!(ring.len(), 2);

    let ring = mesh.outgoing_half_edges(v3);
    assert_eq!(ring.len(), 2);
}

#[test]
fn test_face_loop_traversal() {
    let mut mesh = Mesh::<f64, TestPoint2F64>::new();

    // Build the same two-triangle mesh:
    let v0 = mesh.add_vertex(TestPoint2F64(0.0, 0.0));
    let v1 = mesh.add_vertex(TestPoint2F64(1.0, 0.0));
    let v2 = mesh.add_vertex(TestPoint2F64(0.0, 1.0));
    let v3 = mesh.add_vertex(TestPoint2F64(1.0, 1.0));

    let f0 = mesh.add_triangle(v0, v1, v2);
    let f1 = mesh.add_triangle(v1, v3, v2);

    // Before building boundary loops, traversal around faces is already valid:
    let fe0 = mesh.face_half_edges(f0);
    assert_eq!(fe0.len(), 3);
    // The triangle f0 = (v0, v1, v2) in CCW order gives half-edges targeting (v1, v2, v0)
    let fv0 = mesh.face_vertices(f0);
    assert_eq!(fv0, vec![v1, v2, v0]);

    let fe1 = mesh.face_half_edges(f1);
    assert_eq!(fe1.len(), 3);
    // f1 = (v1, v3, v2) → targets (v3, v2, v1)
    let fv1 = mesh.face_vertices(f1);
    assert_eq!(fv1, vec![v3, v2, v1]);
}

#[test]
fn test_boundary_detection_and_loops() {
    let mut mesh = Mesh::<f64, TestPoint2F64>::new();

    // Build the two‐triangle quad as before:
    let v0 = mesh.add_vertex(TestPoint2F64(0.0, 0.0));
    let v1 = mesh.add_vertex(TestPoint2F64(1.0, 0.0));
    let v2 = mesh.add_vertex(TestPoint2F64(0.0, 1.0));
    let v3 = mesh.add_vertex(TestPoint2F64(1.0, 1.0));

    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v1, v3, v2);

    // Build the boundary‐ghosts
    mesh.build_boundary_loops();

    // 1) Boundary vertices should be all four
    let bverts = mesh.boundary_vertices();
    assert_eq!(bverts.len(), 4);
    for &v in &[v0, v1, v2, v3] {
        assert!(bverts.contains(&v));
    }

    // 2) There should be exactly one boundary loop,
    //    and it should visit each of the four vertices once.
    let loops = mesh.boundary_loops();
    assert_eq!(loops.len(), 1);

    let loop_vs = &loops[0];
    assert_eq!(loop_vs.len(), 4);
    let set: std::collections::HashSet<_> = loop_vs.iter().cloned().collect();
    let expected: std::collections::HashSet<_> = [v0, v1, v2, v3].into_iter().collect();
    assert_eq!(set, expected);
}

#[test]
fn test_edge_flip() {
    let mut mesh = Mesh::<f64, TestPoint2F64>::new();
    let v0 = mesh.add_vertex(TestPoint2F64(0.0, 0.0));
    let v1 = mesh.add_vertex(TestPoint2F64(1.0, 0.0));
    let v2 = mesh.add_vertex(TestPoint2F64(0.0, 1.0));
    let v3 = mesh.add_vertex(TestPoint2F64(1.0, 1.0));

    // build two triangles sharing edge v1–v2
    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v1, v3, v2);
    mesh.build_boundary_loops();

    // find the half-edge for the shared edge (v1→v2)
    let he_shared = *mesh.edge_map.get(&(v1, v2)).unwrap();

    // flip it
    mesh.flip_edge(he_shared).expect("flip must succeed");

    // Now the shared diagonal should be v0–v3
    // face f0 (was v0,v1,v2) becomes (v0,v2,v3)
    let f0_vs = mesh.face_vertices(0);
    let set0: std::collections::HashSet<_> = f0_vs.into_iter().collect();
    assert_eq!(set0, [v0, v2, v3].into_iter().collect());

    // face f1 (was v1,v3,v2) becomes (v0,v3,v1)
    let f1_vs = mesh.face_vertices(1);
    let set1: std::collections::HashSet<_> = f1_vs.into_iter().collect();
    assert_eq!(set1, [v0, v3, v1].into_iter().collect());
}
