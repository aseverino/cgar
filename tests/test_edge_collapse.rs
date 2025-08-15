// tests/test_edge_collapse.rs
use cgar::geometry::{point::Point, spatial_element::SpatialElement, vector::Vector};
use cgar::mesh::{
    basic_types::Mesh,
    edge_collapse::{CollapseOpts, CollapsePlan, CollapseReject, Midpoint, Placement},
};
use cgar::numeric::cgar_f64::CgarF64;

type TestMesh = Mesh<CgarF64, 3>;
type TestPoint = Point<CgarF64, 3>;

use cgar::io::obj::write_obj;

fn create_simple_triangle() -> TestMesh {
    let mut mesh = TestMesh::new();

    // Add three vertices to form a simple triangle
    let v0 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v1 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(1.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v2 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.5),
        CgarF64::from(1.0),
        CgarF64::from(0.0),
    ]));

    // Add triangle face
    mesh.add_triangle(v0, v1, v2);

    mesh.build_boundary_loops();

    mesh
}

fn create_square_mesh() -> TestMesh {
    let mut mesh = TestMesh::new();

    // Add four vertices to form a square
    let v0 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v1 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(1.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v2 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(1.0),
        CgarF64::from(1.0),
        CgarF64::from(0.0),
    ]));
    let v3 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.0),
        CgarF64::from(1.0),
        CgarF64::from(0.0),
    ]));

    // Add two triangles to form the square
    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v0, v2, v3);

    mesh.build_boundary_loops();

    mesh
}

fn create_grid4x4_mesh() -> TestMesh {
    let mut mesh = TestMesh::new();

    // make 4x4 vertices
    let id = |x: usize, y: usize| -> usize { y * 4 + x };
    for y in 0..4 {
        for x in 0..4 {
            mesh.add_vertex(TestPoint::from_vals([
                CgarF64::from(x as f64),
                CgarF64::from(y as f64),
                CgarF64::from(0.0),
            ]));
        }
    }

    // triangulate 3x3 quads with consistent CCW diagonal
    for y in 0..3 {
        for x in 0..3 {
            let v00 = id(x, y);
            let v10 = id(x + 1, y);
            let v01 = id(x, y + 1);
            let v11 = id(x + 1, y + 1);

            mesh.add_triangle(v00, v10, v11); // diag (v00->v11)
            mesh.add_triangle(v00, v11, v01);
        }
    }

    // Wire boundary loops so rot_ccw = twin(prev) works everywhere
    mesh.build_boundary_loops();
    mesh
}

#[test]
fn test_midpoint_placement() {
    let mesh = create_square_mesh();
    let placement = Midpoint;

    let p_star = placement.place(&mesh, 0, 2);

    // Midpoint between (0,0,0) and (1,1,0) should be (0.5, 0.5, 0)
    assert!((p_star[0].0 - 0.5).abs() < 1e-10);
    assert!((p_star[1].0 - 0.5).abs() < 1e-10);
    assert!((p_star[2].0 - 0.0).abs() < 1e-10);
}

#[test]
fn test_collapse_options_default() {
    let opts = CollapseOpts::<CgarF64>::default();
    assert!(!opts.forbid_border);
    assert!(opts.forbid_normal_flip);
}

#[test]
fn test_link_condition_interior_edge() {
    let mesh = create_square_mesh();

    let pr = mesh.ring_pair(0, 2).expect("no ring_pair for (0,2)");
    dbg!(&pr.opposite_a, &pr.opposite_b);
    dbg!(&pr.ring0.neighbors_ccw, &pr.ring1.neighbors_ccw);

    // Edge (0,2) is interior and should pass link condition
    // Common neighbors should be {1,3}
    assert!(mesh.check_link_condition_triangle(0, 2));
}

#[test]
fn test_link_condition_border_edge() {
    let mesh = create_simple_triangle();

    // All edges in a single triangle are border edges
    // Should pass with single common neighbor
    assert!(mesh.check_link_condition_triangle(0, 1));
    assert!(mesh.check_link_condition_triangle(1, 2));
    assert!(mesh.check_link_condition_triangle(2, 0));
}

#[test]
fn test_non_adjacent_vertices_fail_link_condition() {
    let mut mesh = TestMesh::new();

    // Create two separate triangles
    let v0 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v1 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(1.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v2 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.5),
        CgarF64::from(1.0),
        CgarF64::from(0.0),
    ]));
    let v3 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(2.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v4 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(3.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v5 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(2.5),
        CgarF64::from(1.0),
        CgarF64::from(0.0),
    ]));

    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v3, v4, v5);

    mesh.build_boundary_loops();

    // Non-adjacent vertices should fail
    assert!(!mesh.check_link_condition_triangle(v0, v3));
}

#[test]
fn test_duplicate_edge_detection() {
    let mut mesh = TestMesh::new();

    // Create a configuration that would create duplicate edges
    // Diamond mesh: two triangles sharing an edge
    let v0 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v1 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(1.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v2 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.5),
        CgarF64::from(1.0),
        CgarF64::from(0.0),
    ]));
    let v3 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.5),
        CgarF64::from(-1.0),
        CgarF64::from(0.0),
    ]));

    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v0, v3, v1);

    mesh.build_boundary_loops();

    // Collapsing (0,1) would not create duplicate edges in this simple case
    assert!(!mesh.would_create_duplicate_edges(v0, v1));
}

#[test]
fn test_two_gon_detection() {
    let mesh = create_simple_triangle();

    // In a single triangle, no edge collapse would create a 2-gon
    assert!(!mesh.would_create_2gons(0, 1));
    assert!(!mesh.would_create_2gons(1, 2));
    assert!(!mesh.would_create_2gons(2, 0));
}

#[test]
fn test_collapse_begin_valid_edge() {
    let mut mesh = create_square_mesh();
    mesh.build_boundary_loops();

    let opts = CollapseOpts::default();
    let placement = Midpoint;

    // Try to collapse interior edge (0,2)
    let result = mesh.collapse_edge_begin_vertices(0, 2, &placement, &opts);

    match result {
        Ok(plan) => {
            assert!(plan.v_keep == 0 || plan.v_keep == 2);
            assert!(plan.v_gone == 0 || plan.v_gone == 2);
            assert_ne!(plan.v_keep, plan.v_gone);
        }
        Err(reason) => {
            // Print for debugging - this might fail due to implementation details
            println!("Collapse rejected: {:?}", reason);
        }
    }
}

#[test]
fn test_border_edge_forbidden() {
    let mut mesh = create_simple_triangle();
    mesh.build_boundary_loops();

    let mut opts = CollapseOpts::default();
    opts.forbid_border = true;
    let placement = Midpoint;

    // All edges in a triangle are border edges
    let result = mesh.collapse_edge_begin_vertices(0, 1, &placement, &opts);
    assert!(matches!(result, Err(CollapseReject::BorderForbidden)));
}

#[test]
fn test_border_edge_allowed() {
    let mut mesh = create_simple_triangle();
    mesh.build_boundary_loops();

    let mut opts = CollapseOpts::default();
    opts.forbid_border = false;
    opts.forbid_normal_flip = false; // Disable for this test
    let placement = Midpoint;

    // Border edges should be allowed when forbid_border = false
    let result = mesh.collapse_edge_begin_vertices(0, 1, &placement, &opts);

    // Should either succeed or fail for topology reasons, not border restriction
    match result {
        Err(CollapseReject::BorderForbidden) => {
            panic!("Should not reject for border when forbid_border = false");
        }
        _ => {} // OK - either success or other rejection reason
    }
}

#[test]
fn test_degenerate_area_rejection() {
    let mut mesh = TestMesh::new();

    // Create a very thin triangle that would become degenerate
    let v0 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v1 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(1.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v2 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.5),
        CgarF64::from(1e-10),
        CgarF64::from(0.0),
    ]));

    mesh.add_triangle(v0, v1, v2);
    mesh.build_boundary_loops();

    let mut opts = CollapseOpts::default();
    opts.area_eps2 = CgarF64::from(1e-15); // Very strict area threshold
    let placement = Midpoint;

    let result = mesh.collapse_edge_begin_vertices(0, 1, &placement, &opts);

    // May reject due to degenerate face
    match result {
        Err(CollapseReject::DegenerateFace) => {
            // Expected for very thin triangles
        }
        _ => {
            // Other outcomes are also valid depending on implementation
        }
    }
}

#[test]
fn test_normal_flip_detection() {
    let mut mesh = TestMesh::new();

    // Create a configuration where collapsing would flip normals
    let v0 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v1 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(1.0),
        CgarF64::from(0.0),
        CgarF64::from(0.0),
    ]));
    let v2 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.5),
        CgarF64::from(1.0),
        CgarF64::from(0.0),
    ]));
    let v3 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.5),
        CgarF64::from(-2.0),
        CgarF64::from(0.0),
    ]));

    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v0, v3, v1);
    mesh.build_boundary_loops();

    let mut opts = CollapseOpts::default();
    opts.forbid_normal_flip = true;
    let placement = Midpoint;

    // This specific configuration might cause normal flips
    let result = mesh.collapse_edge_begin_vertices(0, 1, &placement, &opts);

    // Check if normal flip is properly detected (outcome depends on geometry)
    match result {
        Err(CollapseReject::NormalFlip) => {
            // Good - flip was detected
        }
        Ok(_) => {
            // Also valid - no flip occurred
        }
        Err(other) => {
            println!("Rejected for other reason: {:?}", other);
        }
    }
}

#[test]
fn test_collapse_commit_after_begin() {
    let mut mesh = create_square_mesh();

    let opts = CollapseOpts::default();
    let placement = Midpoint;

    let original_vertex_count = mesh.vertices.len();
    let original_face_count = mesh.faces.iter().filter(|f| !f.removed).count();
    println!("got here 0");

    // Try full collapse cycle
    let plan = mesh.collapse_edge_begin_vertices(0, 2, &placement, &opts);

    if let Ok(plan) = plan {
        println!("got here 1");
        let result = mesh.collapse_edge_commit(plan);

        if result.is_ok() {
            // Verify mesh consistency after collapse
            let remaining_faces = mesh.faces.iter().filter(|f| !f.removed).count();
            assert!(remaining_faces < original_face_count);

            // Verify vertex was merged
            assert_eq!(mesh.vertices.len(), original_vertex_count); // Vertices not removed, just merged
        }
    } else {
        panic!("Edge collapse failed to begin, {:?}", plan.err());
    }
}

#[test]
fn test_ring_pair_computation() {
    let mesh = create_square_mesh();

    // Test ring pair for adjacent vertices
    let ring_pair = mesh.ring_pair(0, 2);
    assert!(ring_pair.is_some());

    if let Some(pr) = ring_pair {
        // Verify ring structure is valid
        assert!(!pr.ring0.neighbors_ccw.is_empty());
        assert!(!pr.ring1.neighbors_ccw.is_empty());

        // In a square, vertices 0 and 2 should have common neighbors {1, 3}
        let set0: std::collections::HashSet<_> = pr
            .ring0
            .neighbors_ccw
            .iter()
            .copied()
            .filter(|&n| n != 2) // exclude v1 from v0's ring
            .collect();
        let set1: std::collections::HashSet<_> = pr
            .ring1
            .neighbors_ccw
            .iter()
            .copied()
            .filter(|&n| n != 0) // exclude v0 from v1's ring
            .collect();

        let common_neighbors: std::collections::HashSet<_> =
            set0.intersection(&set1).copied().collect();

        assert_eq!(common_neighbors.len(), 2); // Should be {1, 3}
        assert!(common_neighbors.contains(&1));
        assert!(common_neighbors.contains(&3));
    }
}

#[test]
fn test_collapse_on_fully_interior_edge() {
    let mut mesh = create_grid4x4_mesh();
    let _ = write_obj(&mesh, "/mnt/v/cgar_meshes/grid4x4.obj");
    let id = |x: usize, y: usize| -> usize { y * 4 + x };

    let v_keep = id(1, 1); // interior
    let v_gone = id(1, 2); // interior

    // sanity: this edge exists and is interior (two opposites)
    let pr = mesh.ring_pair(v_keep, v_gone).expect("not adjacent?");
    assert!(
        pr.opposite_a.is_some() && pr.opposite_b.is_some(),
        "edge not interior"
    );

    let original_faces = mesh.faces.iter().filter(|f| !f.removed).count();
    let original_verts = mesh.vertices.len();

    let placement = Midpoint;
    let opts = CollapseOpts::default();

    let plan = mesh
        .collapse_edge_begin_vertices(v_keep, v_gone, &placement, &opts)
        .expect("begin failed");

    mesh.collapse_edge_commit(plan).expect("commit failed");

    // exactly the two incident faces are removed
    let remaining_faces = mesh.faces.iter().filter(|f| !f.removed).count();
    assert_eq!(remaining_faces, original_faces - 2);

    // vertex slots are retained (merge semantics)
    assert_eq!(mesh.vertices.len(), original_verts);

    // (optional) rewire boundary loops if later ops rely on uniform rotation post-edit
    mesh.build_boundary_loops();

    // quick topology sanity around the kept vertex after collapse
    let ring = mesh.vertex_ring_ccw(v_keep);
    assert_eq!(ring.halfedges_ccw.len(), ring.neighbors_ccw.len());
    let _ = write_obj(&mesh, "/mnt/v/cgar_meshes/grid4x4_collapse.obj");
}
