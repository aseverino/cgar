use ahash::AHashSet;
// tests/test_topology.rs
use cgar::geometry::{point::Point, vector::Vector};
use cgar::mesh::basic_types::{Mesh, VertexRing};
use cgar::numeric::cgar_f64::CgarF64;

type TestMesh = Mesh<CgarF64, 3>;
type TestPoint = Point<CgarF64, 3>;

use cgar::{geometry::spatial_element::SpatialElement, io::obj::write_obj};

fn create_test_tetrahedron() -> TestMesh
where
    Point<CgarF64, 3>: cgar::geometry::point::PointOps<CgarF64, 3, Vector = Vector<CgarF64, 3>>,
    Vector<CgarF64, 3>: cgar::geometry::vector::VectorOps<CgarF64, 3>,
{
    let mut mesh = TestMesh::new();

    // Add four vertices for tetrahedron
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
        CgarF64::from(0.0),
        CgarF64::from(1.0),
        CgarF64::from(0.0),
    ]));
    let v3 = mesh.add_vertex(TestPoint::from_vals([
        CgarF64::from(0.0),
        CgarF64::from(0.0),
        CgarF64::from(1.0),
    ]));

    // Add four triangular faces
    mesh.add_triangle(v0, v2, v1);
    mesh.add_triangle(v0, v1, v3);
    mesh.add_triangle(v1, v2, v3);
    mesh.add_triangle(v0, v3, v2);

    mesh.build_boundary_loops();

    mesh
}

#[test]
fn test_tetrahedron_creation() {
    let mesh = create_test_tetrahedron();
    assert_eq!(mesh.vertices.len(), 4);
    assert_eq!(mesh.faces.len(), 4);
}

#[test]
fn test_face_vertices() {
    let mesh = create_test_tetrahedron();
    let face_verts = mesh.face_vertices(0);
    assert_eq!(face_verts.len(), 3);
}

#[test]
fn test_vertex_connectivity() {
    let mesh = create_test_tetrahedron();

    // In a tetrahedron, all vertices should be connected to all others
    assert!(mesh.are_vertices_connected(0, 1));
    assert!(mesh.are_vertices_connected(0, 2));
    assert!(mesh.are_vertices_connected(0, 3));
    assert!(mesh.are_vertices_connected(1, 2));
    assert!(mesh.are_vertices_connected(1, 3));
    assert!(mesh.are_vertices_connected(2, 3));
}

type T = CgarF64;
const N: usize = 3;

fn p(x: f64, y: f64, z: f64) -> Point<T, N> {
    Point::from_vals([x, y, z])
}

/// Build a square split into two CCW triangles:
///  v3(0,1) ---- v2(1,1)
///     |  \        |
///     |   \       |
///  v0(0,0) ---- v1(1,0)
///
/// Tris: (v0,v1,v2) and (v0,v2,v3)
fn make_two_tris_square() -> Mesh<T, N> {
    let mut m = Mesh::<T, N>::new();
    let v0 = m.add_vertex(p(0.0, 0.0, 0.0));
    let v1 = m.add_vertex(p(1.0, 0.0, 0.0));
    let v2 = m.add_vertex(p(1.0, 1.0, 0.0));
    let v3 = m.add_vertex(p(0.0, 1.0, 0.0));
    assert_eq!((v0, v1, v2, v3), (0, 1, 2, 3));

    let f0 = m.add_triangle(v0, v1, v2);
    let f1 = m.add_triangle(v0, v2, v3);

    m.build_boundary_loops();

    // basic sanity
    assert!(!m.faces[f0].removed);
    assert!(!m.faces[f1].removed);

    m
}

/// Single triangle for border tests
fn make_single_triangle() -> Mesh<T, N> {
    let mut m = Mesh::<T, N>::new();
    let v0 = m.add_vertex(p(0.0, 0.0, 0.0));
    let v1 = m.add_vertex(p(1.0, 0.0, 0.0));
    let v2 = m.add_vertex(p(0.0, 1.0, 0.0));
    m.add_triangle(v0, v1, v2);
    m
}

/// Helper: assert the index inside the ring corresponds to the exact half-edge id.
fn assert_ring_index_matches_he(
    ring: &VertexRing,
    he_expected: usize,
    idx_opt: Option<usize>,
    neighbor_expected: usize,
    mesh: &Mesh<T, N>,
) {
    let idx = idx_opt.expect("index must be Some in PairRing");
    assert!(idx < ring.halfedges_ccw.len());
    assert_eq!(
        ring.halfedges_ccw[idx], he_expected,
        "ring index must point to the exact half-edge id"
    );
    assert_eq!(
        ring.neighbors_ccw[idx], neighbor_expected,
        "neighbor at ring index must match incidence"
    );
    // Optional extra: faces_ccw lengths align
    assert_eq!(ring.halfedges_ccw.len(), ring.neighbors_ccw.len());
    assert_eq!(ring.halfedges_ccw.len(), ring.faces_ccw.len());
    // And the half-edge itself should have the expected head
    assert_eq!(mesh.half_edges[he_expected].vertex, neighbor_expected);
}

#[test]
fn ring_pair_interior_edge_has_two_opposites_and_not_border() {
    let m = make_two_tris_square();

    let mut h = m.half_edge_between(0, 1).unwrap(); // 0→1 exists
    let mut neighs = Vec::new();
    for _ in 0..3 {
        neighs.push(m.half_edges[h].vertex);
        h = m.rot_ccw_around_vertex(h);
    }
    neighs.sort_unstable();
    assert_eq!(neighs, vec![1, 2, 3]);

    // Interior edge is the diagonal 0 -> 2 (and 2 -> 0)
    let he_0_2 = m.half_edge_between(0, 2).unwrap();
    let he_2_0 = m.half_edge_between(2, 0).unwrap();
    assert!(m.half_edges[he_0_2].face.is_some());
    assert!(
        m.half_edges[he_2_0].face.is_some(),
        "shared edge must have both directions bound to faces"
    );

    let pair = m.ring_pair(0, 2).expect("ring_pair(0,2) should exist");

    // Walk around v0 once and check we see 1,2,3 in some order
    let mut h = m.half_edge_between(0, 1).unwrap(); // seed at 0→1
    let mut neighs = Vec::new();
    for _ in 0..3 {
        neighs.push(m.half_edges[h].vertex);
        h = m.rot_ccw_around_vertex(h);
    }
    neighs.sort_unstable();
    assert_eq!(neighs, vec![1, 2, 3]);

    // 0) Dump the raw rings
    eprintln!("ring0 neighbors = {:?}", pair.ring0.neighbors_ccw);
    eprintln!("ring1 neighbors = {:?}", pair.ring1.neighbors_ccw);

    // 1) Verify the interior edge truly has two valid faces
    let he_0_2 = m.half_edge_between(0, 2).unwrap();
    let he_2_0 = m.half_edge_between(2, 0).unwrap();
    let f_a = m.half_edges[he_0_2].face;
    let f_b = m.half_edges[he_2_0].face;
    eprintln!("faces across 0-2: {:?} / {:?}", f_a, f_b);
    assert!(
        f_a.is_some() && f_b.is_some(),
        "interior edge must have two faces"
    );
    assert!(!m.faces[f_a.unwrap()].removed && !m.faces[f_b.unwrap()].removed);

    // 2) Sanity: rings don’t include removed half-edges and lengths agree
    assert_eq!(
        pair.ring0.halfedges_ccw.len(),
        pair.ring0.neighbors_ccw.len()
    );
    assert!(
        pair.ring0
            .halfedges_ccw
            .iter()
            .all(|&h| !m.half_edges[h].removed)
    );
    assert_eq!(
        pair.ring1.halfedges_ccw.len(),
        pair.ring1.neighbors_ccw.len()
    );
    assert!(
        pair.ring1
            .halfedges_ccw
            .iter()
            .all(|&h| !m.half_edges[h].removed)
    );

    // 3) Recompute common neighbors directly from rings (excluding endpoints)
    let mut s0: AHashSet<_> = pair
        .ring0
        .neighbors_ccw
        .iter()
        .copied()
        .filter(|&x| x != 2)
        .collect();
    let mut s1: AHashSet<_> = pair
        .ring1
        .neighbors_ccw
        .iter()
        .copied()
        .filter(|&x| x != 0)
        .collect();
    let mut commons: Vec<_> = s0.intersection(&s1).copied().collect();
    commons.sort_unstable();
    eprintln!("derived commons = {:?}", commons);

    // Indices inside the rings must point to the exact half-edges
    assert_ring_index_matches_he(&pair.ring0, he_0_2, pair.idx_v1_in_ring0, 2, &m);
    assert_ring_index_matches_he(&pair.ring1, he_2_0, pair.idx_v0_in_ring1, 0, &m);

    // Opposites across the diagonal should be the other two vertices: 1 and 3 (order depends on direction)
    // For he 0->2, the "next" around that face should be vertex 3 or 1 depending on which face we land on.
    // Since faces are (0,1,2) and (0,2,3), we expect:
    //   opposite_a (for 0->2) == 1 OR 3
    //   opposite_b (for 2->0) == 1 OR 3
    // and they must be distinct when both Some.
    let oa = pair
        .opposite_a
        .expect("interior edge must have opposite on side A");
    let ob = pair
        .opposite_b
        .expect("interior edge must have opposite on side B");
    assert!(oa == 1 || oa == 3, "opposite_a must be 1 or 3, got {}", oa);
    assert!(ob == 1 || ob == 3, "opposite_b must be 1 or 3, got {}", ob);
    assert_ne!(
        oa, ob,
        "interior edge must have two distinct opposite vertices"
    );

    // Not a border edge
    assert!(!pair.is_border_edge);

    // Common neighbors of 0 and 2 must be exactly {1, 3} (unordered)
    let mut commons: Vec<_> = pair.common_neighbors.iter().copied().collect();
    commons.sort_unstable();
    assert_eq!(commons, vec![1, 3]);
}

#[test]
fn ring_pair_border_edge_reports_border_and_single_opposite() {
    let m = make_single_triangle();

    // Edge 0-1 is a border (only one incident face)
    let he_0_1 = m.half_edge_between(0, 1).expect("0->1 must exist");
    let he_1_0 = m.half_edge_between(1, 0).expect("1->0 must exist");

    let pair = m.ring_pair(0, 1).expect("ring_pair(0,1) should exist");

    // Indices must align to the exact half-edge ids
    assert_ring_index_matches_he(&pair.ring0, he_0_1, pair.idx_v1_in_ring0, 1, &m);
    assert_ring_index_matches_he(&pair.ring1, he_1_0, pair.idx_v0_in_ring1, 0, &m);

    // On a single-tri border, exactly one side has an opposite (the interior face)
    let a = pair.opposite_a;
    let b = pair.opposite_b;
    assert!(
        a.is_some() ^ b.is_some(),
        "border edge must have exactly one opposite"
    );

    // That opposite must be vertex 2
    let opp = a.or(b).unwrap();
    assert_eq!(opp, 2);

    assert!(pair.is_border_edge);
}

#[test]
fn ring_pair_becomes_border_when_one_face_removed() {
    let mut m = make_two_tris_square();

    // Identify faces that touch the diagonal 0-2.
    let he_0_2 = m.half_edge_between(0, 2).unwrap();
    let he_2_0 = m.half_edge_between(2, 0).unwrap();

    let f_a = m.half_edges[he_0_2].face.expect("should have a face");
    let f_b = m.half_edges[he_2_0]
        .face
        .expect("should have the other face");
    assert_ne!(f_a, f_b);

    // Remove one of the faces
    m.faces[f_b].removed = true;

    let pair = m
        .ring_pair(0, 2)
        .expect("ring_pair(0,2) should still exist");

    // Exactly one opposite should remain
    let has_a = pair.opposite_a.is_some();
    let has_b = pair.opposite_b.is_some();
    assert!(
        has_a ^ has_b,
        "after removing one incident face, exactly one opposite must remain"
    );

    // And it must now be flagged as a border edge
    assert!(pair.is_border_edge);
}

#[test]
fn ring_pair_indices_stay_consistent_even_if_neighbor_values_repeat() {
    // This test protects against indexing by neighbor value.
    // We fabricate a case where the neighbor value could appear twice in the ring’s neighbors
    // (e.g., if someone accidentally emits a removed edge). We still require that indices
    // match the exact half-edge ids for the v0->v1 and v1->v0 incidences.

    let mut m = make_two_tris_square();

    // Diagonal edge
    let he_0_2 = m.half_edge_between(0, 2).unwrap();
    let he_2_0 = m.half_edge_between(2, 0).unwrap();

    // (If your implementation ever emitted removed half-edges, we can simulate it:)
    // Mark the twin-next of he_0_2 as removed to simulate stale entry potential
    let junk = m.half_edges[m.half_edges[he_0_2].twin].next;
    m.half_edges[junk].removed = true;

    let pair = m.ring_pair(0, 2).expect("ring_pair(0,2) should exist");

    // These asserts fail if ring_pair indexed by neighbor value instead of half-edge id.
    assert_ring_index_matches_he(&pair.ring0, he_0_2, pair.idx_v1_in_ring0, 2, &m);
    assert_ring_index_matches_he(&pair.ring1, he_2_0, pair.idx_v0_in_ring1, 0, &m);
}
