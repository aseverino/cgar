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

use ahash::{AHashMap, AHashSet};
use smallvec::SmallVec;

use crate::{
    boolean::boolean::{ApproxPointKey, SplitType, point_key},
    geometry::{
        Point2,
        point::{Point, PointOps},
        spatial_element::SpatialElement,
        vector::{Vector, VectorOps},
    },
    io::obj::write_obj,
    kernel::predicates::orient2d,
    mesh::{
        basic_types::Mesh,
        face::Face,
        half_edge::HalfEdge,
        intersection_segment::{IntersectionEndPoint, IntersectionSegment},
    },
    numeric::{
        cgar_f64::CgarF64,
        scalar::{RefInto, Scalar},
    },
    operations::triangulation::delaunay::{Delaunay, triangles_inside_for_job},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct UEdge(usize, usize);
impl UEdge {
    #[inline]
    fn of(a: usize, b: usize) -> Self {
        if a < b { UEdge(a, b) } else { UEdge(b, a) }
    }
}

/// PSLG job per face, with UV points in f64.
/// You can adapt this type or reuse an existing one.
#[derive(Debug)]
pub struct FaceJobUV<T: Scalar> {
    pub face_id: usize,
    pub verts_global: Vec<usize>,  // global vertex ids (orig + new)
    pub points_uv: Vec<Point2<T>>, // projected to face plane
    pub segments: Vec<[usize; 2]>, // local indices into verts_global / points_uv
}

pub fn rewrite_faces_from_cdt_batch<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    jobs: &[FaceJobUV<T>],
    cdts: &[Delaunay<T>],
) where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    // Remove faces
    let mut to_remove: AHashSet<usize> = AHashSet::with_capacity(jobs.len());
    for j in jobs {
        to_remove.insert(j.face_id);
    }

    let faces_vec: Vec<_> = to_remove.iter().copied().collect();
    let (weld_starts, affected_vs) = mesh.remove_triangles_deferred(&faces_vec);
    // mesh.weld_border_components_from(&weld_starts);
    // mesh.fix_vertices_outgoing_for(&affected_vs);

    println!("BEGINNING REWRITE");

    // let _ = write_obj(&mesh, "/mnt/v/cgar_meshes/before.obj");

    // Let's try simply adding the triangles from cdts.
    for (i, (job, dt)) in jobs.iter().zip(cdts.iter()).enumerate() {
        // let inside = triangles_inside_for_job(&dt.points, &dt.triangles, &job.segments);
        for (ti, t) in dt.triangles.iter().enumerate() {
            // if !inside.contains(&ti) {
            // continue;
            // }
            let (a, b, c) = (t.0, t.1, t.2);
            // convert local indices to global
            let (ga, gb, gc) = (
                job.verts_global[a],
                job.verts_global[b],
                job.verts_global[c],
            );
            mesh.add_triangle(ga, gb, gc);
            // let _ = write_obj(
            //     &mesh,
            //     &format!("/mnt/v/cgar_meshes/before_{}_{}.obj", i, ti),
            // );
        }
    }
}

/// Consume CDT results and rebuild all faces in one batch.
/// `jobs[i]` must correspond to `cdts[i]` (same order/length).
/// Consume CDT results and rebuild all faces in one batch.
/// `jobs[i]` must correspond to `cdts[i]` (same order/length).
pub fn rewrite_faces_from_cdt_batch_2<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    jobs: &[FaceJobUV<T>],
    cdts: &[Delaunay<T>],
) where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    assert_eq!(jobs.len(), cdts.len(), "jobs/cdts size mismatch");

    // Basic validation
    for (ji, (job, dt)) in jobs.iter().zip(cdts.iter()).enumerate() {
        assert_eq!(
            job.points_uv.len(),
            dt.points.len(),
            "CDT points length mismatch job {}: job.points_uv={}, dt.points={}",
            ji,
            job.points_uv.len(),
            dt.points.len()
        );
    }

    // Remove faces
    let mut to_remove: AHashSet<usize> = AHashSet::with_capacity(jobs.len());
    for j in jobs {
        to_remove.insert(j.face_id);
    }

    let faces_vec: Vec<_> = to_remove.iter().copied().collect();
    let (weld_starts, affected_vs) = mesh.remove_triangles_deferred(&faces_vec);
    mesh.weld_border_components_from(&weld_starts);
    mesh.fix_vertices_outgoing_for(&affected_vs);

    // Build filtered triangles with consistent orientation across shared edges
    let mut global_seen_all: AHashSet<(usize, usize, usize)> = AHashSet::default();
    let mut all_triangles: Vec<(usize, usize, usize, usize)> = Vec::new(); // (a,b,c,job_idx)

    // First pass: collect all triangles with shared-edge-aware orientation
    for (job_idx, (job, dt)) in jobs.iter().zip(cdts.iter()).enumerate() {
        let ring_ccw =
            orient2d(&job.points_uv[0], &job.points_uv[1], &job.points_uv[2]).is_positive();
        let inside = triangles_inside_for_job(&dt.points, &dt.triangles, &job.segments);

        for (ti, t) in dt.triangles.iter().enumerate() {
            if !inside.contains(&ti) {
                continue;
            }

            let (la, lb, lc) = (t.0, t.1, t.2);
            if la >= job.points_uv.len() || lb >= job.points_uv.len() || lc >= job.points_uv.len() {
                continue;
            }
            if la == lb || lb == lc || la == lc {
                continue;
            }

            let (ga, gb, gc) = (
                job.verts_global[la],
                job.verts_global[lb],
                job.verts_global[lc],
            );
            if ga == gb || gb == gc || ga == gc {
                continue;
            }

            // Check deduplication
            let mut gkey = [ga, gb, gc];
            gkey.sort_unstable();
            let gkey_t = (gkey[0], gkey[1], gkey[2]);
            if !global_seen_all.insert(gkey_t) {
                continue;
            }

            // Determine orientation - prioritize consistency with shared edges
            let ccw_in_uv =
                orient2d(&job.points_uv[la], &job.points_uv[lb], &job.points_uv[lc]).is_positive();
            let use_ccw = ccw_in_uv == ring_ccw;

            let (final_a, final_b, final_c) = if use_ccw { (ga, gb, gc) } else { (ga, gc, gb) };

            all_triangles.push((final_a, final_b, final_c, job_idx));
        }
    }

    // Sort triangles to prioritize those with more available edges
    all_triangles.sort_by_key(|(a, b, c, _)| {
        let available_edges = [(*a, *b), (*b, *c), (*c, *a)]
            .iter()
            .map(|(u, v)| if edge_ok_now(mesh, *u, *v) { 1 } else { 0 })
            .sum::<i32>();
        -available_edges // Higher availability first
    });

    // Insert triangles in optimized order
    let mut inserted = 0;
    let mut failed = Vec::new();

    for (a, b, c, job_idx) in all_triangles {
        if edge_ok_now(mesh, a, b) && edge_ok_now(mesh, b, c) && edge_ok_now(mesh, c, a) {
            let _f = mesh.add_triangle(a, b, c);
            println!("Added triangle: ({}, {}, {}) from job {}", a, b, c, job_idx);
            inserted += 1;
        } else {
            failed.push((a, b, c, job_idx));
        }
    }

    // Report any failures
    for (a, b, c, job_idx) in &failed {
        let st = |u: usize, v: usize| match mesh.edge_map.get(&(u, v)).copied() {
            None => "absent".to_string(),
            Some(h) => {
                let he = &mesh.half_edges[h];
                if he.removed {
                    "REMOVED".into()
                } else if he.face.is_none() {
                    "BORDER".into()
                } else {
                    format!("INTERIOR(face={})", he.face.unwrap())
                }
            }
        };
        eprintln!(
            "[rewrite] could not insert tri ({},{},{}) from job {}: ({},{})={}, ({},{})={}, ({},{})={}",
            a,
            b,
            c,
            job_idx,
            a,
            b,
            st(*a, *b),
            b,
            c,
            st(*b, *c),
            c,
            a,
            st(*c, *a)
        );
    }

    println!("Inserted {} triangles, failed {}", inserted, failed.len());
}

/// Holds “one side seen” half-edges until the opposite side arrives.
struct TwinTable {
    map: AHashMap<UEdge, usize>, // undirected edge -> half_edge id
}
impl TwinTable {
    fn new() -> Self {
        Self {
            map: AHashMap::default(),
        }
    }
    fn take(&mut self, key: UEdge) -> Option<usize> {
        self.map.remove(&key)
    }
    fn put(&mut self, key: UEdge, he: usize) {
        self.map.insert(key, he);
    }
}

/// 1) Allocate global vertices for all splits (edge OR face) without touching topology.
///    - Reuses your `ApproxPointKey` to dedup.
///    - Updates intersection endpoint vertex_hints in-place.
///    - Returns a map ApproxPointKey -> new global vertex id (for convenience).
pub fn allocate_vertices_for_splits_no_topology<T: Scalar, const N: usize>(
    mesh: &mut Mesh<T, N>,
    intersection_segments: &mut Vec<IntersectionSegment<T, N>>,
) -> AHashMap<ApproxPointKey, usize>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    // Map keyed by (canonical_point_key, half_edge_hint_or_MAX, face_hint_or_MAX, endpoint_flag) -> allocated global vid.
    // endpoint_flag = 1 when the split is essentially at an edge endpoint (u ≈ 0 or u ≈ 1),
    //               = 0 for interior-of-edge, face-interior, or no-edge cases.
    let mut keyed_map: AHashMap<(ApproxPointKey, (usize, usize), [usize; 2], u8), usize> =
        AHashMap::default();

    // Returned convenience map: canonical_point_key -> one representative vid for *no-edge* bucket only
    let mut canonical_map: AHashMap<ApproxPointKey, usize> = AHashMap::default();
    let mut edge_param_map: AHashMap<((usize, usize), i64), usize> = AHashMap::default();

    // threshold for treating half_edge_u_hint as an endpoint
    let tol_t: T = T::point_merge_threshold();
    let tol_f64: f64 = RefInto::<CgarF64>::ref_into(&tol_t).0;

    for seg in intersection_segments.iter_mut() {
        if seg.invalidated {
            continue;
        }
        for (i, ep) in [&mut seg.a, &mut seg.b].iter().enumerate() {
            let ep_pos = &seg.segment[i];
            let canonical = point_key(&ep_pos);

            // Extract edge context
            let he_opt = ep.half_edge_hint;
            let edge_key: (usize, usize) = match he_opt {
                Some(h) => undirected_edge_key(mesh, h).unwrap_or((usize::MAX, usize::MAX)),
                None => (usize::MAX, usize::MAX),
            };

            // Extract face context
            let faces_opt = ep.faces_hint;
            let faces_key = faces_opt.unwrap_or([usize::MAX, usize::MAX]);

            // Determine u along the edge if available
            let u_opt_f64: Option<f64> = ep
                .half_edge_u_hint
                .as_ref()
                .map(|u_t| RefInto::<CgarF64>::ref_into(u_t).0);

            // endpoint_flag = 1 if u is near 0 or near 1 (use tol), else 0
            let endpoint_flag: u8 = match u_opt_f64 {
                Some(u) if (u <= tol_f64) || (u >= 1.0 - tol_f64) => 1,
                _ => 0,
            };
            let map_key = (canonical, edge_key, faces_key, endpoint_flag);

            // only accept an existing vertex_hint as authoritative when
            // it is no-edge (he_key == MAX) OR the endpoint is effectively at the edge endpoint (endpoint_flag == 1).
            if let Some(hint) = ep.vertex_hint {
                if hint[0] != usize::MAX
                    && (edge_key == (usize::MAX, usize::MAX) || endpoint_flag == 1)
                {
                    keyed_map.entry(map_key).or_insert(hint[0]);
                    // don't populate canonical_map here unless it's the no-edge case
                    if edge_key == (usize::MAX, usize::MAX) && faces_key == [usize::MAX, usize::MAX]
                    {
                        canonical_map.entry(canonical).or_insert(hint[0]);
                    }
                    continue;
                }
            }

            // 1b) If this map_key already decided, reuse
            if let Some(&vid) = keyed_map.get(&map_key) {
                continue;
            }

            // 1c) If endpoint_flag==1 and we have half-edge hint, reuse the proper endpoint vertex:
            if endpoint_flag == 1 && he_opt.is_some() {
                let he = he_opt.unwrap();
                if he < mesh.half_edges.len() {
                    // decide whether this is source (u≈0) or dest (u≈1)
                    if let Some(u) = u_opt_f64 {
                        let chosen_vid = if u <= tol_f64 {
                            // source vertex = previous half-edge's vertex
                            let src_he = mesh.half_edges[he].prev;
                            mesh.half_edges[src_he].vertex
                        } else {
                            // dest vertex
                            mesh.half_edges[he].vertex
                        };
                        keyed_map.insert(map_key, chosen_vid);
                        // don't set canonical_map for edge-derived allocations
                        continue;
                    }
                }
            }

            // 1d) Interior-of-edge (he present, not near endpoints): one vertex per (undirected edge, canonical u bucket)
            if let (Some(he), Some(u_raw)) = (he_opt, u_opt_f64) {
                if endpoint_flag == 0 {
                    if let Some(ek) = undirected_edge_key(mesh, he) {
                        let u_can = canonicalize_u_for_edge(mesh, he, ek, u_raw);
                        let ub = bucket_u(u_can, tol_f64);
                        let k = (ek, ub);

                        // Reuse if we’ve already allocated on this edge/param
                        let vid = *edge_param_map.entry(k).or_insert_with(|| {
                            let (vid, existed) = mesh.get_or_insert_vertex(&ep_pos);
                            vid
                        });

                        // Record under your original keyed_map so write-back can find it with this endpoint’s map_key
                        keyed_map.insert(map_key, vid);
                        // no canonical_map update for edge-derived allocations
                        continue;
                    }
                }
            }

            // 1e) If face-interior (face_hint + barycentric_hint), allocate one vertex per (canonical,MAX,face_key,0)
            if faces_opt.is_some() && ep.barycentric_hint.is_some() {
                let map_key_face = (canonical, (usize::MAX, usize::MAX), faces_key, 0u8);
                if let Some(&vid) = keyed_map.get(&map_key_face) {
                    // already allocated for this canonical/face combination
                    continue;
                }
                let new_vid = mesh.get_or_insert_vertex(&ep_pos);
                keyed_map.insert(map_key_face, new_vid.0);
                // do not set canonical_map here (avoid cross-face fallback)
                continue;
            }

            // 1f) No half-edge or face context -> bucket by canonical key (allocate if needed)
            let map_key_nocontext = (
                canonical,
                (usize::MAX, usize::MAX),
                [usize::MAX, usize::MAX],
                0u8,
            );
            if let Some(&vid) = keyed_map.get(&map_key_nocontext) {
                // already allocated for canonical/no-context
                canonical_map.entry(canonical).or_insert(vid);
                continue;
            }
            let new_vid = mesh.get_or_insert_vertex(&ep_pos);
            keyed_map.insert(map_key_nocontext, new_vid.0);
            canonical_map.entry(canonical).or_insert(new_vid.0);
        }
    }

    // 2) Write back vertex_hint for every endpoint referenced by splits.
    //    We set vertex_hint = [vid, usize::MAX] and clear half_edge_hint / u / face / barycentric.
    //    IMPORTANT: do not fallback to canonical_map for arbitrary (canonical,he,face) keys to avoid merging
    //    interior-of-different-half-edge or interior-of-different-face points into a single canonical representative.
    for seg in intersection_segments.iter_mut() {
        if seg.invalidated {
            continue;
        }

        for (i, ep) in [&mut seg.a, &mut seg.b].iter_mut().enumerate() {
            let ep_pos = &seg.segment[i];
            let canonical = point_key(&ep_pos);

            let he_opt = ep.half_edge_hint;
            let edge_key: (usize, usize) = match he_opt {
                Some(h) => undirected_edge_key(mesh, h).unwrap_or((usize::MAX, usize::MAX)),
                None => (usize::MAX, usize::MAX),
            };
            let faces_opt = ep.faces_hint;
            let faces_key = faces_opt.unwrap_or([usize::MAX, usize::MAX]);

            let u_opt_f64: Option<f64> = ep
                .half_edge_u_hint
                .as_ref()
                .map(|u_t| RefInto::<CgarF64>::ref_into(u_t).0);
            let endpoint_flag: u8 = match u_opt_f64 {
                Some(u) if (u <= tol_f64) || (u >= 1.0 - tol_f64) => 1,
                _ => 0,
            };

            let faces_key_for_map = if edge_key != (usize::MAX, usize::MAX) {
                [usize::MAX, usize::MAX]
            } else {
                faces_key
            };
            // prefer the exact (canonical,he,face,endpoint_flag) bucket; fall back to canonical/nocontext only
            let map_key = (canonical, edge_key, faces_key_for_map, endpoint_flag);
            let map_key_nocontext = (canonical, (usize::MAX, usize::MAX), [usize::MAX; 2], 0u8);

            let vid = keyed_map
                .get(&map_key)
                .copied()
                .or_else(|| keyed_map.get(&map_key_nocontext).copied())
                // do NOT use canonical_map as general fallback here
                .unwrap_or_else(|| {
                    // If still missing, allocate a fresh vertex for this exact endpoint to avoid accidental merges.
                    mesh.get_or_insert_vertex(&ep_pos).0
                });

            // Set vertex_hint to vid (slot 0) and clear other geometric hints
            ep.vertex_hint = Some([vid, usize::MAX]);
            if ep.half_edge_hint.is_none() && ep.faces_hint.is_none() {
                ep.faces_hint = Some([seg.initial_face_reference, usize::MAX]);
            }
            ep.half_edge_hint = None;
            ep.half_edge_u_hint = None;
            ep.barycentric_hint = None;
        }
    }

    // 3) Update each segment's resulting_vertices_pair from endpoint vertex_hint
    for seg in intersection_segments.iter_mut() {
        let a_vid = seg.a.vertex_hint.map(|h| h[0]).unwrap_or(usize::MAX);
        let b_vid = seg.b.vertex_hint.map(|h| h[0]).unwrap_or(usize::MAX);
        seg.a.resulting_vertex = Some(a_vid);
        seg.b.resulting_vertex = Some(b_vid);
    }

    canonical_map
}

/// Metric-preserving, sqrt-free UV frame.
/// Guarantees u and v are linearly independent for any non-degenerate triangle.
/// u = (b-a)
/// v = ( ( (b-a) × (c-a) ) × u )  [in-plane and ⟂ u]
pub fn face_uv_frame_metric<T: Scalar, const N: usize>(
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> (Vector<T, N>, Vector<T, N>)
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let e0 = (b - a).as_vector();
    let e1 = (c - a).as_vector();

    // Choose a non-zero u
    let mut u = e0;
    if u.dot(&u) == T::zero() {
        u = e1.clone();
    }

    // Candidate v from double cross: v = (n × u) with n = u × e1
    let n = u.cross(&e1);
    let mut v = n.cross(&u);

    // If v vanished (collinear e0,e1), fall back to Gram–Schmidt: v = e1 - proj_u(e1)
    if v.dot(&v) == T::zero() {
        let uu = u.dot(&u);
        if uu != T::zero() {
            let ue1 = u.dot(&e1);
            let proj = u.scale(&(ue1 / uu));
            v = &e1 - &proj; // in-plane
        }
        // If still zero, the face is degenerate; caller should skip this face.
    }

    (u, v)
}

#[inline(always)]
pub fn project_to_uv_metric<T: Scalar + Clone + PartialEq, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    u: &Vector<T, N>,
    v: &Vector<T, N>,
) -> Point2<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
    for<'x> &'x T: std::ops::Add<&'x T, Output = T>
        + std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let q = (p - a).as_vector();

    // Gram matrix and RHS
    let uu = u.dot(u);
    let uv = u.dot(v);
    let vv = v.dot(v);
    let r0 = q.dot(u);
    let r1 = q.dot(v);

    // Solve [uu uv; uv vv] [α β]^T = [r0 r1]^T
    let den = &uu * &vv - &uv * &uv;

    if den == T::zero() {
        // Degenerate basis; fall back to 1D projection on whichever axis is usable.
        if uu != T::zero() {
            let alpha = r0 / uu.clone();
            return Point2::<T>::from_vals([alpha, T::zero()]);
        } else if vv != T::zero() {
            let beta = r1 / vv.clone();
            return Point2::<T>::from_vals([T::zero(), beta]);
        } else {
            // u and v are both zero -> fully degenerate
            return Point2::<T>::from_vals([T::zero(), T::zero()]);
        }
    }

    let alpha = (&r0 * &vv - &r1 * &uv) / den.clone();
    let beta = (&r1 * &uu - &r0 * &uv) / den;
    Point2::<T>::from_vals([alpha, beta])
}

fn barycentric_uv<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> Point2<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'x> &'x T: std::ops::Add<&'x T, Output = T>
        + std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>,
{
    let v0 = (b - a).as_vector();
    let v1 = (c - a).as_vector();
    let v2 = (p - a).as_vector();

    // Solve v2 = v0 * α + v1 * β in least-squares sense (2×2), but do it by
    // dot products (stable if v0,v1 are independent)
    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);
    let den = &d00 * &d11 - &d01 * &d01;

    if den == T::zero() {
        // extremely degenerate; fall back to a 1D projection
        return Point2::<T>::from_vals([T::zero(), T::zero()]);
    }

    let alpha = (&d20 * &d11 - &d21 * &d01) / den.clone();
    let beta = (&d21 * &d00 - &d20 * &d01) / den;
    Point2::<T>::from_vals([alpha, beta])
}

#[inline]
fn push_vertex_uv<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    g2l: &mut AHashMap<usize, usize>,
    verts_global: &mut Vec<usize>,
    points_uv: &mut Vec<Point2<T>>,
    pa: &Point<T, N>,
    pb: &Point<T, N>,
    pc: &Point<T, N>,
    vid: usize,
) -> usize
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'x> &'x T: std::ops::Add<&'x T, Output = T>
        + std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>,
{
    if let Some(&li) = g2l.get(&vid) {
        return li;
    }
    let p = &mesh.vertices[vid].position;
    let li = points_uv.len();
    points_uv.push(barycentric_uv(p, pa, pb, pc));
    verts_global.push(vid);
    g2l.insert(vid, li);
    li
}

/// 2) Build a PSLG per face: boundary chains + interior points + intersection segments.
///    Requires that `allocate_vertices_for_splits_no_topology` has already run,
///    so every relevant endpoint has a `vertex_hint`.
pub fn build_face_pslgs<T: Scalar + Clone + PartialOrd, const N: usize>(
    mesh: &Mesh<T, N>,
    intersection_segments: &[IntersectionSegment<T, N>],
) -> Vec<FaceJobUV<T>>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    use ahash::{AHashMap, AHashSet as FastSet};

    // Collect intersection segments per face (using resulting vertex ids).
    let mut by_face: AHashMap<usize, Vec<[usize; 2]>> = AHashMap::default();
    for seg in intersection_segments {
        if seg.invalidated {
            continue;
        }
        let a = seg.a.resulting_vertex.unwrap();
        let b = seg.b.resulting_vertex.unwrap();
        if a == usize::MAX || b == usize::MAX || a == b {
            continue;
        }

        by_face
            .entry(seg.initial_face_reference)
            .or_default()
            .push([a, b]);

        // For each endpoint, check if it belongs to adjacent faces
        for (ep_idx, &vertex_id) in [a, b].iter().enumerate() {
            let ep = if ep_idx == 0 { &seg.a } else { &seg.b };

            if let Some(faces) = ep.faces_hint {
                for &face_id in &faces {
                    if face_id != usize::MAX && face_id != seg.initial_face_reference {
                        // This vertex belongs to an adjacent face
                        by_face
                            .entry(face_id)
                            .or_default()
                            .push([vertex_id, usize::MAX]); // Single vertex
                    }
                }
            }
        }
    }

    let mut jobs: Vec<FaceJobUV<T>> = Vec::with_capacity(by_face.len());

    // --- helpers (UV) --------------------------------------------------------

    // build the ordered chain of local indices that lie on segment (i,j) (including i and j)
    fn chain_on_edge<TS: Scalar>(i: usize, j: usize, pts: &[Point2<TS>]) -> Vec<usize>
    where
        for<'x> &'x TS: std::ops::Sub<&'x TS, Output = TS>
            + std::ops::Mul<&'x TS, Output = TS>
            + std::ops::Add<&'x TS, Output = TS>,
    {
        let a = &pts[i];
        let b = &pts[j];
        let mut out: Vec<(f64, usize)> = Vec::new();
        for (k, p) in pts.iter().enumerate() {
            let (on, t) = on_edge_with_t(a, b, p);
            if on {
                out.push((t, k));
            }
        }
        out.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        out.into_iter().map(|(_, k)| k).collect()
    }

    // ------------------------------------------------------------------------

    for (face_id, vertex_pairs) in by_face {
        if mesh.faces[face_id].removed {
            continue;
        }

        // Face geometry
        let [ia, ib, ic] = mesh.face_vertices(face_id);
        let pa = &mesh.vertices[ia].position;
        let pb = &mesh.vertices[ib].position;
        let pc = &mesh.vertices[ic].position;

        // UV frame (metric-preserving, sqrt-free)
        let (u, v) = face_uv_frame_metric(pa, pb, pc);

        // Maps / buffers
        let mut g2l: AHashMap<usize, usize> = AHashMap::default();
        let mut verts_global: Vec<usize> = Vec::new();
        let mut points_uv: Vec<Point2<T>> = Vec::new();

        // boundary verts first: local 0,1,2 map to (ia,ib,ic)
        let boundary_verts = [ia, ib, ic];
        for &bv in &boundary_verts {
            push_vertex_uv(
                mesh,
                &mut g2l,
                &mut verts_global,
                &mut points_uv,
                pa,
                pb,
                pc,
                bv,
            );
        }

        // Resolve each endpoint to the vertex that belongs to THIS face.
        // We also keep the resolved pairs to emit segments later.
        let mut resolved_pairs: Vec<(usize, usize)> = Vec::with_capacity(vertex_pairs.len());
        for &[va, vb] in &vertex_pairs {
            let mut a_res = usize::MAX;
            let mut b_res = usize::MAX;
            // project both candidates in the SAME basis you use for points_uv
            if va != usize::MAX {
                let pa_uv = barycentric_uv(&mesh.vertices[va].position, pa, pb, pc);
                a_res = choose_vertex_on_face(mesh, face_id, &[va, vb], &pa_uv, pa, pb, pc)
                    .unwrap_or(va);
                if !boundary_verts.contains(&a_res) {
                    push_vertex_uv(
                        mesh,
                        &mut g2l,
                        &mut verts_global,
                        &mut points_uv,
                        pa,
                        pb,
                        pc,
                        a_res,
                    );
                }
            }

            if vb != usize::MAX {
                let pb_uv = barycentric_uv(&mesh.vertices[vb].position, pa, pb, pc);
                b_res = choose_vertex_on_face(mesh, face_id, &[vb, va], &pb_uv, pa, pb, pc)
                    .unwrap_or(vb);
                if !boundary_verts.contains(&b_res) {
                    push_vertex_uv(
                        mesh,
                        &mut g2l,
                        &mut verts_global,
                        &mut points_uv,
                        pa,
                        pb,
                        pc,
                        b_res,
                    );
                }
            }

            resolved_pairs.push((a_res, b_res));
        }

        // constraints (PSLG)
        let mut segments: Vec<[usize; 2]> = Vec::new();
        let mut forced: FastSet<(usize, usize)> = FastSet::default();

        // 1) boundary split at on-edge vertices
        add_split_or_chain_uv(0, 1, &mut segments, &points_uv);
        add_split_or_chain_uv(1, 2, &mut segments, &points_uv);
        add_split_or_chain_uv(2, 0, &mut segments, &points_uv);

        // Mark **all** edges along those three chains as forced
        let mark_chain_as_forced = |i0: usize, i1: usize, forced: &mut FastSet<(usize, usize)>| {
            let chain = chain_on_edge(i0, i1, &points_uv);
            for w in chain.windows(2) {
                let (a, b) = (w[0], w[1]);
                let key = if a < b { (a, b) } else { (b, a) };
                forced.insert(key);
            }
        };
        mark_chain_as_forced(0, 1, &mut forced);
        mark_chain_as_forced(1, 2, &mut forced);
        mark_chain_as_forced(2, 0, &mut forced);

        // ---- existing intersection segments ----
        for &(va, vb) in &resolved_pairs {
            if va != usize::MAX && vb != usize::MAX {
                if let (Some(&la), Some(&lb)) = (g2l.get(&va), g2l.get(&vb)) {
                    if la != lb {
                        add_split_or_chain_uv(la, lb, &mut segments, &points_uv);
                    }
                }
            }
        }

        // ---- your fan from boundary vertex 0 ----
        // let mut ring = {
        //     let mut c01 = chain_on_edge(0, 1, &points_uv);
        //     let mut c12 = chain_on_edge(1, 2, &points_uv);
        //     let mut c20 = chain_on_edge(2, 0, &points_uv);
        //     if !c01.is_empty() {
        //         c01.pop();
        //     } // drop '1'
        //     if !c12.is_empty() {
        //         c12.pop();
        //     } // drop '2'
        //     if !c20.is_empty() {
        //         c20.pop();
        //     } // drop '0'
        //     let mut r = Vec::with_capacity(c01.len() + c12.len() + c20.len() + 3);
        //     r.push(0);
        //     r.extend(c01);
        //     r.push(1);
        //     r.extend(c12);
        //     r.push(2);
        //     r.extend(c20);
        //     r
        // };

        // insert spokes (0,k) and mark them as FORCED
        // if ring.len() >= 4 {
        //     let nbr_left = *ring.last().unwrap();
        //     let nbr_right = ring[1];
        //     for &k in &ring[2..ring.len() - 1] {
        //         if k == nbr_left || k == nbr_right {
        //             continue;
        //         }
        //         segments.push([0, k]);
        //         let (a, b) = if 0 < k { (0, k) } else { (k, 0) };
        //         forced.insert((a, b));
        //     }
        // }

        // ---- pruning & dedup: keep forced edges ----
        {
            // dedupe undirected
            let mut seen: FastSet<(usize, usize)> = FastSet::default();
            let mut uniq: Vec<[usize; 2]> = Vec::with_capacity(segments.len());
            for [a, b] in segments.drain(..) {
                if a == b {
                    continue;
                }
                let key = if a < b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    uniq.push([a, b]);
                }
            }
            segments = uniq;
        }

        // relaxed pruning: DO NOT drop fan spokes in `forced`
        segments.retain(|[a, b]| {
            if a == b {
                return false;
            }
            let key = if *a < *b { (*a, *b) } else { (*b, *a) };

            // 1) keep anything we declared forced
            if forced.contains(&key) {
                return true;
            }

            let a_is_boundary = *a < 3;
            let b_is_boundary = *b < 3;

            // 2) boundary–boundary: keep only true outer-edge chords
            if a_is_boundary && b_is_boundary {
                return seg_on_any_outer_edge(*a, *b, &points_uv);
            }

            // 3) boundary–interior: allow only if the segment lies on an outer edge
            if a_is_boundary ^ b_is_boundary {
                return seg_on_any_outer_edge(*a, *b, &points_uv);
            }

            // 4) interior–interior: always ok (intersection chords etc.)
            true
        });

        jobs.push(FaceJobUV {
            face_id,
            verts_global,
            points_uv,
            segments,
        });
    }

    jobs
}

/// Validate correspondence between PSLG jobs and their CDTs.
///
/// Prints concise diagnostics:
///  - job.points_uv.len() vs job.verts_global.len()
///  - any CDT triangle that references an index >= job.points_uv.len()
///  - mapped global triangles (ga,gb,gc)
///  - degenerate mapped triangles (duplicate global vertex ids)
///
/// Returns Ok(()) if no fatal problems found, Err(()) otherwise.
pub fn validate_jobs_cdts<T: Scalar, const N: usize>(
    jobs: &[FaceJobUV<T>],
    cdts: &[Delaunay<T>],
) -> Result<(), ()>
where
    Point2<T>: Clone,
{
    if jobs.len() != cdts.len() {
        eprintln!(
            "validate_jobs_cdts: length mismatch jobs={} cdts={}",
            jobs.len(),
            cdts.len()
        );
        return Err(());
    }

    let mut fatal = false;

    for (i, (job, dt)) in jobs.iter().zip(cdts.iter()).enumerate() {
        let pu = job.points_uv.len();
        let vg = job.verts_global.len();
        if pu != vg {
            eprintln!("Job {}: points_uv={} verts_global={} (mismatch)", i, pu, vg);
            fatal = true;
        }

        // Basic check: ensure no triangle references helper/super points
        for (ti, tri) in dt.triangles.iter().enumerate() {
            let (a, b, c) = (tri.0, tri.1, tri.2);

            if a >= pu || b >= pu || c >= pu {
                eprintln!(
                    "Job {}: CDT triangle {} references out-of-range local index -> tri=({},{},{}) points_uv={}",
                    i, ti, a, b, c, pu
                );
                // Print dt.points for quick trace (only once per job)
                eprintln!("  dt.points.len() = {}", dt.points.len());
                fatal = true;
                continue;
            }

            // Map local -> global
            let ga = job.verts_global[a];
            let gb = job.verts_global[b];
            let gc = job.verts_global[c];

            // Report mapped triangle (concise)
            eprintln!("Job {} tri {} -> global ({},{},{})", i, ti, ga, gb, gc);

            // Degeneracy check
            if ga == gb || gb == gc || ga == gc {
                eprintln!(
                    "  DEG: Job {} tri {} maps to degenerate global triangle ({},{},{})",
                    i, ti, ga, gb, gc
                );
                fatal = true;
            }
        }

        // Report duplicate mapped triangles (optional but useful)
        let mut seen: AHashSet<(usize, usize, usize)> = AHashSet::new();
        for tri in dt.triangles.iter() {
            if tri.0 >= job.points_uv.len()
                || tri.1 >= job.points_uv.len()
                || tri.2 >= job.points_uv.len()
            {
                continue;
            }
            let mut g = [
                job.verts_global[tri.0],
                job.verts_global[tri.1],
                job.verts_global[tri.2],
            ];
            g.sort_unstable();
            let key = (g[0], g[1], g[2]);
            if !seen.insert(key) {
                eprintln!(
                    "Job {}: duplicate mapped triangle (global sorted) {:?}",
                    i, key
                );
                fatal = true;
            }
        }
    }

    if fatal {
        Err(())
    } else {
        eprintln!("validate_jobs_cdts: no obvious mapping problems found");
        Ok(())
    }
}

/// Dump detailed diagnostics for a single PSLG job + CDT.
/// Useful to inspect:
///  - local -> global vertex mapping
///  - whether the same global vertex appears at multiple local indices
///  - CDT local triangles and their mapped global triples
///  - duplicated local triangles and duplicated mapped global triangles
pub fn dump_job_debug<T: Scalar, const N: usize>(
    job_i: usize,
    job: &FaceJobUV<T>,
    dt: &Delaunay<T>,
) {
    use ahash::AHashMap;
    use ahash::AHashSet;

    eprintln!(
        "--- dump_job_debug: job {} face_id={} ---",
        job_i, job.face_id
    );
    eprintln!(
        "points_uv.len() = {}, verts_global.len() = {}",
        job.points_uv.len(),
        job.verts_global.len()
    );
    eprintln!(
        "dt.points.len() = {}, dt.triangles.len() = {}",
        dt.points.len(),
        dt.triangles.len()
    );

    // local -> global listing
    eprintln!("local -> global mapping (local_idx -> global_id):");
    for (li, &gid) in job.verts_global.iter().enumerate() {
        eprintln!("  {} -> {}", li, gid);
    }

    // find collisions: global_id -> list of local indices that map to it
    let mut global_to_locals: AHashMap<usize, Vec<usize>> = AHashMap::default();
    for (li, &gid) in job.verts_global.iter().enumerate() {
        global_to_locals.entry(gid).or_default().push(li);
    }
    let mut had_collisions = false;
    for (gid, locals) in &global_to_locals {
        if locals.len() > 1 {
            had_collisions = true;
            eprintln!(
                "COLLISION: global {} appears at local indices {:?}",
                gid, locals
            );
        }
    }
    if !had_collisions {
        eprintln!("no local->global collisions detected");
    }

    // scan local triangles and mapped globals; collect duplicates
    let mut seen_local: AHashSet<(usize, usize, usize)> = AHashSet::default();
    let mut dup_local: Vec<(usize, (usize, usize, usize))> = Vec::new();
    let mut seen_mapped: AHashSet<(usize, usize, usize)> = AHashSet::default();
    let mut dup_mapped: Vec<(usize, (usize, usize, usize))> = Vec::new();

    for (ti, tri) in dt.triangles.iter().enumerate() {
        let local = (tri.0, tri.1, tri.2);
        eprintln!("tri {} (local) = {:?}", ti, local);

        // duplicate local triangles detection (sorted canonical)
        let mut lkey = [local.0, local.1, local.2];
        lkey.sort_unstable();
        let lkey_t = (lkey[0], lkey[1], lkey[2]);
        if !seen_local.insert(lkey_t) {
            dup_local.push((ti, lkey_t));
        }

        // validate indices and map to global if possible
        if local.0 >= job.verts_global.len()
            || local.1 >= job.verts_global.len()
            || local.2 >= job.verts_global.len()
        {
            eprintln!(
                "  WARNING: tri {} references out-of-range local index (points_uv.len()={})",
                ti,
                job.points_uv.len()
            );
            continue;
        }

        let mapped = (
            job.verts_global[local.0],
            job.verts_global[local.1],
            job.verts_global[local.2],
        );
        eprintln!("  -> mapped global = {:?}", mapped);

        // degenerate mapped triangle?
        if mapped.0 == mapped.1 || mapped.1 == mapped.2 || mapped.0 == mapped.2 {
            eprintln!(
                "  WARNING: tri {} maps to degenerate global triangle {:?}",
                ti, mapped
            );
        }

        // duplicate mapped triangles detection (sorted)
        let mut mkey = [mapped.0, mapped.1, mapped.2];
        mkey.sort_unstable();
        let mkey_t = (mkey[0], mkey[1], mkey[2]);
        if !seen_mapped.insert(mkey_t) {
            dup_mapped.push((ti, mkey_t));
        }
    }

    if !dup_local.is_empty() {
        eprintln!(
            "Found {} duplicate local triangles (first entries):",
            dup_local.len()
        );
        for (ti, key) in dup_local.iter().take(8) {
            eprintln!("  local tri idx {} duplicated canonical {:?}", ti, key);
        }
    } else {
        eprintln!("No duplicate local triangles detected");
    }

    if !dup_mapped.is_empty() {
        eprintln!(
            "Found {} duplicate mapped global triangles (first entries):",
            dup_mapped.len()
        );
        for (ti, key) in dup_mapped.iter().take(8) {
            eprintln!(
                "  tri idx {} maps to duplicated global canonical {:?}",
                ti, key
            );
        }
    } else {
        eprintln!("No duplicate mapped global triangles detected");
    }

    eprintln!("--- end dump_job_debug: job {} ---", job_i);
}

// Compact a single FaceJobUV by deduplicating points_uv (quantized by f64 bit pattern).
// Returns a new FaceJobUV with:
//  - points_uv deduplicated
//  - verts_global remapped to the compacted local indices (one global id per unique local point, first-wins)
//  - segments remapped to the new local indices (degenerate segments removed)
pub fn compact_face_job_for_delaunay<T: Scalar>(job: &FaceJobUV<T>) -> FaceJobUV<T>
where
    Point2<T>: Clone,
{
    // quantizer identical to Delaunay::q (bitwise)
    #[inline]
    fn q(x: f64) -> i64 {
        x.to_bits() as i64
    }

    let n = job.points_uv.len();
    let mut new_points_uv: Vec<Point2<T>> = Vec::with_capacity(n);
    let mut new_verts_global: Vec<usize> = Vec::with_capacity(n);
    let mut old_to_new: Vec<usize> = vec![usize::MAX; n];

    // For each quantized key we keep a small list of (global_id -> new_local_index).
    // We only reuse a local index if the global_id is the same; otherwise we create a distinct local.
    let mut created_by_key: AHashMap<(i64, i64), Vec<(usize, usize)>> = AHashMap::default();

    for old_local in 0..n {
        let p = &job.points_uv[old_local];
        let key = (q(p[0].to_f64().unwrap()), q(p[1].to_f64().unwrap()));
        let global_id = job.verts_global[old_local];

        if let Some(list) = created_by_key.get_mut(&key) {
            // Search for existing entry with same global id
            if let Some((_, new_idx)) = list.iter().find(|(gid, _)| *gid == global_id) {
                old_to_new[old_local] = *new_idx;
                continue;
            } else {
                // No same-global entry → create new local entry for this distinct global vertex
                let new_idx = new_points_uv.len();
                new_points_uv.push(p.clone());
                new_verts_global.push(global_id);
                list.push((global_id, new_idx));
                old_to_new[old_local] = new_idx;
                continue;
            }
        } else {
            // First time we see this quantized key
            let new_idx = new_points_uv.len();
            new_points_uv.push(p.clone());
            new_verts_global.push(global_id);
            created_by_key.insert(key, vec![(global_id, new_idx)]);
            old_to_new[old_local] = new_idx;
        }
    }

    // Remap segments and remove degenerate/duplicate segments
    let mut seg_set: AHashSet<(usize, usize)> = AHashSet::default();
    let mut new_segments: Vec<[usize; 2]> = Vec::with_capacity(job.segments.len());
    for seg in &job.segments {
        let a = old_to_new[seg[0]];
        let b = old_to_new[seg[1]];
        if a == usize::MAX || b == usize::MAX {
            // Shouldn't happen, but skip defensively
            continue;
        }
        if a == b {
            continue; // degenerate
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if seg_set.insert(key) {
            // keep original orientation a->b
            new_segments.push([a, b]);
        }
    }

    FaceJobUV {
        face_id: job.face_id,
        verts_global: new_verts_global,
        points_uv: new_points_uv,
        segments: new_segments,
    }
}

// Compact many jobs (fast, allocation proportional to jobs size).
pub fn compact_jobs_for_delaunay<T: Scalar>(jobs: &[FaceJobUV<T>]) -> Vec<FaceJobUV<T>>
where
    Point2<T>: Clone,
{
    let mut out: Vec<FaceJobUV<T>> = Vec::with_capacity(jobs.len());
    for job in jobs {
        out.push(compact_face_job_for_delaunay(job));
    }

    out
}

pub fn weld_vertices<T: Scalar, const N: usize>(mesh: &mut Mesh<T, N>)
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
{
    use ahash::AHashMap;

    // Map approximate point key -> new vertex index
    let mut canonical: AHashMap<ApproxPointKey, usize> = AHashMap::default();
    let mut new_vertices = Vec::with_capacity(mesh.vertices.len());
    let mut old_to_new: Vec<usize> = vec![usize::MAX; mesh.vertices.len()];

    for (old_i, v) in mesh.vertices.iter().enumerate() {
        let key = point_key(&v.position);
        if let Some(&new_i) = canonical.get(&key) {
            old_to_new[old_i] = new_i;
        } else {
            let new_i = new_vertices.len();
            new_vertices.push(v.clone());
            canonical.insert(key, new_i);
            old_to_new[old_i] = new_i;
        }
    }

    // Replace vertex array with compacted unique list
    mesh.vertices = new_vertices;

    // Remap half-edge vertex references to canonical indices
    for he in mesh.half_edges.iter_mut() {
        let old_vid = he.vertex;
        he.vertex = old_to_new[old_vid];
    }

    // Rebuild edge map to reflect the new vertex ids and half-edge -> (u,v) keys
    // rebuild_edge_map(mesh);
}

#[inline(always)]
fn undirected_edge_key<TS: Scalar, const M: usize>(
    mesh: &Mesh<TS, M>,
    he: usize,
) -> Option<(usize, usize)> {
    if he >= mesh.half_edges.len() {
        return None;
    }
    let h = &mesh.half_edges[he];
    let t = h.twin;
    if t >= mesh.half_edges.len() {
        return None;
    }
    let head = h.vertex; // v
    let tail = mesh.half_edges[t].vertex; // u (origin)
    Some(if tail <= head {
        (tail, head)
    } else {
        (head, tail)
    })
}

#[inline(always)]
fn canonicalize_u_for_edge<TS: Scalar, const M: usize>(
    mesh: &Mesh<TS, M>,
    he: usize,
    edge_key: (usize, usize),
    u_raw: f64,
) -> f64 {
    // half-edge origin
    let tail = mesh.half_edges[mesh.half_edges[he].twin].vertex;
    // If this directed half-edge runs edge_key.0 -> edge_key.1, keep u; else flip.
    if tail == edge_key.0 {
        u_raw
    } else {
        1.0 - u_raw
    }
}

#[inline(always)]
fn bucket_u(u: f64, eps: f64) -> i64 {
    // Quantize to tolerance-sized bins
    ((u / eps).round() as i64).clamp(i64::MIN / 4, i64::MAX / 4)
}

#[inline(always)]
fn edge_ok_now<TS: Scalar, const M: usize>(m: &Mesh<TS, M>, u: usize, v: usize) -> bool {
    match m.edge_map.get(&(u, v)).copied() {
        None => true, // not present yet -> ok to create (u->v)
        Some(h) => {
            if h >= m.half_edges.len() {
                return false;
            }
            let he = &m.half_edges[h];
            if he.removed {
                return true;
            } // safe to recreate
            if he.face.is_none() {
                return true;
            } // BORDER on (u->v) side

            // Our side is already interior. If the *other* side is still BORDER,
            // we should allow the tri that uses (v->u).
            if let Some(&ht) = m.edge_map.get(&(v, u)) {
                if ht < m.half_edges.len() {
                    let twin = &m.half_edges[ht];
                    return !twin.removed && twin.face.is_none(); // BORDER on reverse
                }
            }
            false
        }
    }
}

#[inline(always)]
fn orient_ok_now<TS: Scalar, const M: usize>(
    m: &Mesh<TS, M>,
    a: usize,
    b: usize,
    c: usize,
) -> bool {
    edge_ok_now(m, a, b) && edge_ok_now(m, b, c) && edge_ok_now(m, c, a)
}

pub fn print_face_job_and_dt<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    job: &FaceJobUV<T>,
    dt: &Delaunay<T>,
) {
    println!("FaceJobUV face_id={}", job.face_id);
    println!(
        "  verts_global (len={}): {:?}",
        job.verts_global.len(),
        job.verts_global
    );
    println!("  points_uv (len={}):", job.points_uv.len());
    for (i, p) in job.points_uv.iter().enumerate() {
        println!(
            "    {}: ({}, {})",
            i,
            p[0].to_f64().unwrap(),
            p[1].to_f64().unwrap()
        );
    }
    println!("  segments (len={}):", job.segments.len(),);

    for (i, seg) in job.segments.iter().enumerate() {
        let global_idxs = seg.map(|li| job.verts_global[li]);
        println!("    {}: {:?} / {:?}", i, seg, global_idxs);
        println!("      with positions:");
        for idx in global_idxs {
            let pos = mesh.vertices[idx].position.clone();
            println!("        - {:?}", pos);
        }
    }

    println!("dt triangle positions:");
    for (i, tri) in dt.triangles.iter().enumerate() {
        let global_idxs = [tri.0, tri.1, tri.2].map(|li| job.verts_global[li]);
        println!("  Triangle {}: {:?}", i, tri);
        println!("    with positions:");
        for idx in global_idxs {
            let pos = mesh.vertices[idx].position.clone();
            println!("      - {:?}", pos);
        }
    }
}

fn segment_is_on_outer_edge<T: Scalar>(a: usize, b: usize, pts: &[Point2<T>]) -> bool {
    if a > 2 || b > 2 {
        return false;
    } // both must be boundary indices 0,1,2

    let pairs = [(0, 1), (1, 2), (2, 0)];
    let eps = RefInto::<CgarF64>::ref_into(&T::point_merge_threshold())
        .0
        .max(1e-12);

    let on_line = |i0: usize, i1: usize, k: usize| -> Option<f64> {
        let p0 = &pts[i0];
        let p1 = &pts[i1];
        let pk = &pts[k];
        let ax = RefInto::<CgarF64>::ref_into(&p0[0]).0;
        let ay = RefInto::<CgarF64>::ref_into(&p0[1]).0;
        let bx = RefInto::<CgarF64>::ref_into(&p1[0]).0;
        let by = RefInto::<CgarF64>::ref_into(&p1[1]).0;
        let px = RefInto::<CgarF64>::ref_into(&pk[0]).0;
        let py = RefInto::<CgarF64>::ref_into(&pk[1]).0;

        let ux = bx - ax;
        let uy = by - ay;
        let cx = px - ax;
        let cy = py - ay;
        let len2 = ux * ux + uy * uy;
        if len2 == 0.0 {
            return None;
        }
        let cross = ux * cy - uy * cx;
        if cross.abs() > eps * len2.sqrt() {
            return None;
        }
        let t = (cx * ux + cy * uy) / len2;
        Some(t)
    };

    for &(i0, i1) in &pairs {
        if (a == i0 && b == i1) || (a == i1 && b == i0) {
            return true; // exactly the outer edge
        }
        if (a == i0 || a == i1) && (b == i0 || b == i1) {
            // both endpoints are boundary vertices of the same side (a subdivided edge)
            return true;
        }
    }
    false
}

fn forbid_boundary_interior_connectors<T: Scalar>(
    segments: &mut Vec<[usize; 2]>,
    pts: &[Point2<T>],
) {
    use ahash::AHashSet as FastSet;

    // dedupe undirected
    let mut seen: FastSet<(usize, usize)> = FastSet::default();
    segments.retain(|[a, b]| {
        if a == b {
            return false;
        }
        let k = if a < b { (*a, *b) } else { (*b, *a) };
        seen.insert(k)
    });

    segments.retain(|[a, b]| {
        let a_is_boundary = *a < 3;
        let b_is_boundary = *b < 3;

        if a_is_boundary == b_is_boundary {
            // both boundary or both interior => okay (interior chords are allowed)
            return true;
        }

        // mixed boundary/interior: allow only if it lies on an outer edge chain
        // i.e., boundary index combined with another boundary index (not interior)
        // Since here one is interior, reject.
        false
    });

    // Also keep only true outer-edge chains among boundary vertices
    segments.retain(|[a, b]| {
        let a_is_boundary = *a < 3;
        let b_is_boundary = *b < 3;
        if a_is_boundary && b_is_boundary {
            segment_is_on_outer_edge(*a, *b, pts)
        } else {
            true
        }
    });
}

#[inline]
fn on_segment_uv<T: Scalar + PartialOrd>(a: &Point2<T>, b: &Point2<T>, p: &Point2<T>) -> bool
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    // identical to Delaunay::on_segment
    let o = orient2d(a, b, p);
    if !o.is_zero() {
        return false;
    }
    let (minx, maxx) = if a[0] <= b[0] {
        (a[0].clone(), b[0].clone())
    } else {
        (b[0].clone(), a[0].clone())
    };
    let (miny, maxy) = if a[1] <= b[1] {
        (a[1].clone(), b[1].clone())
    } else {
        (b[1].clone(), a[1].clone())
    };
    p[0] >= minx && p[0] <= maxx && p[1] >= miny && p[1] <= maxy
}

#[inline]
fn param_t_uv<T: Scalar>(a: &Point2<T>, b: &Point2<T>, p: &Point2<T>) -> T
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>,
{
    // monotone parameter for ordering: (p-a)·(b-a)
    let ab0 = &b[0] - &a[0];
    let ab1 = &b[1] - &a[1];
    let ap0 = &p[0] - &a[0];
    let ap1 = &p[1] - &a[1];
    &ap0 * &ab0 + &ap1 * &ab1
}

fn add_split_or_chain_uv<T: Scalar + PartialOrd>(
    la: usize,
    lb: usize,
    out: &mut Vec<[usize; 2]>,
    points_uv: &[Point2<T>],
) where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    if la == lb {
        return;
    }
    let a = &points_uv[la];
    let b = &points_uv[lb];

    // degenerate AB guard
    if a[0] == b[0] && a[1] == b[1] {
        return;
    }

    // collect all local indices strictly between la and lb on segment in UV
    let mut mids: Vec<usize> = (0..points_uv.len())
        .filter(|&k| k != la && k != lb && on_segment_uv(a, b, &points_uv[k]))
        .collect();

    if mids.is_empty() {
        out.push([la, lb]);
        return;
    }

    // sort by projection parameter
    mids.sort_by(|&i, &j| {
        let ti = param_t_uv(a, b, &points_uv[i]);
        let tj = param_t_uv(a, b, &points_uv[j]);
        ti.partial_cmp(&tj).unwrap_or(std::cmp::Ordering::Equal)
    });

    // emit chain
    let mut prev = la;
    for m in mids {
        if prev != m {
            out.push([prev, m]);
        }
        prev = m;
    }
    if prev != lb {
        out.push([prev, lb]);
    }
}

fn decide_flip_uv<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    job: &FaceJobUV<T>,
    tris: &[(usize, usize, usize)],
    ccw: impl Fn(usize, usize, usize) -> bool,
) -> bool {
    // 1) Try a boundary reference that touches a live neighbor.
    for seg in &job.segments {
        let (li, lj) = (seg[0], seg[1]);
        let (gi, gj) = (job.verts_global[li], job.verts_global[lj]);

        if let Some(&h_fwd) = mesh.edge_map.get(&(gi, gj)) {
            let he = &mesh.half_edges[h_fwd];
            if he.removed || he.face.is_some() {
                continue;
            } // not a border spoke on our side
            let t = he.twin;
            if t == usize::MAX {
                continue;
            }
            if let Some(adj_f) = mesh.half_edges[t].face {
                if !mesh.faces[adj_f].removed {
                    // Find any tri that contains this edge (li, lj) locally
                    for &(a, b, c) in tris {
                        let has = |x| x == li || x == lj;
                        if has(a) && has(b) || has(b) && has(c) || has(c) && has(a) {
                            // Reorder to put (li, lj, k) and check CCW
                            let k = if a != li && a != lj {
                                a
                            } else if b != li && b != lj {
                                b
                            } else {
                                c
                            };
                            // If ccw(li, lj, k) is true, UV-CCW already matches (gi->gj) direction.
                            // If false, we should flip the entire job.
                            return !ccw(li, lj, k);
                        }
                    }
                }
            }
        }
    }

    // 2) Fallback: pick the flip that maximizes border promotions across all candidate edges.
    let score = |flip: bool| -> usize {
        let mut s = 0usize;
        for &(la, lb, lc) in tris {
            let (a, b, c) = if ccw(la, lb, lc) ^ flip {
                (la, lb, lc)
            } else {
                (la, lc, lb)
            };
            let edges = [
                (job.verts_global[a], job.verts_global[b]),
                (job.verts_global[b], job.verts_global[c]),
                (job.verts_global[c], job.verts_global[a]),
            ];
            for &(u, v) in &edges {
                if let Some(&h) = mesh.edge_map.get(&(u, v)) {
                    let he = &mesh.half_edges[h];
                    if !he.removed && he.face.is_none() {
                        s += 1;
                    } // would promote border
                }
            }
        }
        s
    };
    score(false) < score(true)
}

fn decide_flip_uv_for_job<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    job: &FaceJobUV<T>,
    tris_local: &[(usize, usize, usize)],
) -> bool {
    // prefer a triangle that touches the boundary ring
    let mut probe = None;
    for &(la, lb, lc) in tris_local {
        let mut boundary_edges = 0;
        for (x, y) in &[(la, lb), (lb, lc), (lc, la)] {
            if *x < 3 && *y < 3 {
                boundary_edges += 1;
            }
        }
        // pick the first with any boundary edge
        if boundary_edges > 0 {
            probe = Some((la, lb, lc));
            break;
        }
    }
    let (la, lb, lc) = probe.unwrap_or_else(|| tris_local[0]);

    let (ga, gb, gc) = (
        job.verts_global[la],
        job.verts_global[lb],
        job.verts_global[lc],
    );

    // try as-is (CCW in UV)
    let as_is_ok =
        edge_ok_now(mesh, ga, gb) && edge_ok_now(mesh, gb, gc) && edge_ok_now(mesh, gc, ga);

    if as_is_ok {
        return false;
    } // no flip

    // try flipped (CW in UV)
    let flipped_ok =
        edge_ok_now(mesh, ga, gc) && edge_ok_now(mesh, gc, gb) && edge_ok_now(mesh, gb, ga);

    // flip if flipped works better
    flipped_ok
}

fn seg_on_any_outer_edge<T: Scalar>(a: usize, b: usize, pts: &[Point2<T>]) -> bool
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>,
{
    // reuse your robust on_edge_with_t from build_face_pslgs
    let pairs = [(0, 1), (1, 2), (2, 0)];
    for &(i0, i1) in &pairs {
        let (ok_a, _) = on_edge_with_t(&pts[i0], &pts[i1], &pts[a]);
        if !ok_a {
            continue;
        }
        let (ok_b, _) = on_edge_with_t(&pts[i0], &pts[i1], &pts[b]);
        if ok_b {
            return true;
        }
    }
    false
}

// return true if p lies on segment ab in UV; also return the parameter t in [0,1] if wanted
#[inline(always)]
pub fn on_edge_with_t<TS: Scalar>(a: &Point2<TS>, b: &Point2<TS>, p: &Point2<TS>) -> (bool, f64)
where
    for<'x> &'x TS: std::ops::Sub<&'x TS, Output = TS>
        + std::ops::Mul<&'x TS, Output = TS>
        + std::ops::Add<&'x TS, Output = TS>,
{
    let ax = (a[0].clone().into() as CgarF64).0;
    let ay = (a[1].clone().into() as CgarF64).0;
    let bx = (b[0].clone().into() as CgarF64).0;
    let by = (b[1].clone().into() as CgarF64).0;
    let px = (p[0].clone().into() as CgarF64).0;
    let py = (p[1].clone().into() as CgarF64).0;
    let ux = bx - ax;
    let uy = by - ay;
    let vx = px - ax;
    let vy = py - ay;
    let cross = ux * vy - uy * vx;
    let eps = 1e-12 * (ux.abs() + uy.abs()).max(1.0);
    if cross.abs() > eps {
        return (false, 0.0);
    }
    let dot = ux * vx + uy * vy;
    let len2 = ux * ux + uy * uy;
    if len2 <= 0.0 {
        return (false, 0.0);
    }
    let t = dot / len2;
    (t >= -1e-12 && t <= 1.0 + 1e-12, t)
}

/// Return the global vertex from `candidates` that is *incident to this face*,
/// else fall back to the nearest-on-this-face edge.
fn choose_vertex_on_face<T: Scalar, const N: usize>(
    mesh: &Mesh<T, N>,
    face_id: usize,
    candidates: &[usize],
    projected_uv: &Point2<T>, // must be barycentric_uv in the SAME (a,b,c) basis
    pa: &Point<T, N>,
    pb: &Point<T, N>,
    pc: &Point<T, N>,
) -> Option<usize>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    // 1) prefer a candidate that touches this face
    for &g in candidates {
        if g < mesh.vertices.len() && mesh.vertex_touches_face(g, face_id) {
            return Some(g);
        }
    }
    // 2) otherwise: minimal barycentric-UV distance
    let mut best: Option<(usize, T)> = None;
    for &g in candidates {
        if g >= mesh.vertices.len() {
            continue;
        }
        let p = &mesh.vertices[g].position;
        let uv = barycentric_uv(p, pa, pb, pc);
        let du = &uv[0] - &projected_uv[0];
        let dv = &uv[1] - &projected_uv[1];
        let d2 = &du * &du + &dv * &dv;
        if best.as_ref().map_or(true, |(_, bd2)| d2 < *bd2) {
            best = Some((g, d2));
        }
    }
    best.map(|(g, _)| g)
}
