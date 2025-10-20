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

use ahash::AHashMap;

use crate::{
    geometry::{
        Point2,
        point::{Point, PointOps},
        spatial_element::SpatialElement,
        util::EPS,
        vector::{Cross3, Vector, VectorOps},
    },
    mesh::{basic_types::Mesh, intersection_segment::IntersectionSegment},
    mesh_processing::boolean::{ApproxPointKey, point_key},
    numeric::{
        cgar_f64::CgarF64,
        scalar::{RefInto, Scalar},
    },
    operations::triangulation::delaunay::Delaunay,
};

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
    Vector<T, N>: VectorOps<T, N> + Cross3<T>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let total_triangles: usize = cdts.iter().map(|dt| dt.triangles.len()).sum();
    let mut face_ids = Vec::with_capacity(jobs.len());
    let mut triangles = Vec::with_capacity(total_triangles);

    for ((job, dt), _) in jobs.iter().zip(cdts.iter()).zip(0..) {
        face_ids.push(job.face_id);
        let verts_global = &job.verts_global;

        for t in &dt.triangles {
            triangles.push((verts_global[t.0], verts_global[t.1], verts_global[t.2]));
        }
    }

    mesh.remove_triangles_deferred(&face_ids);
    mesh.add_triangles_deferred(&triangles);
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
    Vector<T, N>: VectorOps<T, N> + Cross3<T>,
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
            if let Some(&_vid) = keyed_map.get(&map_key) {
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
                            let (vid, _existed) = mesh.get_or_insert_vertex(&ep_pos);
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
                if let Some(&_vid) = keyed_map.get(&map_key_face) {
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
    Vector<T, N>: VectorOps<T, N> + Cross3<T>,
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
    if u.dot(&u).is_zero() {
        u = e1.clone();
    }

    // Candidate v from double cross: v = (n × u) with n = u × e1
    let n = u.cross(&e1);
    let mut v = n.cross(&u);

    // If v vanished (collinear e0,e1), fall back to Gram–Schmidt: v = e1 - proj_u(e1)
    if v.dot(&v).is_zero() {
        let uu = u.dot(&u);
        if !uu.is_zero() {
            let ue1 = u.dot(&e1);
            let proj = u.scale(&(ue1 / uu));
            v = &e1 - &proj; // in-plane
        }
        // If still zero, the face is degenerate; caller should skip this face.
    }

    (u, v)
}

fn barycentric_uv<T: Scalar, const N: usize>(
    p: &Point<T, N>,
    a: &Point<T, N>,
    b: &Point<T, N>,
    c: &Point<T, N>,
) -> Point2<T>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N>,
{
    // Extract ball centers for all coordinates
    let mut p_coords = [0.0; N];
    let mut a_coords = [0.0; N];
    let mut b_coords = [0.0; N];
    let mut c_coords = [0.0; N];

    for i in 0..N {
        p_coords[i] = p[i].ball_center_f64();
        a_coords[i] = a[i].ball_center_f64();
        b_coords[i] = b[i].ball_center_f64();
        c_coords[i] = c[i].ball_center_f64();
    }

    // Compute vectors using f64 arithmetic
    let mut v0 = [0.0; N]; // b - a
    let mut v1 = [0.0; N]; // c - a  
    let mut v2 = [0.0; N]; // p - a

    for i in 0..N {
        v0[i] = b_coords[i] - a_coords[i];
        v1[i] = c_coords[i] - a_coords[i];
        v2[i] = p_coords[i] - a_coords[i];
    }

    // Dot products
    let d00: f64 = v0.iter().zip(&v0).map(|(x, y)| x * y).sum();
    let d01: f64 = v0.iter().zip(&v1).map(|(x, y)| x * y).sum();
    let d11: f64 = v1.iter().zip(&v1).map(|(x, y)| x * y).sum();
    let d20: f64 = v2.iter().zip(&v0).map(|(x, y)| x * y).sum();
    let d21: f64 = v2.iter().zip(&v1).map(|(x, y)| x * y).sum();

    let den = d00 * d11 - d01 * d01;

    if den.abs() < 1e-15 {
        return Point2::<T>::from_vals([T::zero(), T::zero()]);
    }

    let alpha = (d20 * d11 - d21 * d01) / den;
    let beta = (d21 * d00 - d20 * d01) / den;

    Point2::<T>::from_vals([T::from(alpha), T::from(beta)])
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
    Vector<T, N>: VectorOps<T, N>,
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
    Vector<T, N>: VectorOps<T, N> + Cross3<T>,
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
        // let (u, v) = face_uv_frame_metric(pa, pb, pc);

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

#[inline]
fn on_segment_uv_fast<T: Scalar + PartialOrd>(a: &Point2<T>, b: &Point2<T>, p: &Point2<T>) -> bool
where
    for<'x> &'x T: std::ops::Sub<&'x T, Output = T>
        + std::ops::Mul<&'x T, Output = T>
        + std::ops::Add<&'x T, Output = T>
        + std::ops::Div<&'x T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let ax = a[0].ball_center_f64();
    let ay = a[1].ball_center_f64();
    let bx = b[0].ball_center_f64();
    let by = b[1].ball_center_f64();
    let px = p[0].ball_center_f64();
    let py = p[1].ball_center_f64();

    let ux = bx - ax;
    let uy = by - ay;
    let vx = px - ax;
    let vy = py - ay;
    let cross = ux * vy - uy * vx;
    let eps = EPS * (ux.abs() + uy.abs()).max(1.0);

    if cross.abs() > eps {
        return false;
    }

    let (minx, maxx) = if ax <= bx { (ax, bx) } else { (bx, ax) };
    let (miny, maxy) = if ay <= by { (ay, by) } else { (by, ay) };
    px >= minx - eps && px <= maxx + eps && py >= miny - eps && py <= maxy + eps
}

fn param_t_uv<T: Scalar>(a: &Point2<T>, b: &Point2<T>, p: &Point2<T>) -> f64 {
    let ax = a[0].ball_center_f64();
    let ay = a[1].ball_center_f64();
    let bx = b[0].ball_center_f64();
    let by = b[1].ball_center_f64();
    let px = p[0].ball_center_f64();
    let py = p[1].ball_center_f64();

    let abx = bx - ax;
    let aby = by - ay;
    let apx = px - ax;
    let apy = py - ay;
    apx * abx + apy * aby
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
    if (a[0].ball_center_f64() - b[0].ball_center_f64()).abs() < EPS
        && (a[1].ball_center_f64() - b[1].ball_center_f64()).abs() < EPS
    {
        return;
    }

    // collect all local indices strictly between la and lb on segment in UV
    let mut mids: Vec<usize> = (0..points_uv.len())
        .filter(|&k| k != la && k != lb && on_segment_uv_fast(a, b, &points_uv[k]))
        .collect();

    if mids.is_empty() {
        out.push([la, lb]);
        return;
    }

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
pub fn on_edge_with_t<TS: Scalar>(a: &Point2<TS>, b: &Point2<TS>, p: &Point2<TS>) -> (bool, f64) {
    let ax = a[0].ball_center_f64();
    let ay = a[1].ball_center_f64();
    let bx = b[0].ball_center_f64();
    let by = b[1].ball_center_f64();
    let px = p[0].ball_center_f64();
    let py = p[1].ball_center_f64();

    let ux = bx - ax;
    let uy = by - ay;
    let vx = px - ax;
    let vy = py - ay;
    let cross = ux * vy - uy * vx;
    let eps = EPS * (ux.abs() + uy.abs()).max(1.0);

    if cross.abs() > eps {
        return (false, 0.0);
    }

    let dot = ux * vx + uy * vy;
    let len2 = ux * ux + uy * uy;
    if len2 <= 0.0 {
        return (false, 0.0);
    }
    let t = dot / len2;
    (t >= -EPS && t <= 1.0 + EPS, t)
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
    Vector<T, N>: VectorOps<T, N> + Cross3<T>,
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
