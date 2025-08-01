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

use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use crate::{
    geometry::{Point3, spatial_element::SpatialElement},
    mesh::mesh::Mesh,
    numeric::scalar::Scalar,
};

pub fn write_obj<T: Scalar, const N: usize, P: AsRef<Path>>(
    mesh: &Mesh<T, N>,
    path: P,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    // 1) write vertices
    for v in &mesh.vertices {
        let coords = v.position.coords();
        writeln!(
            out,
            "v {:?} {:?} {:?}",
            coords[0].to_f64().unwrap(),
            coords[1].to_f64().unwrap(),
            coords[2].to_f64().unwrap()
        )?;
    }

    // 2) write faces (1-based indices)
    for f in 0..mesh.faces.len() {
        if mesh.faces[f].removed {
            continue; // skip removed faces
        }
        let vs = mesh.face_vertices(f);
        // OBJ is 1-based
        writeln!(out, "f {:?} {:?} {:?}", vs[0] + 1, vs[1] + 1, vs[2] + 1)?;
    }

    out.flush()
}

/// Read a mesh from a Wavefront OBJ file.
/// Only supports `v x y z` and `f i j k` lines; ignores others.
pub fn read_obj<T: Scalar, P: AsRef<Path>>(path: P) -> io::Result<Mesh<T, 3>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut mesh = Mesh::new();
    let mut vertex_map = Vec::new();

    for line in reader.lines() {
        let l = line?;
        let mut parts = l.split_whitespace();
        match parts.next() {
            Some("v") => {
                // parse three floats
                let x: f64 = parts.next().unwrap().parse().unwrap();
                let y: f64 = parts.next().unwrap().parse().unwrap();
                let z: f64 = parts.next().unwrap().parse().unwrap();
                let vid = mesh.add_vertex(Point3::<T>::from_vals([x, y, z]));
                vertex_map.push(vid);
            }
            Some("f") => {
                // parse three vertex indices (1-based OBJ)
                let i: usize = parts.next().unwrap().parse().unwrap();
                let j: usize = parts.next().unwrap().parse().unwrap();
                let k: usize = parts.next().unwrap().parse().unwrap();
                // subtract 1 for zero-based
                mesh.add_triangle(i - 1, j - 1, k - 1);
            }
            _ => {
                // ignore comments, normals, etc.
            }
        }
    }

    // Rebuild twin connectivity & boundary loops
    mesh.build_boundary_loops();
    Ok(mesh)
}
