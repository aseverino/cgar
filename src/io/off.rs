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
    geometry::{
        point::{Point, PointOps},
        spatial_element::SpatialElement,
        vector::{Cross3, Vector, VectorOps},
    },
    mesh::basic_types::Mesh,
    numeric::scalar::Scalar,
};

/// Write a mesh to OFF.
/// Format:
///   OFF
///   <numVertices> <numFaces> <numEdges>
///   x y z
///   ...
///   3 i j k   (zero-based indices)
pub fn write_off<T: Scalar, const N: usize, P: AsRef<Path>>(
    mesh: &Mesh<T, N>,
    path: P,
) -> io::Result<()>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N> + Cross3<T>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    // Count non-removed faces
    let face_count = mesh.faces.iter().filter(|f| !f.removed).count();
    let vert_count = mesh.vertices.len();

    // OFF header (edges count can be 0 if unknown)
    writeln!(out, "OFF")?;
    writeln!(out, "{} {} {}", vert_count, face_count, 0)?;

    // 1) vertices
    for v in &mesh.vertices {
        let c = v.position.coords();
        // OFF requires plain numbers (typically floats)
        writeln!(
            out,
            "{} {} {}",
            c[0].to_f64().unwrap(),
            c[1].to_f64().unwrap(),
            c[2].to_f64().unwrap()
        )?;
    }

    // 2) triangular faces (zero-based indices)
    for f in 0..mesh.faces.len() {
        if mesh.faces[f].removed {
            continue;
        }
        let vs = mesh.face_vertices(f);
        // OFF face line: "<n> v0 v1 v2"
        writeln!(out, "3 {} {} {}", vs[0], vs[1], vs[2])?;
    }

    out.flush()
}

/// Read a mesh from OFF.
/// Supports:
///   OFF
///   <V> <F> <E>
///   V lines of "x y z"
///   F lines of "k i j k [ ... ]" (only k==3 is accepted; others are skipped)
pub fn read_off<T: Scalar, P: AsRef<Path>>(path: P) -> io::Result<Mesh<T, 3>>
where
    Point<T, 3>: PointOps<T, 3, Vector = Vector<T, 3>>,
    Vector<T, 3>: VectorOps<T, 3>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Helper: iterate non-empty, non-comment tokens across lines
    fn tokenize<R: BufRead>(r: R) -> io::Result<Vec<String>> {
        let mut toks = Vec::new();
        for line in r.lines() {
            let l = line?;
            let trimmed = l.split('#').next().unwrap_or("").trim();
            if trimmed.is_empty() {
                continue;
            }
            toks.extend(trimmed.split_whitespace().map(|s| s.to_string()));
        }
        Ok(toks)
    }

    let toks = tokenize(reader)?;
    if toks.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "OFF: empty file",
        ));
    }

    let mut it = toks.into_iter();

    // Header token must be OFF
    let header = it
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "OFF: missing header"))?;
    if header != "OFF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("OFF: expected 'OFF', got '{}'", header),
        ));
    }

    // Counts
    let vcount: usize = it
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "OFF: missing vertex count"))?
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad vertex count"))?;
    let fcount: usize = it
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "OFF: missing face count"))?
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad face count"))?;
    // edges count (ignored)
    let _ecount: usize = it
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "OFF: missing edge count"))?
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad edge count"))?;

    // Vertices
    let mut mesh = Mesh::new();
    for _ in 0..vcount {
        let x: f64 = it
            .next()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "OFF: incomplete vertex (x)")
            })?
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad vertex x"))?;
        let y: f64 = it
            .next()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "OFF: incomplete vertex (y)")
            })?
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad vertex y"))?;
        let z: f64 = it
            .next()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "OFF: incomplete vertex (z)")
            })?
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad vertex z"))?;

        // Assuming Point3::<T>::from_vals accepts f64s as in your OBJ reader
        let _vid = mesh.add_vertex(Point::<T, 3>::from_vals([x, y, z]));
        // We don't need to store a map because OFF face indices are zero-based
    }

    // Faces
    for _ in 0..fcount {
        // First token is the polygon size
        let poly_size: usize = it
            .next()
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "OFF: face missing vertex count")
            })?
            .parse()
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "OFF: bad face vertex count")
            })?;

        if poly_size == 3 {
            let a: usize = it
                .next()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "OFF: face missing index a")
                })?
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad index a"))?;
            let b: usize = it
                .next()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "OFF: face missing index b")
                })?
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad index b"))?;
            let c: usize = it
                .next()
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "OFF: face missing index c")
                })?
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "OFF: bad index c"))?;
            mesh.add_triangle(a, b, c);
        } else {
            // Skip non-triangle indices
            for _ in 0..poly_size {
                let _ = it.next(); // discard
            }
        }
    }

    Ok(mesh)
}
