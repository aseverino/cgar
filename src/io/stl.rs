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
        Point3,
        point::{Point, PointOps},
        spatial_element::SpatialElement,
        vector::{Vector, VectorOps},
    },
    mesh::basic_types::Mesh,
    numeric::scalar::Scalar,
};

/// Write a mesh to ASCII STL format.
/// Format:
///   solid <name>
///   facet normal nx ny nz
///     outer loop
///       vertex x1 y1 z1
///       vertex x2 y2 z2
///       vertex x3 y3 z3
///     endloop
///   endfacet
///   endsolid <name>
pub fn write_stl<T: Scalar, const N: usize, P: AsRef<Path>>(
    mesh: &Mesh<T, N>,
    path: P,
) -> io::Result<()>
where
    Point<T, N>: PointOps<T, N, Vector = Vector<T, N>>,
    Vector<T, N>: VectorOps<T, N, Cross = Vector<T, N>>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);

    writeln!(out, "solid mesh")?;

    for f in 0..mesh.faces.len() {
        if mesh.faces[f].removed {
            continue;
        }

        let vs = mesh.face_vertices(f);
        let v0 = &mesh.vertices[vs[0]].position;
        let v1 = &mesh.vertices[vs[1]].position;
        let v2 = &mesh.vertices[vs[2]].position;

        // Calculate face normal
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.as_vector().cross(&edge2.as_vector()).normalize();
        let n_coords = normal.coords();

        writeln!(
            out,
            "  facet normal {} {} {}",
            n_coords[0].to_f64().unwrap(),
            n_coords[1].to_f64().unwrap(),
            n_coords[2].to_f64().unwrap()
        )?;
        writeln!(out, "    outer loop")?;

        for &vi in &vs {
            let coords = mesh.vertices[vi].position.coords();
            writeln!(
                out,
                "      vertex {} {} {}",
                coords[0].to_f64().unwrap(),
                coords[1].to_f64().unwrap(),
                coords[2].to_f64().unwrap()
            )?;
        }

        writeln!(out, "    endloop")?;
        writeln!(out, "  endfacet")?;
    }

    writeln!(out, "endsolid mesh")?;
    out.flush()
}

/// Read a mesh from ASCII STL format.
/// Supports:
///   solid <name>
///   facet normal nx ny nz
///     outer loop
///       vertex x y z
///       vertex x y z
///       vertex x y z
///     endloop
///   endfacet
///   endsolid <name>
pub fn read_stl<T: Scalar, P: AsRef<Path>>(path: P) -> io::Result<Mesh<T, 3>>
where
    Point<T, 3>: PointOps<T, 3, Vector = Vector<T, 3>>,
    Vector<T, 3>: VectorOps<T, 3, Cross = Vector<T, 3>>,
    for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
        + std::ops::Mul<&'a T, Output = T>
        + std::ops::Add<&'a T, Output = T>
        + std::ops::Div<&'a T, Output = T>
        + std::ops::Neg<Output = T>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut mesh = Mesh::new();
    let mut vertices = Vec::new();

    for line in reader.lines() {
        let l = line?;
        let trimmed = l.trim();
        let mut parts = trimmed.split_whitespace();

        match parts.next() {
            Some("solid") => {
                // Start of STL file
            }
            Some("facet") => {
                // Start of facet, normal is ignored during reading
                vertices.clear();
            }
            Some("vertex") => {
                let x: f64 = parts
                    .next()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "STL: missing vertex x")
                    })?
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "STL: bad vertex x"))?;
                let y: f64 = parts
                    .next()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "STL: missing vertex y")
                    })?
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "STL: bad vertex y"))?;
                let z: f64 = parts
                    .next()
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "STL: missing vertex z")
                    })?
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "STL: bad vertex z"))?;

                let vid = mesh.add_vertex(Point::<T, 3>::from_vals([x, y, z]));
                vertices.push(vid);
            }
            Some("endfacet") => {
                if vertices.len() == 3 {
                    mesh.add_triangle(vertices[0], vertices[1], vertices[2]);
                }
                vertices.clear();
            }
            Some("endsolid") => {
                // End of STL file
                break;
            }
            _ => {
                // Ignore other lines (outer loop, endloop, etc.)
            }
        }
    }

    Ok(mesh)
}
