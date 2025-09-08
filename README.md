# CGAR

[![Crates.io](https://img.shields.io/crates/v/cgar.svg)](https://crates.io/crates/cgar)
[![Documentation](https://docs.rs/cgar/badge.svg)](https://docs.rs/cgar)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CGAR** (Computational Geometry Algorithms for Rust) is an early-stage project
aimed at becoming a **functional equivalent to [CGAL](https://www.cgal.org/)** in Rust.  

The goal is to provide robust and efficient algorithms for computational geometry,
meshes, and geometric predicates—fully in safe Rust.

---

## 🚧 Project Status

⚠️ **This library is experimental.**  
There is not yet a stable or usable release. APIs may change at any time.

That said, the project already includes a solid foundation for building
geometry-heavy applications and welcomes early adopters and contributors.

---

## ✨ Features (so far)

- **Scalar types**
  - 64-bit floats (`f64`)
  - Exact rationals (`rug::Rational`)
  - Lazy-exact scalars (approximate until exact evaluation is required)

- **Algorithms**
  - Constrained Delaunay Triangulation (CDT)
  - Mesh corefinement
  - Boolean operations: difference, union, intersection
  - Common predicates (e.g. point-in-mesh, point-on-border, plane side tests, etc)
  - AABB trees and spatial search structures
  - Topologically consistent mesh opperations (e.g.: edge/face splitting, edge collapse)

---

## 🔮 Roadmap

Planned areas of development include:

- Robust 2D/3D polygon mesh operations
- Isotropic remeshing
- Mesh simplification
- Poisson reconstruction
- Alpha wrapping
- More exact geometric predicates
- Performance improvements (parallelization, SIMD, etc.)

---

## 📦 Installation

Add **CGAR** to your `Cargo.toml`:

```toml
[dependencies]
cgar = "0.1"