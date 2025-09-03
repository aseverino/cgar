CGAR is a project aimed to be a functional equivalent to C++'s Computational Geometry Algorithm Library (CGAL) in Rust.

There is yet no usable release.

The library currently features:

- 64 bit floats, exact (rug::Rational) and lazy-exact scalar types;
- Constrained Delaunay Triangulation (CDT);
- Mesh corefinement and Boolean (diff, union, intersection);
- Predicates in general (plane side, point in mesh, etc).