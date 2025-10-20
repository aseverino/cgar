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

pub mod basic_types;
pub mod core;
pub mod edge_collapse;
pub mod face;
pub mod half_edge;
pub mod intersection_segment;
pub mod spatial_hash;
pub mod topology;
pub mod triangle;
pub mod vertex;

#[macro_export]
macro_rules! impl_mesh {
    ($($items:item)*) => {
        impl<T: crate::numeric::scalar::Scalar, const N: usize> crate::mesh::basic_types::Mesh<T, N>
        where
            crate::geometry::point::Point<T, N>: crate::geometry::point::PointOps<T, N, Vector = crate::geometry::vector::Vector<T, N>>,
            crate::geometry::vector::Vector<T, N>: crate::geometry::vector::VectorOps<T, N> +  crate::geometry::vector::Cross3<T>,
            for<'a> &'a T: std::ops::Sub<&'a T, Output = T>
                + std::ops::Mul<&'a T, Output = T>
                + std::ops::Add<&'a T, Output = T>
                + std::ops::Div<&'a T, Output = T>
                + std::ops::Neg<Output = T>,
        {
            $($items)*
        }
    };
}
