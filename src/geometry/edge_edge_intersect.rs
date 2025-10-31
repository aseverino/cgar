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

use std::array::from_fn;

use crate::{
    geometry::{point::Point, spatial_element::SpatialElement},
    numeric::scalar::Scalar,
};

pub fn edge_edge_intersection_2<T: Scalar, const N: usize>(
    p1: &Point<T, N>,
    p2: &Point<T, N>,
    p3: &Point<T, N>,
    p4: &Point<T, N>,
) -> Option<Point<T, N>>
where
    Point<T, N>: SpatialElement<T, N>,
{
    // Standard 2D line-line intersection using determinants
    let d1 = (p2[0].clone() - p1[0].clone()) * (p4[1].clone() - p3[1].clone())
        - (p2[1].clone() - p1[1].clone()) * (p4[0].clone() - p3[0].clone());

    if d1.is_zero() {
        return None; // Parallel lines
    }

    let d2 = (p1[0].clone() - p3[0].clone()) * (p4[1].clone() - p3[1].clone())
        - (p1[1].clone() - p3[1].clone()) * (p4[0].clone() - p3[0].clone());

    let t = d2 / d1.clone();

    // Check if intersection is within both line segments
    if t >= T::zero() && t <= T::one() {
        let x = p1[0].clone() + t.clone() * (p2[0].clone() - p1[0].clone());
        let y = p1[1].clone() + t * (p2[1].clone() - p1[1].clone());
        Some(Point::<T, N>::from_vals(from_fn(|i| match i {
            0 => x.clone(),
            1 => y.clone(),
            _ => T::zero(),
        })))
    } else {
        None
    }
}
