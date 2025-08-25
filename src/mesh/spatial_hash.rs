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

use crate::{geometry::point::Point, impl_mesh, numeric::cgar_f64::CgarF64};

impl_mesh! {
    pub fn with_default_hash(mut self) -> Self {
        // derive from kernel tolerance once, as f64
        let thr: CgarF64 = T::point_merge_threshold().ref_into();
        let mut cell = (0.9 * thr.0).max(1e-12); // clamp to avoid INF
        if !cell.is_finite() { cell = 1e-5; }
        self.cell = cell;
        self.hash_inv = 1.0 / cell;
        self.vertex_spatial_hash = AHashMap::default();
        self
    }

    #[inline(always)]
    fn floor_sat_i64(x: f64) -> i64 {
        if !x.is_finite() { return if x.is_sign_positive() { i64::MAX } else { i64::MIN }; }
        let i = x as i64;
        i - ((i as f64 > x) as i64)
    }

    #[inline(always)]
    fn pack_key3(kx: i64, ky: i64, kz: i64) -> u128 {
        // 3 × 42-bit signed lanes into 126 bits (fits typical ranges).
        let mask = (1u128 << 42) - 1;
        let ux = (kx as i128 as u128) & mask;
        let uy = (ky as i128 as u128) & mask;
        let uz = (kz as i128 as u128) & mask;
        ux | (uy << 42) | (uz << 84)
    }

    #[inline(always)]
    pub fn position_to_hash_key(&self, pos: &Point<T, N>) -> (i64, i64, i64) {
        let inv = self.hash_inv;

        let ax = if 0 < N { let a: CgarF64 = (&pos[0]).ref_into(); a.0 } else { 0.0 };
        let ay = if 1 < N { let a: CgarF64 = (&pos[1]).ref_into(); a.0 } else { 0.0 };
        let az = if 2 < N { let a: CgarF64 = (&pos[2]).ref_into(); a.0 } else { 0.0 };

        (
            Self::floor_sat_i64(ax * inv),
            Self::floor_sat_i64(ay * inv),
            Self::floor_sat_i64(az * inv),
        )
    }

    #[inline(always)]
    pub fn position_to_packed_key(&self, pos: &Point<T, N>) -> u128 {
        let (x, y, z) = self.position_to_hash_key(pos);
        Self::pack_key3(x, y, z)
    }
}
