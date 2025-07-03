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

use crate::geometry::Point2;
use crate::geometry::point::Point3;
use crate::geometry::vector::Vector3;
use crate::operations::{Abs, Pow, Sqrt};
use std::ops::{Add, Div, Mul, Sub};

/// Returns:
/// - >0 if counter-clockwise
/// - <0 if clockwise
/// - =0 if collinear
pub fn orient2d<T>(a: &Point2<T>, b: &Point2<T>, c: &Point2<T>) -> T
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    &(&(&b.x - &a.x) * &(&c.y - &a.y)) - &(&(&b.y - &a.y) * &(&c.x - &a.x))
}

pub fn orient3d<T>(a: &Point3<T>, b: &Point3<T>, c: &Point3<T>, d: &Point3<T>) -> T
where
    T: Clone + PartialOrd + Abs + Pow + Sqrt,
    for<'a> &'a T: Add<&'a T, Output = T>
        + Sub<&'a T, Output = T>
        + Mul<&'a T, Output = T>
        + Div<&'a T, Output = T>,
{
    let ab = Vector3 {
        x: &b.x - &a.x,
        y: &b.y - &a.y,
        z: &b.z - &a.z,
    };

    let ac = Vector3 {
        x: &c.x - &a.x,
        y: &c.y - &a.y,
        z: &c.z - &a.z,
    };

    let ad = Vector3 {
        x: &d.x - &a.x,
        y: &d.y - &a.y,
        z: &d.z - &a.z,
    };

    ab.cross(&ac).dot(&ad)
}

#[cfg(test)]
mod tests {
    use crate::geometry::Point2;
    use crate::geometry::point::Point3;
    use crate::kernel::orientation::orient2d;
    use crate::kernel::orientation::orient3d;

    #[test]
    fn ccw_test() {
        let a = Point2 { x: 0.0, y: 0.0 };
        let b = Point2 { x: 1.0, y: 0.0 };
        let c = Point2 { x: 0.0, y: 1.0 };

        assert!(orient2d(&a, &b, &c) > 0.0); // Counter-clockwise
    }

    #[test]
    fn orientation_3d_positive_volume() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);
        let d = Point3::new(0.0, 0.0, 1.0); // above the abc plane

        let vol = orient3d(&a, &b, &c, &d);
        assert!(vol > 0.0);
    }

    #[test]
    fn orientation_3d_negative_volume() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);
        let d = Point3::new(0.0, 0.0, -1.0); // below the abc plane

        let vol = orient3d(&a, &b, &c, &d);
        assert!(vol < 0.0);
    }

    #[test]
    fn orientation_3d_coplanar() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 0.0);
        let c = Point3::new(0.0, 1.0, 0.0);
        let d = Point3::new(1.0, 1.0, 0.0); // lies in the same z=0 plane

        let vol = orient3d(&a, &b, &c, &d);
        assert!(vol.abs() < 1e-12); // small epsilon to account for floating point
    }
}
