use cgar::geometry::Point2;
use cgar::geometry::point::Point3;
use cgar::kernel::orientation::orient2d;
use cgar::kernel::orientation::orient3d;

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
