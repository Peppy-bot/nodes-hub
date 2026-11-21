use super::*;

/// A base's URDF: `parts` inside the robot `base`, whose SRDF gives nothing.
fn base(parts: &str) -> RobotDescription {
    RobotDescription::parse(
        &format!(r#"<robot name="base">{parts}</robot>"#),
        r#"<robot name="base"/>"#,
    )
    .unwrap_or_else(|error| panic!("{error}: {parts}"))
}

/// A wheel `name` on a continuous joint at `xyz` about `axis`, its link a
/// cylinder about the joint's axis of `radius`.
fn wheel(name: &str, xyz: &str, axis: &str, radius: f64) -> String {
    format!(
        r#"<link name="{name}_link"><collision><origin rpy="1.5707963267948966 0 0" xyz="0 0 0"/>
             <geometry><cylinder length="0.04" radius="{radius}"/></geometry></collision></link>
           <joint name="{name}" type="continuous"><origin rpy="0 0 0" xyz="{xyz}"/>
             <parent link="base_footprint"/><child link="{name}_link"/><axis xyz="{axis}"/></joint>"#
    )
}

/// A chassis box 0.4 m by 0.3 m on base_footprint, the root link.
const CHASSIS: &str = r#"<link name="base_footprint"><collision><origin rpy="0 0 0" xyz="0 0 0.2"/>
      <geometry><box size="0.4 0.3 0.1"/></geometry></collision></link>"#;

fn two_wheels() -> String {
    wheel("left", "0 0.2 0.05", "0 1 0", 0.05) + &wheel("right", "0 -0.2 0.05", "0 1 0", 0.05)
}

fn refusal(parts: &str) -> String {
    BaseModel::parse(base(parts))
        .expect_err("a description that gives no base this node drives")
        .to_string()
}

/// How far two lengths (m) or two unit vectors may differ and count as the
/// same: what writing a decimal leaves of one value.
const SAME: f64 = 1e-9;

fn assert_near(actual: [f64; 3], expected: [f64; 3]) {
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() <= SAME),
        "{actual:?} != {expected:?}"
    );
}

// The bundled botshare base: wheels 0.1 m in radius on a 0.4 m track, and
// the chassis's 0.78 m by 0.58 m outline, which holds the wheels, the casters
// and the lidar.
#[test]
fn the_bundled_base_gives_its_wheels_and_its_outline() {
    let BundledBase {
        model: bundled,
        base: model,
    } = BaseModel::of("botshare_base").expect("the bundled base");
    assert_eq!(bundled.id(), "botshare_base");
    assert_eq!(
        model.wheels,
        WheelGeometry::parse(0.4, 0.1, 0.1).expect("a wheel geometry")
    );
    let footprint: Vec<[f64; 2]> = model.footprint.points().to_vec();
    let expected = [[0.39, -0.29], [0.39, 0.29], [-0.39, 0.29], [-0.39, -0.29]];
    for (corner, expected) in footprint.iter().zip(expected) {
        assert_near([corner[0], corner[1], 0.0], [expected[0], expected[1], 0.0]);
    }
}

#[test]
fn a_model_robot_description_lacks_or_one_with_no_base_is_refused() {
    let unknown = BaseModel::of("no_such_base").expect_err("no such model");
    assert!(
        unknown
            .to_string()
            .contains("no robot model is named \"no_such_base\""),
        "{unknown}"
    );
    let arm = BaseModel::of("so101").expect_err("an arm");
    assert!(
        arm.to_string()
            .starts_with("model so101 is no differential base this node drives: its wheels are"),
        "{arm}"
    );
    assert!(matches!(
        arm,
        ModelError::NoBase {
            reason: BaseError::NoWheels(NoDifferentialBase::WheelCount { .. }),
            ..
        }
    ));
}

// Each wheel keeps its own radius in the wheel geometry.
#[test]
fn wheels_of_two_radii_give_a_geometry_of_both() {
    let unequal = CHASSIS.to_owned()
        + &wheel("left", "0 0.2 0.05", "0 1 0", 0.05)
        + &wheel("right", "0 -0.2 0.05", "0 1 0", 0.06);
    let model = BaseModel::parse(base(&unequal)).expect("a base");
    assert_eq!(
        model.wheels,
        WheelGeometry::parse(0.4, 0.05, 0.06).expect("a wheel geometry")
    );
}

#[test]
fn a_base_of_two_wheels_and_a_chassis_gives_its_geometry() {
    let model = BaseModel::parse(base(&(CHASSIS.to_owned() + &two_wheels()))).expect("a base");
    assert_eq!(
        model.wheels,
        WheelGeometry::parse(0.4, 0.05, 0.05).expect("a wheel geometry")
    );
    // The wheels' outer faces stand 0.22 m to each side, past the chassis.
    let corners = model.footprint.points();
    assert_near([corners[0][0], corners[0][1], 0.0], [0.2, -0.22, 0.0]);
    assert_near([corners[2][0], corners[2][1], 0.0], [-0.2, 0.22, 0.0]);
}

// A description whose wheels robot_description refuses gives no base, the
// refusal naming them ([`RobotDescription::differential_base`]).
#[test]
fn a_base_without_two_wheels_is_refused_naming_them() {
    let one = CHASSIS.to_owned() + &wheel("left", "0 0.2 0.05", "0 1 0", 0.05);
    let named = NoDifferentialBase::WheelCount {
        root: "base_footprint".to_owned(),
        found: vec!["left".to_owned()],
        grouped: Vec::new(),
    };
    assert_eq!(refusal(&one), named.to_string());
    assert_eq!(
        named.to_string(),
        "its wheels are the two continuous joints that no group of its SRDF holds, on a link \
         that fixed joints alone join to its root link base_footprint; it has [\"left\"], and \
         its SRDF's groups hold []"
    );
}

#[test]
fn a_base_whose_parts_have_a_mesh_has_no_outline() {
    let parts = CHASSIS.to_owned()
        + &two_wheels()
        + r#"<link name="bumper"><collision><geometry><mesh filename="bumper.stl"/></geometry></collision></link>
             <joint name="bumper_joint" type="fixed"><origin rpy="0 0 0" xyz="0.2 0 0.1"/>
               <parent link="base_footprint"/><child link="bumper"/></joint>"#;
    assert_eq!(
        refusal(&parts),
        "link bumper: the outline holds boxes, cylinders and spheres, and one of its collision elements is none of them"
    );
}

// A base whose root link is not base_footprint is refused, naming its root
// link: odom places base_footprint on transforms.
#[test]
fn a_base_whose_root_link_is_not_base_footprint_is_refused() {
    let parts = (CHASSIS.to_owned() + &two_wheels()).replace("base_footprint", "base_link");
    assert_eq!(
        refusal(&parts),
        "its root link is base_link: the root link of a mobile base is its robot frame, base_footprint, which odom places on transforms; name the URDF's root link base_footprint"
    );
}
