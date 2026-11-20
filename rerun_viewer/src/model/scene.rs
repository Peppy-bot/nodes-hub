//! The planning scene of the motion planner: its objects as their shapes, the objects an
//! arm carries, the pairs allowed to touch, and the shell of the clearance around each
//! shape.

use frame_tree::Transform;

use crate::model::placement::{Placement, translation};

/// What a shape is, centered on its own frame.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeKind {
    /// Full side lengths along x, y and z (m).
    Box {
        size: [f64; 3],
    },
    Sphere {
        radius: f64,
    },
    /// Its height along z and its radius (m).
    Cylinder {
        height: f64,
        radius: f64,
    },
    /// Vertices in the shape's frame (m), and three vertex indices a triangle.
    Mesh {
        vertices: Vec<[f64; 3]>,
        triangles: Vec<[u32; 3]>,
    },
}

impl ShapeKind {
    /// The kind's name as motion_planning spells it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Box { .. } => "box",
            Self::Sphere { .. } => "sphere",
            Self::Cylinder { .. } => "cylinder",
            Self::Mesh { .. } => "mesh",
        }
    }
}

/// A shape of an object, placed in the object's frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub kind: ShapeKind,
    pub placement: Placement,
}

/// An object that no link carries, placed in the robot frame.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneObject {
    pub id: String,
    pub placement: Placement,
    pub shapes: Vec<Shape>,
}

/// An object a link carries, placed in the frame of that link.
#[derive(Debug, Clone, PartialEq)]
pub struct AttachedObject {
    pub id: String,
    pub link: String,
    pub placement: Placement,
    pub shapes: Vec<Shape>,
}

/// Everything the planning scene holds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlanningScene {
    /// Sorted by id.
    pub objects: Vec<SceneObject>,
    /// Sorted by id.
    pub attached: Vec<AttachedObject>,
    /// Each pair once, the first name before the second in sort order.
    pub allowed_pairs: Vec<(String, String)>,
    /// The clearance (m) a plan keeps from each object.
    pub clearance_m: f64,
}

impl PlanningScene {
    /// The scene with nothing in it and the clearance of `other`: what the first read of
    /// a scene is compared with, so that it logs its objects and pairs and not its
    /// clearance.
    pub fn empty_like(other: &PlanningScene) -> Self {
        Self {
            clearance_m: other.clearance_m,
            ..Self::default()
        }
    }
}

/// A shape to draw: what it is, and where its frame sits in the frame it is drawn in.
#[derive(Debug, Clone)]
pub struct PlacedShape {
    pub kind: ShapeKind,
    pub pose: Transform,
}

/// The shapes of `object`, each in the robot frame.
pub fn object_shapes(object: &SceneObject) -> Vec<PlacedShape> {
    placed(&object.placement, &object.shapes)
}

/// The shapes of `attached`, each in the frame of the link that carries it.
pub fn attached_shapes(attached: &AttachedObject) -> Vec<PlacedShape> {
    placed(&attached.placement, &attached.shapes)
}

fn placed(object: &Placement, shapes: &[Shape]) -> Vec<PlacedShape> {
    shapes
        .iter()
        .map(|shape| PlacedShape {
            kind: shape.kind.clone(),
            pose: object.pose.compose(&shape.placement.pose),
        })
        .collect()
}

/// The shell of the clearance `clearance_m` around `shape`: each point within the
/// clearance of the shape lies inside it. A sphere grows its radius by the clearance; a
/// box grows each side by twice the clearance, and a cylinder its height by twice the
/// clearance and its radius by the clearance, so their shells stand a little wider than
/// the clearance at their edges and corners. A mesh's shell is its bounding box, grown
/// as a box grows.
pub fn shell(shape: &PlacedShape, clearance_m: f64) -> PlacedShape {
    let grow = 2.0 * clearance_m;
    match &shape.kind {
        ShapeKind::Box { size } => PlacedShape {
            kind: ShapeKind::Box {
                size: size.map(|side| side + grow),
            },
            pose: shape.pose,
        },
        ShapeKind::Sphere { radius } => PlacedShape {
            kind: ShapeKind::Sphere {
                radius: radius + clearance_m,
            },
            pose: shape.pose,
        },
        ShapeKind::Cylinder { height, radius } => PlacedShape {
            kind: ShapeKind::Cylinder {
                height: height + grow,
                radius: radius + clearance_m,
            },
            pose: shape.pose,
        },
        ShapeKind::Mesh { vertices, .. } => {
            let (center, size) = bounding_box(vertices);
            PlacedShape {
                kind: ShapeKind::Box {
                    size: size.map(|side| side + grow),
                },
                pose: shape.pose.compose(&translation(center)),
            }
        }
    }
}

/// The center and the side lengths of the smallest box along the axes that holds
/// `vertices`; a box of no size at the origin for none.
fn bounding_box(vertices: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let Some(first) = vertices.first() else {
        return ([0.0; 3], [0.0; 3]);
    };
    let (mut low, mut high) = (*first, *first);
    for vertex in vertices {
        for axis in 0..3 {
            low[axis] = low[axis].min(vertex[axis]);
            high[axis] = high[axis].max(vertex[axis]);
        }
    }
    let center = std::array::from_fn(|axis| (low[axis] + high[axis]) / 2.0);
    let size = std::array::from_fn(|axis| high[axis] - low[axis]);
    (center, size)
}

/// The normal of each vertex of a mesh, for the light to shade it: the sum of the
/// normals of the triangles that hold the vertex, each as long as twice its triangle's
/// area, scaled to length 1. A triangle faces the side that its vertices turn
/// counterclockwise on. A triangle with an index past the vertices adds nothing. A
/// vertex whose triangles add up to no normal (none has an area, or they face opposite
/// ways) takes +z.
pub fn vertex_normals(vertices: &[[f64; 3]], triangles: &[[u32; 3]]) -> Vec<[f64; 3]> {
    let mut sums = vec![[0.0; 3]; vertices.len()];
    let corner = |index: u32| vertices.get(index as usize).copied();
    for triangle in triangles {
        let (Some(a), Some(b), Some(c)) = (
            corner(triangle[0]),
            corner(triangle[1]),
            corner(triangle[2]),
        ) else {
            continue;
        };
        let normal = cross(difference(b, a), difference(c, a));
        for &index in triangle {
            let sum = &mut sums[index as usize];
            for axis in 0..3 {
                sum[axis] += normal[axis];
            }
        }
    }
    sums.into_iter()
        .map(|sum| {
            let length = sum.iter().map(|c| c * c).sum::<f64>().sqrt();
            if length > 0.0 {
                sum.map(|c| c / length)
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

fn difference(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model::placement::tests::{close, quarter_turn_about_z};

    pub(crate) const IDENTITY: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

    pub(crate) fn at(position: [f64; 3]) -> Placement {
        Placement::parse(position, IDENTITY).unwrap()
    }

    pub(crate) fn boxed(id: &str, position: [f64; 3]) -> SceneObject {
        SceneObject {
            id: id.to_string(),
            placement: at(position),
            shapes: vec![Shape {
                kind: ShapeKind::Box {
                    size: [0.1, 0.2, 0.3],
                },
                placement: at([0.0; 3]),
            }],
        }
    }

    /// The box of `boxed`, carried by the link `link` 5 cm along its z.
    pub(crate) fn held(id: &str, link: &str) -> AttachedObject {
        let object = boxed(id, [0.0, 0.0, 0.05]);
        AttachedObject {
            id: object.id,
            link: link.to_string(),
            placement: object.placement,
            shapes: object.shapes,
        }
    }

    fn placed_at_origin(kind: ShapeKind) -> PlacedShape {
        PlacedShape {
            kind,
            pose: translation([0.0; 3]),
        }
    }

    #[test]
    fn an_empty_scene_like_another_keeps_its_clearance_only() {
        let scene = PlanningScene {
            objects: vec![boxed("table", [0.5, 0.0, 0.0])],
            attached: vec![],
            allowed_pairs: vec![("table".to_string(), "openarm_left_tcp".to_string())],
            clearance_m: 0.04,
        };
        let empty = PlanningScene::empty_like(&scene);
        assert!(empty.objects.is_empty() && empty.allowed_pairs.is_empty());
        assert_eq!(empty.clearance_m, 0.04);
    }

    #[test]
    fn a_shape_is_placed_through_its_object() {
        // The object stands at (1, 0, 0) turned a quarter about z; its shape stands
        // 0.1 m along the object's x, which the turn points along the robot's y.
        let object = SceneObject {
            id: "brain/lime_1".to_string(),
            placement: Placement::parse([1.0, 0.0, 0.0], quarter_turn_about_z()).unwrap(),
            shapes: vec![Shape {
                kind: ShapeKind::Sphere { radius: 0.03 },
                placement: at([0.1, 0.0, 0.0]),
            }],
        };
        let shapes = object_shapes(&object);
        assert_eq!(shapes.len(), 1);
        assert!(close(shapes[0].pose.origin(), [1.0, 0.1, 0.0], 1e-12));
    }

    #[test]
    fn an_attached_object_is_placed_in_its_links_frame() {
        let attached = AttachedObject {
            id: "brain/lime_1".to_string(),
            link: "openarm_right_tcp".to_string(),
            placement: at([0.0, 0.0, 0.05]),
            shapes: vec![Shape {
                kind: ShapeKind::Sphere { radius: 0.03 },
                placement: at([0.0, 0.0, 0.01]),
            }],
        };
        assert!(close(
            attached_shapes(&attached)[0].pose.origin(),
            [0.0, 0.0, 0.06],
            1e-12
        ));
    }

    #[test]
    fn a_sphere_shell_grows_its_radius_by_the_clearance() {
        let grown = shell(&placed_at_origin(ShapeKind::Sphere { radius: 0.05 }), 0.04);
        assert_eq!(
            grown.kind,
            ShapeKind::Sphere {
                radius: 0.05 + 0.04
            }
        );
    }

    #[test]
    fn a_box_shell_grows_each_side_by_twice_the_clearance() {
        let grown = shell(
            &placed_at_origin(ShapeKind::Box {
                size: [0.1, 0.2, 0.3],
            }),
            0.04,
        );
        assert_eq!(
            grown.kind,
            ShapeKind::Box {
                size: [0.1 + 0.08, 0.2 + 0.08, 0.3 + 0.08]
            }
        );
    }

    #[test]
    fn a_cylinder_shell_grows_its_height_by_twice_and_its_radius_by_once_the_clearance() {
        let grown = shell(
            &placed_at_origin(ShapeKind::Cylinder {
                height: 0.2,
                radius: 0.05,
            }),
            0.04,
        );
        assert_eq!(
            grown.kind,
            ShapeKind::Cylinder {
                height: 0.2 + 0.08,
                radius: 0.05 + 0.04
            }
        );
    }

    #[test]
    fn a_mesh_shell_is_its_bounding_box_grown_around_its_center() {
        // A tetrahedron from (0, 0, 0) to 1 m along each axis, its shape frame at
        // (2, 0, 0) of the robot.
        let mesh = PlacedShape {
            kind: ShapeKind::Mesh {
                vertices: vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                triangles: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            },
            pose: translation([2.0, 0.0, 0.0]),
        };
        let grown = shell(&mesh, 0.1);
        assert_eq!(
            grown.kind,
            ShapeKind::Box {
                size: [1.0 + 0.2, 1.0 + 0.2, 1.0 + 0.2]
            }
        );
        assert!(close(grown.pose.origin(), [2.5, 0.5, 0.5], 1e-12));
    }

    /// The corner of a cube at the origin, and the ends of its three edges from there.
    const CORNER: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    #[test]
    fn a_triangle_faces_the_side_its_vertices_turn_counterclockwise_on() {
        let up = vertex_normals(&CORNER, &[[0, 1, 2]]);
        assert_eq!(up[..3], [[0.0, 0.0, 1.0]; 3]);
        let down = vertex_normals(&CORNER, &[[0, 2, 1]]);
        assert_eq!(down[..3], [[0.0, 0.0, -1.0]; 3]);
    }

    #[test]
    fn a_vertex_of_two_triangles_takes_the_normal_between_theirs() {
        // The floor of the corner faces +z, its wall in the xz plane faces -y; both
        // have the same area, and they share the edge from vertex 0 to vertex 1.
        let normals = vertex_normals(&CORNER, &[[0, 1, 2], [0, 1, 3]]);
        let between = [
            0.0,
            -std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ];
        assert!(close(normals[0], between, 1e-12));
        assert!(close(normals[1], between, 1e-12));
        assert_eq!(normals[2], [0.0, 0.0, 1.0]);
        assert_eq!(normals[3], [0.0, -1.0, 0.0]);
    }

    #[test]
    fn a_larger_triangle_weighs_more_in_the_normal_of_a_shared_vertex() {
        // The floor reaches 3 m along y: three times the area of the wall.
        let vertices = [CORNER[0], CORNER[1], [0.0, 3.0, 0.0], CORNER[3]];
        let normals = vertex_normals(&vertices, &[[0, 1, 2], [0, 1, 3]]);
        let length = 10.0_f64.sqrt();
        assert!(close(normals[0], [0.0, -1.0 / length, 3.0 / length], 1e-12));
    }

    #[test]
    fn a_vertex_with_no_normal_takes_up_and_a_triangle_past_the_vertices_adds_nothing() {
        let normals = vertex_normals(
            &CORNER,
            // A triangle with no area, one past the vertices, and the floor.
            &[[3, 3, 0], [0, 1, 9], [0, 1, 2]],
        );
        assert_eq!(
            normals[3],
            [0.0, 0.0, 1.0],
            "only in the triangle with no area"
        );
        assert_eq!(normals[1], [0.0, 0.0, 1.0], "the floor alone");
        assert_eq!(
            vertex_normals(&CORNER, &[[0, 1, 2], [0, 2, 1]])[0],
            [0.0, 0.0, 1.0],
            "two triangles that face opposite ways add up to no normal"
        );
    }
}
