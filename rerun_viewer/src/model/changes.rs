//! What changed in the planning scene between two reads of it, as lines of a log.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::model::placement::Placement;
use crate::model::scene::{AttachedObject, PlanningScene, SceneObject, Shape};

/// One change of the planning scene between two reads.
#[derive(Debug, Clone, PartialEq)]
pub enum SceneChange {
    Added(String),
    Removed(String),
    Moved(String),
    Reshaped(String),
    Attached { id: String, link: String },
    Detached { id: String, link: String },
    PairAllowed(String, String),
    PairDropped(String, String),
    ClearanceChanged { from: f64, to: f64 },
}

impl fmt::Display for SceneChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Added(id) => write!(f, "added {id}"),
            Self::Removed(id) => write!(f, "removed {id}"),
            Self::Moved(id) => write!(f, "moved {id}"),
            Self::Reshaped(id) => write!(f, "reshaped {id}"),
            Self::Attached { id, link } => write!(f, "attached {id} to {link}"),
            Self::Detached { id, link } => write!(f, "detached {id} from {link}"),
            Self::PairAllowed(a, b) => write!(f, "allowed {a} to touch {b}"),
            Self::PairDropped(a, b) => write!(f, "stopped allowing {a} to touch {b}"),
            Self::ClearanceChanged { from, to } => {
                write!(f, "clearance changed from {from} m to {to} m")
            }
        }
    }
}

/// The changes that take the scene `old` to the scene `new`, the objects in the order
/// of their ids, then the allowed pairs, then the clearance. An object that a link
/// starts to carry is attached, not removed; one that a link stops carrying is
/// detached, and also removed when the scene no longer holds it.
pub fn scene_changes(old: &PlanningScene, new: &PlanningScene) -> Vec<SceneChange> {
    let old_free = by_id(&old.objects, |object| &object.id);
    let new_free = by_id(&new.objects, |object| &object.id);
    let old_held = by_id(&old.attached, |attached| &attached.id);
    let new_held = by_id(&new.attached, |attached| &attached.id);
    let ids: BTreeSet<&str> = [&old_free, &new_free]
        .into_iter()
        .flat_map(|objects| objects.keys().copied())
        .chain(
            [&old_held, &new_held]
                .into_iter()
                .flat_map(|held| held.keys().copied()),
        )
        .collect();
    let mut changes = Vec::new();
    for id in ids {
        let before = Seen::of(old_free.get(id).copied(), old_held.get(id).copied());
        let after = Seen::of(new_free.get(id).copied(), new_held.get(id).copied());
        changes.extend(object_changes(id, before, after));
    }
    changes.extend(pair_changes(&old.allowed_pairs, &new.allowed_pairs));
    if old.clearance_m != new.clearance_m {
        changes.push(SceneChange::ClearanceChanged {
            from: old.clearance_m,
            to: new.clearance_m,
        });
    }
    changes
}

/// One object in one read of the scene: absent, free, or carried by a link.
#[derive(Clone, Copy)]
enum Seen<'a> {
    Absent,
    Free(&'a SceneObject),
    Held(&'a AttachedObject),
}

impl<'a> Seen<'a> {
    /// The scene holds an id once, free or carried; a carried one wins should both come.
    fn of(free: Option<&'a SceneObject>, held: Option<&'a AttachedObject>) -> Self {
        match (free, held) {
            (_, Some(held)) => Self::Held(held),
            (Some(free), None) => Self::Free(free),
            (None, None) => Self::Absent,
        }
    }
}

fn object_changes(id: &str, before: Seen, after: Seen) -> Vec<SceneChange> {
    let id = id.to_string();
    let attached = |held: &AttachedObject| SceneChange::Attached {
        id: id.clone(),
        link: held.link.clone(),
    };
    let detached = |held: &AttachedObject| SceneChange::Detached {
        id: id.clone(),
        link: held.link.clone(),
    };
    match (before, after) {
        (Seen::Absent, Seen::Absent) => vec![],
        (Seen::Absent, Seen::Free(_)) => vec![SceneChange::Added(id.clone())],
        (Seen::Absent, Seen::Held(new)) => vec![SceneChange::Added(id.clone()), attached(new)],
        (Seen::Free(_), Seen::Absent) => vec![SceneChange::Removed(id.clone())],
        (Seen::Held(old), Seen::Absent) => vec![detached(old), SceneChange::Removed(id.clone())],
        (Seen::Free(_), Seen::Held(new)) => vec![attached(new)],
        (Seen::Held(old), Seen::Free(_)) => vec![detached(old)],
        (Seen::Free(old), Seen::Free(new)) => moved_or_reshaped(
            &id,
            (&old.placement, &new.placement),
            (&old.shapes, &new.shapes),
        ),
        (Seen::Held(old), Seen::Held(new)) if old.link != new.link => {
            vec![detached(old), attached(new)]
        }
        (Seen::Held(old), Seen::Held(new)) => moved_or_reshaped(
            &id,
            (&old.placement, &new.placement),
            (&old.shapes, &new.shapes),
        ),
    }
}

/// A moved object when its placement changed (old, new), a reshaped one when its
/// shapes changed (old, new).
fn moved_or_reshaped(
    id: &str,
    placements: (&Placement, &Placement),
    shapes: (&[Shape], &[Shape]),
) -> Vec<SceneChange> {
    let mut changes = Vec::new();
    if placements.0 != placements.1 {
        changes.push(SceneChange::Moved(id.to_string()));
    }
    if shapes.0 != shapes.1 {
        changes.push(SceneChange::Reshaped(id.to_string()));
    }
    changes
}

fn pair_changes(old: &[(String, String)], new: &[(String, String)]) -> Vec<SceneChange> {
    let old: BTreeSet<&(String, String)> = old.iter().collect();
    let new: BTreeSet<&(String, String)> = new.iter().collect();
    let allowed = new
        .difference(&old)
        .map(|(a, b)| SceneChange::PairAllowed(a.clone(), b.clone()));
    let dropped = old
        .difference(&new)
        .map(|(a, b)| SceneChange::PairDropped(a.clone(), b.clone()));
    allowed.chain(dropped).collect()
}

fn by_id<'a, T>(items: &'a [T], id: impl Fn(&'a T) -> &'a String) -> BTreeMap<&'a str, &'a T> {
    items.iter().map(|item| (id(item).as_str(), item)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::scene::ShapeKind;
    use crate::model::scene::tests::{at, boxed, held};

    fn scene(objects: Vec<SceneObject>, attached: Vec<AttachedObject>) -> PlanningScene {
        PlanningScene {
            objects,
            attached,
            allowed_pairs: vec![],
            clearance_m: 0.04,
        }
    }

    fn lines(changes: Vec<SceneChange>) -> Vec<String> {
        changes.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn the_same_scene_has_no_changes() {
        let a = scene(vec![boxed("table", [0.5, 0.0, 0.0])], vec![]);
        assert!(scene_changes(&a, &a.clone()).is_empty());
    }

    #[test]
    fn an_object_is_added_then_removed() {
        let empty = scene(vec![], vec![]);
        let with_lime = scene(vec![boxed("brain/lime_1", [0.4, -0.2, 0.1])], vec![]);
        assert_eq!(
            lines(scene_changes(&empty, &with_lime)),
            ["added brain/lime_1"]
        );
        assert_eq!(
            lines(scene_changes(&with_lime, &empty)),
            ["removed brain/lime_1"]
        );
    }

    #[test]
    fn an_object_moves_or_changes_its_shapes() {
        let before = scene(vec![boxed("brain/lime_1", [0.4, -0.2, 0.1])], vec![]);
        let moved = scene(vec![boxed("brain/lime_1", [0.4, -0.2, 0.2])], vec![]);
        assert_eq!(
            lines(scene_changes(&before, &moved)),
            ["moved brain/lime_1"]
        );
        let mut reshaped = before.clone();
        reshaped.objects[0].shapes.push(Shape {
            kind: ShapeKind::Sphere { radius: 0.05 },
            placement: at([0.0; 3]),
        });
        assert_eq!(
            lines(scene_changes(&before, &reshaped)),
            ["reshaped brain/lime_1"]
        );
    }

    #[test]
    fn an_object_is_attached_then_detached_then_removed() {
        let free = scene(vec![boxed("brain/lime_1", [0.4, -0.2, 0.1])], vec![]);
        let carried = scene(vec![], vec![held("brain/lime_1", "openarm_right_tcp")]);
        let placed = scene(vec![boxed("brain/lime_1", [0.3, 0.1, 0.1])], vec![]);
        let empty = scene(vec![], vec![]);
        assert_eq!(
            lines(scene_changes(&free, &carried)),
            ["attached brain/lime_1 to openarm_right_tcp"]
        );
        assert_eq!(
            lines(scene_changes(&carried, &placed)),
            ["detached brain/lime_1 from openarm_right_tcp"]
        );
        assert_eq!(
            lines(scene_changes(&carried, &empty)),
            [
                "detached brain/lime_1 from openarm_right_tcp",
                "removed brain/lime_1"
            ]
        );
        assert_eq!(
            lines(scene_changes(&empty, &carried)),
            [
                "added brain/lime_1",
                "attached brain/lime_1 to openarm_right_tcp"
            ]
        );
    }

    #[test]
    fn a_handover_detaches_then_attaches() {
        let right = scene(vec![], vec![held("brain/lime_1", "openarm_right_tcp")]);
        let left = scene(vec![], vec![held("brain/lime_1", "openarm_left_tcp")]);
        assert_eq!(
            lines(scene_changes(&right, &left)),
            [
                "detached brain/lime_1 from openarm_right_tcp",
                "attached brain/lime_1 to openarm_left_tcp"
            ]
        );
    }

    #[test]
    fn pairs_and_the_clearance_change() {
        let before = scene(vec![], vec![]);
        let mut after = before.clone();
        after.allowed_pairs = vec![(
            "brain/lime_1".to_string(),
            "openarm_right_finger".to_string(),
        )];
        after.clearance_m = 0.05;
        assert_eq!(
            lines(scene_changes(&before, &after)),
            [
                "allowed brain/lime_1 to touch openarm_right_finger",
                "clearance changed from 0.04 m to 0.05 m"
            ]
        );
        assert_eq!(
            lines(scene_changes(&after, &before)),
            [
                "stopped allowing brain/lime_1 to touch openarm_right_finger",
                "clearance changed from 0.05 m to 0.04 m"
            ]
        );
    }

    #[test]
    fn the_changes_of_several_objects_come_in_the_order_of_their_ids() {
        let before = scene(vec![boxed("b", [0.0; 3])], vec![]);
        let after = scene(vec![boxed("a", [0.0; 3]), boxed("c", [0.0; 3])], vec![]);
        assert_eq!(
            lines(scene_changes(&before, &after)),
            ["added a", "removed b", "added c"]
        );
    }
}
