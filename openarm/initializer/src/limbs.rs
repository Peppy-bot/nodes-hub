//! What this robot's limbs are, on both sides of the seat.
//!
//! Toward the robot a limb is a pairing slot (`left_arm`, `right_gripper`);
//! toward the simulation it is a position in the arrays of the
//! `simulation_robot` contract, in the order the engine listed the model's
//! limbs when the robot attached. Everything that crosses the seat is
//! translated here, so no loop below carries an index it did not resolve.

use std::sync::{Arc, Mutex};

use peppygen::consumed_actions::simulation::attach;
use peppygen::consumed_services::simulation::command;

/// The arms of an OpenArm, by the slot that drives each and the name its
/// model gives it.
pub const ARMS: [(&str, &str); 2] = [("left_arm_link", "left"), ("right_arm_link", "right")];
/// The grippers, likewise.
pub const GRIPPERS: [(&str, &str); 2] = [
    ("left_gripper_link", "left"),
    ("right_gripper_link", "right"),
];
/// Joints of one OpenArm arm.
pub const ARM_DOF: usize = 7;

/// The setpoint of one arm, as the engine takes it.
#[derive(Clone, Debug, PartialEq)]
pub struct ArmCommand {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
}

/// The setpoint of one gripper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GripperCommand {
    pub opening: f64,
    pub max_effort: f64,
}

/// The latest setpoint of every limb, in the engine's order: what the next
/// command carries. A limb nothing has commanded yet holds the pose the
/// robot joined at.
#[derive(Debug)]
pub struct Commands {
    arms: Vec<Option<ArmCommand>>,
    grippers: Vec<Option<GripperCommand>>,
}

impl Commands {
    fn new(arms: usize, grippers: usize) -> Self {
        Self {
            arms: vec![None; arms],
            grippers: vec![None; grippers],
        }
    }

    /// The command the engine reads: every limb of the model, in its order.
    pub fn request(&self) -> command::Request {
        command::Request::new(
            self.arms
                .iter()
                .map(|arm| match arm {
                    Some(arm) => command::CommandArmsItem {
                        positions: arm.positions.clone(),
                        velocities: arm.velocities.clone(),
                    },
                    None => command::CommandArmsItem {
                        positions: Vec::new(),
                        velocities: Vec::new(),
                    },
                })
                .collect(),
            self.grippers
                .iter()
                .map(|gripper| command::CommandGrippersItem {
                    commanded: gripper.is_some(),
                    opening: gripper.map_or(0.0, |g| g.opening),
                    max_effort: gripper.map_or(0.0, |g| g.max_effort),
                })
                .collect(),
        )
    }
}

/// This robot's limbs as the engine numbered them: which entry of the
/// contract's arrays each slot drives and is measured by, and the setpoints
/// waiting to go out.
#[derive(Clone, Debug)]
pub struct Seat {
    /// The engine's index of each arm slot, in [`ARMS`] order.
    arms: [usize; ARMS.len()],
    /// The engine's index of each gripper slot, in [`GRIPPERS`] order.
    grippers: [usize; GRIPPERS.len()],
    commands: Arc<Mutex<Commands>>,
}

impl Seat {
    /// Resolves the limbs of the model the engine seated this robot as. A
    /// model that carries none of this robot's limbs is refused here rather
    /// than driving the wrong arm.
    pub fn of(response: &attach::GoalResponseData) -> Result<Self, String> {
        let index = |limbs: &[String], (slot, name): (&str, &str)| {
            limbs
                .iter()
                .position(|limb| limb == name)
                .ok_or_else(|| {
                    format!(
                        "the simulation seats this robot with limbs [{}], which carry no '{name}' for slot '{slot}'",
                        limbs.join(", ")
                    )
                })
        };
        let arms = ARMS
            .map(|arm| index(&response.arm_names, arm))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        for (&index, (slot, name)) in arms.iter().zip(ARMS) {
            let joints = *response
                .arm_joints
                .get(index)
                .ok_or_else(|| format!("the simulation named no joint count for arm '{name}'"))?;
            if joints as usize != ARM_DOF {
                return Err(format!(
                    "slot '{slot}' drives an OpenArm arm of {ARM_DOF} joints, and the simulation seats this robot with a '{name}' of {joints}"
                ));
            }
        }
        let grippers = GRIPPERS
            .map(|gripper| index(&response.gripper_names, gripper))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            arms: arms.try_into().expect("one index per arm slot"),
            grippers: grippers.try_into().expect("one index per gripper slot"),
            commands: Arc::new(Mutex::new(Commands::new(
                response.arm_names.len(),
                response.gripper_names.len(),
            ))),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Commands> {
        self.commands.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The command every limb's latest setpoint makes.
    pub fn request(&self) -> command::Request {
        self.lock().request()
    }

    /// Holds an arm slot's setpoint until the next command goes out.
    pub fn set_arm(&self, slot: &str, command: ArmCommand) {
        let Some(index) = self.arm_index(slot) else {
            return;
        };
        self.lock().arms[index] = Some(command);
    }

    /// Holds a gripper slot's setpoint until the next command goes out.
    pub fn set_gripper(&self, slot: &str, command: GripperCommand) {
        let Some(index) = self.gripper_index(slot) else {
            return;
        };
        self.lock().grippers[index] = Some(command);
    }

    /// The effort ceiling this robot last set on a gripper slot, which the
    /// engine holds until the next command replaces it. 0 while nothing has
    /// commanded one, the value the contract reserves for no effort control.
    pub fn gripper_max_effort(&self, slot: &str) -> f64 {
        self.gripper_index(slot)
            .and_then(|index| self.lock().grippers[index])
            .map_or(0.0, |gripper| gripper.max_effort)
    }

    fn arm_index(&self, slot: &str) -> Option<usize> {
        ARMS.iter()
            .position(|(name, _)| *name == slot)
            .map(|slot| self.arms[slot])
    }

    fn gripper_index(&self, slot: &str) -> Option<usize> {
        GRIPPERS
            .iter()
            .position(|(name, _)| *name == slot)
            .map(|slot| self.grippers[slot])
    }

    /// The measured state of an arm slot in a state the engine sent back.
    pub fn arm_state<'a>(
        &self,
        slot: &str,
        feedback: &'a attach::FeedbackMessage,
    ) -> Option<&'a attach::SimulationRobotAttachActionFeedbackMessageArmsItem> {
        feedback.arms.get(self.arm_index(slot)?)
    }

    /// The measured state of a gripper slot.
    pub fn gripper_state<'a>(
        &self,
        slot: &str,
        feedback: &'a attach::FeedbackMessage,
    ) -> Option<&'a attach::SimulationRobotAttachActionFeedbackMessageGrippersItem> {
        feedback.grippers.get(self.gripper_index(slot)?)
    }
}

/// The catalogue model of an OpenArm generation.
pub fn model_of(hardware_version: &str) -> Result<String, String> {
    match hardware_version {
        "v1" | "v2" => Ok(format!("openarm_{hardware_version}")),
        other => Err(format!(
            "hardware_version must be v1 or v2, and this robot's is '{other}'"
        )),
    }
}

/// A setpoint the engine can use: one finite position per joint, the
/// velocities only when they match too.
pub fn arm_command(positions: Vec<f64>, velocities: Vec<f64>) -> Option<ArmCommand> {
    if positions.len() != ARM_DOF || !positions.iter().all(|v| v.is_finite()) {
        return None;
    }
    let velocities = if velocities.len() == ARM_DOF && velocities.iter().all(|v| v.is_finite()) {
        velocities
    } else {
        Vec::new()
    };
    Some(ArmCommand {
        positions,
        velocities,
    })
}

/// A gripper setpoint the engine can use: a finite opening and a finite,
/// non-negative effort cap.
pub fn gripper_command(opening: f64, max_effort: f64) -> Option<GripperCommand> {
    (opening.is_finite() && max_effort.is_finite() && max_effort >= 0.0).then_some(GripperCommand {
        opening,
        max_effort,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seated(arms: &[&str], grippers: &[&str]) -> Result<Seat, String> {
        seated_with(arms, &vec![ARM_DOF as u32; arms.len()], grippers)
    }

    fn seated_with(arms: &[&str], joints: &[u32], grippers: &[&str]) -> Result<Seat, String> {
        Seat::of(&attach::GoalResponseData::new(
            arms.iter().map(|s| (*s).to_owned()).collect(),
            joints.to_vec(),
            grippers.iter().map(|s| (*s).to_owned()).collect(),
        ))
    }

    fn positions() -> Vec<f64> {
        (0..ARM_DOF).map(|j| j as f64 / 10.0).collect()
    }

    #[test]
    fn a_seat_refuses_a_model_whose_arms_carry_other_joints() {
        let refused =
            seated_with(&["left", "right"], &[ARM_DOF as u32, 6], &["left", "right"]).unwrap_err();
        assert!(
            refused.contains("slot 'right_arm_link' drives an OpenArm arm of 7 joints")
                && refused.contains("a 'right' of 6"),
            "{refused}"
        );
        let missing =
            seated_with(&["left", "right"], &[ARM_DOF as u32], &["left", "right"]).unwrap_err();
        assert!(
            missing.contains("no joint count for arm 'right'"),
            "{missing}"
        );
    }

    #[test]
    fn a_seat_maps_each_slot_onto_the_limb_the_engine_numbered() {
        // The engine lists the model's limbs in its own order, which is not
        // the order of this robot's slots.
        let seat = seated(&["right", "left"], &["right", "left"]).unwrap();
        seat.set_arm(
            "left_arm_link",
            arm_command(positions(), Vec::new()).unwrap(),
        );
        seat.set_gripper("right_gripper_link", gripper_command(0.5, 1.0).unwrap());
        let request = seat.request();
        assert_eq!(
            request.arms[1].positions,
            positions(),
            "left is the engine's second arm"
        );
        assert!(request.arms[0].positions.is_empty());
        assert!(request.grippers[0].commanded && !request.grippers[1].commanded);
        assert_eq!(request.grippers[0].opening, 0.5);

        // States come back in the same order and are read by slot.
        let feedback = attach::FeedbackMessage {
            timestamp: std::time::SystemTime::UNIX_EPOCH,
            arms: vec![
                attach::SimulationRobotAttachActionFeedbackMessageArmsItem {
                    positions: vec![1.0; ARM_DOF],
                    velocities: Vec::new(),
                },
                attach::SimulationRobotAttachActionFeedbackMessageArmsItem {
                    positions: positions(),
                    velocities: Vec::new(),
                },
            ],
            grippers: vec![
                attach::SimulationRobotAttachActionFeedbackMessageGrippersItem {
                    opening: 0.5,
                    effort: 0.0,
                },
                attach::SimulationRobotAttachActionFeedbackMessageGrippersItem {
                    opening: 0.25,
                    effort: 0.0,
                },
            ],
        };
        assert_eq!(
            seat.arm_state("left_arm_link", &feedback)
                .unwrap()
                .positions,
            positions()
        );
        assert_eq!(
            seat.arm_state("right_arm_link", &feedback)
                .unwrap()
                .positions,
            vec![1.0; ARM_DOF]
        );
        assert_eq!(
            seat.gripper_state("left_gripper_link", &feedback)
                .unwrap()
                .opening,
            0.25
        );
        assert_eq!(
            seat.gripper_state("right_gripper_link", &feedback)
                .unwrap()
                .opening,
            0.5
        );
        assert!(seat.arm_state("nope", &feedback).is_none());
    }

    #[test]
    fn a_model_without_this_robots_limbs_is_refused() {
        let err = seated(&["left"], &["left", "right"]).unwrap_err();
        assert!(
            err.contains("no 'right' for slot 'right_arm_link'"),
            "{err}"
        );
        let err = seated(&["left", "right"], &["left"]).unwrap_err();
        assert!(
            err.contains("no 'right' for slot 'right_gripper_link'"),
            "{err}"
        );
    }

    #[test]
    fn an_uncommanded_limb_holds_what_the_robot_joined_with() {
        let seat = seated(&["left", "right"], &["left", "right"]).unwrap();
        let request = seat.request();
        assert!(request.arms.iter().all(|arm| arm.positions.is_empty()));
        assert!(request.grippers.iter().all(|gripper| !gripper.commanded));
    }

    #[test]
    fn setpoints_need_one_finite_value_per_joint() {
        assert!(arm_command(positions(), Vec::new()).is_some());
        assert!(arm_command(vec![0.0; ARM_DOF - 1], Vec::new()).is_none());
        let mut nan = positions();
        nan[3] = f64::NAN;
        assert!(arm_command(nan, Vec::new()).is_none());
        // Velocities that do not match the joints are dropped, the
        // positions stand.
        let short = arm_command(positions(), vec![1.0]).unwrap();
        assert!(short.velocities.is_empty());
        let full = arm_command(positions(), vec![1.0; ARM_DOF]).unwrap();
        assert_eq!(full.velocities, vec![1.0; ARM_DOF]);

        assert_eq!(
            gripper_command(0.5, 1.0),
            Some(GripperCommand {
                opening: 0.5,
                max_effort: 1.0
            })
        );
        assert!(gripper_command(f64::NAN, 1.0).is_none());
        assert!(gripper_command(0.5, -1.0).is_none());
    }

    #[test]
    fn the_model_follows_the_hardware_generation() {
        assert_eq!(model_of("v1").unwrap(), "openarm_v1");
        assert_eq!(model_of("v2").unwrap(), "openarm_v2");
        assert!(model_of("v3").unwrap_err().contains("must be v1 or v2"));
    }
}
