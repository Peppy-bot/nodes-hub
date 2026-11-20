//! The listeners that fill the node's inputs: base_states into the odometry, with a read
//! of describe_base at each new odometry start; and the lidar's scans to the worker, with
//! the lidar's place on the base looked up once its frame is known.

use std::sync::Arc;
use std::time::Duration;

use camera_core::sensor_mounts::{DESCRIBE_SENSORS_TIMEOUT, DescribedSensors, LIDAR_2D};
use control_core::fault::Fault;
use control_core::footprint::Footprint;
use control_core::non_negative_finite::NonNegativeFinite;
use control_core::sync::lock_ignoring_poison;
use frame_tree::Transform;
use navigation_core::PaddedFootprint;
use peppygen::NodeRunner;
use peppygen::consumed_services::base::describe_base;
use peppygen::consumed_services::sensor_mounts::describe_sensors;
use peppygen::consumed_services::transform_lookup::lookup_transform;
use peppygen::consumed_topics::base::base_states;
use peppygen::consumed_topics::lidar::scan_stream;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::inputs::{BaseFacts, FactsRead, Inputs, Mount};
use crate::lidar::ArrivedScan;
use crate::messages::ns_of;
use crate::odometry::{Measurement, OdometryStart};

/// The frame the lidar's place is looked up in.
const BASE_FRAME: &str = "base_footprint";
/// How long (s) a lookup of the lidar's place waits, transform_lookup's longest.
const LOOKUP_TIMEOUT_S: f64 = 1.0;

/// Follows base_states until they end.
pub async fn follow_base(
    runner: Arc<NodeRunner>,
    inputs: Arc<Inputs>,
    padding: NonNegativeFinite,
    source_timeout: Duration,
) -> peppygen::Result<()> {
    let mut states = base_states::subscribe(&runner).await?;
    while let Some((_, state)) = states.next().await? {
        let start = OdometryStart(ns_of(state.odometry_start_timestamp));
        let Ok(odom_from_base) = Transform::new(state.pose.position, state.pose.orientation) else {
            warn!("a base_states holds a pose that is no transform; it is dropped");
            continue;
        };
        let measurement = Measurement {
            at_ns: ns_of(state.timestamp),
            odom_from_base,
        };
        let arrived = inputs.arrival_of(measurement.at_ns);
        let started = lock_ignoring_poison(&inputs.odometry).take(start, measurement);
        *lock_ignoring_poison(&inputs.base_arrived_ns) = Some(arrived);
        inputs.changed.notify_one();
        if started {
            info!("the base's odometry started: describe_base is read");
            inputs.facts.send_replace(FactsRead::Reading(start));
            tokio::spawn(read_facts(
                runner.clone(),
                inputs.clone(),
                start,
                padding,
                source_timeout,
            ));
        }
    }
    Ok(())
}

/// Reads describe_base for the odometry start `start`, unless a newer start began.
async fn read_facts(
    runner: Arc<NodeRunner>,
    inputs: Arc<Inputs>,
    start: OdometryStart,
    padding: NonNegativeFinite,
    source_timeout: Duration,
) {
    let answered = describe_base::poll(
        &runner,
        describe_base::bound_producer(&runner),
        source_timeout,
    )
    .await;
    let read = match answered {
        Ok(response) => match facts_of(&response.data, padding) {
            Ok(facts) => FactsRead::Read(facts),
            Err(why) => FactsRead::Unanswered(why),
        },
        Err(error) => FactsRead::Unanswered(format!(
            "describe_base did not answer within the source timeout: {error}"
        )),
    };
    inputs.facts.send_if_modified(|current| {
        let ours = matches!(current, FactsRead::Reading(reading) if *reading == start);
        if ours {
            *current = read;
        }
        ours
    });
    inputs.changed.notify_one();
}

/// What `answer` says of the base, its footprint padded by `padding`.
fn facts_of(
    answer: &describe_base::ResponseData,
    padding: NonNegativeFinite,
) -> Result<BaseFacts, String> {
    let commanded = match answer.odometry_source.as_str() {
        "measured" => false,
        "commanded" => true,
        other => {
            return Err(format!(
                "describe_base names the odometry source {other:?}, which base_state does not"
            ));
        }
    };
    let points: Vec<[f64; 2]> = answer
        .footprint
        .points
        .iter()
        .map(|point| [point.position[0], point.position[1]])
        .collect();
    let footprint = Footprint::parse(&points)
        .map_err(|e| format!("describe_base's footprint is no outline: {e}"))?;
    let footprint = PaddedFootprint::new(&footprint, padding).map_err(|e| {
        format!("footprint_padding_m pads describe_base's footprint to no outline: {e}")
    })?;
    Ok(BaseFacts {
        commanded,
        footprint,
    })
}

/// Follows the lidar's scans until they end, sending each to the worker. The first scan's
/// frame is checked against sensor_mounts, a lidar a part carries stopping the node, and
/// looked up in base_footprint until the lookup answers.
pub async fn follow_lidar(
    runner: Arc<NodeRunner>,
    inputs: Arc<Inputs>,
    worker: mpsc::UnboundedSender<ArrivedScan>,
    fault: Fault,
) -> peppygen::Result<()> {
    let mut scans = scan_stream::subscribe(&runner).await?;
    let mut fixed: Option<String> = None;
    while let Some((_, scan)) = scans.next().await? {
        let arrival = inputs.scan_arrives();
        *lock_ignoring_poison(&inputs.scan_arrived_ns) =
            Some(inputs.arrival_of(ns_of(scan.header.timestamp)));
        let frame = scan.header.frame.clone();
        let placed = matches!(
            &*lock_ignoring_poison(&inputs.mount),
            Mount::Placed { frame: placed, .. } if *placed == frame
        );
        if !placed {
            if fixed.as_ref() != Some(&frame) {
                if !fixed_to_the_base(&runner, &inputs, &frame, &fault).await {
                    continue;
                }
                fixed = Some(frame.clone());
            }
            place_lidar(&runner, &inputs, &frame).await;
        }
        let arrived = ArrivedScan {
            arrival,
            stamp_ns: ns_of(scan.header.timestamp),
            frame,
            min_angle_rad: scan.min_angle_rad,
            angle_increment_rad: scan.angle_increment_rad,
            time_increment_s: scan.time_increment_s,
            min_range_m: scan.min_range_m,
            max_range_m: scan.max_range_m,
            ranges: scan.ranges,
        };
        if worker.send(arrived).is_err() {
            return Ok(());
        }
    }
    Ok(())
}

/// Whether sensor_mounts names `frame` a 2-D lidar fixed to the base. A lidar a part
/// carries raises `fault`; a describe_sensors that does not answer leaves the mount failing.
async fn fixed_to_the_base(
    runner: &Arc<NodeRunner>,
    inputs: &Arc<Inputs>,
    frame: &str,
    fault: &Fault,
) -> bool {
    let answered = describe_sensors::poll(
        runner,
        describe_sensors::bound_producer(runner),
        DESCRIBE_SENSORS_TIMEOUT,
    )
    .await;
    match answered {
        Ok(response) => match carried_check(&described(response.data), frame) {
            Ok(()) => true,
            Err(why) => {
                fault.raise(why, || runner.cancellation_token().cancel());
                false
            }
        },
        Err(error) => {
            failing(inputs, format!("describe_sensors did not answer: {error}"));
            false
        }
    }
}

/// The mount of the lidar failing, for `why`, which the log gives.
fn failing(inputs: &Inputs, why: String) {
    warn!("the lidar's place on the base: {why}");
    *lock_ignoring_poison(&inputs.mount) = Mount::Failing(why);
}

/// Looks the lidar's frame up in base_footprint on transform_lookup; a lookup that fails
/// leaves the mount failing, its failure logged.
async fn place_lidar(runner: &Arc<NodeRunner>, inputs: &Arc<Inputs>, frame: &str) {
    let timeout = Duration::from_secs_f64(LOOKUP_TIMEOUT_S);
    let request = lookup_transform::Request::new(
        BASE_FRAME.to_string(),
        frame.to_string(),
        None,
        None,
        LOOKUP_TIMEOUT_S,
    );
    let answered = lookup_transform::poll(
        runner,
        lookup_transform::bound_producer(runner),
        timeout + timeout,
        request,
    )
    .await;
    match answered {
        Ok(response) => match Transform::new(response.data.position, response.data.orientation) {
            Ok(base_from_lidar) => {
                info!("the lidar {frame} stands on the base");
                *lock_ignoring_poison(&inputs.mount) = Mount::Placed {
                    frame: frame.to_string(),
                    base_from_lidar,
                };
            }
            Err(error) => failing(
                inputs,
                format!("the lookup of {frame} gave no transform: {error}"),
            ),
        },
        Err(error) => failing(
            inputs,
            format!(
                "the lookup of {frame} in {BASE_FRAME} failed: {}",
                lookup_transform::Failure::of(&error)
                    .map_or_else(|| error.to_string(), |failure| failure.to_string())
            ),
        ),
    }
}

/// The four arrays of `answer` that place each sensor.
fn described(answer: describe_sensors::ResponseData) -> DescribedSensors {
    DescribedSensors {
        sensor_names: answer.sensor_names,
        sensor_kinds: answer.sensor_kinds,
        sensor_frames: answer.sensor_frames,
        carried_by: answer.carried_by,
    }
}

/// Refuses a lidar that sensor_mounts does not name, or names carried by a part, and an
/// answer whose arrays do not give each sensor one kind, one frame and one carrier.
fn carried_check(answer: &DescribedSensors, frame: &str) -> Result<(), String> {
    let sensors = answer.sensors().map_err(|e| e.to_string())?;
    let lidar = sensors
        .iter()
        .find(|sensor| sensor.frame == frame && LIDAR_2D.kinds.contains(&sensor.kind))
        .ok_or_else(|| {
            format!(
                "sensor_mounts names no lidar of the frame {frame}, a sensor of {}: bind the \
                 lidar link to a lidar of the robot whose sensor_mounts the node reads",
                LIDAR_2D.kinds.join(" or ")
            )
        })?;
    if lidar.carried_by.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the lidar {} is carried by {}: room_mapper maps with a lidar fixed to the base, as \
         slam_toolbox and AMCL read a laser's pose once; bind a lidar whose carried_by is empty",
        lidar.name, lidar.carried_by
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    /// A chest camera, and a lidar on the frame `lidar_frame` that `carried_by` carries.
    fn answer(carried_by: &str) -> DescribedSensors {
        DescribedSensors {
            sensor_names: strings(&["chest", "lidar"]),
            sensor_kinds: strings(&["rgbd_camera", "lidar_2d"]),
            sensor_frames: strings(&["chest_color_optical_frame", "lidar_frame"]),
            carried_by: strings(&["", carried_by]),
        }
    }

    #[test]
    fn a_lidar_fixed_to_the_base_passes_and_a_carried_one_is_refused() {
        assert_eq!(carried_check(&answer(""), "lidar_frame"), Ok(()));
        let refusal = carried_check(&answer("left_arm"), "lidar_frame").expect_err("carried");
        assert!(
            refusal.starts_with("the lidar lidar is carried by left_arm"),
            "{refusal}"
        );
    }

    #[test]
    fn a_frame_no_lidar_of_sensor_mounts_names_is_refused() {
        for frame in ["chest_color_optical_frame", "rear_lidar_frame"] {
            assert_eq!(
                carried_check(&answer(""), frame),
                Err(format!(
                    "sensor_mounts names no lidar of the frame {frame}, a sensor of lidar_2d: \
                     bind the lidar link to a lidar of the robot whose sensor_mounts the node \
                     reads"
                ))
            );
        }
    }

    #[test]
    fn an_answer_whose_arrays_differ_in_length_is_refused() {
        let mut short = answer("");
        short.carried_by.pop();
        assert_eq!(
            carried_check(&short, "lidar_frame"),
            Err(
                "sensor_mounts' describe_sensors names 2 sensors and gives 2 sensor_kinds, 2 \
                 sensor_frames and 1 carried_by: one of each for each sensor"
                    .to_string()
            )
        );
    }
}
