//! The services of room_map:v1, each answered by its member on a task of its own until the
//! node stops: a member's refusal is the error reply of its kind.

use std::sync::Arc;

use control_core::serving;
use mapping_core::MapPose;
use peppygen::NodeRunner;
use peppygen::exposed_services::room_map::{
    finish_mapping, get_map_state, list_maps, load_map, reinitialize_global_localization,
    remove_map, set_initial_pose, start_mapping,
};

use crate::messages::{extent_of, ns_of, orientation_of, position_of, time_of};
use crate::rooms::members::{MAP_FRAME, Members, TargetRequest};
use crate::rooms::state::{Refusal, SavedMap};

/// The error reply of `refusal`, in the failure kinds of the member `$member`.
macro_rules! failure_of {
    ($member:ident, $refusal:expr) => {{
        let (kind, message) = kind_and_message($refusal);
        $member::Failure::new(
            $member::FailureKind::parse(kind).expect("room_map declares the kind for its member"),
            message,
        )
    }};
}

/// A saved map's object as the members of room_map give it, with its extent.
macro_rules! saved_map {
    ($item:path, $object:expr) => {{
        let object: SavedMap = $object;
        $item {
            map_name: object.name,
            cell_size_m: object.resolution_m,
            extent: extent_of(&object.extent),
            scans: object.scans,
            mapping_start_timestamp: time_of(object.mapping_start_unix_ns),
        }
    }};
}

fn kind_and_message(refusal: Refusal) -> (&'static str, String) {
    match refusal {
        Refusal::Rejected(message) => ("rejected", message),
        Refusal::PreconditionFailed(message) => ("precondition_failed", message),
        Refusal::ResourceUnavailable(message) => ("resource_unavailable", message),
        Refusal::AttemptFailed(message) => ("attempt_failed", message),
        Refusal::NoMatch(message) => ("no_match", message),
        Refusal::Extrapolation(message) => ("extrapolation", message),
    }
}

/// Serves the eight members of room_map.
pub fn serve(runner: &Arc<NodeRunner>, members: Members) {
    macro_rules! spawn {
        ($name:literal, $member:ident, $handler:expr) => {{
            let runner = runner.clone();
            let members = members.clone();
            let token = runner.cancellation_token().clone();
            tokio::spawn(async move {
                serving::serve($name, token.cancelled(), || {
                    $member::handle_requests(&runner, |request| $handler(&members, request))
                })
                .await;
            });
        }};
    }
    spawn!("get_map_state", get_map_state, |members: &Members, _| {
        let state = members.get_map_state();
        Ok(get_map_state::Response {
            map: state.map.map(|object| get_map_state::ResponseMap {
                map_name: object.name,
                cell_size_m: object.resolution_m,
                extent: object.extent.as_ref().map(extent_of),
                scans: object.scans,
                mapping_start_timestamp: time_of(object.mapping_start_unix_ns),
            }),
            mapping: state.mapping,
            searching: state.searching,
            waiting: state.waiting,
            localized: state.localized,
        })
    });
    spawn!(
        "start_mapping",
        start_mapping,
        |members: &Members, request: start_mapping::Request| {
            members
                .start_mapping(&request.data.map_name)
                .map_err(|refusal| failure_of!(start_mapping, refusal))
        }
    );
    spawn!("finish_mapping", finish_mapping, |members: &Members, _| {
        members
            .finish_mapping()
            .map(|object| finish_mapping::Response {
                map: saved_map!(finish_mapping::ResponseMap, object),
            })
            .map_err(|refusal| failure_of!(finish_mapping, refusal))
    });
    spawn!("list_maps", list_maps, |members: &Members, _| {
        members
            .list_maps()
            .map(|objects| list_maps::Response {
                maps: objects
                    .into_iter()
                    .map(|object| saved_map!(list_maps::ResponseMapsItem, object))
                    .collect(),
            })
            .map_err(|refusal| failure_of!(list_maps, refusal))
    });
    spawn!(
        "remove_map",
        remove_map,
        |members: &Members, request: remove_map::Request| {
            members
                .remove_map(&request.data.map_name)
                .map_err(|refusal| failure_of!(remove_map, refusal))
        }
    );
    spawn!(
        "load_map",
        load_map,
        |members: &Members, request: load_map::Request| {
            members
                .load_map(&request.data.map_name)
                .map(|object| load_map::Response {
                    map: saved_map!(load_map::ResponseMap, object),
                })
                .map_err(|refusal| failure_of!(load_map, refusal))
        }
    );
    spawn!(
        "set_initial_pose",
        set_initial_pose,
        |members: &Members, request: set_initial_pose::Request| {
            let data = request.data;
            let target = TargetRequest {
                frame: data.target.frame,
                at_ns: data.target.timestamp.map(ns_of),
                position: data.target.position,
                orientation: data.target.orientation,
            };
            members
                .set_initial_pose(&target, data.search_radius_m, data.search_angle_rad)
                .map(|matched: MapPose| set_initial_pose::Response {
                    current_pose: set_initial_pose::ResponseCurrentPose {
                        frame: MAP_FRAME.to_string(),
                        timestamp: time_of(matched.stamp_ns),
                        position: position_of(matched.pose),
                        orientation: orientation_of(matched.pose),
                    },
                })
                .map_err(|refusal| failure_of!(set_initial_pose, refusal))
        }
    );
    spawn!(
        "reinitialize_global_localization",
        reinitialize_global_localization,
        |members: &Members, _| {
            members
                .reinitialize_global_localization()
                .map_err(|refusal| failure_of!(reinitialize_global_localization, refusal))
        }
    );
}
