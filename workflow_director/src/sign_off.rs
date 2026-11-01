//! The sign-off records of the director. A trial that signs off a bundle
//! keeps what it tested: the bundle digest, the core node of the stack, the
//! binding with the identity of each bound robot, the versions of the
//! director, of the nodes of the bound robots and of the anomaly monitor,
//! the incidents policy and the peppy version. An unattended run on physical
//! robots runs only when a record admits it, and `"auto"` binds physical
//! robots only as one whole record binds them. The records stay on the disk
//! of the director, one file per record, and the director reads them at its
//! start; it removes a record only at the start of a trial of the same
//! bundle and binding, and at `forget_robot`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use workflow_documents::lock::{Lock, LockedNode};
use workflow_documents::run::Incidents;
use workflow_documents::{Name, NameTag, RobotName, Sha256};

use crate::fleet::Snapshot;
use crate::records::RecordDir;

/// A bound robot as a sign-off names it: its name, the model it reports
/// (empty with none) and the core node of its machine.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundRobot {
    pub name: RobotName,
    pub model: String,
    pub core_node: String,
}

/// What a sign-off names of a run and of its stack.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignOff {
    /// The digest of the bundle, without its trials.
    pub bundle: Sha256,
    pub core_node: String,
    pub binding: BTreeMap<Name, BoundRobot>,
    /// The director, the node of the anomaly monitor and each node of the
    /// copy of each bound robot.
    pub nodes: BTreeMap<NameTag, LockedNode>,
    pub incidents: Incidents,
    pub peppy: String,
}

impl SignOff {
    /// The sign-off of a run of the bundle `bundle` that binds `binding` on
    /// the stack of `snapshot`, with the versions `versions` and the
    /// incidents policy `incidents`.
    pub fn of_run(
        bundle: Sha256,
        snapshot: &Snapshot,
        binding: &BTreeMap<Name, RobotName>,
        versions: &Lock,
        incidents: &Incidents,
    ) -> SignOff {
        let bound = binding
            .iter()
            .map(|(role, robot)| {
                let core_node = snapshot
                    .fleet
                    .robots
                    .get(robot.as_str())
                    .map(|listed| listed.core_node.clone())
                    .unwrap_or_default();
                let model = snapshot
                    .models
                    .get(robot.as_str())
                    .cloned()
                    .unwrap_or_default();
                let identity = BoundRobot {
                    name: robot.clone(),
                    model,
                    core_node,
                };
                (role.clone(), identity)
            })
            .collect();
        SignOff {
            bundle,
            core_node: snapshot.fleet.core_node.clone(),
            binding: bound,
            nodes: versions.nodes.clone(),
            incidents: incidents.clone(),
            peppy: versions.peppy.clone(),
        }
    }

    /// Each part of the run `run` that differs from this sign-off. A run
    /// matches it when each part is the same, or when only its incidents
    /// policy differs and is stricter: the same monitor, and each severity
    /// of its `continue_on_severity` also in that of the trial.
    pub fn differences(&self, run: &SignOff) -> Vec<String> {
        let mut found = Vec::new();
        if run.bundle != self.bundle {
            found.push(format!(
                "the bundle is `{}`, and the trial signed off `{}`",
                run.bundle, self.bundle
            ));
        }
        if run.core_node != self.core_node {
            found.push(format!(
                "the core node of the stack is `{}`, and the trial ran on `{}`",
                run.core_node, self.core_node
            ));
        }
        found.extend(binding_differences(&self.binding, &run.binding));
        found.extend(node_differences(&self.nodes, &run.nodes));
        if !is_as_strict(&run.incidents, &self.incidents) {
            found.push(
                "the incidents policy is neither that of the trial nor stricter: give the monitor of the trial, and only severities of its continue_on_severity".to_owned(),
            );
        }
        if run.peppy != self.peppy {
            found.push(format!(
                "the peppy version is `{}`, and the trial ran `{}`",
                run.peppy, self.peppy
            ));
        }
        found
    }

    /// The robot names of the binding, role by role in the order of the
    /// role names: the order of the records that `"auto"` reads.
    fn robot_names(&self) -> Vec<&RobotName> {
        self.binding.values().map(|robot| &robot.name).collect()
    }

    /// Whether this sign-off binds each role to the robot of `binding`, and
    /// no other role.
    pub fn has_binding(&self, binding: &BTreeMap<Name, RobotName>) -> bool {
        self.binding.len() == binding.len()
            && binding
                .iter()
                .all(|(role, robot)| self.binding.get(role).is_some_and(|b| b.name == *robot))
    }

    /// Whether this sign-off binds the robot named `robot`.
    pub fn binds(&self, robot: &str) -> bool {
        self.binding
            .values()
            .any(|bound| bound.name.as_str() == robot)
    }
}

/// The differences of the binding of a run from that of a trial, role by
/// role: another robot, or another model or core node of the same robot.
fn binding_differences(
    trial: &BTreeMap<Name, BoundRobot>,
    run: &BTreeMap<Name, BoundRobot>,
) -> Vec<String> {
    let mut found = Vec::new();
    for (role, signed) in trial {
        let Some(bound) = run.get(role) else {
            found.push(format!(
                "the run binds no robot to the role `{role}`, which the trial bound to `{}`",
                signed.name
            ));
            continue;
        };
        if bound.name != signed.name {
            found.push(format!(
                "the role `{role}` binds `{}`, and the trial bound `{}`",
                bound.name, signed.name
            ));
            continue;
        }
        if bound.model != signed.model {
            found.push(format!(
                "`{}` reports the model `{}`, and reported `{}` in the trial",
                bound.name, bound.model, signed.model
            ));
        }
        if bound.core_node != signed.core_node {
            found.push(format!(
                "`{}` runs on the core node `{}`, and ran on `{}` in the trial",
                bound.name, bound.core_node, signed.core_node
            ));
        }
    }
    for role in run.keys().filter(|role| !trial.contains_key(*role)) {
        found.push(format!("the trial bound no robot to the role `{role}`"));
    }
    found
}

/// The differences of the nodes of a run from those of a trial: a node the
/// trial did not run, a node the run does not use, and a node at another
/// sha256 of its manifest or another commit.
fn node_differences(
    trial: &BTreeMap<NameTag, LockedNode>,
    run: &BTreeMap<NameTag, LockedNode>,
) -> Vec<String> {
    let at = |node: &LockedNode| match &node.commit {
        Some(commit) => format!("{} at the commit {commit}", node.sha256),
        None => format!("{} with no commit", node.sha256),
    };
    let mut found = Vec::new();
    for (name, signed) in trial {
        match run.get(name) {
            None => found.push(format!(
                "the run uses no node `{name}`, which the trial used"
            )),
            Some(used) if used != signed || used.commit.is_none() => found.push(format!(
                "the node `{name}` is {}, and the trial used {}",
                at(used),
                at(signed)
            )),
            Some(_) => {}
        }
    }
    for name in run.keys().filter(|name| !trial.contains_key(*name)) {
        found.push(format!(
            "the run uses the node `{name}`, which the trial did not use"
        ));
    }
    found
}

/// Whether the incidents policy `run` is that of the trial, `trial`, or
/// stricter: the same monitor, and each severity that lets the run go on
/// also lets the trial go on.
fn is_as_strict(run: &Incidents, trial: &Incidents) -> bool {
    match (run, trial) {
        (Incidents::None, Incidents::None) => true,
        (
            Incidents::Monitored {
                monitor,
                continue_on_severity,
            },
            Incidents::Monitored {
                monitor: monitor_of_trial,
                continue_on_severity: of_trial,
            },
        ) => monitor == monitor_of_trial && continue_on_severity.is_subset(of_trial),
        _ => false,
    }
}

/// A sign-off that a trial kept.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignOffRecord {
    /// The trial that signed off, `t<N>`.
    pub trial: String,
    pub sign_off: SignOff,
}

/// Why no sign-off record admits an unattended run on a physical robot
/// whose sign-off is `run`, given the records `records` of its bundle; none
/// when one admits it.
pub fn refusal_of_no_record(records: &[SignOffRecord], run: &SignOff) -> Option<String> {
    if records.is_empty() {
        return Some(format!(
            "no trial signed off the bundle `{}`: an unattended run on a physical robot needs a sign-off record of a trial of its bundle on this stack; run a trial of it",
            run.bundle
        ));
    }
    let mut differing = Vec::new();
    for record in records {
        let differences = record.sign_off.differences(run);
        if differences.is_empty() {
            return None;
        }
        differing.push(format!(
            "the record of trial {} differs: {}",
            record.trial,
            differences.join("; ")
        ));
    }
    Some(format!(
        "no sign-off record of the bundle `{}` matches the run, so a new trial of it on this stack is needed: {}",
        run.bundle,
        differing.join(". ")
    ))
}

/// The sign-off records of the director, each in its file
/// `<journal_root>/workflow_director/sign_offs/<trial>.json`.
pub struct SignOffs {
    dir: RecordDir<SignOffRecord>,
    records: Mutex<BTreeMap<String, SignOffRecord>>,
}

impl SignOffs {
    /// Opens the records under `root` and reads each. A record that does
    /// not read refuses the start, naming its file.
    pub fn open(root: &Path) -> std::io::Result<SignOffs> {
        let dir = RecordDir::open(root, "sign_offs", "sign-off record")?;
        let records = dir
            .read_all()?
            .into_iter()
            .map(|record: SignOffRecord| (record.trial.clone(), record))
            .collect();
        Ok(SignOffs {
            dir,
            records: Mutex::new(records),
        })
    }

    /// The records of the bundle `bundle`, ordered by the robot names of
    /// their bindings, role by role with the roles in name order.
    pub fn of_bundle(&self, bundle: &Sha256) -> Vec<SignOffRecord> {
        let records = self.records.lock().expect("no sign-off panics");
        let mut of_bundle: Vec<SignOffRecord> = records
            .values()
            .filter(|record| record.sign_off.bundle == *bundle)
            .cloned()
            .collect();
        of_bundle.sort_by(|a, b| {
            (a.sign_off.robot_names(), &a.trial).cmp(&(b.sign_off.robot_names(), &b.trial))
        });
        of_bundle
    }

    /// Keeps `record` in its file.
    pub fn keep(&self, record: SignOffRecord) -> std::io::Result<()> {
        let mut records = self.records.lock().expect("no sign-off panics");
        self.dir.write(&record.trial, &record)?;
        records.insert(record.trial.clone(), record);
        Ok(())
    }

    /// Removes each record that `removes` holds, and gives how many.
    fn remove_each(&self, removes: impl Fn(&SignOffRecord) -> bool) -> std::io::Result<u32> {
        let mut records = self.records.lock().expect("no sign-off panics");
        let trials: Vec<String> = records
            .values()
            .filter(|record| removes(record))
            .map(|record| record.trial.clone())
            .collect();
        for trial in &trials {
            self.dir.remove(trial)?;
            records.remove(trial);
        }
        Ok(trials.len() as u32)
    }

    /// Removes each record that binds the robot named `robot`.
    pub fn forget_robot(&self, robot: &str) -> std::io::Result<u32> {
        self.remove_each(|record| record.sign_off.binds(robot))
    }

    /// Removes each record of the bundle `bundle` with the binding `binding`.
    pub fn forget_binding(
        &self,
        bundle: &Sha256,
        binding: &BTreeMap<Name, RobotName>,
    ) -> std::io::Result<u32> {
        self.remove_each(|record| {
            record.sign_off.bundle == *bundle && record.sign_off.has_binding(binding)
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::json;
    use workflow_documents::run::{Monitor, Severity};

    use super::*;
    use crate::testing::{name, robot_name};

    pub(crate) const BUNDLE: &str =
        "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    /// The sign-off of a run of the bundle `BUNDLE` that binds each role of
    /// `binding` to its robot, of the model `so101`, on the core `core`.
    pub(crate) fn sign_off(binding: &[(&str, &str)]) -> SignOff {
        SignOff {
            bundle: Sha256::parse(BUNDLE).unwrap(),
            core_node: "core".to_owned(),
            binding: binding
                .iter()
                .map(|(role, robot)| {
                    let bound = BoundRobot {
                        name: robot_name(robot),
                        model: "so101".to_owned(),
                        core_node: "core".to_owned(),
                    };
                    (name(role), bound)
                })
                .collect(),
            nodes: [(
                NameTag::parse("so101_driver:v1").unwrap(),
                LockedNode {
                    sha256: Sha256::parse(SHA_A).unwrap(),
                    commit: Some("c1".to_owned()),
                },
            )]
            .into(),
            incidents: Incidents::None,
            peppy: "0.36.0".to_owned(),
        }
    }

    pub(crate) fn record(trial: &str, binding: &[(&str, &str)]) -> SignOffRecord {
        SignOffRecord {
            trial: trial.to_owned(),
            sign_off: sign_off(binding),
        }
    }

    fn monitored(severities: &[Severity]) -> Incidents {
        Incidents::Monitored {
            monitor: Monitor {
                contract: NameTag::parse("anomaly:v1").unwrap(),
                member: name("look"),
                stack_instance: "monitor_inst".to_owned(),
            },
            continue_on_severity: severities.iter().copied().collect(),
        }
    }

    #[test]
    fn a_run_matches_a_sign_off_when_each_part_is_the_same() {
        let signed = sign_off(&[("arm", "alpha")]);
        assert!(signed.differences(&signed.clone()).is_empty());
    }

    /// A run on another stack, robot, model, node or peppy version needs a
    /// new trial, and each such difference is named.
    #[test]
    fn each_difference_of_a_run_from_its_sign_off_is_named() {
        let signed = sign_off(&[("arm", "alpha")]);
        let mut run = signed.clone();
        run.core_node = "other".to_owned();
        run.binding.get_mut(&name("arm")).unwrap().model = "so100".to_owned();
        run.nodes
            .values_mut()
            .for_each(|node| node.commit = Some("c2".to_owned()));
        run.peppy = "0.37.0".to_owned();
        assert_eq!(
            signed.differences(&run),
            [
                "the core node of the stack is `other`, and the trial ran on `core`",
                "`alpha` reports the model `so100`, and reported `so101` in the trial",
                &format!(
                    "the node `so101_driver:v1` is {SHA_A} at the commit c2, and the trial used {SHA_A} at the commit c1"
                ),
                "the peppy version is `0.37.0`, and the trial ran `0.36.0`",
            ]
        );
        let other_robot = sign_off(&[("arm", "beta")]);
        assert_eq!(
            signed.differences(&other_robot),
            ["the role `arm` binds `beta`, and the trial bound `alpha`"]
        );
    }

    /// A node with no commit matches no sign-off, as two local builds can
    /// hold different code.
    #[test]
    fn a_node_with_no_commit_matches_no_sign_off() {
        let mut signed = sign_off(&[("arm", "alpha")]);
        signed
            .nodes
            .values_mut()
            .for_each(|node| node.commit = None);
        assert_eq!(signed.differences(&signed.clone()).len(), 1);
    }

    /// Only an incidents policy that is stricter matches: the same monitor,
    /// and fewer severities that let the run go on.
    #[test]
    fn a_stricter_incidents_policy_matches_and_any_other_does_not() {
        let mut signed = sign_off(&[("arm", "alpha")]);
        signed.incidents = monitored(&[Severity::Low, Severity::Medium]);
        let with = |incidents: Incidents| SignOff {
            incidents,
            ..signed.clone()
        };
        assert!(
            signed
                .differences(&with(monitored(&[Severity::Low])))
                .is_empty()
        );
        assert!(signed.differences(&with(monitored(&[]))).is_empty());
        for looser in [monitored(&[Severity::High]), Incidents::None] {
            assert_eq!(signed.differences(&with(looser)).len(), 1);
        }
        let mut other_monitor = monitored(&[]);
        if let Incidents::Monitored { monitor, .. } = &mut other_monitor {
            monitor.stack_instance = "other_monitor".to_owned();
        }
        assert_eq!(signed.differences(&with(other_monitor)).len(), 1);
    }

    /// With no record of the bundle, the refusal asks for a trial; with
    /// records, it names how each differs; a record that matches admits.
    #[test]
    fn the_refusal_of_no_record_names_how_each_record_differs() {
        let run = sign_off(&[("arm", "alpha")]);
        let none = refusal_of_no_record(&[], &run).unwrap();
        assert!(
            none.starts_with(&format!("no trial signed off the bundle `{BUNDLE}`")),
            "{none}"
        );
        let other = record("t1", &[("arm", "beta")]);
        let differing = refusal_of_no_record(std::slice::from_ref(&other), &run).unwrap();
        assert!(
            differing.ends_with(
                "the record of trial t1 differs: the role `arm` binds `alpha`, and the trial bound `beta`"
            ),
            "{differing}"
        );
        assert_eq!(
            refusal_of_no_record(&[other, record("t2", &[("arm", "alpha")])], &run),
            None
        );
    }

    /// Writes `record` into the directory of the records under `root`, as a
    /// trial keeps it.
    pub(crate) fn write(root: &Path, record: &SignOffRecord) {
        let dir: RecordDir<SignOffRecord> =
            RecordDir::open(root, "sign_offs", "sign-off record").unwrap();
        dir.write(&record.trial, record).unwrap();
    }

    /// The records stay across an open of the store, come in the order of
    /// the robot names of their bindings, and go at `forget_robot`, each
    /// with no other.
    #[test]
    fn the_records_come_in_the_order_of_their_robots_and_forget_robot_removes_them() {
        let dir = tempfile::tempdir().unwrap();
        let bundle = Sha256::parse(BUNDLE).unwrap();
        for kept in [
            record("t3", &[("left", "bravo"), ("right", "alpha")]),
            record("t1", &[("left", "alpha"), ("right", "charlie")]),
            record("t2", &[("left", "alpha"), ("right", "bravo")]),
        ] {
            write(dir.path(), &kept);
        }
        let store = SignOffs::open(dir.path()).unwrap();
        let trials = |store: &SignOffs| -> Vec<String> {
            store
                .of_bundle(&bundle)
                .into_iter()
                .map(|record| record.trial)
                .collect()
        };
        assert_eq!(trials(&store), ["t2", "t1", "t3"]);
        assert_eq!(store.forget_robot("charlie").unwrap(), 1);
        assert_eq!(store.forget_robot("delta").unwrap(), 0);
        assert_eq!(trials(&SignOffs::open(dir.path()).unwrap()), ["t2", "t3"]);
    }

    /// A kept record stays across an open of the store, and a trial of the
    /// same bundle and binding removes it and no other.
    #[test]
    fn a_kept_record_stays_until_a_trial_of_its_bundle_and_binding() {
        let dir = tempfile::tempdir().unwrap();
        let store = SignOffs::open(dir.path()).unwrap();
        let bundle = Sha256::parse(BUNDLE).unwrap();
        store
            .keep(record("t1", &[("left", "alpha"), ("right", "bravo")]))
            .unwrap();
        store
            .keep(record("t2", &[("left", "bravo"), ("right", "alpha")]))
            .unwrap();
        let reopened = SignOffs::open(dir.path()).unwrap();
        assert_eq!(reopened.of_bundle(&bundle).len(), 2);
        let binding = BTreeMap::from([
            (name("left"), robot_name("alpha")),
            (name("right"), robot_name("bravo")),
        ]);
        let mut other_bundle = binding.clone();
        other_bundle.insert(name("courier"), robot_name("charlie"));
        assert_eq!(reopened.forget_binding(&bundle, &other_bundle).unwrap(), 0);
        assert_eq!(reopened.forget_binding(&bundle, &binding).unwrap(), 1);
        let left: Vec<String> = SignOffs::open(dir.path())
            .unwrap()
            .of_bundle(&bundle)
            .into_iter()
            .map(|record| record.trial)
            .collect();
        assert_eq!(left, ["t2"]);
    }

    #[test]
    fn a_record_that_does_not_read_names_its_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = SignOffs::open(dir.path()).unwrap();
        std::fs::write(
            store.dir.dir().join("t9.json"),
            json!({ "trial": "t9" }).to_string(),
        )
        .unwrap();
        let error = SignOffs::open(dir.path())
            .err()
            .expect("refused")
            .to_string();
        assert!(error.contains("t9.json"), "{error}");
        assert!(error.contains("move this sign-off record out"), "{error}");
    }
}
