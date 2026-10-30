//! The configuration of the director, parsed from its parameters once at its
//! start.

use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use workflow_documents::{Bound, NameTag};

/// The contracts that runs may command and observe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllowedContracts(BTreeSet<NameTag>);

impl AllowedContracts {
    /// Parses `name:tag` items separated by commas, with optional spaces. A
    /// blank text allows no contract.
    pub fn parse(text: &str) -> Result<AllowedContracts, String> {
        if text.trim().is_empty() {
            return Ok(AllowedContracts(BTreeSet::new()));
        }
        let mut allowed = BTreeSet::new();
        for item in text.split(',').map(str::trim) {
            if item.is_empty() {
                return Err(format!(
                    "allowed_contracts `{text}` has an empty item; separate `name:tag` items with one comma"
                ));
            }
            let contract =
                NameTag::parse(item).map_err(|error| format!("allowed_contracts: {error}"))?;
            if !allowed.insert(contract) {
                return Err(format!("allowed_contracts names `{item}` twice"));
            }
        }
        Ok(AllowedContracts(allowed))
    }

    pub fn allows(&self, contract: &NameTag) -> bool {
        self.0.contains(contract)
    }

    pub fn allows_name_tag(&self, name: &str, tag: &str) -> bool {
        self.0
            .iter()
            .any(|c| c.name.as_str() == name && c.tag() == tag)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for AllowedContracts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let items: Vec<String> = self.0.iter().map(NameTag::to_string).collect();
        f.write_str(&items.join(", "))
    }
}

/// The configuration of the director.
#[derive(Clone, Debug)]
pub struct Config {
    pub allowed: AllowedContracts,
    pub poll: Bound,
    pub cancel_timeout: Duration,
    pub journal_root: PathBuf,
    pub journal_max_runs: u32,
    pub journal_max_bytes: u64,
}

impl Config {
    pub fn new(
        allowed_contracts: &str,
        poll_s: f64,
        cancel_timeout_s: f64,
        journal_root: &str,
        journal_max_runs: u32,
        journal_max_mib: u32,
    ) -> Result<Config, String> {
        let allowed = AllowedContracts::parse(allowed_contracts)?;
        let poll = Bound::from_seconds(poll_s).map_err(|reason| format!("poll_s: {reason}"))?;
        let cancel_timeout = Bound::from_seconds(cancel_timeout_s)
            .map_err(|reason| format!("cancel_timeout_s: {reason}"))?;
        let journal_root = PathBuf::from(journal_root);
        if !journal_root.is_absolute() {
            return Err(format!(
                "journal_root `{}` is no absolute path",
                journal_root.display()
            ));
        }
        if journal_max_runs == 0 || journal_max_mib == 0 {
            return Err("journal_max_runs and journal_max_mib keep at least one journal: give each 1 or more".to_owned());
        }
        Ok(Config {
            allowed,
            poll,
            cancel_timeout: Duration::from_nanos(cancel_timeout.nanos()),
            journal_root,
            journal_max_runs,
            journal_max_bytes: u64::from(journal_max_mib) * 1024 * 1024,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_contracts_parse_and_print_back_sorted() {
        let allowed =
            AllowedContracts::parse(" postures:v1,limb_motion:v1 , item_perception:v1").unwrap();
        assert_eq!(
            allowed.to_string(),
            "item_perception:v1, limb_motion:v1, postures:v1"
        );
        assert_eq!(
            AllowedContracts::parse(&allowed.to_string()).unwrap(),
            allowed
        );
        assert!(allowed.allows_name_tag("postures", "v1"));
        assert!(!allowed.allows_name_tag("postures", "v2"));
        assert!(AllowedContracts::parse("").unwrap().is_empty());
    }

    #[test]
    fn a_malformed_or_repeated_item_is_refused_by_name() {
        assert!(
            AllowedContracts::parse("postures:v1,,limb_motion:v1")
                .unwrap_err()
                .contains("empty item")
        );
        assert!(
            AllowedContracts::parse("postures")
                .unwrap_err()
                .contains("`postures`")
        );
        assert!(
            AllowedContracts::parse("postures:v1, postures:v1")
                .unwrap_err()
                .contains("twice")
        );
    }

    #[test]
    fn a_config_refuses_each_bad_parameter_by_name() {
        assert!(
            Config::new("", 0.0, 10.0, "/var/tmp/j", 10, 10)
                .unwrap_err()
                .starts_with("poll_s")
        );
        assert!(
            Config::new("", 0.5, 10.0, "journals", 10, 10)
                .unwrap_err()
                .contains("absolute")
        );
        assert!(
            Config::new("", 0.5, 10.0, "/var/tmp/j", 0, 10)
                .unwrap_err()
                .contains("at least one")
        );
        assert!(Config::new("postures:v1", 0.5, 10.0, "/var/tmp/j", 10, 10).is_ok());
    }

    /// Any set of contracts survives its text, with any spaces around each of its items.
    #[hegel::test]
    fn the_allowed_contracts_survive_their_text_with_any_spaces(tc: hegel::TestCase) {
        use hegel::generators as gs;
        let names = [
            "postures",
            "limb_state",
            "item_perception",
            "robot_identity",
            "test_arm",
        ];
        let drawn: BTreeSet<(usize, u8)> = tc.draw(
            gs::btree_sets(gs::tuples2(
                gs::integers::<usize>().max_value(names.len() - 1),
                gs::integers::<u8>().min_value(1).max_value(3),
            ))
            .max_size(5),
        );
        let items: Vec<String> = drawn
            .iter()
            .map(|(name, tag)| format!("{}:v{tag}", names[*name]))
            .collect();
        let spaced: Vec<String> = items
            .iter()
            .map(|item| {
                let before = " ".repeat(tc.draw(gs::integers::<usize>().max_value(2)));
                let after = " ".repeat(tc.draw(gs::integers::<usize>().max_value(2)));
                format!("{before}{item}{after}")
            })
            .collect();
        let allowed = AllowedContracts::parse(&spaced.join(",")).unwrap();
        assert_eq!(
            AllowedContracts::parse(&allowed.to_string()).unwrap(),
            allowed
        );
        assert_eq!(allowed.0.len(), items.len());
    }
}
