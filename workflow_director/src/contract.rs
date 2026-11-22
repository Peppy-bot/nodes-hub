//! A pinned contract document as the checker and the engine read it: each
//! member with its kind, its marks and the formats of its sides, each field
//! with its type, its enum, its marks and whether a message can leave it
//! out. It reads the formats in the serde form of the codec's parser.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use peppy_codec::contract::{InterfaceKind, MessageFormat, PeppyContract, PeppyContractParser};
use peppy_codec::value;
use serde_json::Value as Json;
use workflow_documents::expression::{FieldType, Type};
use workflow_documents::marks::ACTIVITY;

use crate::engine::plan::MemberKind;

/// The type of a field of a message format.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldKind {
    Bool,
    /// An integer; `wide` for the 64-bit ones, which a message carries as a
    /// decimal string.
    Int {
        wide: bool,
    },
    Double,
    String,
    Enum(BTreeSet<String>),
    Bytes,
    Time,
    List(Box<Field>),
    Object(Format),
}

/// A field of a message format.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    pub kind: FieldKind,
    pub optional: bool,
    pub marks: BTreeSet<String>,
}

/// A message format: each field by name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Format(pub BTreeMap<String, Field>);

/// A service or an action of a contract.
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub kind: MemberKind,
    /// The request of a service, the goal of an action.
    pub input: Format,
    /// The response of a service, the result of an action.
    pub output: Format,
    pub marks: BTreeSet<String>,
    /// The string field at the top of the feedback message of an action
    /// with the mark `activity`, the sentence of what the robot does now;
    /// `None` for a service, and for an action with no such field.
    pub activity: Option<String>,
}

/// A pinned contract document.
#[derive(Clone, Debug)]
pub struct ContractShape {
    pub name: String,
    pub tag: String,
    pub sha256: String,
    /// The text of the document.
    pub document: String,
    pub parsed: Arc<PeppyContract>,
    pub members: BTreeMap<String, Member>,
    pub topics: BTreeMap<String, Format>,
}

impl ContractShape {
    /// Parses the document of the contract pinned as `sha256`.
    pub fn parse(document: &str, sha256: &str) -> Result<ContractShape, String> {
        let parsed =
            PeppyContractParser::from_content(document).map_err(|error| error.to_string())?;
        let format = |format: Option<&MessageFormat>| -> Result<Format, String> {
            match format {
                Some(format) => {
                    format_of(&serde_json::to_value(format).map_err(|e| e.to_string())?)
                }
                None => Ok(Format::default()),
            }
        };
        let mut members = BTreeMap::new();
        for service in &parsed.interfaces.services {
            members.insert(
                service.name.clone(),
                Member {
                    kind: MemberKind::Service,
                    input: format(service.request_message_format.as_ref())?,
                    output: format(service.response_message_format.as_ref())?,
                    marks: marks_of(&parsed, InterfaceKind::Service, &service.name),
                    activity: None,
                },
            );
        }
        for action in &parsed.interfaces.actions {
            let input = action
                .goal_service
                .as_ref()
                .and_then(|goal| goal.request_message_format.as_ref());
            let output = action
                .result_service
                .as_ref()
                .and_then(|result| result.response_message_format.as_ref());
            let feedback = action
                .feedback_topic
                .as_ref()
                .map(|feedback| &feedback.message_format);
            members.insert(
                action.name.clone(),
                Member {
                    kind: MemberKind::Action,
                    input: format(input)?,
                    output: format(output)?,
                    marks: marks_of(&parsed, InterfaceKind::Action, &action.name),
                    activity: activity_of(&format(feedback)?),
                },
            );
        }
        let mut topics = BTreeMap::new();
        for topic in &parsed.interfaces.topics {
            topics.insert(topic.name.clone(), format(topic.message_format.as_ref())?);
        }
        Ok(ContractShape {
            name: parsed.manifest.name.to_string(),
            tag: parsed.manifest.tag.to_string(),
            sha256: sha256.to_owned(),
            document: document.to_owned(),
            parsed: Arc::new(parsed),
            members,
            topics,
        })
    }

    pub fn name_tag(&self) -> String {
        format!("{}:{}", self.name, self.tag)
    }
}

fn marks_of(contract: &PeppyContract, kind: InterfaceKind, name: &str) -> BTreeSet<String> {
    let marks = contract.marks.of(kind, name);
    serde_json::to_value(&marks)
        .ok()
        .and_then(|value| serde_json::from_value::<Vec<String>>(value).ok())
        .unwrap_or_default()
        .into_iter()
        .collect()
}

/// The string field at the top of `feedback` with the mark `activity`.
fn activity_of(feedback: &Format) -> Option<String> {
    feedback
        .0
        .iter()
        .find(|(_, field)| field.kind == FieldKind::String && field.marks.contains(ACTIVITY))
        .map(|(name, _)| name.clone())
}

/// A format from its serde form: an object from field names to fields.
fn format_of(value: &Json) -> Result<Format, String> {
    let Json::Object(fields) = value else {
        return Err(format!("a message format is an object, not {value}"));
    };
    fields
        .iter()
        .map(|(name, field)| field_of(field).map(|field| (name.clone(), field)))
        .collect::<Result<_, _>>()
        .map(Format)
}

/// A field from its serde form: a type name, or an object with a `$type`
/// and its modifiers.
fn field_of(value: &Json) -> Result<Field, String> {
    let token_kind = |token: &str| -> Result<FieldKind, String> {
        Ok(match token {
            "bool" => FieldKind::Bool,
            "u8" | "u16" | "u32" | "i8" | "i16" | "i32" => FieldKind::Int { wide: false },
            "u64" | "i64" => FieldKind::Int { wide: true },
            "f32" | "f64" => FieldKind::Double,
            "string" => FieldKind::String,
            "bytes" => FieldKind::Bytes,
            "time" => FieldKind::Time,
            other => return Err(format!("`{other}` is no type of a field")),
        })
    };
    let plain = |kind| Field {
        kind,
        optional: false,
        marks: BTreeSet::new(),
    };
    let Json::Object(object) = value else {
        let token = value
            .as_str()
            .ok_or_else(|| format!("{value} is no field"))?;
        return Ok(plain(token_kind(token)?));
    };
    let optional = object
        .get("$optional")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let marks: BTreeSet<String> = object
        .get("$marks")
        .and_then(Json::as_array)
        .map(|marks| {
            marks
                .iter()
                .filter_map(Json::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let kind = match object.get("$type").and_then(Json::as_str) {
        Some("array") => FieldKind::List(Box::new(field_of(
            object.get("$items").ok_or("an array has `$items`")?,
        )?)),
        Some("object") | None => {
            let fields: serde_json::Map<String, Json> = object
                .iter()
                .filter(|(key, _)| !key.starts_with('$'))
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            FieldKind::Object(format_of(&Json::Object(fields))?)
        }
        Some(token) => match object.get("$enum").and_then(Json::as_array) {
            Some(values) => FieldKind::Enum(
                values
                    .iter()
                    .filter_map(Json::as_str)
                    .map(str::to_owned)
                    .collect(),
            ),
            None => token_kind(token)?,
        },
    };
    Ok(Field {
        kind,
        optional,
        marks,
    })
}

impl Field {
    /// The type an expression reads this field as.
    pub fn ty(&self) -> Type {
        match &self.kind {
            FieldKind::Bool => Type::Bool,
            FieldKind::Int { .. } => Type::Int,
            FieldKind::Double => Type::Double,
            FieldKind::String => Type::String,
            FieldKind::Enum(values) => Type::Enum(values.clone()),
            FieldKind::Bytes => Type::Bytes,
            FieldKind::Time => Type::Time,
            FieldKind::List(item) => Type::List(Box::new(item.ty())),
            FieldKind::Object(format) => format.ty(),
        }
    }

    /// `literal` in the canonical form of this field, as the codec takes
    /// it: a 64-bit integer as its decimal text; or why it does not fit.
    pub fn canonical(&self, literal: &Json) -> Result<Json, String> {
        let mismatch = || format!("{literal} is no value of a {} field", self.ty());
        match (&self.kind, literal) {
            (FieldKind::Bool, Json::Bool(_)) => Ok(literal.clone()),
            (FieldKind::Int { wide }, Json::Number(n))
                if n.as_i64().is_some() || n.as_u64().is_some() =>
            {
                Ok(match wide {
                    true => Json::String(n.to_string()),
                    false => literal.clone(),
                })
            }
            (FieldKind::Int { wide: true }, Json::String(text)) if text.parse::<i128>().is_ok() => {
                Ok(literal.clone())
            }
            (FieldKind::Double, Json::Number(_)) => Ok(literal.clone()),
            (FieldKind::String | FieldKind::Bytes, Json::String(_)) => Ok(literal.clone()),
            (FieldKind::Time, Json::String(_)) => value::value_time(literal, "time")
                .map(|_| literal.clone())
                .map_err(|_| {
                    format!(
                        "{literal} is no time: give RFC 3339 text in UTC, such as \"2026-11-18T12:00:00Z\", or leave the timestamp of a command out for the latest time"
                    )
                }),
            (FieldKind::Enum(values), Json::String(value)) if values.contains(value) => {
                Ok(literal.clone())
            }
            (FieldKind::Enum(values), Json::String(value)) => Err(format!(
                "'{value}' is not a value of {}",
                Type::Enum(values.clone())
            )),
            (FieldKind::List(item), Json::Array(items)) => items
                .iter()
                .map(|item_value| item.canonical(item_value))
                .collect::<Result<Vec<_>, _>>()
                .map(Json::Array),
            (FieldKind::Object(format), Json::Object(_)) => format.canonical(literal),
            _ => Err(mismatch()),
        }
    }
}

impl Format {
    pub fn ty(&self) -> Type {
        Type::Object(
            self.0
                .iter()
                .map(|(name, field)| {
                    (
                        name.clone(),
                        FieldType {
                            ty: field.ty(),
                            optional: field.optional,
                        },
                    )
                })
                .collect(),
        )
    }

    /// An object literal in the canonical form of this format.
    pub fn canonical(&self, literal: &Json) -> Result<Json, String> {
        let Json::Object(object) = literal else {
            return Err(format!("{literal} is no object"));
        };
        object
            .iter()
            .map(|(key, value)| {
                let field = self
                    .0
                    .get(key)
                    .ok_or_else(|| format!("the format has no field `{key}`"))?;
                field.canonical(value).map(|value| (key.clone(), value))
            })
            .collect::<Result<serde_json::Map<_, _>, _>>()
            .map(Json::Object)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A contract of an arm with each form of field the checker reads.
    /// The arm of the tests of the director, which its harness tests serve from the fleet too.
    pub(crate) const TEST_ARM: &str = include_str!("../tests/fixtures/test_arm.json5");

    #[test]
    fn a_contract_reads_into_its_members_formats_and_marks() {
        let shape = ContractShape::parse(TEST_ARM, &"a".repeat(64)).unwrap();
        assert_eq!(shape.name_tag(), "test_arm:v1");
        let grab = &shape.members["grab"];
        assert_eq!(grab.kind, MemberKind::Action);
        assert!(grab.marks.contains("moves_robot") && grab.marks.contains("reaches_for_item"));
        assert_eq!(grab.activity.as_deref(), Some("activity"));
        assert_eq!(shape.members["home"].activity, None);
        assert_eq!(shape.members["find"].activity, None);
        assert!(grab.input.0["item_id"].marks.contains("item_id"));
        let find = &shape.members["find"].output;
        assert_eq!(find.0["count"].kind, FieldKind::Int { wide: true });
        assert!(matches!(find.0["kind"].kind, FieldKind::Enum(_)));
        assert!(find.0["note"].optional);
        assert_eq!(shape.topics["state"].0["busy"].kind, FieldKind::Bool);
    }

    #[test]
    fn a_literal_takes_the_canonical_form_of_its_field() {
        let shape = ContractShape::parse(TEST_ARM, &"a".repeat(64)).unwrap();
        let find = &shape.members["find"].output;
        assert_eq!(
            find.0["count"].canonical(&serde_json::json!(7)),
            Ok(serde_json::json!("7"))
        );
        assert!(
            find.0["kind"]
                .canonical(&serde_json::json!("plate"))
                .unwrap_err()
                .contains("'plate'")
        );
        let grab = &shape.members["grab"].input;
        assert_eq!(
            grab.0["speed"].canonical(&serde_json::json!(3)),
            Ok(serde_json::json!(3))
        );
        assert!(
            grab.0["speed"]
                .canonical(&serde_json::json!("fast"))
                .is_err()
        );
    }

    /// A time is RFC 3339 text in UTC, as the codec writes one.
    #[test]
    fn a_time_literal_is_rfc_3339_text() {
        let shape = ContractShape::parse(TEST_ARM, &"a".repeat(64)).unwrap();
        let FieldKind::Object(target) = &shape.members["grab"].input.0["target"].kind else {
            panic!("the target of a grab is a record");
        };
        let timestamp = &target.0["timestamp"];
        for time in ["2026-11-18T12:00:00Z", "2026-11-18T12:00:00.250000000Z"] {
            let literal = serde_json::json!(time);
            assert_eq!(timestamp.canonical(&literal), Ok(literal));
        }
        for not_a_time in ["0", "2026-11-18", "2026-11-18T12:00:00+01:00"] {
            let refused = timestamp
                .canonical(&serde_json::json!(not_a_time))
                .unwrap_err();
            assert_eq!(
                refused,
                format!(
                    "\"{not_a_time}\" is no time: give RFC 3339 text in UTC, such as \"2026-11-18T12:00:00Z\", or leave the timestamp of a command out for the latest time"
                )
            );
        }
    }
}
