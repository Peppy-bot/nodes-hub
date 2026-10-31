//! A directory of JSON records of the director, one file `<id>.json` per
//! record: the sign-off records and the records of the trials. Each record
//! is written whole, to a file of its own that then takes its name, so a
//! read never sees half of one; a read leaves out each other file, such as
//! the `<id>.json.part` of a write that did not end.

use std::fs;
use std::io;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

/// The records of one kind, each in its file
/// `<root>/workflow_director/<name>/<id>.json`.
pub struct RecordDir<T> {
    dir: PathBuf,
    /// What one record is, as the refusal of a record that does not read
    /// names it.
    what: &'static str,
    records: PhantomData<fn() -> T>,
}

impl<T: Serialize + DeserializeOwned> RecordDir<T> {
    /// The directory `name` under `root`, made when missing, of records
    /// that a refusal names `what`.
    pub fn open(root: &Path, name: &str, what: &'static str) -> io::Result<RecordDir<T>> {
        let dir = root.join("workflow_director").join(name);
        fs::create_dir_all(&dir)?;
        Ok(RecordDir {
            dir,
            what,
            records: PhantomData,
        })
    }

    /// The directory of the records.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The id of each record of the directory, in no set order.
    pub fn ids(&self) -> io::Result<Vec<String>> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.strip_suffix(".json") {
                ids.push(id.to_owned());
            }
        }
        Ok(ids)
    }

    /// Each record of the directory, in no set order. A record that does
    /// not read refuses the read, naming its file.
    pub fn read_all(&self) -> io::Result<Vec<T>> {
        let mut records = Vec::new();
        for id in self.ids()? {
            let path = self.path(&id);
            let text = fs::read_to_string(&path)?;
            let record = serde_json::from_str(&text).map_err(|error| {
                io::Error::other(format!(
                    "{}: {error}; move this {} out of its directory, and the director starts without it",
                    path.display(),
                    self.what
                ))
            })?;
            records.push(record);
        }
        Ok(records)
    }

    /// Writes `record` as the record `id`.
    pub fn write(&self, id: &str, record: &T) -> io::Result<()> {
        let text = serde_json::to_string_pretty(record).map_err(io::Error::other)?;
        let path = self.path(id);
        let written = path.with_extension("json.part");
        fs::write(&written, text)?;
        fs::rename(&written, &path)
    }

    /// Removes the record `id`.
    pub fn remove(&self, id: &str) -> io::Result<()> {
        fs::remove_file(self.path(id))
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Note {
        text: String,
    }

    /// A record written reads back, a file of a write that did not end is
    /// left out, a removed record is gone, and a record that does not read
    /// names its file and what it is.
    #[test]
    fn records_read_back_whole_and_a_bad_one_names_its_file() {
        let root = tempfile::tempdir().unwrap();
        let notes: RecordDir<Note> = RecordDir::open(root.path(), "notes", "note").unwrap();
        let note = Note {
            text: "kept".to_owned(),
        };
        notes.write("n1", &note).unwrap();
        fs::write(notes.dir().join("n2.json.part"), "{ half").unwrap();
        assert_eq!(notes.ids().unwrap(), ["n1"]);
        assert_eq!(notes.read_all().unwrap(), [note]);
        notes.remove("n1").unwrap();
        assert!(notes.read_all().unwrap().is_empty());
        fs::write(notes.dir().join("n3.json"), "{ half").unwrap();
        let error = notes.read_all().unwrap_err().to_string();
        assert!(error.contains("n3.json"), "{error}");
        assert!(
            error.contains("move this note out of its directory"),
            "{error}"
        );
    }
}
