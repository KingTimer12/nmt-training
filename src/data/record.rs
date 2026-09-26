//! Canonical JSONL records (`data/canonical/<pair>/<corpus>.jsonl`).

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::tokenizer::Lang;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairRecord {
    pub id: String,
    pub src_lang: Lang,
    pub tgt_lang: Lang,
    pub src: String,
    pub tgt: String,
    pub corpus: String,
    pub doc: Option<String>,
    pub seg: Option<u64>,
}

pub fn load_jsonl(path: impl AsRef<Path>) -> io::Result<Vec<PairRecord>> {
    let path = path.as_ref();
    let reader = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let rec: PairRecord = serde_json::from_str(&line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {e}", path.display(), i + 1),
            )
        })?;
        out.push(rec);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_line() {
        let line = r#"{"id": "tatoeba:1-2", "src_lang": "en", "tgt_lang": "pt_BR", "src": "Go.", "tgt": "Vai.", "corpus": "tatoeba", "doc": null, "seg": null}"#;
        let r: PairRecord = serde_json::from_str(line).unwrap();
        assert_eq!(r.src_lang, Lang::En);
        assert_eq!(r.tgt_lang, Lang::PtBr);
        assert_eq!(r.tgt, "Vai.");
        assert!(r.doc.is_none() && r.seg.is_none());
    }
}
