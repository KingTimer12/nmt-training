//! Self-contained release build (`--features bundle`).
//!
//! The corpus and the vocabulary are embedded in the executable. On startup we `chdir` to
//! the executable's directory and write them there if missing, so every relative path
//! (`data/`, `assets/`, `artifacts/`) resolves next to the binary regardless of where it
//! was launched from.

use std::io;
use std::path::Path;

use crate::CORPUS;
use crate::tokenizer::DEFAULT_VOCAB_PATH;

const FILES: &[(&str, &[u8])] = &[
    (
        CORPUS,
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/canonical/en-pt_BR/tatoeba.jsonl"
        )),
    ),
    (
        DEFAULT_VOCAB_PATH,
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/rwkv_vocab_v20230424.txt"
        )),
    ),
];

pub fn prepare() -> io::Result<()> {
    let exe = std::env::current_exe()?;
    let dir = exe.parent().expect("executable has a parent directory");
    std::env::set_current_dir(dir)?;
    println!("Working directory: {}", dir.display());

    for (rel, bytes) in FILES {
        let path = Path::new(rel);
        let up_to_date = std::fs::metadata(path).is_ok_and(|m| m.len() == bytes.len() as u64);
        if up_to_date {
            continue;
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
        println!("Extracted {rel}");
    }
    Ok(())
}
