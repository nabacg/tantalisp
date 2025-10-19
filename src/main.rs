use anyhow::Result;
use tantalisp::rep;
use std::{env, fs, io::{self, Read}};
use anyhow::bail;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    match &args[..] {
        [_cmd, input_file] => {
            // Read from file
            let sources = fs::read_to_string(input_file)?;
            rep(&sources)
        }
        [_cmd] => {
            // Read from stdin
            let mut sources = String::new();
            io::stdin().read_to_string(&mut sources)?;
            rep(&sources)
        }
        _ => bail!("Usage: {} [filename]\nIf no filename provided, reads from stdin", args[0])
    }
}