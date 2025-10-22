use anyhow::Result;
use clap::Parser;
use tantalisp::{Tantalisp};
use std::{env, fs, io::{self, Read, Write, BufRead}};
use anyhow::bail;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = None)]
    file: Option<String>,
    
    #[arg(short, long, default_value_t = false)]
    debug: bool,
    
    #[arg(short, long, default_value_t = false)]
    interpreter: bool


}

fn main() -> Result<()> {
    // let args: Vec<String> = env::args().collect();
    let args = Args::parse();

    let mut lisp = Tantalisp::new(args.debug, args.interpreter);
 
    match args.file {
        Some(file) => {
            // Read from file
            let sources = fs::read_to_string(file)?;
 
            lisp.rep(&sources)
        },
        None => {
            // Check if stdin is a terminal (interactive) or a pipe
            if atty::is(atty::Stream::Stdin) {
                // Interactive REPL mode
                lisp.repl()
            } else {
                // Piped input mode
                let mut sources = String::new();
                io::stdin().read_to_string(&mut sources)?;
                lisp.rep(&sources)
            }
        }
    }
}

