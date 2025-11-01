use anyhow::Result;
use clap::Parser;
use tantalisp::{Tantalisp, set_gc_debug_mode, start_gc_monitor};
use std::{env, fs, io::{self, Read}};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = None)]
    file: Option<String>,

    #[arg(short, long, default_value_t = false)]
    debug: bool,

    #[arg(short, long, default_value_t = false)]
    interpreter: bool,

    #[arg(long, default_value_t = false)]
    print_ir: bool,
}

fn main() -> Result<()> {
    // Enable GC debug mode if GC_DEBUG env var is set
    if env::var("GC_DEBUG").is_ok() {
        set_gc_debug_mode(true);

        // Start background thread to log memory stats every 100ms
        start_gc_monitor(100, "gc_debug.log");
        println!("GC monitoring enabled. Writing stats to gc_debug.log every 100ms.");
    }

    // let args: Vec<String> = env::args().collect();
    let args = Args::parse();

    let mut lisp = Tantalisp::new(args.debug, args.interpreter, args.print_ir);
 
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

