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
                repl(&mut lisp)
            } else {
                // Piped input mode
                let mut sources = String::new();
                io::stdin().read_to_string(&mut sources)?;
                lisp.rep(&sources)
            }
        }
    }
}

fn repl(lisp: &mut Tantalisp) -> Result<()> {
    println!("Tantalisp REPL");
    println!("Type expressions to evaluate. Press Ctrl+D (Unix) or Ctrl+Z (Windows) to exit.");
    println!("Special commands: :quit or :q to exit");
    println!();

    
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        print!("tantalisp> ");
        io::stdout().flush()?;

        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // EOF (Ctrl+D)
                println!("\nBye!");
                break;
            }
            Ok(_) => {
                let input = line.trim();

                // Skip empty lines
                if input.is_empty() {
                    continue;
                }

                // Handle special REPL commands
                if input == ":quit" || input == ":q" {
                    println!("Bye!");
                    break;
                }

                // Evaluate the expression with persistent environment
                match lisp.rep_with_env(input) {
                    Ok(_) => {},
                    Err(e) => eprintln!("Error: {}", e),
                }
                println!(); // Blank line between evaluations
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }

    Ok(())
}