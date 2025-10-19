use anyhow::Result;
use tantalisp::{rep, rep_with_env, Environment};
use std::{env, fs, io::{self, Read, Write, BufRead}};
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
            // Check if stdin is a terminal (interactive) or a pipe
            if atty::is(atty::Stream::Stdin) {
                // Interactive REPL mode
                repl()
            } else {
                // Piped input mode
                let mut sources = String::new();
                io::stdin().read_to_string(&mut sources)?;
                rep(&sources)
            }
        }
        _ => bail!("Usage: {} [filename]\nIf no filename provided, starts REPL or reads from stdin", args[0])
    }
}

fn repl() -> Result<()> {
    println!("Tantalisp REPL");
    println!("Type expressions to evaluate. Press Ctrl+D (Unix) or Ctrl+Z (Windows) to exit.");
    println!("Special commands: :quit or :q to exit");
    println!();

    // Create persistent environment for the REPL session
    let mut env = Environment::global();

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
                match rep_with_env(&mut env, input) {
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