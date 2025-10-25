mod lexer;
mod parser;
mod evaluator;
mod codegen;

use std::io::{self, BufRead, Write};

use anyhow::{anyhow, Result};
use inkwell::context::Context;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use lexer::tokenize;
use parser::parse;
use evaluator::{eval, eval_with_env};
use codegen::CodeGen;

// Re-export Environment for REPL
pub use evaluator::Environment;

pub struct Tantalisp<'l> {
    debug_mode: bool,
    interpreted_mode: bool,
    interpreter_env: Option<Environment<'l>>,
    codegen: Option<CodeGen<'l>>
}

impl<'l> Tantalisp<'l> {
    pub fn new(debug_mode: bool, interpreted_mode: bool) -> Self {
        if interpreted_mode {

            Self { 
                debug_mode, 
                interpreted_mode,
                interpreter_env: Some(Environment::global()), 
                codegen: None,
            }
        } else {
            // allocate Ctx and don't store it, it will outlive codegen
            // TODO - find a more correct solution, but it's late 
            let ctx = Box::leak(Box::new(Context::create()));

            Self { 
                debug_mode, 
                interpreted_mode,
                interpreter_env: None, 
                codegen: Some(CodeGen::new(ctx, debug_mode))
            }
        }
    }

   pub fn repl(&mut self) -> Result<()> {
        let mode = if self.interpreted_mode { "interpreter" } else { "JIT compiler" };
        let debug = if self.debug_mode { "ON" } else { "OFF" };

        println!("Tantalisp REPL");
        println!("Mode: {} | Debug: {}", mode, debug);

        // Load prelude (standard library)
        self.load_prelude()?;

        println!("Type expressions to evaluate.");
        println!("Exit: Ctrl+C, Ctrl+D, or type :quit");
        println!();

        // Create rustyline editor with history support
        let mut rl = DefaultEditor::new()?;

        // Optionally load history from file
        let history_file = ".tantalisp_history";
        let _ = rl.load_history(history_file); // Ignore error if file doesn't exist

        loop {
            match rl.readline("tantalisp> ") {
                Ok(line) => {
                    let input = line.trim();

                    // Skip empty lines
                    if input.is_empty() {
                        continue;
                    }

                    // Add to history
                    let _ = rl.add_history_entry(&line);

                    // Handle special REPL commands
                    if input == ":quit" || input == ":q" {
                        println!("Bye!");
                        break;
                    }

                    // Evaluate the expression with persistent environment
                    match self.rep_with_env(input) {
                        Ok(_) => {},
                        Err(e) => eprintln!("Error: {}", e),
                    }
                    println!(); // Blank line between evaluations
                }
                Err(ReadlineError::Interrupted) => {
                    // Ctrl+C - Exit gracefully
                    println!("\nBye!");
                    break;
                }
                Err(ReadlineError::Eof) => {
                    // Ctrl+D
                    println!("\nBye!");
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {:?}", err);
                    break;
                }
            }
        }

        // Save history to file
        let _ = rl.save_history(history_file);

        Ok(())
    }

    pub fn rep(&mut self, input: &str) -> Result<()> {
        let tokens = tokenize(input)?;
        if self.debug_mode {
            println!("Tokens:\n{}", tokens.iter().map(|t| format!("{:?}", t)).collect::<Vec<_>>().join("\n"));
        }
        let ast = parse(&tokens[..])?;
        if self.debug_mode {
            println!("AST:\n{}", ast.iter().map(|t| format!("{:?}", t)).collect::<Vec<_>>().join("\n"));
        }

        if self.interpreted_mode {
            let result = eval(ast)?;
            println!("Result:\n{}", result);
        } else {
            let codegen = self.codegen.as_mut().ok_or(anyhow!("compiled_mode but empty codegen"))?;

            let result = codegen.compile_and_run(&ast)?;
            println!("Result:\n{}", result);

        }

        Ok(())
    }

    // REPL-friendly version that takes a mutable environment
    pub fn rep_with_env(&mut self, input: &str) -> Result<()> {

        let tokens = tokenize(input)?;
        let ast = parse(&tokens[..])?;
        if self.interpreted_mode {
            let env = self.interpreter_env.as_mut().ok_or(anyhow!("interpreter_mode: ON but missing interpreter_env!"))?;
            let result = eval_with_env(env, ast)?;
            println!("{}", result);

        } else {
            let codegen = self.codegen.as_mut().ok_or(anyhow!("compiled_mode but empty codegen"))?;

            let result = codegen.repl_compile(&ast)?;
            println!("{}", result);
        }

        Ok(())
    }

    // Load standard library (prelude) on REPL startup
    fn load_prelude(&mut self) -> Result<()> {
        use std::fs;
        use std::path::Path;

        // Try to find prelude.tlsp in current directory or parent directory
        let prelude_paths = ["prelude.tlsp", "../prelude.tlsp", "../../prelude.tlsp"];

        for prelude_path in &prelude_paths {
            if Path::new(prelude_path).exists() {
                let prelude_source = fs::read_to_string(prelude_path)
                    .map_err(|e| anyhow!("Failed to read prelude: {}", e))?;

                // Parse and evaluate each expression in the prelude
                let tokens = tokenize(&prelude_source)?;
                let ast = parse(&tokens[..])?;

                if self.interpreted_mode {
                    let env = self.interpreter_env.as_mut()
                        .ok_or(anyhow!("interpreter_mode: ON but missing interpreter_env!"))?;
                    for expr in ast {
                        let _ = eval_with_env(env, vec![expr])?;
                    }
                } else {
                    let codegen = self.codegen.as_mut()
                        .ok_or(anyhow!("compiled_mode but empty codegen"))?;
                    for expr in ast {
                        let _ = codegen.repl_compile(&[expr])?;
                    }
                }

                println!("Loaded standard library from {}", prelude_path);
                return Ok(());
            }
        }

        // Prelude not found - continue without it
        println!("Note: prelude.tlsp not found, starting without standard library");
        Ok(())
    }

}