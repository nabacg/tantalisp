mod lexer;
mod parser;
mod evaluator;
mod codegen;

use std::io::{self, BufRead, Write};

use anyhow::{anyhow, Result};
use inkwell::context::Context;
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
                    match self.rep_with_env(input) {
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

}