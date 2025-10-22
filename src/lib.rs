mod lexer;
mod parser;
mod evaluator;
mod codegen;

use anyhow::{anyhow, Result};
use lexer::tokenize;
use parser::parse;
use evaluator::{eval, eval_with_env};
use codegen::compile_and_run;

// Re-export Environment for REPL
pub use evaluator::Environment;

pub struct Tantalisp<'l> {
    debug_mode: bool,
    interpreted_mode: bool,
    interpreter_env: Environment<'l>
}

impl<'l> Tantalisp<'l> {
    pub fn new(debug_mode: bool, interpreted_mode: bool) -> Self {
        let interpreter_env  = 
        if interpreted_mode {
            Environment::global()
        } else {
            Environment::new()
        };
        Self { 
            debug_mode, 
            interpreted_mode,
            interpreter_env
        }
    }


    pub fn rep(&self, input: &str) -> Result<()> {
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
            let result = compile_and_run(&ast, self.debug_mode)?;
            println!("Result:\n{}", result);

        }

        Ok(())
    }

    // REPL-friendly version that takes a mutable environment
    pub fn rep_with_env(&mut self, input: &str) -> Result<()> {
        
        let tokens = tokenize(input)?;
        let ast = parse(&tokens[..])?;
        if self.interpreted_mode {
            let result = eval_with_env(&mut  self.interpreter_env, ast)?;
            println!("{}", result);

        } else {

            let result = compile_and_run(&ast, self.debug_mode)?;
            println!("{}", result);
        }

        Ok(())
    }

}