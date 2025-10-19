mod lexer;
mod parser;
mod evaluator;

use anyhow::Result;
use lexer::tokenize;
use parser::parse;
use evaluator::{eval, eval_with_env};

// Re-export Environment for REPL
pub use evaluator::Environment;

pub fn rep(input: &str) -> Result<()> {
    let tokens = tokenize(input)?;
    println!("Tokens:\n{}", tokens.iter().map(|t| format!("{:?}", t)).collect::<Vec<_>>().join("\n"));
    let ast = parse(&tokens[..])?;
    println!("AST:\n{}", ast.iter().map(|t| format!("{:?}", t)).collect::<Vec<_>>().join("\n"));

    let result = eval(ast)?;
    println!("Result:\n{}", result);

    Ok(())
}

// REPL-friendly version that takes a mutable environment
pub fn rep_with_env(env: &mut Environment, input: &str) -> Result<()> {
    let tokens = tokenize(input)?;
    let ast = parse(&tokens[..])?;
    let result = eval_with_env(env, ast)?;
    println!("{}", result);

    Ok(())
}


pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
