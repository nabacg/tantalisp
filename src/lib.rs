mod lexer;

use anyhow::Result;
use lexer::tokenize;

pub fn rep(input: &str) -> Result<()> {
    let tokens = tokenize(input)?;
    println!("Tokens:\n{}", tokens.iter().map(|t| format!("{:?}", t)).collect::<Vec<_>>().join("\n"));
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
