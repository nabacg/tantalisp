
use anyhow::{bail, Result};

use crate::lexer::Token;

pub fn parse(tokens: &[Token]) -> Result<Vec<SExpr>> {
    Parser::new(tokens).collect()

}


#[derive(Debug, PartialEq)]
pub enum SExpr {
    Int(i32),
    String(String),
    Symbol(String),
    List(Vec<SExpr>),
    Vector(Vec<SExpr>)
}

pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize
}

impl<'a> Iterator for Parser<'a> {
    type Item = Result<SExpr>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
        match self.consume()? {
            Token::Int(v) => Ok(SExpr::Int(*v)), 
            Token::String(s) => Ok(SExpr::String(s.clone())),
            Token::Symbol(sym) => Ok(SExpr::Symbol(sym.clone())),
            Token::LeftBracket => self.consume_vector(),
            Token::LeftParen => self.consume_list(),
            c => Err(anyhow::anyhow!("Invalid Token stream, found: {:?}", c))
        })
    }
}


impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Parser {
            tokens: tokens, 
            pos: 0
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    
    fn consume(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.pos);
        self.pos += 1;
        tok
    }

    fn peek_ahead(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset)
    }

    fn consume_while<F>(&mut self, predicate: F) -> Result<Vec<SExpr>> where F: Fn(&Token) -> bool {
        let mut res = vec![];
        while let Some(t) = self.peek() {
            if predicate(t) {
                res.push(self.next().unwrap()?);
            } else  {
                break;
            }
        }
        Ok(res)
    }
    
    fn consume_vector(&mut self) -> Result<SExpr> {
        let vec_contents = self.consume_while(|t| t != &Token::RightBracket);
        if let Some(&Token::RightBracket) = self.peek() {
            // consume the ']'
            self.consume();
            Ok(SExpr::Vector(vec_contents?))
        } else {
            bail!("Unclosed vector, expected ')', found: {:?}", 
                self.peek()
                    .map_or(" END_OF_TOKENS".to_string(), 
                    |t| format!("{:?}", t)));
        }
    }
    
    fn consume_list(&mut self) -> Result<SExpr> {
          let vec_contents = self.consume_while(|t| t != &Token::RightParen);
          if let Some(&Token::RightParen) = self.peek() {
              // consume the ')'
              self.consume();
              Ok(SExpr::List(vec_contents?))
          } else {
              bail!("Unclosed list, expected ')', found: {:?}", 
                self.peek()
                    .map_or(" END_OF_TOKENS".to_string(), 
                    |t| format!("{:?}", t)));
          }
    }

}

#[cfg(test)]
mod parser_tests {
    use super::*;

    // Helper to parse and get first result
    fn parse_one(tokens: &[Token]) -> SExpr {
        parse(tokens).unwrap().into_iter().next().unwrap()
    }

    // Helper to parse and get all results
    fn parse_ok(tokens: &[Token]) -> Vec<SExpr> {
        parse(tokens).unwrap()
    }

    // ===== Scalar Values =====

    #[test]
    fn parse_int() {
        assert_eq!(parse_one(&[Token::Int(42)]), SExpr::Int(42));
    }

    #[test]
    fn parse_string() {
        assert_eq!(
            parse_one(&[Token::String("BOOM!".to_string())]),
            SExpr::String("BOOM!".to_string())
        );
    }

    #[test]
    fn parse_symbol() {
        assert_eq!(
            parse_one(&[Token::Symbol("fn".to_string())]),
            SExpr::Symbol("fn".to_string())
        );
    }

    // ===== Lists =====

    #[test]
    fn parse_empty_list() {
        assert_eq!(
            parse_one(&[Token::LeftParen, Token::RightParen]),
            SExpr::List(vec![])
        );
    }

    #[test]
    fn parse_list_with_ints() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Int(1),
                Token::Int(2),
                Token::Int(3),
                Token::RightParen
            ]),
            SExpr::List(vec![SExpr::Int(1), SExpr::Int(2), SExpr::Int(3)])
        );
    }

    #[test]
    fn parse_nested_list() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::LeftParen,
                Token::Symbol("*".to_string()),
                Token::Int(2),
                Token::Int(3),
                Token::RightParen,
                Token::Int(4),
                Token::RightParen
            ]),
            SExpr::List(vec![
                SExpr::Symbol("+".to_string()),
                SExpr::List(vec![
                    SExpr::Symbol("*".to_string()),
                    SExpr::Int(2),
                    SExpr::Int(3)
                ]),
                SExpr::Int(4)
            ])
        );
    }

    // ===== Vectors =====

    #[test]
    fn parse_empty_vector() {
        assert_eq!(
            parse_one(&[Token::LeftBracket, Token::RightBracket]),
            SExpr::Vector(vec![])
        );
    }

    #[test]
    fn parse_vector_with_ints() {
        assert_eq!(
            parse_one(&[
                Token::LeftBracket,
                Token::Int(1),
                Token::Int(2),
                Token::Int(3),
                Token::RightBracket
            ]),
            SExpr::Vector(vec![SExpr::Int(1), SExpr::Int(2), SExpr::Int(3)])
        );
    }

    #[test]
    fn parse_nested_vectors() {
        assert_eq!(
            parse_one(&[
                Token::LeftBracket,
                Token::LeftBracket,
                Token::Int(1),
                Token::RightBracket,
                Token::LeftBracket,
                Token::Int(2),
                Token::RightBracket,
                Token::RightBracket
            ]),
            SExpr::Vector(vec![
                SExpr::Vector(vec![SExpr::Int(1)]),
                SExpr::Vector(vec![SExpr::Int(2)])
            ])
        );
    }

    #[test]
    fn parse_list_with_vector() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("vec".to_string()),
                Token::LeftBracket,
                Token::Int(1),
                Token::Int(2),
                Token::RightBracket,
                Token::RightParen
            ]),
            SExpr::List(vec![
                SExpr::Symbol("vec".to_string()),
                SExpr::Vector(vec![SExpr::Int(1), SExpr::Int(2)])
            ])
        );
    }

    #[test]
    fn parse_vector_with_list() {
        assert_eq!(
            parse_one(&[
                Token::LeftBracket,
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::Int(1),
                Token::Int(2),
                Token::RightParen,
                Token::RightBracket
            ]),
            SExpr::Vector(vec![
                SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::Int(1),
                    SExpr::Int(2)
                ])
            ])
        );
    }

    // ===== Multiple Expressions =====

    #[test]
    fn parse_multiple_expressions() {
        assert_eq!(
            parse_ok(&[
                Token::Int(42),
                Token::Symbol("foo".to_string()),
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::Int(1),
                Token::Int(2),
                Token::RightParen
            ]),
            vec![
                SExpr::Int(42),
                SExpr::Symbol("foo".to_string()),
                SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::Int(1),
                    SExpr::Int(2)
                ])
            ]
        );
    }

    // ===== Error Cases =====

    #[test]
    fn unclosed_list() {
        let result = parse(&[Token::LeftParen, Token::Int(1)]);
        assert!(result.is_err());
    }

    #[test]
    fn unclosed_vector() {
        let result = parse(&[Token::LeftBracket, Token::Int(1)]);
        assert!(result.is_err());
    }

    #[test]
    fn unexpected_closing_paren() {
        let result = parse(&[Token::RightParen]);
        assert!(result.is_err());
    }

    #[test]
    fn unexpected_closing_bracket() {
        let result = parse(&[Token::RightBracket]);
        assert!(result.is_err());
    }

    #[test]
    fn deeply_nested_lists() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::LeftParen,
                Token::LeftParen,
                Token::Int(42),
                Token::RightParen,
                Token::RightParen,
                Token::RightParen
            ]),
            SExpr::List(vec![
                SExpr::List(vec![
                    SExpr::List(vec![SExpr::Int(42)])
                ])
            ])
        );
    }
}