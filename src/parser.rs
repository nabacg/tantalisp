
use anyhow::{bail, anyhow, Result};

use crate::lexer::Token;

pub fn parse(tokens: &[Token]) -> Result<Vec<SExpr>> {
    Parser::new(tokens).collect()

}


#[derive(Clone)]
pub enum SExpr {
    Int(i32),
    Bool(bool),
    String(String),
    Symbol(String),
    IfExpr(Box<SExpr>, Vec<SExpr>, Vec<SExpr>),
    LambdaExpr(Vec<SExpr>, Vec<SExpr>),
    DefExpr(String, Box<SExpr>),
    List(Vec<SExpr>),
    Vector(Vec<SExpr>),
    BuiltinFn(String, fn(&[SExpr]) -> anyhow::Result<SExpr>),
}

// Manual Debug implementation since function pointers don't auto-derive
impl std::fmt::Debug for SExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SExpr::Int(n) => write!(f, "Int({})", n),
            SExpr::Bool(b) => write!(f, "Bool({})", b),
            SExpr::String(s) => write!(f, "String({:?})", s),
            SExpr::Symbol(s) => write!(f, "Symbol({})", s),
            SExpr::IfExpr(p, t, e) => write!(f, "IfExpr({:?}, {:?}, {:?})", p, t, e),
            SExpr::LambdaExpr(p, b) => write!(f, "LambdaExpr({:?}, {:?})", p, b),
            SExpr::DefExpr(n, v) => write!(f, "DefExpr({}, {:?})", n, v),
            SExpr::List(l) => write!(f, "List({:?})", l),
            SExpr::Vector(v) => write!(f, "Vector({:?})", v),
            SExpr::BuiltinFn(name, _) => write!(f, "BuiltinFn({})", name),
        }
    }
}

// Manual PartialEq - compare builtins by name
impl PartialEq for SExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SExpr::Int(a), SExpr::Int(b)) => a == b,
            (SExpr::Bool(a), SExpr::Bool(b)) => a == b,
            (SExpr::String(a), SExpr::String(b)) => a == b,
            (SExpr::Symbol(a), SExpr::Symbol(b)) => a == b,
            (SExpr::IfExpr(p1, t1, e1), SExpr::IfExpr(p2, t2, e2)) => {
                p1 == p2 && t1 == t2 && e1 == e2
            }
            (SExpr::LambdaExpr(p1, b1), SExpr::LambdaExpr(p2, b2)) => p1 == p2 && b1 == b2,
            (SExpr::DefExpr(n1, v1), SExpr::DefExpr(n2, v2)) => n1 == n2 && v1 == v2,
            (SExpr::List(a), SExpr::List(b)) => a == b,
            (SExpr::Vector(a), SExpr::Vector(b)) => a == b,
            (SExpr::BuiltinFn(n1, _), SExpr::BuiltinFn(n2, _)) => n1 == n2,
            _ => false,
        }
    }
}

// Display implementation for Lisp-style pretty printing
impl std::fmt::Display for SExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SExpr::Int(n) => write!(f, "{}", n),
            SExpr::Bool(b) => write!(f, "{}", if *b { "true" } else { "false" }),
            SExpr::String(s) => write!(f, "\"{}\"", s),
            SExpr::Symbol(s) => write!(f, "{}", s),
            SExpr::List(exprs) => {
                write!(f, "(")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, ")")
            }
            SExpr::Vector(exprs) => {
                write!(f, "[")?;
                for (i, expr) in exprs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", expr)?;
                }
                write!(f, "]")
            }
            SExpr::IfExpr(pred, then_exprs, else_exprs) => {
                write!(f, "(if {} ", pred)?;
                if then_exprs.len() == 1 {
                    write!(f, "{}", then_exprs[0])?;
                } else {
                    write!(f, "(do")?;
                    for expr in then_exprs {
                        write!(f, " {}", expr)?;
                    }
                    write!(f, ")")?;
                }
                write!(f, " ")?;
                if else_exprs.len() == 1 {
                    write!(f, "{}", else_exprs[0])?;
                } else {
                    write!(f, "(do")?;
                    for expr in else_exprs {
                        write!(f, " {}", expr)?;
                    }
                    write!(f, ")")?;
                }
                write!(f, ")")
            }
            SExpr::LambdaExpr(params, body) => {
                write!(f, "(fn [")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, "]")?;
                for expr in body {
                    write!(f, " {}", expr)?;
                }
                write!(f, ")")
            }
            SExpr::DefExpr(name, value) => {
                write!(f, "(def {} {})", name, value)
            }
            SExpr::BuiltinFn(name, _) => write!(f, "#<builtin:{}>", name),
        }
    }
}

pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize
}

impl<'a> Iterator for Parser<'a> {
    type Item = Result<SExpr>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
        match self.peek()? {
            Token::Int(_) => self.consume_int(), 
            Token::String(_) => self.consume_string(),
            Token::Symbol(_) => self.consume_symbol(),
            Token::LeftBracket => self.consume_vector(),
            Token::LeftParen if self.is_def() => self.consume_def(),
            Token::LeftParen if self.is_if_expr() => self.consume_if(),
            Token::LeftParen if self.is_lambda_expr() => self.consume_lambda(),
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
        self.consume(); // consume `[`
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

    fn close_list_and_return(&mut self, ret_val: SExpr) -> Result<SExpr> {
        if let Some(&Token::RightParen) = self.peek() {
            // consume the ')'
            self.consume();
            Ok(ret_val)
        } else {
            bail!("Unclosed list, expected ')', found: {:?}", 
              self.peek()
                  .map_or(" END_OF_TOKENS".to_string(), 
                  |t| format!("{:?}", t)));
        }
    }
    
    fn consume_list(&mut self) -> Result<SExpr> {
        self.consume(); // consume `(`
          let vec_contents = self.consume_while(|t| t != &Token::RightParen);
          self.close_list_and_return(SExpr::List(vec_contents?))
    }
    
    fn is_def(&self) -> bool {
        match self.peek_ahead(1) {
            Some(Token::Symbol(sym)) if sym == "def" => true,
            _ => false
        }
    }
    
    fn consume_def(&mut self) -> Result<SExpr> {
        self.consume(); // consume `(`
        self.consume(); // consume 'def' keyword
        let var = self.consume().ok_or(anyhow!("InvalidDefinition - Failed to consume variable name"))?;
        let var_name = 
            match var {
                Token::Symbol(var_name) => var_name.to_string(),
                _ => bail!("InvalidDefinition variable name, expected symbol, got: {:?}", var) 
            };

        let val = self.next().ok_or(anyhow!("InvalidDefinition - Failed to get variable value"))?;
        match val {
            Ok(val_expr) => {
                let def_expr = SExpr::DefExpr(var_name.to_string(), Box::new(val_expr));
                self.close_list_and_return(def_expr)
            },
            Err(e) => bail!("InvalidDefinition bad variable value, error {} ", e)
        }
    }
    
    fn is_if_expr(&self) -> bool {
        match self.peek_ahead(1) {
            Some(Token::Symbol(sym)) if sym == "if" => true,
            _ => false
        }
    }


    
    fn consume_if(&mut self) -> Result<SExpr> {
        self.consume(); // consume `(`
        self.consume(); // consume 'if' keyword

        let pred_expr = self.next().ok_or(anyhow!("InvalidIf - Failed to get predicate expr"))??;
   
   //TODO - should really parse a list (block) for truthy and falsy
        let truthy_exprs = self.next().ok_or(anyhow!("InvalidIf - Failed to get truthy expr"))??;
        let falsy_expr = self.next().ok_or(anyhow!("InvalidIf - Failed to get falsy expr"))??;

        let if_expr = SExpr::IfExpr(Box::new(pred_expr), vec![truthy_exprs], vec![falsy_expr]);
        self.close_list_and_return(if_expr)
    }
    
    fn is_lambda_expr(&self) -> bool {
        match self.peek_ahead(1) {
            Some(Token::Symbol(sym)) if sym == "fn" => true,
            _ => false
        }
    }
    
    fn consume_lambda(&mut self) -> Result<SExpr> {
        self.consume(); // consume `(`
        self.consume(); // consume 'fn' keyword

        if let SExpr::Vector(param_vec_expr) = self.consume_vector()? {
            // Consume body expressions until we hit ')'
            let body = self.consume_while(|t| t != &Token::RightParen)?;
            let lambda_expr = SExpr::LambdaExpr(param_vec_expr, body);
            self.close_list_and_return(lambda_expr)
        } else {
            bail!("InvalidLambdaExpr - param list needs to be a vector")
        }
    }
    
    fn consume_int(&mut self) -> Result<SExpr> {
        if let Some(Token::Int(v)) =  self.consume() {
            Ok(SExpr::Int(*v))
        } else {
            bail!("Failed to parse IntToken, expected int.")
        }
    }
    
    fn consume_string(&mut self) -> Result<SExpr> {
        if let Some(Token::String(s)) =  self.consume() {
            Ok(SExpr::String(s.clone()))
        } else {
            bail!("Failed to parse String, expected int.")
        }
    }
    
    fn consume_symbol(&mut self) -> std::result::Result<SExpr, anyhow::Error> {
        if let Some(Token::Symbol(s)) =  self.consume() {
            Ok(SExpr::Symbol(s.clone()))
        } else {
            bail!("Failed to parse String, expected int.")
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
    fn parse_empty_body_fn() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("fn".to_string()),
                Token::LeftBracket,
                Token::Symbol("x".to_string()),
                Token::RightBracket,
                Token::Int(42),
                Token::RightParen
            ]),
            SExpr::LambdaExpr(vec![SExpr::Symbol("x".to_string())], vec![SExpr::Int(42)]));
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

    // ===== Lambda Expression Tests =====

    #[test]
    fn parse_simple_lambda() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("fn".to_string()),
                Token::LeftBracket,
                Token::Symbol("x".to_string()),
                Token::RightBracket,
                Token::Int(42),
                Token::RightParen
            ]),
            SExpr::LambdaExpr(
                vec![SExpr::Symbol("x".to_string())],
                vec![SExpr::Int(42)]
            )
        );
    }

    #[test]
    fn parse_lambda_multiple_params() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("fn".to_string()),
                Token::LeftBracket,
                Token::Symbol("x".to_string()),
                Token::Symbol("y".to_string()),
                Token::RightBracket,
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::Symbol("x".to_string()),
                Token::Symbol("y".to_string()),
                Token::RightParen,
                Token::RightParen
            ]),
            SExpr::LambdaExpr(
                vec![SExpr::Symbol("x".to_string()), SExpr::Symbol("y".to_string())],
                vec![SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::Symbol("x".to_string()),
                    SExpr::Symbol("y".to_string())
                ])]
            )
        );
    }

    #[test]
    fn parse_lambda_no_params() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("fn".to_string()),
                Token::LeftBracket,
                Token::RightBracket,
                Token::Int(42),
                Token::RightParen
            ]),
            SExpr::LambdaExpr(
                vec![],
                vec![SExpr::Int(42)]
            )
        );
    }

    // ===== If Expression Tests =====

    #[test]
    fn parse_simple_if() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("if".to_string()),
                Token::Int(1),
                Token::Int(2),
                Token::Int(3),
                Token::RightParen
            ]),
            SExpr::IfExpr(
                Box::new(SExpr::Int(1)),
                vec![SExpr::Int(2)],
                vec![SExpr::Int(3)]
            )
        );
    }

    #[test]
    fn parse_if_with_expression() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("if".to_string()),
                Token::LeftParen,
                Token::Symbol(">".to_string()),
                Token::Symbol("x".to_string()),
                Token::Int(0),
                Token::RightParen,
                Token::Symbol("x".to_string()),
                Token::Int(0),
                Token::RightParen
            ]),
            SExpr::IfExpr(
                Box::new(SExpr::List(vec![
                    SExpr::Symbol(">".to_string()),
                    SExpr::Symbol("x".to_string()),
                    SExpr::Int(0)
                ])),
                vec![SExpr::Symbol("x".to_string())],
                vec![SExpr::Int(0)]
            )
        );
    }

    // ===== Def Expression Tests =====

    #[test]
    fn parse_simple_def() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("def".to_string()),
                Token::Symbol("x".to_string()),
                Token::Int(42),
                Token::RightParen
            ]),
            SExpr::DefExpr(
                "x".to_string(),
                Box::new(SExpr::Int(42))
            )
        );
    }

    #[test]
    fn parse_def_with_expression() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("def".to_string()),
                Token::Symbol("y".to_string()),
                Token::LeftParen,
                Token::Symbol("+".to_string()),
                Token::Int(2),
                Token::Int(3),
                Token::RightParen,
                Token::RightParen
            ]),
            SExpr::DefExpr(
                "y".to_string(),
                Box::new(SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::Int(2),
                    SExpr::Int(3)
                ]))
            )
        );
    }

    #[test]
    fn parse_def_lambda() {
        assert_eq!(
            parse_one(&[
                Token::LeftParen,
                Token::Symbol("def".to_string()),
                Token::Symbol("square".to_string()),
                Token::LeftParen,
                Token::Symbol("fn".to_string()),
                Token::LeftBracket,
                Token::Symbol("x".to_string()),
                Token::RightBracket,
                Token::LeftParen,
                Token::Symbol("*".to_string()),
                Token::Symbol("x".to_string()),
                Token::Symbol("x".to_string()),
                Token::RightParen,
                Token::RightParen,
                Token::RightParen
            ]),
            SExpr::DefExpr(
                "square".to_string(),
                Box::new(SExpr::LambdaExpr(
                    vec![SExpr::Symbol("x".to_string())],
                    vec![SExpr::List(vec![
                        SExpr::Symbol("*".to_string()),
                        SExpr::Symbol("x".to_string()),
                        SExpr::Symbol("x".to_string())
                    ])]
                ))
            )
        );
    }
}