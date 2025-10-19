use std::{iter::Peekable, str::Chars};

use anyhow::{bail, Result};


pub struct Tokenizer<'a> {
    chars: Peekable<Chars<'a>>
}

impl<'a> Tokenizer<'a> {
    pub fn new(input:&'a str) -> Self {
        Tokenizer { chars: input.chars().peekable() }
    }

    fn peek(&mut self) -> Option<&char> {
        self.chars.peek()
    }

    fn consume(&mut self) -> Option<char> {
        self.chars.next()
    }

    fn consume_while<F>(&mut self, predicate: F) -> String where F: Fn(char) -> bool {
        let mut res = String::new();
        while let Some(&ch) = self.peek() {
            if predicate(ch) {
                res.push(self.consume().unwrap());
            } else {
                break;
            }
        }
        res
    }

    fn consume_int(&mut self) -> Result<Token> {
        let num_char = self.consume_while(|ch| ch.is_ascii_digit());
        let num = num_char.parse()?;
        Ok(Token::Int(num))
    }

    fn consume_string(&mut self) -> Result<Token> {
        // consume opening quote
        self.consume();
        let str_contents = self.consume_while(|ch| ch != '"');
        if self.consume() != Some('"') {
            bail!("Encountered unterminated string!");
        }

        Ok(Token::String(str_contents)) // TODO need to handle quoted strings "adssa \" asdas "

    }

    fn consume_symbol(&mut self) -> Result<Token> {
        let sym_contents = self.consume_while(|c|is_symbol_char(&c));
        Ok(Token::Symbol(sym_contents))
    }

    fn consume_paren(&mut self) -> Result<Token> {
        match self.consume() {
            None => bail!("Expected '(' or ')' but found end of Stream" ),
            Some('(') => Ok(Token::LeftParen),
            Some(')') => Ok(Token::RightParen),
            Some(c) => bail!("Expected '(' or ')' but found: {}", c )
        }
    }
    
    fn consume_bracket(&mut self) -> std::result::Result<Token, anyhow::Error> {
        match self.consume() {
            None => bail!("Expected '[' or ']' but found end of Stream" ),
            Some('[') => Ok(Token::LeftBracket),
            Some(']') => Ok(Token::RightBracket),
            Some(c) => bail!("Expected '[' or ']' but found: {}", c )
        }
    }
}

#[derive(PartialEq, Debug)]
pub enum Token {
    Int(i32),
    String(String),
    Symbol(String),
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket
}

fn is_symbol_char(c: &char) -> bool {
    match c {
        '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'' | ',' | ';' => false,
        c if c.is_whitespace() => false,
        _ => true
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = Result<Token>;
    
    fn next(&mut self) -> Option<Self::Item> {
        // consume all whitespace
        while matches!(self.peek(), Some(' ') | Some('\t') | Some('\n')) {
            self.consume();
        }

        Some(
        match self.peek()? {
            '(' | ')' => self.consume_paren(),
            '[' | ']' => self.consume_bracket(),
            '"' => self.consume_string(),
            '0'..='9' => self.consume_int(),
            c if is_symbol_char(c) => self.consume_symbol(),
            c => Err(anyhow::anyhow!("Unexpected character: {}", c))
        })
    }

    
}


pub fn tokenize(input: &str) -> Result<Vec<Token>> {

    Tokenizer::new(input)
        .collect()
}


#[cfg(test)]
mod lexer_tests {
    use super::*;
    use Token::*;

    #[test]
    fn empty_input_empty_token() {
        assert_eq!(tokenize("").expect("Error in tokenize"), vec![]);
    }

    #[test]
    fn empty_paren() {
        assert_eq!(tokenize("()").unwrap(), vec![Token::LeftParen, Token::RightParen])
    }


    #[test]
    fn list_of_ints() {
        assert_eq!(tokenize("(1 2 3 4 5)").unwrap(), vec![LeftParen, 
                                                            Int(1), 
                                                            Int(2),
                                                            Int(3),
                                                            Int(4),
                                                            Int(5), 
                                                            RightParen])
    }


    #[test]
    fn multidigit_ints() {
        assert_eq!(tokenize("12345").unwrap(), vec![Int(12345)])
    }

    // ===== Integer Tests =====

    #[test]
    fn single_digit_int() {
        assert_eq!(tokenize("7").unwrap(), vec![Int(7)])
    }

    #[test]
    fn zero() {
        assert_eq!(tokenize("0").unwrap(), vec![Int(0)])
    }

    #[test]
    fn large_int() {
        assert_eq!(tokenize("2147483647").unwrap(), vec![Int(2147483647)])
    }

    #[test]
    fn multiple_ints_with_spaces() {
        assert_eq!(tokenize("1 22 333").unwrap(), vec![Int(1), Int(22), Int(333)])
    }

    #[test]
    fn int_overflow() {
        // i32::MAX is 2147483647, this should overflow
        let result = tokenize("99999999999");
        assert!(result.is_err());
    }

    // ===== Parenthesis Tests =====

    #[test]
    fn nested_parens() {
        assert_eq!(tokenize("(())").unwrap(), vec![LeftParen, LeftParen, RightParen, RightParen])
    }

    #[test]
    fn multiple_nested_parens() {
        assert_eq!(tokenize("((()))").unwrap(),
            vec![LeftParen, LeftParen, LeftParen, RightParen, RightParen, RightParen])
    }

    #[test]
    fn parens_with_whitespace() {
        assert_eq!(tokenize("(  )").unwrap(), vec![LeftParen, RightParen])
    }

    // ===== Bracket Tests =====

    #[test]
    fn empty_brackets() {
        assert_eq!(tokenize("[]").unwrap(), vec![LeftBracket, RightBracket])
    }

    #[test]
    fn brackets_with_int() {
        assert_eq!(tokenize("[42]").unwrap(), vec![LeftBracket, Int(42), RightBracket])
    }

    #[test]
    fn mixed_parens_and_brackets() {
        assert_eq!(tokenize("([])").unwrap(),
            vec![LeftParen, LeftBracket, RightBracket, RightParen])
    }

    // ===== Symbol Tests =====

    #[test]
    fn single_letter_symbol() {
        assert_eq!(tokenize("x").unwrap(), vec![Symbol("x".to_string())])
    }

    #[test]
    fn simple_symbol() {
        assert_eq!(tokenize("foo").unwrap(), vec![Symbol("foo".to_string())])
    }

    #[test]
    fn symbol_with_parens() {
        assert_eq!(tokenize("(foo)").unwrap(),
            vec![LeftParen, Symbol("foo".to_string()), RightParen])
    }

    #[test]
    fn multiple_symbols() {
        assert_eq!(tokenize("foo bar baz").unwrap(),
            vec![Symbol("foo".to_string()), Symbol("bar".to_string()), Symbol("baz".to_string())])
    }

    #[test]
    fn symbol_ends_at_paren() {
        // Symbol should stop consuming when it hits '('
        assert_eq!(tokenize("foo(bar)").unwrap(),
            vec![Symbol("foo".to_string()), LeftParen, Symbol("bar".to_string()), RightParen])
    }

    // ===== String Tests =====

    #[test]
    fn empty_string() {
        assert_eq!(tokenize(r#""""#).unwrap(), vec![String("".to_string())])
    }

    #[test]
    fn simple_string() {
        assert_eq!(tokenize(r#""hello""#).unwrap(), vec![String("hello".to_string())])
    }

    #[test]
    fn string_with_spaces() {
        assert_eq!(tokenize(r#""hello world""#).unwrap(), vec![String("hello world".to_string())])
    }

    #[test]
    fn string_in_expression() {
        assert_eq!(tokenize(r#"("hello")"#).unwrap(),
            vec![LeftParen, String("hello".to_string()), RightParen])
    }

    // ===== Whitespace Tests =====

    #[test]
    fn only_spaces() {
        assert_eq!(tokenize("   ").unwrap(), vec![])
    }

    #[test]
    fn only_tabs() {
        assert_eq!(tokenize("\t\t\t").unwrap(), vec![])
    }

    #[test]
    fn only_newlines() {
        assert_eq!(tokenize("\n\n\n").unwrap(), vec![])
    }

    #[test]
    fn mixed_whitespace() {
        assert_eq!(tokenize(" \t\n \t\n ").unwrap(), vec![])
    }

    #[test]
    fn whitespace_between_tokens() {
        assert_eq!(tokenize("( \t\n 42 \n\t )").unwrap(),
            vec![LeftParen, Int(42), RightParen])
    }

    // ===== Complex Expressions =====

    #[test]
    fn simple_addition() {
        assert_eq!(tokenize("(+ 1 2)").unwrap(),
            vec![LeftParen, Symbol("+".to_string()), Int(1), Int(2), RightParen])
    }

    #[test]
    fn nested_expression() {
        assert_eq!(tokenize("(+ (* 2 3) 4)").unwrap(),
            vec![LeftParen, Symbol("+".to_string()),
                 LeftParen, Symbol("*".to_string()), Int(2), Int(3), RightParen,
                 Int(4), RightParen])
    }

    #[test]
    fn multiple_expressions() {
        assert_eq!(tokenize("(foo 1) (bar 2)").unwrap(),
            vec![LeftParen, Symbol("foo".to_string()), Int(1), RightParen,
                 LeftParen, Symbol("bar".to_string()), Int(2), RightParen])
    }

    // ===== Edge Cases =====

    #[test]
    fn no_space_between_int_and_paren() {
        assert_eq!(tokenize("(42)").unwrap(),
            vec![LeftParen, Int(42), RightParen])
    }

    #[test]
    fn no_space_between_symbol_and_paren() {
        assert_eq!(tokenize("(foo)").unwrap(),
            vec![LeftParen, Symbol("foo".to_string()), RightParen])
    }

    #[test]
    fn int_at_start() {
        assert_eq!(tokenize("42 foo").unwrap(),
            vec![Int(42), Symbol("foo".to_string())])
    }

    #[test]
    fn symbol_at_start() {
        assert_eq!(tokenize("foo 42").unwrap(),
            vec![Symbol("foo".to_string()), Int(42)])
    }

    // ===== Known Limitations (TODO) =====

    #[test]
    fn unclosed_string() {
        let result = tokenize(r#""hello"#);
        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Not yet implemented
    fn string_with_escaped_quote() {
        assert_eq!(tokenize(r#""hello \"world\"""#).unwrap(),
            vec![String(r#"hello "world""#.to_string())])
    }
}