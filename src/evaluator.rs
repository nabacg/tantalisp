use std::collections::HashMap;
use crate::parser::SExpr;
use anyhow::{bail, anyhow, Result};
use std::rc::Rc;
use std::cell::RefCell;

// ===== Builtin Functions =====

fn builtin_add(args: &[SExpr]) -> Result<SExpr> {
    let mut sum = 0;
    for arg in args {
        match arg {
            SExpr::Int(n) => sum += n,
            _ => bail!("+ expects integers, got: {:?}", arg),
        }
    }
    Ok(SExpr::Int(sum))
}

fn builtin_sub(args: &[SExpr]) -> Result<SExpr> {
    if args.is_empty() {
        bail!("- requires at least 1 argument");
    }
    match &args[0] {
        SExpr::Int(first) => {
            let mut result = *first;
            if args.len() == 1 {
                // Unary negation: (- 5) => -5
                return Ok(SExpr::Int(-result));
            }
            for arg in &args[1..] {
                match arg {
                    SExpr::Int(n) => result -= n,
                    _ => bail!("- expects integers, got: {:?}", arg),
                }
            }
            Ok(SExpr::Int(result))
        }
        _ => bail!("- expects integers, got: {:?}", args[0]),
    }
}

fn builtin_mul(args: &[SExpr]) -> Result<SExpr> {
    let mut product = 1;
    for arg in args {
        match arg {
            SExpr::Int(n) => product *= n,
            _ => bail!("* expects integers, got: {:?}", arg),
        }
    }
    Ok(SExpr::Int(product))
}

fn builtin_div(args: &[SExpr]) -> Result<SExpr> {
    if args.is_empty() {
        bail!("/ requires at least 1 argument");
    }
    match &args[0] {
        SExpr::Int(first) => {
            let mut result = *first;
            for arg in &args[1..] {
                match arg {
                    SExpr::Int(n) => {
                        if *n == 0 {
                            bail!("Division by zero");
                        }
                        result /= n;
                    }
                    _ => bail!("/ expects integers, got: {:?}", arg),
                }
            }
            Ok(SExpr::Int(result))
        }
        _ => bail!("/ expects integers, got: {:?}", args[0]),
    }
}

fn builtin_list(args: &[SExpr]) -> Result<SExpr> {
    Ok(SExpr::List(args.to_vec()))
}

fn builtin_vec(args: &[SExpr]) -> Result<SExpr> {
    Ok(SExpr::Vector(args.to_vec()))
}

fn builtin_print(args: &[SExpr]) -> Result<SExpr> {
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        // Use Display for pretty printing
        print!("{}", arg);
    }
    println!();
    Ok(SExpr::Int(0)) // Return dummy value
}

// ===== Comparison Functions =====

fn builtin_eq(args: &[SExpr]) -> Result<SExpr> {
    if args.len() < 2 {
        bail!("= requires at least 2 arguments");
    }

    match &args[0] {
        SExpr::Int(first) => {
            for arg in &args[1..] {
                match arg {
                    SExpr::Int(n) if n == first => continue,
                    SExpr::Int(_) => return Ok(SExpr::Bool(false)),
                    _ => bail!("= expects integers, got: {:?}", arg),
                }
            }
            Ok(SExpr::Bool(true))
        }
        _ => bail!("= expects integers, got: {:?}", args[0]),
    }
}

fn builtin_ne(args: &[SExpr]) -> Result<SExpr> {
    if args.len() != 2 {
        bail!("!= requires exactly 2 arguments");
    }

    match (&args[0], &args[1]) {
        (SExpr::Int(a), SExpr::Int(b)) => Ok(SExpr::Bool(a != b)),
        _ => bail!("!= expects integers"),
    }
}

fn builtin_lt(args: &[SExpr]) -> Result<SExpr> {
    if args.len() != 2 {
        bail!("< requires exactly 2 arguments");
    }

    match (&args[0], &args[1]) {
        (SExpr::Int(a), SExpr::Int(b)) => Ok(SExpr::Bool(a < b)),
        _ => bail!("< expects integers"),
    }
}

fn builtin_gt(args: &[SExpr]) -> Result<SExpr> {
    if args.len() != 2 {
        bail!("> requires exactly 2 arguments");
    }

    match (&args[0], &args[1]) {
        (SExpr::Int(a), SExpr::Int(b)) => Ok(SExpr::Bool(a > b)),
        _ => bail!("> expects integers"),
    }
}

fn builtin_le(args: &[SExpr]) -> Result<SExpr> {
    if args.len() != 2 {
        bail!("<= requires exactly 2 arguments");
    }

    match (&args[0], &args[1]) {
        (SExpr::Int(a), SExpr::Int(b)) => Ok(SExpr::Bool(a <= b)),
        _ => bail!("<= expects integers"),
    }
}

fn builtin_ge(args: &[SExpr]) -> Result<SExpr> {
    if args.len() != 2 {
        bail!(">= requires exactly 2 arguments");
    }

    match (&args[0], &args[1]) {
        (SExpr::Int(a), SExpr::Int(b)) => Ok(SExpr::Bool(a >= b)),
        _ => bail!(">= expects integers"),
    }
}


pub fn eval(exprs: Vec<SExpr>) -> Result<SExpr> {
    let mut env = Environment::global();
    eval_with_env(&mut env, exprs)
}

pub fn eval_with_env(env: &mut Environment, exprs: Vec<SExpr>) -> Result<SExpr> {
    exprs
        .into_iter()
        .map(|e| eval_expr(env, e))
        .last()
        .unwrap()
}


fn eval_expr(env: &mut Environment, expr: SExpr) -> Result<SExpr> {
    match expr {
        SExpr::Int(n) => Ok(SExpr::Int(n)),
        SExpr::Bool(b) => Ok(SExpr::Bool(b)),
        SExpr::String(s) => Ok(SExpr::String(s)),
        SExpr::Symbol(sym) => env.get(&sym)
                                            .map(|e| e.clone()),
  
        SExpr::Vector(sexprs) =>  { 
          let vec_of_results: Result<Vec<SExpr>> =  sexprs.into_iter().map(|e| eval_expr(env, e)).collect();
          Ok(SExpr::Vector(vec_of_results?))
          
        },
        SExpr::IfExpr(pred, truthy_exprs, falsy_exprs) => {
            let pred_value = eval_expr(env, *pred)?;
            if is_expr_true(pred_value) {
                truthy_exprs.into_iter().map(|e| eval_expr(env, e)).last().ok_or(anyhow!("empty truthy exprs"))?
            } else {
                falsy_exprs.into_iter().map(|e| eval_expr(env, e)).last().ok_or(anyhow!("empty falsy exprs"))?
            }
        },
        SExpr::LambdaExpr(params, body) => Ok(SExpr::LambdaExpr(params, body)),
        SExpr::BuiltinFn(name, func) => Ok(SExpr::BuiltinFn(name, func)),
        SExpr::DefExpr(var_name, var_value) => {
            let val = eval_expr(env, *var_value)?;
            let res = val.clone();
            env.define(var_name, val);
            Ok(res)
        },
        SExpr::List(sexprs) => {
            if sexprs.is_empty() {
                bail!("Cannot evaluate empty list ()");
            }

            let fn_value = eval_expr(env, sexprs[0].clone())?;

            // Evaluate arguments
            let args: Result<Vec<SExpr>> = sexprs[1..]
                .iter()
                .map(|e| eval_expr(env, e.clone()))
                .collect();
            let args = args?;

            match fn_value {
                SExpr::BuiltinFn(_, func) => func(&args),
                SExpr::LambdaExpr(params, body) => {
                    if params.len() != args.len() {
                        bail!(
                            "Arity mismatch: expected {} args, got {}",
                            params.len(),
                            args.len()
                        );
                    }

                    let mut new_env = Environment::with_parent(env);

                    for (i, p) in params.iter().enumerate() {
                        if let SExpr::Symbol(param_sym) = p {
                            new_env.define(param_sym.to_string(), args[i].clone());
                        } else {
                            bail!("Function parameter must be a symbol, got: {:?}", p);
                        }
                    }

                    body.iter()
                        .map(|e| eval_expr(&mut new_env, e.clone()))
                        .last()
                        .ok_or(anyhow!("Empty function body"))?
                }
                _ => bail!("Not a function: {:?}", fn_value),
            }
        },
    }
}

//TODO - this needs some actual design and thinking
fn is_expr_true(pred_value: SExpr) -> bool {
    match pred_value {
        SExpr::Bool(b) => b,
        SExpr::List(vs) if vs.len() == 0 => false,
        SExpr::Int(0) => false,
        _ => true
    }
}


// Make Environment public so REPL can maintain state
#[derive(Debug, Clone)]
pub struct Environment<'a> {
    bindings: HashMap<String, SExpr>,
    parent: Option<&'a Environment<'a>>
}

impl<'a> Environment<'a> {
    pub fn new() -> Self {
        Environment {
            bindings: HashMap::new(),
            parent: None,
        }
    }

    pub fn with_parent(parent: &'a Environment<'a>) -> Self {
        Environment {
            bindings: HashMap::new(),
            parent: Some(parent),
        }
    }

    // Create global environment with built-ins
    pub fn global() -> Self {
        let mut env = Environment::new();

        // Arithmetic operations
        env.define("+".to_string(), SExpr::BuiltinFn("+".to_string(), builtin_add));
        env.define("-".to_string(), SExpr::BuiltinFn("-".to_string(), builtin_sub));
        env.define("*".to_string(), SExpr::BuiltinFn("*".to_string(), builtin_mul));
        env.define("/".to_string(), SExpr::BuiltinFn("/".to_string(), builtin_div));

        // Comparison operations
        env.define("=".to_string(), SExpr::BuiltinFn("=".to_string(), builtin_eq));
        env.define("!=".to_string(), SExpr::BuiltinFn("!=".to_string(), builtin_ne));
        env.define("<".to_string(), SExpr::BuiltinFn("<".to_string(), builtin_lt));
        env.define(">".to_string(), SExpr::BuiltinFn(">".to_string(), builtin_gt));
        env.define("<=".to_string(), SExpr::BuiltinFn("<=".to_string(), builtin_le));
        env.define(">=".to_string(), SExpr::BuiltinFn(">=".to_string(), builtin_ge));

        // Data structure constructors
        env.define("list".to_string(), SExpr::BuiltinFn("list".to_string(), builtin_list));
        env.define("vec".to_string(), SExpr::BuiltinFn("vec".to_string(), builtin_vec));

        // I/O
        env.define("print".to_string(), SExpr::BuiltinFn("print".to_string(), builtin_print));

        env
    }

    // Define a new variable in the current scope
    pub fn define(&mut self, name: String, value: SExpr) {
        self.bindings.insert(name, value);
    }

    // Look up a variable (searches parent scopes)
    pub fn get(&self, name: &str) -> Result<&SExpr> {
        self.bindings.get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.get(name).ok()))
            .ok_or_else(|| anyhow::anyhow!("Undefined variable: {}", name))
    }

    // Update an existing variable
    pub fn set(&mut self, name: &str, value: SExpr) -> Result<()> {
        if self.bindings.contains_key(name) {
            self.bindings.insert(name.to_string(), value);
            Ok(())
        } else if let Some(parent) = &mut self.parent {
             bail!("Cannot mutate parent through shared ref")
        } else {
            bail!("Undefined variable: {}", name)
        }
    }
}

#[cfg(test)]
mod evaluator_tests {
    use super::*;
    use crate::lexer::tokenize;
    use crate::parser::parse;

    // Helper to tokenize, parse, and eval in one go
    fn eval_str(input: &str) -> Result<SExpr> {
        let tokens = tokenize(input)?;
        let ast = parse(&tokens)?;
        eval(ast)
    }

    // Helper to get Display string
    fn eval_display(input: &str) -> String {
        eval_str(input).unwrap().to_string()
    }

    // ===== Arithmetic Tests =====

    #[test]
    fn eval_addition() {
        assert_eq!(eval_display("(+ 1 2 3)"), "6");
    }

    #[test]
    fn eval_addition_single() {
        assert_eq!(eval_display("(+ 42)"), "42");
    }

    #[test]
    fn eval_subtraction() {
        assert_eq!(eval_display("(- 10 3)"), "7");
    }

    #[test]
    fn eval_subtraction_multiple() {
        assert_eq!(eval_display("(- 10 3 2)"), "5");
    }

    #[test]
    fn eval_unary_negation() {
        assert_eq!(eval_display("(- 5)"), "-5");
    }

    #[test]
    fn eval_multiplication() {
        assert_eq!(eval_display("(* 2 3 4)"), "24");
    }

    #[test]
    fn eval_division() {
        assert_eq!(eval_display("(/ 24 2 3)"), "4");
    }

    #[test]
    fn eval_nested_arithmetic() {
        assert_eq!(eval_display("(* (+ 2 3) (- 10 4))"), "30");
    }

    // ===== Data Structure Tests =====

    #[test]
    fn eval_list() {
        assert_eq!(eval_display("(list 1 2 3)"), "(1 2 3)");
    }

    #[test]
    fn eval_list_with_expression() {
        assert_eq!(eval_display("(list 1 2 (+ 3 4))"), "(1 2 7)");
    }

    #[test]
    fn eval_vector() {
        assert_eq!(eval_display("(vec 1 2 3)"), "[1 2 3]");
    }

    #[test]
    fn eval_nested_list_vector() {
        assert_eq!(eval_display("(list (list 1 2) (vec 3 4) 5)"), "((1 2) [3 4] 5)");
    }

    #[test]
    fn eval_empty_list() {
        assert_eq!(eval_display("(list)"), "()");
    }

    #[test]
    fn eval_empty_vector() {
        assert_eq!(eval_display("(vec)"), "[]");
    }

    // ===== Literal Tests =====

    #[test]
    fn eval_integer() {
        assert_eq!(eval_display("42"), "42");
    }

    #[test]
    fn eval_string() {
        assert_eq!(eval_display(r#""hello""#), r#""hello""#);
    }

    #[test]
    fn eval_vector_literal() {
        assert_eq!(eval_display("[1 2 3]"), "[1 2 3]");
    }

    // ===== Complex Expressions =====

    #[test]
    fn eval_nested_lists() {
        assert_eq!(eval_display("(list (+ 1 2) (* 3 4) (- 10 5))"), "(3 12 5)");
    }

    #[test]
    fn eval_mixed_operations() {
        assert_eq!(eval_display("(+ (* 2 3) (/ 8 2) (- 10 5))"), "15");
    }

    // ===== Error Tests =====

    #[test]
    fn eval_division_by_zero() {
        let result = eval_str("(/ 10 0)");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Division by zero"));
    }

    #[test]
    fn eval_undefined_variable() {
        let result = eval_str("undefined_var");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Undefined variable"));
    }

    #[test]
    fn eval_type_error_addition() {
        let result = eval_str(r#"(+ 1 "string")"#);
        assert!(result.is_err());
    }

    #[test]
    fn eval_empty_list_call() {
        let result = eval_str("()");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty list"));
    }

    #[test]
    fn eval_not_a_function() {
        let result = eval_str("(42 1 2)");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not a function"));
    }

    // ===== Display Format Tests =====

    #[test]
    fn display_builtin_function() {
        let env = Environment::global();
        let plus = env.get("+").unwrap();
        assert_eq!(plus.to_string(), "#<builtin:+>");
    }

    #[test]
    fn display_nested_structure() {
        let result = eval_str("(list 1 (vec 2 3) (list 4 5))").unwrap();
        assert_eq!(result.to_string(), "(1 [2 3] (4 5))");
    }

    // ===== Comparison Tests =====

    #[test]
    fn eval_eq_true() {
        assert_eq!(eval_display("(= 5 5)"), "true");
    }

    #[test]
    fn eval_eq_false() {
        assert_eq!(eval_display("(= 5 6)"), "false");
    }

    #[test]
    fn eval_eq_multiple_true() {
        assert_eq!(eval_display("(= 5 5 5 5)"), "true");
    }

    #[test]
    fn eval_eq_multiple_false() {
        assert_eq!(eval_display("(= 5 5 6 5)"), "false");
    }

    #[test]
    fn eval_ne_true() {
        assert_eq!(eval_display("(!= 5 6)"), "true");
    }

    #[test]
    fn eval_ne_false() {
        assert_eq!(eval_display("(!= 5 5)"), "false");
    }

    #[test]
    fn eval_lt_true() {
        assert_eq!(eval_display("(< 3 5)"), "true");
    }

    #[test]
    fn eval_lt_false() {
        assert_eq!(eval_display("(< 5 3)"), "false");
    }

    #[test]
    fn eval_lt_equal() {
        assert_eq!(eval_display("(< 5 5)"), "false");
    }

    #[test]
    fn eval_gt_true() {
        assert_eq!(eval_display("(> 5 3)"), "true");
    }

    #[test]
    fn eval_gt_false() {
        assert_eq!(eval_display("(> 3 5)"), "false");
    }

    #[test]
    fn eval_gt_equal() {
        assert_eq!(eval_display("(> 5 5)"), "false");
    }

    #[test]
    fn eval_le_true_less() {
        assert_eq!(eval_display("(<= 3 5)"), "true");
    }

    #[test]
    fn eval_le_true_equal() {
        assert_eq!(eval_display("(<= 5 5)"), "true");
    }

    #[test]
    fn eval_le_false() {
        assert_eq!(eval_display("(<= 5 3)"), "false");
    }

    #[test]
    fn eval_ge_true_greater() {
        assert_eq!(eval_display("(>= 5 3)"), "true");
    }

    #[test]
    fn eval_ge_true_equal() {
        assert_eq!(eval_display("(>= 5 5)"), "true");
    }

    #[test]
    fn eval_ge_false() {
        assert_eq!(eval_display("(>= 3 5)"), "false");
    }

    #[test]
    fn eval_comparison_with_expressions() {
        assert_eq!(eval_display("(< (+ 2 3) (* 2 4))"), "true");
    }

    #[test]
    fn eval_comparison_nested() {
        assert_eq!(eval_display("(= (+ 1 2) (- 5 2))"), "true");
    }

    // ===== Bool Literal Tests =====

    #[test]
    fn eval_bool_literal() {
        let result = eval_str("(= 5 5)").unwrap();
        match result {
            SExpr::Bool(true) => (),
            _ => panic!("Expected Bool(true), got {:?}", result),
        }
    }

    #[test]
    fn display_bool_true() {
        let b = SExpr::Bool(true);
        assert_eq!(b.to_string(), "true");
    }

    #[test]
    fn display_bool_false() {
        let b = SExpr::Bool(false);
        assert_eq!(b.to_string(), "false");
    }

    // ===== Comparison Error Tests =====

    #[test]
    fn eval_eq_insufficient_args() {
        let result = eval_str("(= 5)");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires at least 2"));
    }

    #[test]
    fn eval_ne_wrong_arity() {
        let result = eval_str("(!= 5 6 7)");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 2"));
    }

    #[test]
    fn eval_lt_wrong_arity() {
        let result = eval_str("(< 5)");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exactly 2"));
    }

    #[test]
    fn eval_eq_type_error() {
        let result = eval_str(r#"(= 5 "hello")"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expects integers"));
    }

    #[test]
    fn eval_lt_type_error() {
        let result = eval_str(r#"(< 5 "hello")"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expects integers"));
    }
}