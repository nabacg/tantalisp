use std::{any, collections::HashMap, result};

use inkwell::{basic_block::BasicBlock, builder::Builder, context::{self, Context}, execution_engine::{self, JitFunction}, module::Module, values::{FunctionValue, IntValue}};
use anyhow::{anyhow, bail, Result};
use crate::parser::SExpr;


pub fn compile_and_run(exprs: &[SExpr]) -> Result<i32> {
    let ctx = Context::create();
    let mut codegen = CodeGen::new(&ctx);

    codegen.compile_and_run(exprs)
}

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,

    //variables in current scope, dummy environment
    environment: HashMap<String, IntValue<'ctx>>,

    //track function parameters during lambda compilation
    current_function: Option<FunctionValue<'ctx>>

}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(ctx: &'ctx Context) -> Self {
        let module = ctx.create_module("tantalisp_main");
        let builder = ctx.create_builder();

        Self {
            context : ctx,
            module,
            builder,
            environment: HashMap::new(),
            current_function: None
        }
    }

    pub fn compile_and_run(&mut self, exprs: &[SExpr])  -> Result<i32> {
        let i32_type = self.context.i32_type();
        let main_fn_type = i32_type.fn_type(&[], false);
        let main_fn_name = "main";
        let main_fn = self.module.add_function(main_fn_name, main_fn_type, None);
        let entry_bb = self.context.append_basic_block(main_fn, "main_entry");
        // set main function as current and entry_bb
        self.builder.position_at_end(entry_bb);
        self.current_function = Some(main_fn);

        // compile main body
        
        let compiled_body: Result<Vec<_>> = exprs.iter().map(|e| self.emit_expr(e)).collect();
        let binding = compiled_body?;
        let result = binding.last().ok_or(anyhow!("empty body after compilation"))?;

        // return result 
        self.builder.build_return(Some(result))?;

        // print LLVM IR 
        println!("-------- LLVM IR ---------");
        self.module.print_to_stderr();
        println!("--------------------------");


        let execution_engine = self.module
            .create_jit_execution_engine(inkwell::OptimizationLevel::None);

        if let Ok(engine) = execution_engine {
            unsafe {
                type MainFunc = unsafe extern "C" fn() -> i32;
                //get handle to the main function
                let jit_function: JitFunction<MainFunc> = engine.get_function(main_fn_name)?;
    
                //call it
                let result = jit_function.call();
                Ok(result)
            }
        }    else {
            bail!("Failed to create_jit_execution_engine")
        }



    }

    fn emit_expr(&mut self, expr: &SExpr) -> Result<IntValue<'ctx>> {
        match expr {
            SExpr::Int(i) => Ok(self.context.i32_type().const_int(*i as u64, true)),
            SExpr::Bool(b) => Ok(if *b {
                self.context.i8_type().const_int(1, true)
             } else { 
                self.context.i8_type().const_zero()
            }),
            SExpr::String(_) => todo!(),
            SExpr::Symbol(id) => self.environment
                                            .get(id)
                                            .copied()
                                            .ok_or_else(|| anyhow::anyhow!("Undefined variable: {}", id)),
            SExpr::IfExpr(pred_expr, truthy_exprs, falsy_exprs) => {
                // let current_function = self.current_function.ok_or_else(anyhow!("Cannot codegen IfExpr with out current_function!"))?;
                // let pred_value = self.codegen_expr(*pred_expr)?;
                // // TODO check if Bool
                // let truthy_vals: Result<Vec<IntValue<'ctx>>> =    truthy_exprs
                // .iter()
                // .map(|e| self.codegen_expr(e))
                // .collect();


                // let falsy_vals: Result<Vec<IntValue<'ctx>>> =    falsy_exprs
                // .iter()
                // .map(|e| self.codegen_expr(e))
                // .collect();

                // let truthy_block = self.context.append_basic_block(current_function, "truthy_branch");

                // let falsy_block = self.context.append_basic_block(current_function, "falsy_branch");

                // let merge_block = self.context.append_basic_block(current_function, "merge");
                // // TODO the PHI node 
                // self.builder.build_conditional_branch(pred_value, truthy_block, falsy_block);

                

                todo!()
            },
            SExpr::LambdaExpr(sexprs, sexprs1) => todo!(),
            SExpr::DefExpr(_, sexpr) => todo!(),
            SExpr::List(sexprs) => self.emit_call(&sexprs),
            SExpr::Vector(sexprs) => todo!(),
            SExpr::BuiltinFn(_, _) => todo!(),
        }
    }

    fn emit_call(&mut self, items:&[SExpr]) -> Result<IntValue<'ctx>> {
        let func = &items[0];
        let args = &items[1..];

        if let SExpr::Symbol(op) = func {
            match op.as_str() {
                "+" => return self.compile_add(args),
                "*" => return self.compile_mul(args),
                "-" => return self.compile_sub(args),
                "/" => return self.compile_div(args),
                _ => {}
            }
        }

        if let SExpr::LambdaExpr(params, body) = func { 
            let llvm_function = self.emit_lambda(params, body);
            let args: Vec<_> = args.iter().map(|a| self.emit_expr(a)).collect::<Result<Vec<_>>>()?;

            // Convert to BasicMetadataValueEnum for call
            let args: Vec<_> = args.iter().map(|a| (*a).into()).collect();   

            let call_site = self.builder.build_call(llvm_function?, &args, "lambda_call")?;

            let result = call_site.try_as_basic_value().left().ok_or(anyhow!("Failed to get lambda_call result"));

            let result = result?.into_int_value();

            return Ok(result)
        }

        bail!("FunctionCalls for lisp functions not implemented yet!")

    }

    fn emit_lambda(&mut self, params: &[SExpr], body: &[SExpr]) -> Result<FunctionValue<'ctx>> {
        let param_names: Result<Vec<String>> = params.iter().map(|p|  match p {
            SExpr::Symbol(id) => Ok(id.clone()),
            e => bail!("Only symbols are allowed in function parameters, found: {}", e)
        }).collect();
        let param_names = param_names?;


        // create function type: (i32, i32, ..) -> i32
        let i32_type = self.context.i32_type();
        let param_types:Vec<_> = param_names
                            .iter()
                            .map(|_| i32_type.into())
                            .collect();

        let fn_type = i32_type.fn_type(&param_types, false);
        
        // create function in current module
        // TODO - should generate unique names for lambdas
        let fn_name = "lambda_324";
        let new_lambda = self.module.add_function(fn_name, fn_type, None);
        let entry_bb = self.context.append_basic_block(new_lambda, &format!("{}_body", fn_name));


        // save current CodeGen state
        let current_env = self.environment.clone();
        let current_fn = self.current_function;

        //also save current insertion point
        let current_block = self.builder.get_insert_block().ok_or(anyhow!("Empty current_block"))?;

        // position builder at start of the function
        self.builder.position_at_end(entry_bb);


        //we're setting current_function to newly created lambda
        // so that all body expr compile in context of that function, not the previous fn
        self.current_function = Some(new_lambda);

        // create new_lamda environment
        self.environment.clear();
        for (i, param_name) in param_names.iter().enumerate() {
            let param_value = new_lambda
                        .get_nth_param(i as u32)
                        .ok_or(anyhow!("Param no: {} not found on llvm function", i))?;
            let param_value = param_value.into_int_value();
            self.environment.insert(param_name.clone(), param_value);
        }

        let body_value: Result<Vec<IntValue>> = body.iter().map(|e| self.emit_expr(e)).collect();

        let body_value = body_value?;
        let result = body_value.last().ok_or(anyhow!("Empty body value"))?;


        self.builder.build_return(Some(result))?;


        // restore previous environment and function
        self.environment = current_env;
        self.current_function = current_fn;
        self.builder.position_at_end(current_block);

        if new_lambda.verify(true) {
            Ok(new_lambda)
        } else {
            bail!("Function verification failed")
        }
    }
    


}

// tmp built ins
impl<'ctx> CodeGen<'ctx> {
    fn compile_add(&mut self, args: &[SExpr]) -> Result<IntValue<'ctx>> {
        if args.len() != 2 {
            bail!("+ requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;

        Ok(self.builder.build_int_add(arg0, arg1, "builtin_addint")?)
    }

    fn compile_sub(&mut self, args: &[SExpr]) -> Result<IntValue<'ctx>> {
        if args.len() != 2 {
            bail!("- requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;

        Ok(self.builder.build_int_sub(arg0, arg1, "builtin_subint")?)
    }


    fn compile_mul(&mut self, args: &[SExpr]) -> Result<IntValue<'ctx>> {
        if args.len() != 2 {
            bail!("* requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;

        Ok(self.builder.build_int_mul(arg0, arg1, "builtin_mulint")?)
    }


    fn compile_div(&mut self, args: &[SExpr]) -> Result<IntValue<'ctx>> {
        if args.len() != 2 {
            bail!("/ requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;

        Ok(self.builder.build_int_signed_div(arg0, arg1, "builtin_divint")?)
    }
}


#[cfg(test)]
mod compiler_tests {
    use super::*;


    #[test]
    fn scalar_int_expr() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx);
        let res = compiler.emit_expr(&SExpr::Int(42));

        assert!(res.is_ok());
        assert_eq!(res.unwrap(), ctx.i32_type().const_int(42, true));
         
    }

    #[test]
    fn scalar_bool_expr() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx);
        let res = compiler.emit_expr(&SExpr::Bool(false));

        assert_eq!(res.unwrap(), ctx.i8_type().const_zero());
        assert_eq!(compiler.emit_expr(&SExpr::Bool(true)).unwrap(), ctx.i8_type().const_int(1, true));
         
    }

    #[test]
    fn integer_math() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx);
        let res = compiler.emit_expr(&SExpr::List(
            vec![SExpr::Symbol("+".to_string()), SExpr::Int(41), SExpr::Int(1)]));
        println!("res={:?}", res);
        assert!(res.is_ok());
    }

    #[test]
    #[ignore]
    fn scalar_conditional_expr() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx);
        let res = compiler.emit_expr(&SExpr::Bool(false));

        assert!(res.is_ok());
         
    }

    #[test]
    fn test_jit() {
        // Create context
        let context = Context::create();
        let mut codegen = CodeGen::new(&context);
        
        // Test 1: Simple arithmetic
        // (+ 40 2)
        let expr1 = SExpr::List(vec![
            SExpr::Symbol("+".to_string()),
            SExpr::Int(40),
            SExpr::Int(2),
        ]);
        
        match codegen.compile_and_run(&[expr1]) {
            Ok(result) => assert_eq!(42, result),  // Should print 42
            Err(e) => eprintln!("Error: {}", e),
        }
        
        // Test 2: Lambda call - ((fn [x y] (+ x y)) 41 1)
        let add_lambda = SExpr::LambdaExpr(
            vec![SExpr::Symbol("x".to_string()), SExpr::Symbol("y".to_string())],
            vec![SExpr::List(vec![
                SExpr::Symbol("+".to_string()),
                SExpr::Symbol("x".to_string()),
                SExpr::Symbol("y".to_string()),
            ])],
        );
        
        let expr2 = SExpr::List(vec![
            add_lambda,
            SExpr::Int(41),
            SExpr::Int(1),
        ]);
        
        // Create fresh compiler for second test
        let mut compiler2 = CodeGen::new(&context);
        match compiler2.compile_and_run(&[expr2]) {
            Ok(result) => assert_eq!(42, result),  // Should print 42
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}