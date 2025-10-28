use std::{collections::HashMap, ffi::CStr, os::raw::c_char, slice, sync::atomic};

use inkwell::{builder::Builder, context::Context, execution_engine::{ExecutionEngine, JitFunction}, module::Module, types::{PointerType, StructType}, values::{BasicMetadataValueEnum, FunctionValue, IntValue, PointerValue}, AddressSpace, IntPredicate};
use anyhow::{anyhow, bail, Result};
#[cfg(target_pointer_width = "64")]
use runtime::ConsCellLayout;
use crate::parser::SExpr;
mod runtime;
pub use runtime::garbage_collector::{set_gc_debug_mode, start_gc_monitor};
use runtime::{alloc_cons_cell, alloc_lisp_val, alloc_string, lisp_val_decref, lisp_val_incref, lisp_val_print_refcount, runtime_get_var, runtime_set_var, LispValLayout, TAG_BOOL, TAG_INT, TAG_LAMBDA, TAG_LIST, TAG_NIL, TAG_STRING};


pub struct CodeGen<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,

    //the boxed type LispVal
    lisp_val_type: StructType<'ctx>,

    //Cons cell type for constructing lisp lists
    cons_cell_type: StructType<'ctx>,

    //handle to ptr_type, because it's used all over the place
    ptr_type: PointerType<'ctx>,

    //necessary Runtime functions (malloc, free, etc.)
    alloc_lisp_val_fn: FunctionValue<'ctx>,
    alloc_cons_cell_fn: FunctionValue<'ctx>,
    alloc_string_fn: FunctionValue<'ctx>,
    incref_fn: FunctionValue<'ctx>,
    decref_fn: FunctionValue<'ctx>,
    print_refcount_fn: FunctionValue<'ctx>,

    memcpy_fn: FunctionValue<'ctx>,

    //variables in current scope, dummy environment
    local_env: HashMap<String, PointerValue<'ctx>>,

    //track function parameters during lambda compilation
    current_function: Option<FunctionValue<'ctx>>,
    // print debug info like LLVM IR
    debug: bool,

    // Runtime symbol table - use raw pointer for stable address
    runtime_env: *mut HashMap<String, *mut LispValLayout>,

}


impl<'ctx> CodeGen<'ctx> {
   
   
    pub fn new(ctx: &'ctx Context, debug_mode: bool) -> Self {
        let module = ctx.create_module("tantalisp_main");
        let builder = ctx.create_builder();

        // Ptr type
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let lisp_val_type = ctx.opaque_struct_type("LispVal");
        lisp_val_type.set_body(&[
            ctx.i8_type().into(), // tag
            ctx.i64_type().into(), // union { i32, bool, ptr(String), ptr(List)}
            ctx.i32_type().into() // reference count
        ], false);

        let cons_cell_type = ctx.opaque_struct_type("ConsCell");
        cons_cell_type.set_body(&[
            ptr_type.into(),
            ptr_type.into()
        ], false);

        //Declare alloc_* fns for LispVal, ConsCell, string etc.
        let alloc_lisp_val_type = ptr_type.fn_type(&[], false);
        let alloc_lisp_val_fn = module.add_function("alloc_lisp_val", alloc_lisp_val_type, None);

        let calloc_cons_cell_type = ptr_type.fn_type(&[], false);
        let alloc_cons_cell_fn = module.add_function("alloc_cons_cell", calloc_cons_cell_type, None);

        let alloc_string_type = ptr_type.fn_type(&[ctx.i64_type().into()], false);
        let alloc_string_fn = module.add_function("alloc_string", alloc_string_type, None);


        // Declare memcpy: (i8* dest, i8* src, i64 size) -> i8*
        // void* memcpy(void* dest, const void* src, size_t n)
        let memcpy_fn_type = ptr_type.fn_type(&[
            ptr_type.into(), 
            ptr_type.into(),
            ctx.i64_type().into() 
        ], false);
        let memcpy = module.add_function("memcpy", memcpy_fn_type, None);

        let incref_type =  ctx.void_type().fn_type(&[ptr_type.into()], false);
        let incref_fn = module.add_function("lisp_val_incref", incref_type, None);

        let decref_type = ctx.void_type().fn_type(&[ptr_type.into()], false);
        let decref_fn = module.add_function("lisp_val_decref", decref_type, None);

        let print_refcount_type = ctx.void_type().fn_type(&[ptr_type.into()], false);
        let print_refcount_fn = module.add_function("lisp_val_print_refcount", print_refcount_type, None);


        let mut c =Self {
            ctx,
            module,
            builder,
            ptr_type,
            lisp_val_type,
            cons_cell_type,
            alloc_lisp_val_fn,
            alloc_cons_cell_fn,
            alloc_string_fn,
            incref_fn,
            decref_fn,
            print_refcount_fn,
            memcpy_fn: memcpy,
            local_env: HashMap::new(),
            current_function: None,
            runtime_env: Box::into_raw(Box::new(HashMap::new())),
            debug: debug_mode
        };
        c.declare_runtime_functions();

        c
    }

    // compile_and_run version for REPL, doesn't create fresh execution engine each time,
    // thus allows defining global vars and re-using them in subsequent executions
    pub fn repl_compile(&mut self, exprs: &[SExpr]) -> Result<String> {
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());

        //Generate unique function name for each REAPL line
        static COUNTER: atomic::AtomicUsize = 
            atomic::AtomicUsize::new(0);

        let repl_line_fn_name = format!("repl_eval_fn_{}", COUNTER.fetch_add(1, atomic::Ordering::SeqCst));
        let repl_fn_type = ptr_type.fn_type(&[], false);
        let repl_fn = self.module.add_function(&repl_line_fn_name, repl_fn_type, None);
        let entry_bb = self.ctx.append_basic_block(repl_fn, "repl_entry");

        self.builder.position_at_end(entry_bb);
        self.current_function = Some(repl_fn);

        //Clear local environment 
        self.local_env.clear();
        
        // Compile exprs
        let compiled_repl_exprs: Result<Vec<_>> = exprs.iter().map(|e| self.emit_expr(e)).collect();
        let compiled_exprs = compiled_repl_exprs?;
        let result_ptr = compiled_exprs.last().ok_or(anyhow!("empty body"))?;

        self.builder.build_return(Some(result_ptr))?;

        if !repl_fn.verify(true) {
            bail!("Function verification failed");
        }


        if self.debug {
            // print LLVM IR 
            println!("-------- LLVM IR ---------");
            self.module.print_to_stderr();
            println!("--------------------------");
        }

        let engine = self.create_execution_engine_for_repl()?;

        unsafe {
            type MainFunc = unsafe extern "C" fn() -> *mut LispValLayout;
            //get handle to the main function
            let jit_function: JitFunction<MainFunc> = engine.get_function(&repl_line_fn_name)?;

            //call it
            let lisp_val_ptr = jit_function.call();
            if lisp_val_ptr.is_null() {
                bail!("JIT returned null pointer");
            }
            lisp_val_to_string(lisp_val_ptr)

        }
     
    }

    pub fn compile_and_run(&mut self, exprs: &[SExpr])  -> Result<String> {
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());
        let main_fn_type = ptr_type.fn_type(&[], false);  // Returns ptr
        let main_fn_name = "main";
        let main_fn = self.module.add_function(main_fn_name, main_fn_type, None);
        let entry_bb = self.ctx.append_basic_block(main_fn, "main_entry");
        // set main function as current and entry_bb
        self.builder.position_at_end(entry_bb);
        self.current_function = Some(main_fn);

        // compile main body

        let compiled_body: Result<Vec<_>> = exprs.iter().map(|e| self.emit_expr(e)).collect();
        let binding = compiled_body?;
        let result_ptr = binding.last().ok_or(anyhow!("empty body after compilation"))?;

        // return the pointer to LispVal
        self.builder.build_return(Some(result_ptr))?;

        if self.debug {
            // print LLVM IR 
            println!("-------- LLVM IR ---------");
            self.module.print_to_stderr();
            println!("--------------------------");
        }


        let engine = self.create_execution_engine()?;


        
        unsafe {
            type MainFunc = unsafe extern "C" fn() -> *mut LispValLayout;
            //get handle to the main function
            let jit_function: JitFunction<MainFunc> = engine.get_function(main_fn_name)?;

            //call it
            let lisp_val_ptr = jit_function.call();
            if lisp_val_ptr.is_null() {
                bail!("JIT returned null pointer");
            }
            lisp_val_to_string(lisp_val_ptr)
        }
    }


    // TODO = move this to CodeGen::new so we can store handled to FunctionValues in self
    fn declare_runtime_functions(&mut self) {
        let ptr_type = self.ptr_type;
        
        //declare fn: ptr runtime_get_var(ptr env, ptr name)
        let get_var_type = ptr_type.fn_type(
            &[
                ptr_type.into(), 
                ptr_type.into()], false);

        self.module.add_function("runtime_get_var", get_var_type, None);

        // declare: void runtime_set_var(ptr env, ptr name, ptr value)
        let set_var_type = ptr_type.fn_type(&[
            ptr_type.into(),
            ptr_type.into(),
            ptr_type.into()
        ], false);

        self.module.add_function("runtime_set_var", set_var_type, None);
    }

    fn emit_incref(&mut self, val: PointerValue<'ctx>) -> Result<()> {
        self.builder.build_call(self.incref_fn, &[val.into()], "incref_call")?;
        Ok(())
    }

    fn emit_decref(&mut self, val: PointerValue<'ctx>) -> Result<()> {
        self.builder.build_call(self.decref_fn, &[val.into()], "incref_call")?;
        Ok(())
    }

    fn emit_print_refcount(&mut self, val: PointerValue<'ctx>) -> Result<()> {
        self.builder.build_call(self.print_refcount_fn, &[val.into()], "print_refcount_call")?;
        Ok(())
    }



    fn create_execution_engine(&mut self) -> Result<ExecutionEngine<'ctx>> {
        self.module.create_jit_execution_engine(inkwell::OptimizationLevel::Aggressive)
                .map_err(|e| anyhow!("Failed to create JIT execution engine: {}", e))
    }

    fn create_execution_engine_for_repl(&mut self) -> Result<ExecutionEngine<'ctx>> {

             // Clone the module so the original isn't consumed
        let module_clone = self.module.clone();

        // Create execution engine from the clone
        let  engine = module_clone.create_jit_execution_engine(inkwell::OptimizationLevel::Aggressive)
            .map_err(|e| anyhow!("Failed to create JIT execution engine: {}", e))?;

        // Register our runtime helper functions with the JIT
        // This tells the JIT where to find these symbols at runtime

        // Get the LLVM function declarations
        let get_var_fn = module_clone.get_function("runtime_get_var")
            .ok_or(anyhow!("runtime_get_var not found in module"))?;
        let set_var_fn = module_clone.get_function("runtime_set_var")
            .ok_or(anyhow!("runtime_set_var not found in module"))?;

        // Map them to the actual Rust function addresses
        engine.add_global_mapping(
        &get_var_fn,
        runtime_get_var as usize
        );

        engine.add_global_mapping(
            &set_var_fn,
            runtime_set_var as usize
        );  

        engine.add_global_mapping(
            &self.incref_fn,
            lisp_val_incref as usize
            );

        engine.add_global_mapping(
            &self.decref_fn,
            lisp_val_decref as usize
        );

        // Map allocation functions
        engine.add_global_mapping(
            &self.alloc_lisp_val_fn,
            alloc_lisp_val as usize
        );

        engine.add_global_mapping(
            &self.alloc_cons_cell_fn,
            alloc_cons_cell as usize
        );

        engine.add_global_mapping(
            &self.alloc_string_fn,
            alloc_string as usize
        );

        engine.add_global_mapping(
            &self.print_refcount_fn,
            lisp_val_print_refcount as usize
        );

        Ok(engine)
    }

    fn emit_expr(&mut self, expr: &SExpr) -> Result<PointerValue<'ctx>> {
        match expr {
            SExpr::Int(i) =>  self.box_int(self.ctx.i32_type().const_int(*i as u64, true)),
            SExpr::Bool(b) => self.box_bool(
                        if *b {
                         self.ctx.bool_type().const_int(1, true)
                     } else { 
                        self.ctx.bool_type().const_zero()
                    }),
            SExpr::String(str) => self.box_string(str),
            SExpr::Symbol(id) => self.emit_var_lookup(id),
            SExpr::DefExpr(id, val_expr) => self.emit_def(id, val_expr),
            SExpr::IfExpr(pred_expr, truthy_exprs, falsy_exprs) => {
                        self.emit_if(pred_expr, truthy_exprs, falsy_exprs)
                    },
            SExpr::LambdaExpr(params, body) => { 
                        let lambda_val = self.emit_lambda(params, body)?;
                        self.box_lambda(lambda_val)
                    },
            SExpr::List(xs) if xs.is_empty() => self.box_nil(),
            SExpr::List(sexprs) if CodeGen::is_builtin_proc(&sexprs) => self.emit_builtin_proc_call(&sexprs),
            SExpr::List(sexprs) => self.emit_call(&sexprs),
            SExpr::Vector(sexprs) => todo!(),
            SExpr::BuiltinFn(_, _) => todo!(),
            SExpr::Quoted(sexpr) => {
                // Special case: '() should emit nil, not a list containing nil
                match sexpr.as_ref() {
                    SExpr::List(xs) if xs.is_empty() => self.box_nil(),
                    _ => self.emit_list(slice::from_ref(sexpr.as_ref()))
                }
            },
        }
    }

fn emit_var_lookup(&mut self, id: &String) -> std::result::Result<PointerValue<'ctx>, anyhow::Error> {
        // first check in local env
        if let Some(&val) = self.local_env.get(id) {
            self.emit_incref(val)?;
            return Ok(val)
        }
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());
        // create a pointer to Rust's self.runtime_env
        let env_addr = self.runtime_env as u64;
        let env_ptr = self.builder.build_int_to_ptr(
            self.ctx.i64_type().const_int(env_addr, false), 
            ptr_type,
                "env_ptr")?;
        //create C string for id (var_name)
        let name_str_val = self.ctx.const_string(id.as_bytes(), true);
        let name_global = self.module.add_global(name_str_val.get_type(), 
            None,
        &format!("varname_{}", id) );
        name_global.set_initializer(&name_str_val);
        // Finally call runtime_get_var 
        let get_var_fn = self.module.get_function("runtime_get_var")
        .ok_or(anyhow!("cannot find `runtime_get_var` function"))?;
        // TODO - need constatns for those name
        let call_site  =      self.builder.build_call(    
        get_var_fn,     
        &[
            env_ptr.into(),
            name_global.as_pointer_value().into()
        ], "get_var_call")?;
        let result = call_site.try_as_basic_value()
            .left()
            .ok_or(anyhow!("runtime_get_var didn't return a value for var_name: {}", id))?
            .into_pointer_value();
        // TODO - build null check 
        let is_null = self.builder.build_is_null(result, "is_get_var_result_null")?;


        self.emit_incref(result)?;
        Ok(result)
    }

fn emit_def(&mut self, id: &String, val_expr: &Box<SExpr>) -> std::result::Result<PointerValue<'ctx>, anyhow::Error> {
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());
        let val_ptr = self.emit_expr(&*val_expr)?;
        // create a pointer to Rust's self.runtime_env
        let env_addr = self.runtime_env as u64;
        let env_ptr = self.builder.build_int_to_ptr(
            self.ctx.i64_type().const_int(env_addr, false), 
            ptr_type,
             "env_ptr")?;
        //create C string for id (var_name)
        let name_str_val = self.ctx.const_string(id.as_bytes(), true);
        let name_global = self.module.add_global(name_str_val.get_type(), 
        None,
    &format!("varname_{}", id) );
        name_global.set_initializer(&name_str_val);
        // Finally call runtime_ser_var 
        let set_var_fn = self.module.get_function("runtime_set_var")
            .ok_or(anyhow!("cannot find `runtime_set_var` function"))?;
        // TODO - need constatns for those name
        self.builder.build_call(
            set_var_fn, 
            &[
                env_ptr.into(),
                name_global.as_pointer_value().into(),
                val_ptr.into()
            ], "set_var_vall")?;
        self.emit_incref(val_ptr)?;
        Ok(val_ptr)
    }

fn emit_if(&mut self, pred_expr: &Box<SExpr>, truthy_exprs: &Vec<SExpr>, falsy_exprs: &Vec<SExpr>) -> std::result::Result<PointerValue<'ctx>, anyhow::Error> {
        let current_function = self.current_function
            .ok_or(anyhow!("Cannot codegen IfExpr with out current_function!"))?;

        //eval predicate to LispVal pointer
        let pred_value_ptr = self.emit_expr(&*pred_expr)?;
        // convert to truthy/falsy for conditional branching
        let pred_val_bool = self.to_truthy(pred_value_ptr)?;
              
        //no longer need pred_value_ptr
        self.emit_decref(pred_value_ptr)?;

        // create BBs for branches
        let truthy_block = self.ctx.append_basic_block(current_function, "truthy_branch");
        let falsy_block = self.ctx.append_basic_block(current_function, "falsy_branch");
        let merge_block = self.ctx.append_basic_block(current_function, "merge");
    
        self.builder.build_conditional_branch(pred_val_bool, truthy_block, falsy_block)?;
    
        // then branch
        self.builder.position_at_end(truthy_block);
    
        let truthy_vals: Result<Vec<PointerValue<'ctx>>> =    truthy_exprs
                            .iter()
                            .map(|e| self.emit_expr(e))
                            .collect();
        let truthy_val = truthy_vals?;
        let truthy_result = truthy_val.last().ok_or(anyhow!("Empty IFExpr then block!"))?;
    
        self.builder.build_unconditional_branch(merge_block)?;
        let truthy_bb_end = self.builder.get_insert_block().unwrap();
    
        // else branch             
        self.builder.position_at_end(falsy_block);
        let falsy_vals: Result<Vec<PointerValue<'ctx>>> =    falsy_exprs
                                                            .iter()
                                                            .map(|e| self.emit_expr(e))
                                                            .collect();
        let falsy_val = falsy_vals?;
        let falsy_result = falsy_val.last().ok_or(anyhow!("Empty IfExpr else block"))?;
        self.builder.build_unconditional_branch(merge_block)?;
        let falsy_bb_end = self.builder.get_insert_block().unwrap();
    
    
        // merge with Phi node
        self.builder.position_at_end(merge_block);
        let phi_node = self.builder.build_phi(self.ctx.ptr_type(AddressSpace::default()), "if_phi_node")?;
    
        phi_node.add_incoming(&[(truthy_result, truthy_bb_end), (falsy_result, falsy_bb_end)]);
    
        let cond_result = phi_node.as_basic_value().into_pointer_value();
    
        Ok(cond_result)
    }


    fn call_alloc_fn(&self, alloc_fn: FunctionValue<'ctx>, args: &[BasicMetadataValueEnum<'ctx>]) -> Result<PointerValue<'ctx>> {
        let ptr = self.builder
            .build_call(alloc_fn,  args, "call_alloc_fn")?.try_as_basic_value()
            .left()
            .ok_or(anyhow!("Failed to call call_alloc_fn" ))?;
    
        let val_ptr = self.builder
            .build_pointer_cast(ptr.into_pointer_value(), 
                self.ptr_type, 
                "cast_to_ptr")?;
    
         Ok(val_ptr)   
    }
    
    //Allocate a LispVal on the heap and return a pointer to it
    fn alloc_lisp_val(&self) -> Result<PointerValue<'ctx>> {
        let val_ptr = self.call_alloc_fn(self.alloc_lisp_val_fn, &[])?;
    
         Ok(val_ptr)   

    }

    fn alloc_cons_cell(&self) -> Result<PointerValue<'ctx>>  { 
        let val_ptr = self.call_alloc_fn(self.alloc_cons_cell_fn, &[])?;
    
        Ok(val_ptr)   
    }

    fn alloc_string(&self, size: IntValue<'ctx>) -> Result<PointerValue<'ctx>>  { 
        let val_ptr = self.call_alloc_fn(self.alloc_string_fn, &[size.into()])?;
    
        Ok(val_ptr)   
    }

    fn box_int(&mut self, value: IntValue<'ctx>) -> Result<PointerValue<'ctx>> {
        // allocate new value on the heap
        let new_lisp_val_ptr = self.alloc_lisp_val()?;
        // set BOOL tag
        self.lisp_val_set_tag(new_lisp_val_ptr, TAG_INT)?;

        //but first need to extend i32 to i64, as that's the type of data field 
        // build_int_s_extend - Extends by copying the sign bit (the most significant bit) into all the new high-order bits. This is for signed integers.
        let i64_value = self.builder.build_int_s_extend(value, self.ctx.i64_type(), "extend")?;
        self.lisp_val_set_int_data(new_lisp_val_ptr, i64_value)?;

        Ok(new_lisp_val_ptr)
    }

    fn box_bool(&mut self, value: IntValue<'ctx>) -> Result<PointerValue<'ctx>> {
        // allocate new value on the heap
        let new_lisp_val_ptr = self.alloc_lisp_val()?;
        // set BOOL tag
        self.lisp_val_set_tag(new_lisp_val_ptr, TAG_BOOL)?;

        //same for box_int
        //but first need to extend i32 to i64, as that's the type of data field 
        // build_int_z_extend:        Extends by filling the new high-order bits with zeros. This is for unsigned integers.
        let i64_value = self.builder.build_int_z_extend(value, self.ctx.i64_type(), "extend")?;
        self.lisp_val_set_int_data(new_lisp_val_ptr, i64_value)?;


        Ok(new_lisp_val_ptr)
    }

    fn box_string(&mut self, value: &str) -> Result<PointerValue<'ctx>> {
        let new_lisp_val_ptr = self.alloc_lisp_val()?;

        // write STRING_TAG 
        self.lisp_val_set_tag(new_lisp_val_ptr, TAG_STRING)?;


        //allocate space for the string itself (null-terminated )
        let str_len = value.len() + 1; //+1 for null
        let str_size = self.ctx.i64_type().const_int(str_len as u64, false);

        let str_ptr = self.alloc_string(str_size)?;

        // Create a global constant for the source string, as it will come from literal values in source code (list "a" )
        // This creates a read-only string in the data section
        let src_str = self.builder.build_global_string_ptr(value, "str_literal")?;

        //Copy the string data using memcpy
        //memcpy(dest, src, size)
        self.builder.build_call(self.memcpy_fn, 
            &[
            str_ptr.into(),
            src_str.as_pointer_value().into(),
            str_size.into() 
        ], "memcpy_call")?;
        

        // set actual data
        self.lisp_val_set_ptr_data(new_lisp_val_ptr, str_ptr)?;

        Ok(new_lisp_val_ptr)

    }

    fn box_nil(&mut self) -> Result<PointerValue<'ctx>> {
        let new_lisp_val_ptr = self.alloc_lisp_val()?;
        // write NIL_TAG 
        self.lisp_val_set_tag(new_lisp_val_ptr, TAG_NIL)?;

        Ok(new_lisp_val_ptr)
    }

    fn cons_set_head(&mut self, cons_cell: PointerValue<'ctx>, value: PointerValue<'ctx> ) -> Result<()> {
        let head_ptr = self.builder
            .build_struct_gep(self.cons_cell_type, cons_cell, 0, "cons_head_ptr")?;
        self.builder.build_store(head_ptr, value)?;

        Ok(())
    }

    fn cons_set_tail(&mut self, cons_cell: PointerValue<'ctx>, value: PointerValue<'ctx>) -> Result<()> {
        let head_ptr = self.builder
        .build_struct_gep(self.cons_cell_type, cons_cell, 1, "cons_head_ptr")?;
        self.builder.build_store(head_ptr, value)?;

        Ok(())
    }

    fn cons_into_lisp_val(&mut self, value: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        let lisp_val = self.alloc_lisp_val()?;
        // write LIST_TAG 
        self.lisp_val_set_tag(lisp_val, TAG_LIST)?;

        let data_ptr = self.builder
            .build_struct_gep(self.lisp_val_type, lisp_val, 1, "data_ptr")?;
        self.builder.build_store(data_ptr, value)?;

        Ok(lisp_val)
    }

    fn box_list(&mut self, exprs: &[PointerValue<'ctx>]) -> Result<PointerValue<'ctx>> {
        // Build list from right to left so tails can be LispVals
        // (1 2 3) = cons(1, cons(2, cons(3, nil)))

        let mut tail = self.box_nil()?;  // Start with nil LispVal

        // Iterate backwards through elements
        for e in exprs.iter().rev() {
            let cons_cell = self.alloc_cons_cell()?;
            self.cons_set_head(cons_cell, *e)?;
            self.cons_set_tail(cons_cell, tail)?;  // tail is a LispVal

            // Wrap this cons cell in a LispVal for the next iteration
            tail = self.cons_into_lisp_val(cons_cell)?;
        }

        Ok(tail)  // The final tail is the complete list
    }

    fn box_lambda(&mut self, func: FunctionValue<'ctx>) -> Result<PointerValue<'ctx>> {
        let lisp_val_ptr = self.alloc_lisp_val()?;
  
        // Set tag to 2 (function)
        let tag_ptr = self.builder.build_struct_gep(
            self.lisp_val_type,
            lisp_val_ptr,
            0,
            "tag_ptr"
        )?;
        self.builder.build_store(tag_ptr, self.ctx.i8_type().const_int(TAG_LAMBDA as u64, false))?;
  
        // Convert function pointer to i64 and store in data field
        let func_ptr = func.as_global_value().as_pointer_value();
        let func_as_int = self.builder.build_ptr_to_int(
            func_ptr,
            self.ctx.i64_type(),
            "func_to_int"
        )?;
  
        let data_ptr = self.builder.build_struct_gep(
            self.lisp_val_type,
            lisp_val_ptr,
            1,
            "data_ptr"
        )?;
        self.builder.build_store(data_ptr, func_as_int)?;
  
        Ok(lisp_val_ptr)
    }

    // Box a string from a runtime i8* pointer and length
    fn box_string_from_ptr(
        &mut self,
        src_ptr: PointerValue<'ctx>,
        length: IntValue<'ctx>,
    ) -> anyhow::Result<PointerValue<'ctx>> {
        let lisp_val_ptr = self.alloc_lisp_val()?;
        
        // Allocate length + 1 for null terminator
        let one = self.ctx.i64_type().const_int(1, false);
        let size_with_null = self.builder.build_int_add(length, one, "size_with_null")?;
        
        let dest_ptr = self.builder.build_call(
            self.alloc_lisp_val_fn,
            &[size_with_null.into()],
            "malloc_str"
        )?.try_as_basic_value()
            .left()
            .unwrap()
            .into_pointer_value();
        
        // Copy the string data
        self.builder.build_call(
            self.memcpy_fn,
            &[
                dest_ptr.into(),
                src_ptr.into(),
                length.into(), // Don't copy the null terminator from source yet
            ],
            "memcpy_call"
        )?;
        
        // Manually add null terminator
        let null_pos_ptr = unsafe {
            self.builder.build_gep(
                self.ctx.i8_type(),
                dest_ptr,
                &[length],
                "null_pos"
            )?
        };
        let null_byte = self.ctx.i8_type().const_int(0, false);
        self.builder.build_store(null_pos_ptr, null_byte)?;
        
        // Set tag
        let tag_ptr = self.builder.build_struct_gep(
            self.lisp_val_type,
            lisp_val_ptr,
            0,
            "tag_ptr"
        )?;
        let tag = self.ctx.i8_type().const_int(TAG_STRING as u64, false);
        self.builder.build_store(tag_ptr, tag)?;
        
        // Set data
        let data_ptr = self.builder.build_struct_gep(
            self.lisp_val_type,
            lisp_val_ptr,
            1,
            "data_ptr"
        )?;
        let ptr_as_int = self.builder.build_ptr_to_int(
            dest_ptr,
            self.ctx.i64_type(),
            "ptr_to_int"
        )?;
        self.builder.build_store(data_ptr, ptr_as_int)?;
        
        Ok(lisp_val_ptr)
    }

    // UNBOXING from LispVal

    fn lisp_val_tag(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<IntValue<'ctx>> {
        let tag_ptr = self.builder.build_struct_gep(
            self.lisp_val_type,
            lisp_val_ptr,
            0,
            "tag_ptr"
        )?;

        let tag = self.builder.build_load(
            self.ctx.i8_type(),
            tag_ptr, "load_tag")?;


        Ok(tag.into_int_value())
    }

    fn unbox_int(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<IntValue<'ctx>>  {
        self.lisp_val_check_tag(lisp_val_ptr, TAG_INT)?;
        let int_val = self.lisp_val_data(lisp_val_ptr)?;
        // we still need to truncate i64 to i32
        let value = self.builder.build_int_truncate(int_val, self.ctx.i32_type(), "trunc_i64_i32")?;

        Ok(value)
    }

    fn unbox_bool(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<IntValue<'ctx>>  {
        self.lisp_val_check_tag(lisp_val_ptr, TAG_BOOL)?;
        let int_val = self.lisp_val_data(lisp_val_ptr)?;

        // we still need to truncate i64 to i32
        let value = self.builder.build_int_truncate(int_val, self.ctx.bool_type(), "trunc_i64_i8")?;

        Ok(value)
    }

    /// Convert any LispVal to a boolean for use in conditionals
    /// Falsy values: nil (tag 4) and :false (tag 1 with data=0)
    /// Truthy values: everything else (ints, non-empty strings, lists, lambdas, :true)
    fn to_truthy(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<IntValue<'ctx>> {
        let tag = self.lisp_val_tag(lisp_val_ptr)?;

        // Check if tag == TAG_NIL (4)
        let is_nil = self.builder.build_int_compare(
            IntPredicate::EQ,
            tag,
            self.ctx.i8_type().const_int(TAG_NIL as u64, false),
            "is_nil"
        )?;

        // Check if tag == TAG_BOOL (1) AND data == 0
        let is_bool = self.builder.build_int_compare(
            IntPredicate::EQ,
            tag,
            self.ctx.i8_type().const_int(TAG_BOOL as u64, false),
            "is_bool_tag"
        )?;

        let data = self.lisp_val_data(lisp_val_ptr)?;
        let is_zero = self.builder.build_int_compare(
            IntPredicate::EQ,
            data,
            self.ctx.i64_type().const_int(0, false),
            "is_zero_data"
        )?;

        let is_false_bool = self.builder.build_and(is_bool, is_zero, "is_false_bool")?;

        // Falsy if: is_nil OR is_false_bool
        let is_falsy = self.builder.build_or(is_nil, is_false_bool, "is_falsy")?;

        // Truthy is the opposite of falsy
        let is_truthy = self.builder.build_not(is_falsy, "is_truthy")?;

        Ok(is_truthy)
    }

    /// --- Candidates for a separate type (struct) for LispValue related operations, but need access &mut self

    fn lisp_val_data(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<IntValue<'ctx>> {
        let int_ptr = self.builder.build_struct_gep(self.lisp_val_type, lisp_val_ptr, 1, "data_ptr")?;
        let data = self.builder.build_load(self.ctx.i64_type(), int_ptr, "load_string_ptr")?
        .into_int_value();
        Ok(data)
    }

    fn lisp_val_data_as_ptr(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        let data = self.lisp_val_data(lisp_val_ptr)?;
        // need to cast that int to pointer for complex types (String, ConsCell etc)
        let data_ptr = self.builder.build_int_to_ptr(data, self.ctx.ptr_type(AddressSpace::default()), "int_data_to_ptr")?;
        Ok(data_ptr)
    }

    fn lisp_val_check_tag(&mut self, lisp_val_ptr: PointerValue<'ctx>, expected_tag: u8) -> Result<()> {
        let tag = self.lisp_val_tag(lisp_val_ptr)?;
        let expected_tag = self.ctx.i8_type().const_int(expected_tag as u64, false);
        self.builder.build_int_compare(IntPredicate::EQ, tag, expected_tag, "is_correct_tag")?;
        //TODO - branch and throw error, leaving for now
        Ok(())
    }

    fn lisp_val_set_tag(&mut self, lisp_val_ptr: PointerValue<'ctx>, tag: u8) -> Result<()> {
        // get element pointer (GEP) to the first field of the LispVal struct, i.e. the tag in tagged union
        // build_struct_gep computes the address of the tag field.
        let tag_ptr = self.builder.build_struct_gep(self.lisp_val_type, 
            lisp_val_ptr, 0, "tag_ptr")?;
        // tag value    
        let tag = self.ctx.i8_type().const_int(tag as u64, false);
        // write set tag to 0 on new_lisp_val
        self.builder.build_store(tag_ptr, tag)?;
        Ok(())
    }

    fn lisp_val_set_int_data(&mut self, lisp_val: PointerValue<'ctx>, int_val: IntValue<'ctx>) -> Result<()> {
        //same for actual int value
        let data_ptr = self.builder
            .build_struct_gep(self.lisp_val_type, lisp_val, 1, "data_ptr")?;

        self.builder.build_store(data_ptr, int_val)?;

        Ok(())
    }

    fn lisp_val_set_ptr_data(&mut self, lisp_val: PointerValue<'ctx>, value_ptr: PointerValue<'ctx>) -> Result<()> {
        let data_ptr = self.builder.build_struct_gep(
            self.lisp_val_type,
            lisp_val, 1   , "data_ptr")?;

        // cast our value_ptr to i64 so it can be stored in second field
        let ptr_as_int = self
            .builder
            .build_ptr_to_int(value_ptr, self.ctx.i64_type(), "ptr_to_int")?;
        
        self.builder.build_store(data_ptr, ptr_as_int)?;
        Ok(())
    }
    /// ^^ Candidates for a separate type (struct) for LispValue related operations, but need access &mut self


    fn unbox_string(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        self.lisp_val_check_tag(lisp_val_ptr, TAG_STRING)?;

        let value_ptr = self.lisp_val_data_as_ptr(lisp_val_ptr)?;

        Ok(value_ptr)
    }

    fn unbox_list(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        self.lisp_val_check_tag(lisp_val_ptr, TAG_LIST)?;
        let value_ptr = self.lisp_val_data_as_ptr(lisp_val_ptr)?;
        Ok(value_ptr)

    }

    fn unbox_lambda(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        self.lisp_val_check_tag(lisp_val_ptr, TAG_LAMBDA)?;

        let value_ptr = self.lisp_val_data_as_ptr(lisp_val_ptr)?;

        Ok(value_ptr)
    }

    fn emit_call(&mut self, items:&[SExpr]) -> Result<PointerValue<'ctx>> {
        let func = &items[0];
        let args = &items[1..];

        let func = self.emit_expr(func)?;
        let func_ptr = self.unbox_lambda(func)?;
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());

        let arg_exprs: Vec<_> = args.iter().map(|a| self.emit_expr(a)).collect::<Result<Vec<_>>>()?;

        let args: Vec<_> = arg_exprs.iter().map(|a| (*a).into()).collect();   
        let param_types:Vec<_> = args
            .iter()
            .map(|_| ptr_type.into())
            .collect();

        let fn_type = ptr_type.fn_type(&param_types, false);

        let call_res = self.builder
            .build_indirect_call(fn_type, func_ptr, &args, "lambda_call")
            .map_err(|e| anyhow!("error making indirect_call: {}", e))?;

        let res = call_res
            .try_as_basic_value()
            .left()
            .ok_or(anyhow!("error converting indirect_call CallSite to PointerValue"))?;

        //can decref on args and lambda
        self.emit_decref(func)?;
        for arg in arg_exprs.into_iter() {
            self.emit_decref(arg)?;
        }

        Ok(res.into_pointer_value())

    }

    fn is_builtin_proc(args:&[SExpr]) -> bool {
        let func = &args.get(0);
        if let Some(SExpr::Symbol(op)) = func {
            match op.as_str() {
                "+" | "-" | "*" | "/" | "=" | "!=" | "<" | ">" | "<=" | ">=" => true,
                "list" | "cons" | "head" | "car" | "cdr" | "tail"  => true,
                _ => false
            }
        } else {
            false
        }
    }

    fn emit_builtin_proc_call(&mut self, args:&[SExpr]) -> Result<PointerValue<'ctx>> {
        let func = &args[0];
        let args = &args[1..];
        if let SExpr::Symbol(op) = func {
            match op.as_str() {
                "+" =>  self.emit_add(args),
                "*" =>  self.emit_mul(args),
                "-" =>  self.emit_sub(args),
                "/" =>  self.emit_div(args),
                "=" =>  self.emit_eq(args),
                "!=" =>  self.emit_ne(args),
                "<" =>  self.emit_lt(args),
                ">" =>  self.emit_gt(args),
                "<=" =>  self.emit_le(args),
                ">=" =>  self.emit_ge(args),
                "list" => self.emit_list(args),
                "head" | "car" => self.emit_car(args),
                "tail" | "cdr" => self.emit_cdr(args),
                "cons"  => self.emit_cons(args),
                _ => bail!("Unsupported builtin proc provided, unknown op: {}", op)
            }
        } else {
            bail!("Invalid builtin proc call, expected Symbol as first element, got: {:?}", args)
        }

        
    }

    fn emit_lambda(&mut self, params: &[SExpr], body: &[SExpr]) -> Result<FunctionValue<'ctx>> {
        let param_names: Result<Vec<String>> = params.iter().map(|p|  match p {
            SExpr::Symbol(id) => Ok(id.clone()),
            e => bail!("Only symbols are allowed in function parameters, found: {}", e)
        }).collect();
        let param_names = param_names?;


        // create function type: (ptr, ptr, ..) -> ptr
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());

        let param_types:Vec<_> = param_names
                            .iter()
                            .map(|_| ptr_type.into())
                            .collect();

        let fn_type = ptr_type.fn_type(&param_types, false);
        
        // create function in current module
        // TODO - should generate unique names for lambdas
        let fn_name = "lambda_324";
        let new_lambda = self.module.add_function(fn_name, fn_type, None);
        let entry_bb = self.ctx.append_basic_block(new_lambda, &format!("{}_body", fn_name));


        // save current CodeGen state
        let current_env = self.local_env.clone();
        let current_fn = self.current_function;

        //also save current insertion point
        let current_block = self.builder.get_insert_block().ok_or(anyhow!("Empty current_block"))?;

        // position builder at start of the function
        self.builder.position_at_end(entry_bb);


        //we're setting current_function to newly created lambda
        // so that all body expr compile in context of that function, not the previous fn
        self.current_function = Some(new_lambda);

        // create new_lamda environment
        self.local_env.clear();
        for (i, param_name) in param_names.iter().enumerate() {
            let param_value = new_lambda
                        .get_nth_param(i as u32)
                        .ok_or(anyhow!("Param no: {} not found on llvm function", i))?;
            let param_value = param_value.into_pointer_value();
            self.local_env.insert(param_name.clone(), param_value);
        }

        let body_value: Result<Vec<PointerValue>> = body.iter().map(|e| self.emit_expr(e)).collect();

        let body_value = body_value?;
        let result = body_value.last().ok_or(anyhow!("Empty body value"))?;


        self.builder.build_return(Some(result))?;
        // restore previous environment and function
        self.local_env = current_env;
        self.current_function = current_fn;
        self.builder.position_at_end(current_block);


        if new_lambda.verify(true) {
            Ok(new_lambda)
        } else {
            bail!("Function verification failed")
        }
    }
    
    fn emit_list(&mut self, sexpr: &[SExpr]) -> Result<PointerValue<'ctx>> {
        let elements: Result<Vec<PointerValue<'ctx>>> =  sexpr.iter()
            .map(|e| self.emit_expr(e))
            .collect();

        let elements = elements?;
        self.box_list(&elements)
    }
    
    fn emit_car(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {

        match args {
            [lst] => {
                let list_res = self.emit_expr(lst)?;
                let cons_cell_ptr = self.unbox_list(list_res)?;
                let head_ptr = self.emit_list_car(cons_cell_ptr)?;
                self.emit_incref(head_ptr)?;  // incref head since we're returning it
                self.emit_decref(list_res)?;   // decref input list
                Ok(head_ptr)

            },
            _ => bail!("head expects single argument!")
        }
    }

    fn emit_cdr(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {

        match args {
            [lst] => {
                let list_res = self.emit_expr(lst)?;
                let cons_cell_ptr = self.unbox_list(list_res)?;
                let tail_ptr = self.emit_list_cdr(cons_cell_ptr)?;
                // Tail is already a LispVal, incref it since we're returning it
                self.emit_incref(tail_ptr)?;
                // Decref input list - this is now safe! The tail has its own refcount.
                self.emit_decref(list_res)?;
                Ok(tail_ptr)

            },
            _ => bail!("tail expects single argument!")
        }
    }
    
    fn emit_list_car(&mut self, cons_cell_ptr: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());
        let head_ptr = self.builder.build_struct_gep(self.cons_cell_type, cons_cell_ptr, 0, "cons_cell_head_ptr")?;
        let head_load = self.builder.build_load(ptr_type, 
        head_ptr, "cons_cell_load_head")?;

        Ok(head_load.into_pointer_value())
    }

    fn emit_list_cdr(&mut self, cons_cell_ptr: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());

        // Tail is now a LispVal (either another list or nil), just load and return it!
        let tail_ptr = self.builder
            .build_struct_gep(self.cons_cell_type, cons_cell_ptr, 1, "cons_cell_tail_ptr")?;
        let tail_lisp_val = self.builder.build_load(ptr_type,
            tail_ptr, "cons_cell_load_tail")?;

        Ok(tail_lisp_val.into_pointer_value())
    }
    
    fn emit_cons(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        match args {
            [x, xs] => {
                let new_elem_expr = self.emit_expr(x)?;
                let tail_expr = self.emit_expr(xs)?;

                // Since tail is now a LispVal, we can just use tail_expr directly (nil or list)
                let new_cons = self.alloc_cons_cell()?;
                self.cons_set_head(new_cons, new_elem_expr)?;
                self.cons_set_tail(new_cons, tail_expr)?;  // tail_expr is a LispVal (nil or list)

                let result = self.cons_into_lisp_val(new_cons)?;

                // Cons cell now owns references to both new_elem_expr and tail_expr
                // Don't decref them - they're now owned by the cons cell

                Ok(result)
            },
            _ => bail!("cons expects 2 arguments!")
        }
    }
    


}

#[cfg(target_pointer_width = "64")]
fn cons_cell_ptr_from_data_ptr(addr: i64) -> *const ConsCellLayout {
    let addr: usize  = addr as u64 as usize;
    addr as *const ConsCellLayout
}

fn list_val_data_to_str<'a>(addr: i64) -> Option<&'a str> {
    let addr: usize = addr as u64 as usize;
    let data_ptr = addr as *const c_char; 
    if data_ptr.is_null() {
        None
    } else {
        let str = unsafe { CStr::from_ptr(data_ptr).to_str().ok()? };
        Some(str)
    }

}

fn lisp_val_to_string(lisp_val_ptr: *mut LispValLayout) -> Result<String> {
    let lisp_val = unsafe { &*lisp_val_ptr };
    match lisp_val.tag {
        0 => Ok(format!("{}", lisp_val.data as i32)), // Int
        1 => {
            // Bool - display as :true or :false
            if lisp_val.data != 0 {
                Ok(":true".to_string())
            } else {
                Ok(":false".to_string())
            }
        },
        2 => {
            let str_ptr = list_val_data_to_str(lisp_val.data).ok_or(anyhow!("Failed to read String out of lisp_val"))?;
            Ok(str_ptr.to_string())
        },
        3 => {
            // TODO write a Rust Iterator impl for this !!
            let cons_cell = cons_cell_ptr_from_data_ptr(lisp_val.data);
            let mut cons_cell_maybe = unsafe { cons_cell.as_ref()};
            let mut list_contents = vec![];
            while let Some(cons_cell_ref) = cons_cell_maybe {

                let current_str = lisp_val_to_string(cons_cell_ref.head)?;
                list_contents.push(current_str);

                // Tail is now a LispVal, check if it's another list or nil
                let tail_lisp_val = unsafe { cons_cell_ref.tail.as_ref() };
                if let Some(tail_val) = tail_lisp_val {
                    if tail_val.tag == TAG_LIST {
                        // Extract the cons cell from the tail LispVal
                        let next_cons_cell = cons_cell_ptr_from_data_ptr(tail_val.data);
                        cons_cell_maybe = unsafe { next_cons_cell.as_ref() };
                    } else {
                        // Tail is nil or something else, stop iteration
                        cons_cell_maybe = None;
                    }
                } else {
                    cons_cell_maybe = None;
                }
            }

            Ok(format!("({})", list_contents.join(" ")))
        }, 
        4 => Ok("nil".to_string()),
        5 => Ok("#<lambda>".to_string()), // Lambda

        _ => bail!("Returning other LispVal types not implemented yet, found: {}", lisp_val.tag)
    }
}

impl<'ctx> Drop for CodeGen<'ctx> {
    fn drop(&mut self) {
        //Clean up the leaked runtime_env
        unsafe {
            let _ = Box::from_raw(self.runtime_env);
        }
    }
}

// tmp built ins
impl<'ctx> CodeGen<'ctx> {
    fn emit_add(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("+ requires exactly 2 arguments");
        }
        
        let arg0 = self.emit_expr(&args[0])?;
        let arg1 = self.emit_expr(&args[1])?;

        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;

        let result = self.builder.build_int_add(int0, int1, "builtin_addint")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_int(result)
    }

    fn emit_sub(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("- requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;


        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;


        let result = self.builder.build_int_sub(int0, int1, "builtin_subint")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;

        self.box_int(result)
    }


    fn emit_mul(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("* requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;


        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;


        let result = self.builder.build_int_mul(int0, int1, "builtin_mulint")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_int(result)
    }


    fn emit_div(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("/ requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;

        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;


        let result = self.builder.build_int_signed_div(int0, int1, "builtin_divint")?;

        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_int(result)
    }

        
    fn emit_eq(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("= requires exactly 2 arguments");
        }
        let arg0 = self.emit_expr(&args[0])?;
        let arg1 = self.emit_expr(&args[1])?;

        // TODO - how about non-int values?
        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;

        let comp_result = self.builder.build_int_compare(IntPredicate::EQ, int0, int1, "int_eq")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_bool(comp_result)
    }

    fn emit_ne(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("!= requires exactly 2 arguments");
        }
        let arg0 = self.emit_expr(&args[0])?;
        let arg1 = self.emit_expr(&args[1])?;

        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;

        let comp_result = self.builder.build_int_compare(IntPredicate::NE, int0, int1, "int_ne")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_bool(comp_result)
    }

    fn emit_lt(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("< requires exactly 2 arguments");
        }
        let arg0 = self.emit_expr(&args[0])?;
        let arg1 = self.emit_expr(&args[1])?;

        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;

        let comp_result = self.builder.build_int_compare(IntPredicate::SLT, int0, int1, "int_lt")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_bool(comp_result)
    }

    fn emit_gt(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("> requires exactly 2 arguments");
        }
        let arg0 = self.emit_expr(&args[0])?;
        let arg1 = self.emit_expr(&args[1])?;

        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;

        let comp_result = self.builder.build_int_compare(IntPredicate::SGT, int0, int1, "int_gt")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_bool(comp_result)
    }

    fn emit_le(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("<= requires exactly 2 arguments");
        }
        let arg0 = self.emit_expr(&args[0])?;
        let arg1 = self.emit_expr(&args[1])?;

        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;

        let comp_result = self.builder.build_int_compare(IntPredicate::SLE, int0, int1, "int_le")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_bool(comp_result)
    }

    fn emit_ge(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!(">= requires exactly 2 arguments");
        }
        let arg0 = self.emit_expr(&args[0])?;
        let arg1 = self.emit_expr(&args[1])?;

        let int0 = self.unbox_int(arg0)?;
        let int1 = self.unbox_int(arg1)?;

        let comp_result = self.builder.build_int_compare(IntPredicate::SGE, int0, int1, "int_ge")?;
        self.emit_decref(arg0)?;
        self.emit_decref(arg1)?;
        self.box_bool(comp_result)
    }
}


#[cfg(test)]
mod codegen_tests {
    use super::*;

    // Set to true to enable LLVM IR debug output for all tests
    const DEBUG_MODE: bool = false;

    #[test]
    fn test_list_fn() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        let expr = SExpr::List(vec![
            SExpr::Symbol("list".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);

        let result = compiler.repl_compile(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "(1 2)");
    }

    #[test]
    fn test_nested_list_fn() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        let expr = SExpr::List(vec![
            SExpr::Symbol("list".to_string()),
            SExpr::Int(11),
            SExpr::List(vec![
                SExpr::Symbol("list".to_string()),
                SExpr::Int(1),
                SExpr::Int(2)
            ])
        ]);

        let result = compiler.repl_compile(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "(11 (1 2))");
    }


    #[test]
    fn test_list_head() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        let list_expr = SExpr::List(vec![
            SExpr::Symbol("list".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);
        let expr = SExpr::List(vec![SExpr::Symbol("head".to_string()), list_expr.clone()]);


        let result = compiler.repl_compile(&[expr]);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "1");

        let expr = SExpr::List(vec![SExpr::Symbol("car".to_string()), list_expr]);


        let result = compiler.repl_compile(&[expr]);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "1");
    }

    #[test]
    fn test_list_tail() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        let list_expr = SExpr::List(vec![
            SExpr::Symbol("list".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);
        let expr = SExpr::List(vec![SExpr::Symbol("tail".to_string()), list_expr.clone()]);


        let result = compiler.repl_compile(&[expr]);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "(2)");

        let expr = SExpr::List(vec![SExpr::Symbol("cdr".to_string()), list_expr]);


        let result = compiler.repl_compile(&[expr]);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "(2)");
    }

    #[test]
    fn test_list_cons() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        let list_expr = SExpr::List(vec![
            SExpr::Symbol("list".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);
        let expr = SExpr::List(vec![SExpr::Symbol("cons".to_string()), SExpr::Int(47), list_expr]);


        let result = compiler.repl_compile(&[expr]);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "(47 1 2)");
    }

    #[test]
    fn test_cons_onto_nil() {
        // Test: (cons 1 '()) should return (1), not (1 nil)
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        // Test 1: (cons 1 '()) -> (1)
        let nil_expr = SExpr::List(vec![]);
        let expr = SExpr::List(vec![
            SExpr::Symbol("cons".to_string()),
            SExpr::Int(1),
            nil_expr
        ]);

        let result = compiler.repl_compile(&[expr]);
        if let Err(e) = &result {
            eprintln!("Error: {:?}", e);
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "(1)", "(cons 1 '()) should produce (1), not (1 nil)");

        // Test 2: (cons 1 (cons 2 '())) -> (1 2)
        let nil_expr2 = SExpr::List(vec![]);
        let inner_cons = SExpr::List(vec![
            SExpr::Symbol("cons".to_string()),
            SExpr::Int(2),
            nil_expr2
        ]);
        let expr2 = SExpr::List(vec![
            SExpr::Symbol("cons".to_string()),
            SExpr::Int(1),
            inner_cons
        ]);

        let result2 = compiler.repl_compile(&[expr2]);
        assert!(result2.is_ok(), "Failed to evaluate (cons 1 (cons 2 '())): {:?}", result2.err());
        assert_eq!(result2.unwrap(), "(1 2)", "(cons 1 (cons 2 '())) should produce (1 2)");

        // Test 3: (cons 1 (list 2 3)) -> (1 2 3)
        let list_expr = SExpr::List(vec![
            SExpr::Symbol("list".to_string()),
            SExpr::Int(2),
            SExpr::Int(3)
        ]);
        let expr3 = SExpr::List(vec![
            SExpr::Symbol("cons".to_string()),
            SExpr::Int(1),
            list_expr
        ]);

        let result3 = compiler.repl_compile(&[expr3]);
        assert!(result3.is_ok(), "Failed to evaluate (cons 1 (list 2 3)): {:?}", result3.err());
        assert_eq!(result3.unwrap(), "(1 2 3)", "(cons 1 (list 2 3)) should produce (1 2 3)");

        // Test 4: '() -> nil
        let quoted_nil = SExpr::Quoted(Box::new(SExpr::List(vec![])));
        let result4 = compiler.repl_compile(&[quoted_nil]);
        assert!(result4.is_ok(), "Failed to evaluate '(): {:?}", result4.err());
        assert_eq!(result4.unwrap(), "nil", "'() should produce nil");
    }

    #[test]
    fn test_scalar_int_expr() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        let expr = SExpr::Int(42);

        let result = compiler.compile_and_run(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "42");
    }

    #[test]
    fn test_bool_expr() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        // Test false - in your runtime, bools might return as 0/1
        let expr = SExpr::Bool(false);
        let result = compiler.compile_and_run(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ":false");

        // Test true
        let mut compiler2 = CodeGen::new(&ctx, DEBUG_MODE);
        let expr = SExpr::Bool(true);
        let result = compiler2.compile_and_run(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ":true");
    }

    #[test]
    fn test_integer_math() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);

        let expr = SExpr::List(
            vec![SExpr::Symbol("+".to_string()), SExpr::Int(41), SExpr::Int(1)]
        );

        let result = compiler.compile_and_run(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "42");
    }

    #[test]
    #[ignore]
    fn test_quoted_empty_list() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, DEBUG_MODE);
        let res = compiler.repl_compile(&[SExpr::Quoted(Box::new(SExpr::List(vec![])))]);

        assert!(res.is_ok());
         
    }

    #[test]
    fn test_jit_integer_math() {
        // Create context
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);
        
        // Test 1: Simple arithmetic
        // (+ 40 2)
        let expr1 = SExpr::List(vec![
            SExpr::Symbol("+".to_string()),
            SExpr::Int(40),
            SExpr::Int(2),
        ]);
        
        match codegen.compile_and_run(&[expr1]) {
            Ok(result) => assert_eq!("42", result),  // Should print 42
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
        let mut compiler2 = CodeGen::new(&context, DEBUG_MODE);
        let result =  compiler2.compile_and_run(&[expr2]).unwrap();
        assert_eq!("42", result); 
    }

    #[test]
    fn test_repl_global_variables() {
        // Test that global variables persist across REPL evaluations
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        // First REPL line: (def x 42)
        let def_expr = SExpr::DefExpr(
            "x".to_string(),
            Box::new(SExpr::Int(42))
        );

        let result1 = codegen.repl_compile(&[def_expr]);
        assert!(result1.is_ok(), "Failed to define x: {:?}", result1.err());
        assert_eq!(result1.unwrap(), "42");

        // Second REPL line: x (should return 42)
        let read_expr = SExpr::Symbol("x".to_string());

        let result2 = codegen.repl_compile(&[read_expr]);
        assert!(result2.is_ok(), "Failed to read x: {:?}", result2.err());
        assert_eq!(result2.unwrap(), "42");
    }

    #[test]
    fn test_eq_operator() {
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        // Test: (= 1 1) should return true (1)
        let eq_expr = SExpr::List(vec![
            SExpr::Symbol("=".to_string()),
            SExpr::Int(1),
            SExpr::Int(1)
        ]);

        let result = codegen.repl_compile(&[eq_expr]);
        assert!(result.is_ok(), "Failed to evaluate (= 1 1): {:?}", result.err());
        assert_eq!(result.unwrap(), ":true", "(= 1 1) should return 1 (true)");

        // Test: (= 1 2) should return false (0)
        let mut codegen2 = CodeGen::new(&context, DEBUG_MODE);
        let ne_expr = SExpr::List(vec![
            SExpr::Symbol("=".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);

        let result2 = codegen2.repl_compile(&[ne_expr]);
        assert!(result2.is_ok(), "Failed to evaluate (= 1 2): {:?}", result2.err());
        assert_eq!(result2.unwrap(), ":false", "(= 1 2) should return 0 (false)");
    }

    #[test]
    fn test_if_true_branch() {
        // Test: (if (= 1 1) 42 1) should return 42
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        let if_expr = SExpr::IfExpr(
            Box::new(SExpr::List(vec![
                SExpr::Symbol("=".to_string()),
                SExpr::Int(1),
                SExpr::Int(1)
            ])),
            vec![SExpr::Int(42)],
            vec![SExpr::Int(1)]
        );

        let result = codegen.repl_compile(&[if_expr]);
        assert!(result.is_ok(), "Failed to evaluate if: {:?}", result.err());
        assert_eq!(result.unwrap(), "42", "(if (= 1 1) 42 1) should return 42");
    }

    #[test]
    fn test_if_false_branch() {
        // Test: (if (= 3 1) 1 42) should return 42
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        let if_expr = SExpr::IfExpr(
            Box::new(SExpr::List(vec![
                SExpr::Symbol("=".to_string()),
                SExpr::Int(3),
                SExpr::Int(1)
            ])),
            vec![SExpr::Int(1)],
            vec![SExpr::Int(42)]
        );

        let result = codegen.repl_compile(&[if_expr]);
        assert!(result.is_ok(), "Failed to evaluate if: {:?}", result.err());
        assert_eq!(result.unwrap(), "42", "(if (= 3 1) 1 42) should return 42");
    }

    #[test]
    fn test_ne_operator() {
        let context = Context::create();

        // Test: (!= 1 2) should return true (1)
        let mut codegen1 = CodeGen::new(&context, DEBUG_MODE);
        let expr1 = SExpr::List(vec![
            SExpr::Symbol("!=".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);
        let result1 = codegen1.repl_compile(&[expr1]);
        assert!(result1.is_ok(), "Failed to evaluate (!= 1 2): {:?}", result1.err());
        assert_eq!(result1.unwrap(), ":true", "(!= 1 2) should return 1 (true)");

        // Test: (!= 1 1) should return false (0)
        let mut codegen2 = CodeGen::new(&context, DEBUG_MODE);
        let expr2 = SExpr::List(vec![
            SExpr::Symbol("!=".to_string()),
            SExpr::Int(1),
            SExpr::Int(1)
        ]);
        let result2 = codegen2.repl_compile(&[expr2]);
        assert!(result2.is_ok(), "Failed to evaluate (!= 1 1): {:?}", result2.err());
        assert_eq!(result2.unwrap(), ":false", "(!= 1 1) should return 0 (false)");
    }

    #[test]
    fn test_lt_operator() {
        let context = Context::create();

        // Test: (< 1 2) should return true (1)
        let mut codegen1 = CodeGen::new(&context, DEBUG_MODE);
        let expr1 = SExpr::List(vec![
            SExpr::Symbol("<".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);
        let result1 = codegen1.repl_compile(&[expr1]);
        assert!(result1.is_ok(), "Failed to evaluate (< 1 2): {:?}", result1.err());
        assert_eq!(result1.unwrap(), ":true", "(< 1 2) should return 1 (true)");

        // Test: (< 2 1) should return false (0)
        let mut codegen2 = CodeGen::new(&context, DEBUG_MODE);
        let expr2 = SExpr::List(vec![
            SExpr::Symbol("<".to_string()),
            SExpr::Int(2),
            SExpr::Int(1)
        ]);
        let result2 = codegen2.repl_compile(&[expr2]);
        assert!(result2.is_ok(), "Failed to evaluate (< 2 1): {:?}", result2.err());
        assert_eq!(result2.unwrap(), ":false", "(< 2 1) should return 0 (false)");
    }

    #[test]
    fn test_gt_operator() {
        let context = Context::create();

        // Test: (> 2 1) should return true (1)
        let mut codegen1 = CodeGen::new(&context, DEBUG_MODE);
        let expr1 = SExpr::List(vec![
            SExpr::Symbol(">".to_string()),
            SExpr::Int(2),
            SExpr::Int(1)
        ]);
        let result1 = codegen1.repl_compile(&[expr1]);
        assert!(result1.is_ok(), "Failed to evaluate (> 2 1): {:?}", result1.err());
        assert_eq!(result1.unwrap(), ":true", "(> 2 1) should return 1 (true)");

        // Test: (> 1 2) should return false (0)
        let mut codegen2 = CodeGen::new(&context, DEBUG_MODE);
        let expr2 = SExpr::List(vec![
            SExpr::Symbol(">".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);
        let result2 = codegen2.repl_compile(&[expr2]);
        assert!(result2.is_ok(), "Failed to evaluate (> 1 2): {:?}", result2.err());
        assert_eq!(result2.unwrap(), ":false", "(> 1 2) should return 0 (false)");
    }

    #[test]
    fn test_le_operator() {
        let context = Context::create();

        // Test: (<= 1 2) should return true (1)
        let mut codegen1 = CodeGen::new(&context, DEBUG_MODE);
        let expr1 = SExpr::List(vec![
            SExpr::Symbol("<=".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);
        let result1 = codegen1.repl_compile(&[expr1]);
        assert!(result1.is_ok(), "Failed to evaluate (<= 1 2): {:?}", result1.err());
        assert_eq!(result1.unwrap(), ":true", "(<= 1 2) should return 1 (true)");

        // Test: (<= 2 2) should return true (1)
        let mut codegen2 = CodeGen::new(&context, DEBUG_MODE);
        let expr2 = SExpr::List(vec![
            SExpr::Symbol("<=".to_string()),
            SExpr::Int(2),
            SExpr::Int(2)
        ]);
        let result2 = codegen2.repl_compile(&[expr2]);
        assert!(result2.is_ok(), "Failed to evaluate (<= 2 2): {:?}", result2.err());
        assert_eq!(result2.unwrap(), ":true", "(<= 2 2) should return 1 (true)");

        // Test: (<= 2 1) should return false (0)
        let mut codegen3 = CodeGen::new(&context, DEBUG_MODE);
        let expr3 = SExpr::List(vec![
            SExpr::Symbol("<=".to_string()),
            SExpr::Int(2),
            SExpr::Int(1)
        ]);
        let result3 = codegen3.repl_compile(&[expr3]);
        assert!(result3.is_ok(), "Failed to evaluate (<= 2 1): {:?}", result3.err());
        assert_eq!(result3.unwrap(), ":false", "(<= 2 1) should return 0 (false)");
    }

    #[test]
    fn test_ge_operator() {
        let context = Context::create();

        // Test: (>= 2 1) should return true (1)
        let mut codegen1 = CodeGen::new(&context, DEBUG_MODE);
        let expr1 = SExpr::List(vec![
            SExpr::Symbol(">=".to_string()),
            SExpr::Int(2),
            SExpr::Int(1)
        ]);
        let result1 = codegen1.repl_compile(&[expr1]);
        assert!(result1.is_ok(), "Failed to evaluate (>= 2 1): {:?}", result1.err());
        assert_eq!(result1.unwrap(), ":true", "(>= 2 1) should return 1 (true)");

        // Test: (>= 2 2) should return true (1)
        let mut codegen2 = CodeGen::new(&context, DEBUG_MODE);
        let expr2 = SExpr::List(vec![
            SExpr::Symbol(">=".to_string()),
            SExpr::Int(2),
            SExpr::Int(2)
        ]);
        let result2 = codegen2.repl_compile(&[expr2]);
        assert!(result2.is_ok(), "Failed to evaluate (>= 2 2): {:?}", result2.err());
        assert_eq!(result2.unwrap(), ":true", "(>= 2 2) should return 1 (true)");

        // Test: (>= 1 2) should return false (0)
        let mut codegen3 = CodeGen::new(&context, DEBUG_MODE);
        let expr3 = SExpr::List(vec![
            SExpr::Symbol(">=".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);
        let result3 = codegen3.repl_compile(&[expr3]);
        assert!(result3.is_ok(), "Failed to evaluate (>= 1 2): {:?}", result3.err());
        assert_eq!(result3.unwrap(), ":false", "(>= 1 2) should return 0 (false)");
    }

    #[test]
    fn test_repl_lambda_with_globals() {
        // Test: (def x 40) (def f (fn [x] (* x 10))) (f (+ 2 x))
        // Expected: 420
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        // Line 1: (def x 40)
        let def_x = SExpr::DefExpr(
            "x".to_string(),
            Box::new(SExpr::Int(40))
        );
        // Line 2: (def f (fn [x] (* x 10)))
        let lambda = SExpr::LambdaExpr(
            vec![SExpr::Symbol("x".to_string())],
            vec![SExpr::List(vec![
                SExpr::Symbol("*".to_string()),
                SExpr::Symbol("x".to_string()),
                SExpr::Int(10),
            ])]
        );
        let def_f = SExpr::DefExpr(
            "f".to_string(),
            Box::new(lambda)
        );

                // Line 3: (f (+ 2 x))
        // This is: (f (+ 2 40)) = (f 42) = (* 42 10) = 420
        let call_expr = SExpr::List(vec![
            SExpr::Symbol("f".to_string()),
            SExpr::List(vec![
                SExpr::Symbol("+".to_string()),
                SExpr::Int(2),
                SExpr::Symbol("x".to_string()),
            ])
        ]);

        let result1 = codegen.repl_compile(&[def_x, def_f, call_expr]);
        assert!(result1.is_ok(), "Failed to define x: {:?}", result1.err());
        assert_eq!(result1.unwrap(), "420");

    }

    #[test]
    fn test_fibonacci_recursive() {
        // Test: (def fib (fn [x] (if (<= x 1) 1 (+ (fib (- x 1)) (fib (- x 2))))))
        // Then test fib(1), fib(2), fib(5), fib(10)
        // Expected: 1, 2, 8, 89
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        // Define the fibonacci function
        // (def fib (fn [x] (if (<= x 1) 1 (+ (fib (- x 1)) (fib (- x 2))))))
        let fib_lambda = SExpr::LambdaExpr(
            vec![SExpr::Symbol("x".to_string())],
            vec![SExpr::IfExpr(
                Box::new(SExpr::List(vec![
                    SExpr::Symbol("<=".to_string()),
                    SExpr::Symbol("x".to_string()),
                    SExpr::Int(1),
                ])),
                vec![SExpr::Int(1)],
                vec![SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::List(vec![
                        SExpr::Symbol("fib".to_string()),
                        SExpr::List(vec![
                            SExpr::Symbol("-".to_string()),
                            SExpr::Symbol("x".to_string()),
                            SExpr::Int(1),
                        ])
                    ]),
                    SExpr::List(vec![
                        SExpr::Symbol("fib".to_string()),
                        SExpr::List(vec![
                            SExpr::Symbol("-".to_string()),
                            SExpr::Symbol("x".to_string()),
                            SExpr::Int(2),
                        ])
                    ]),
                ])]
            )]
        );

        let def_fib = SExpr::DefExpr(
            "fib".to_string(),
            Box::new(fib_lambda)
        );

        // Define fib first
        let result = codegen.repl_compile(&[def_fib]);
        assert!(result.is_ok(), "Failed to define fib: {:?}", result.err());

        // Test fib(1) = 1
        let call_fib_1 = SExpr::List(vec![
            SExpr::Symbol("fib".to_string()),
            SExpr::Int(1),
        ]);
        let result1 = codegen.repl_compile(&[call_fib_1]);
        assert!(result1.is_ok(), "Failed to evaluate fib(1): {:?}", result1.err());
        assert_eq!(result1.unwrap(), "1", "fib(1) should return 1");

        // Test fib(2) = 2
        let call_fib_2 = SExpr::List(vec![
            SExpr::Symbol("fib".to_string()),
            SExpr::Int(2),
        ]);
        let result2 = codegen.repl_compile(&[call_fib_2]);
        assert!(result2.is_ok(), "Failed to evaluate fib(2): {:?}", result2.err());
        assert_eq!(result2.unwrap(), "2", "fib(2) should return 2");

        // Test fib(5) = 8
        let call_fib_5 = SExpr::List(vec![
            SExpr::Symbol("fib".to_string()),
            SExpr::Int(5),
        ]);
        let result5 = codegen.repl_compile(&[call_fib_5]);
        assert!(result5.is_ok(), "Failed to evaluate fib(5): {:?}", result5.err());
        assert_eq!(result5.unwrap(), "8", "fib(5) should return 8");

        // Test fib(10) = 89
        let call_fib_10 = SExpr::List(vec![
            SExpr::Symbol("fib".to_string()),
            SExpr::Int(10),
        ]);
        let result10 = codegen.repl_compile(&[call_fib_10]);
        assert!(result10.is_ok(), "Failed to evaluate fib(10): {:?}", result10.err());
        assert_eq!(result10.unwrap(), "89", "fib(10) should return 89");
    }

    #[test]
    fn test_truthy_nil_is_falsy() {
        // Test: (if '() 1 2) should return 2 (nil/empty list is falsy)
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        // Empty list represents nil
        let expr = SExpr::IfExpr(
            Box::new(SExpr::List(vec![])),
            vec![SExpr::Int(1)],
            vec![SExpr::Int(2)]
        );

        let result = codegen.repl_compile(&[expr]);
        assert!(result.is_ok(), "Failed to evaluate (if '() 1 2): {:?}", result.err());
        assert_eq!(result.unwrap(), "2", "(if '() 1 2) should return 2 because nil is falsy");
    }

    #[test]
    fn test_truthy_false_is_falsy() {
        // Test: (if :false 1 2) should return 2 (:false is falsy)
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        let expr = SExpr::IfExpr(
            Box::new(SExpr::Bool(false)),
            vec![SExpr::Int(1)],
            vec![SExpr::Int(2)]
        );

        let result = codegen.repl_compile(&[expr]);
        assert!(result.is_ok(), "Failed to evaluate (if :false 1 2): {:?}", result.err());
        assert_eq!(result.unwrap(), "2", "(if :false 1 2) should return 2 because :false is falsy");
    }

    #[test]
    fn test_truthy_true_is_truthy() {
        // Test: (if :true 1 2) should return 1 (:true is truthy)
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        let expr = SExpr::IfExpr(
            Box::new(SExpr::Bool(true)),
            vec![SExpr::Int(1)],
            vec![SExpr::Int(2)]
        );

        let result = codegen.repl_compile(&[expr]);
        assert!(result.is_ok(), "Failed to evaluate (if :true 1 2): {:?}", result.err());
        assert_eq!(result.unwrap(), "1", "(if :true 1 2) should return 1 because :true is truthy");
    }

    #[test]
    fn test_truthy_integer_is_truthy() {
        // Test: (if 0 1 2) should return 1 (even 0 is truthy!)
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        let expr = SExpr::IfExpr(
            Box::new(SExpr::Int(0)),
            vec![SExpr::Int(1)],
            vec![SExpr::Int(2)]
        );

        let result = codegen.repl_compile(&[expr]);
        assert!(result.is_ok(), "Failed to evaluate (if 0 1 2): {:?}", result.err());
        assert_eq!(result.unwrap(), "1", "(if 0 1 2) should return 1 because integers are truthy");

        // Test with non-zero integer
        let mut codegen2 = CodeGen::new(&context, DEBUG_MODE);
        let expr2 = SExpr::IfExpr(
            Box::new(SExpr::Int(42)),
            vec![SExpr::Int(1)],
            vec![SExpr::Int(2)]
        );

        let result2 = codegen2.repl_compile(&[expr2]);
        assert!(result2.is_ok(), "Failed to evaluate (if 42 1 2): {:?}", result2.err());
        assert_eq!(result2.unwrap(), "1", "(if 42 1 2) should return 1 because integers are truthy");
    }

    #[test]
    fn test_truthy_list_is_truthy() {
        // Test: (if (list 1 2) 42 99) should return 42 (lists are truthy, even with elements)
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        let list_expr = SExpr::List(vec![
            SExpr::Symbol("list".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);

        let expr = SExpr::IfExpr(
            Box::new(list_expr),
            vec![SExpr::Int(42)],
            vec![SExpr::Int(99)]
        );

        let result = codegen.repl_compile(&[expr]);
        assert!(result.is_ok(), "Failed to evaluate (if (list 1 2) 42 99): {:?}", result.err());
        assert_eq!(result.unwrap(), "42", "(if (list 1 2) 42 99) should return 42 because lists are truthy");
    }

    #[test]
    fn test_truthy_lambda_is_truthy() {
        // Test: (if (fn [x] x) 1 2) should return 1 (lambdas are truthy)
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        let lambda = SExpr::LambdaExpr(
            vec![SExpr::Symbol("x".to_string())],
            vec![SExpr::Symbol("x".to_string())]
        );

        let expr = SExpr::IfExpr(
            Box::new(lambda),
            vec![SExpr::Int(1)],
            vec![SExpr::Int(2)]
        );

        let result = codegen.repl_compile(&[expr]);
        assert!(result.is_ok(), "Failed to evaluate (if (fn [x] x) 1 2): {:?}", result.err());
        assert_eq!(result.unwrap(), "1", "(if (fn [x] x) 1 2) should return 1 because lambdas are truthy");
    }

    #[test]
    fn test_truthy_comparison_result() {
        // Test: (if (= 1 1) 42 99) should return 42 (comparison returns :true which is truthy)
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, DEBUG_MODE);

        let comparison = SExpr::List(vec![
            SExpr::Symbol("=".to_string()),
            SExpr::Int(1),
            SExpr::Int(1)
        ]);

        let expr = SExpr::IfExpr(
            Box::new(comparison),
            vec![SExpr::Int(42)],
            vec![SExpr::Int(99)]
        );

        let result = codegen.repl_compile(&[expr]);
        assert!(result.is_ok(), "Failed to evaluate (if (= 1 1) 42 99): {:?}", result.err());
        assert_eq!(result.unwrap(), "42", "(if (= 1 1) 42 99) should return 42");

        // Test with false comparison
        let mut codegen2 = CodeGen::new(&context, DEBUG_MODE);
        let comparison2 = SExpr::List(vec![
            SExpr::Symbol("=".to_string()),
            SExpr::Int(1),
            SExpr::Int(2)
        ]);

        let expr2 = SExpr::IfExpr(
            Box::new(comparison2),
            vec![SExpr::Int(42)],
            vec![SExpr::Int(99)]
        );

        let result2 = codegen2.repl_compile(&[expr2]);
        assert!(result2.is_ok(), "Failed to evaluate (if (= 1 2) 42 99): {:?}", result2.err());
        assert_eq!(result2.unwrap(), "99", "(if (= 1 2) 42 99) should return 99");
    }
}
