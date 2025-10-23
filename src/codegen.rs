use std::{collections::HashMap, ops::Add, sync::atomic};

use inkwell::{builder::Builder, context::Context, execution_engine::{ExecutionEngine, JitFunction}, module::Module, types::StructType, values::{FunctionValue, GlobalValue, IntValue, PointerValue}, AddressSpace, IntPredicate};
use anyhow::{anyhow, bail, Result};
use crate::parser::SExpr;


pub fn compile_and_run(exprs: &[SExpr], debug_mode: bool) -> Result<i32> {
    let ctx = Context::create();
    let mut codegen = CodeGen::new(&ctx, debug_mode);

    codegen.compile_and_run(exprs)
}

pub fn repl_compile(ctx: &Context, exprs: &[SExpr], debug_mode: bool) -> Result<i32> {
    let mut codegen = CodeGen::new(ctx, debug_mode);

    codegen.repl_compile(exprs)
}

//TODO - move to runtime.rs ?

#[unsafe(no_mangle)]
pub extern "C" fn runtime_get_var(
    env_ptr: *mut std::ffi::c_void,
    name_ptr: *const std::os::raw::c_char
) -> *mut LispValLayout {
    unsafe {
        let env = &*(env_ptr as *mut HashMap<String, *mut LispValLayout>);
        let name = std::ffi::CStr::from_ptr(name_ptr)
            .to_str(); //TODO - check errors?
        if let Ok(name) = name {
            env.get(name).copied().unwrap_or(std::ptr::null_mut())

        } else {
            std::ptr::null_mut()
        }

    }
}

#[unsafe(no_mangle)]
pub extern "C" fn runtime_set_var(
    env_ptr: *mut std::ffi::c_void,
    name_ptr: *const std::os::raw::c_char,
    value_ptr: *mut LispValLayout
) {
    unsafe {
        let env = &mut *(env_ptr as *mut HashMap<String, *mut LispValLayout>);
        let name = std::ffi::CStr::from_ptr(name_ptr)
            .to_str(); //TODO - check errors?
        if let Ok(name) = name {
            env.insert(name.to_string(), value_ptr);

        } 

    }
}

#[repr(C)]
pub struct LispValLayout {
    tag: u8,
    _padding: [u8; 7],  // Explicit padding to align the next field to 8 bytes
    data: i64,
}

pub struct CodeGen<'ctx> {
    ctx: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,

    //the boxed type LispVal
    lisp_val_type: StructType<'ctx>,

    //necessary Runtime functions (malloc, free, etc.)
    malloc_fn: FunctionValue<'ctx>,
    free_fn: FunctionValue<'ctx>,
    memcpy_fn: FunctionValue<'ctx>,

    //variables in current scope, dummy environment
    local_env: HashMap<String, PointerValue<'ctx>>,

    // Track global variables defined in REPL
    global_env: HashMap<String, GlobalValue<'ctx>>,

    //track function parameters during lambda compilation
    current_function: Option<FunctionValue<'ctx>>,
    // print debug info like LLVM IR
    debug: bool,

    // Runtime symbol table - use raw pointer for stable address
    runtime_env: *mut HashMap<String, *mut LispValLayout>,

}

const TAG_INT: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_STRING: u8 = 2;
const TAG_LIST: u8 = 3;
const TAG_NIL: u8 = 4;
const TAG_LAMBDA: u8 = 5;


impl<'ctx> CodeGen<'ctx> {
   
    pub fn new(ctx: &'ctx Context, debug_mode: bool) -> Self {
        let module = ctx.create_module("tantalisp_main");
        let builder = ctx.create_builder();

        // ┌─────────────────┐
        // │ LispVal         │
        // ├─────────────────┤
        // │ tag: u8         │ ← Type discriminator (0=Int, 1=Bool, 2=String, 3=List, etc.)
        // │ data: union {   │
        // │   i32           │
        // │   bool          │
        // │   ptr (String)  │
        // │   ptr (List)    │
        // │ }               │
        // └─────────────────┘

        let lisp_value_type = ctx.opaque_struct_type("LispVal");
        lisp_value_type.set_body(&[
            ctx.i8_type().into(), // tag
            ctx.i64_type().into() // union { i32, bool, ptr(String), ptr(List)}
        ], false);

        // Ptr type
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        //Declare malloc: i64 -> i8*
        let malloc_fn_type = ptr_type.fn_type(&[ctx.i64_type().into()], false);
        let malloc_fn = module.add_function("malloc", malloc_fn_type, None);

        //Declare free: i8 -> void
        let free_fn_type = ptr_type.fn_type(&[ptr_type.into()], false);
        let free_fn = module.add_function("free", free_fn_type, None);

        // Declare memcpy: (i8* dest, i8* src, i64 size) -> i8*
        // void* memcpy(void* dest, const void* src, size_t n)
        let memcpy_fn_type = ptr_type.fn_type(&[
            ptr_type.into(), 
            ptr_type.into(),
            ctx.i64_type().into() 
        ], false);
        let memcpy = module.add_function("memcpy", memcpy_fn_type, None);

        let mut c =Self {
            ctx,
            module,
            builder,
            lisp_val_type: lisp_value_type,
            malloc_fn,
            free_fn,
            memcpy_fn: memcpy,
            local_env: HashMap::new(),
            global_env: HashMap::new(),
            current_function: None,
            runtime_env: Box::into_raw(Box::new(HashMap::new())),
            debug: debug_mode
        };
        c.declare_runtime_functions();

        c
    }

    // compile_and_run version for REPL, doesn't create fresh execution engine each time, 
    // thus allows defining global vars and re-using them in subsequent executions
    pub fn repl_compile(&mut self, exprs: &[SExpr]) -> Result<i32> {
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
            let lisp_val = &*lisp_val_ptr;
            match lisp_val.tag {
                0 => Ok(lisp_val.data as i32), // Int
                1 => Ok(lisp_val.data as i32), // Bool
                _ => bail!("Returning other LispVal types not implemented yet, found: {}", lisp_val.tag)
            }

        }
     


    }


    fn declare_runtime_functions(&mut self) {
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());
        
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

    fn create_execution_engine(&mut self) -> Result<ExecutionEngine<'ctx>> {
        self.module.create_jit_execution_engine(inkwell::OptimizationLevel::None)
                .map_err(|e| anyhow!("Failed to create JIT execution engine: {}", e))
    }

    fn create_execution_engine_for_repl(&mut self) -> Result<ExecutionEngine<'ctx>> {

             // Clone the module so the original isn't consumed
        let module_clone = self.module.clone();

        // Create execution engine from the clone
        let  engine = module_clone.create_jit_execution_engine(inkwell::OptimizationLevel::None)
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

        Ok(engine)
    }

    pub fn compile_and_run(&mut self, exprs: &[SExpr])  -> Result<i32> {
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
            let lisp_val = &*lisp_val_ptr;
            match lisp_val.tag {
                0 => Ok(lisp_val.data as i32), // Int
                1 => Ok(lisp_val.data as i32), // Bool
                _ => bail!("Other LispVal types not implemented yet, found: {}", lisp_val.tag)
            }


        }


    }

    fn emit_expr(&mut self, expr: &SExpr) -> Result<PointerValue<'ctx>> {
        match expr {
            SExpr::Int(i) =>  self.box_int(self.ctx.i32_type().const_int(*i as u64, true)),
            SExpr::Bool(b) => self.box_bool(
                if *b {
                 self.ctx.i8_type().const_int(1, true)
             } else { 
                self.ctx.i8_type().const_zero()
            }),
            SExpr::String(str) => self.box_string(str),
            SExpr::Symbol(id) => { 
                
                // first check in local env
                if let Some(val) = self.local_env.get(id) {
                    return Ok(*val)
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
                .ok_or(anyhow!("cannot find `runtime_get_var` function"))?; // TODO - need constatns for those name

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


                Ok(result)
            },
            SExpr::DefExpr(id, val_expr) => {
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
                    .ok_or(anyhow!("cannot find `runtime_set_var` function"))?; // TODO - need constatns for those name

                self.builder.build_call(
                    set_var_fn, 
                    &[
                        env_ptr.into(),
                        name_global.as_pointer_value().into(),
                        val_ptr.into()
                    ], "set_var_vall")?;
                Ok(val_ptr)
            },
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
            SExpr::LambdaExpr(params, body) => { 
                let lambda_val = self.emit_lambda(params, body)?;
                self.box_lambda(lambda_val)
            },
            SExpr::List(xs) if xs.is_empty() => self.box_nil(),
            SExpr::List(sexprs) => self.emit_call(&sexprs),
            SExpr::Vector(sexprs) => todo!(),
            SExpr::BuiltinFn(_, _) => todo!(),
        }
    }


    //Allocate a LispVal on the heap and return a pointer to it
    fn alloc_lisp_val(&self) -> Result<PointerValue<'ctx>> {
        let size = self.lisp_val_type.size_of().ok_or(anyhow!("Failed to get LispVal size!"))?;

        //Call malloc
        let ptr = self.builder
        .build_call(self.malloc_fn, &[size.into()], "malloc")?.try_as_basic_value()
        .left().ok_or(anyhow!("Failed to call malloc"))?;

        let lisp_val_ptr = self.builder
            .build_pointer_cast(ptr.into_pointer_value(), 
                self.ctx.ptr_type(AddressSpace::default()), 
                "cast_to_lisp_value")?;

         Ok(lisp_val_ptr)       
    }

    fn box_int(&mut self, value: IntValue<'ctx>) -> Result<PointerValue<'ctx>> {
        // allocate new value on the heap
        let new_lisp_val_ptr = self.alloc_lisp_val()?;


        // get element pointer (GEP) to the first field of the LispVal struct, i.e. the tag in tagged union
        // build_struct_gep computes the address of the tag field.
        let tag_ptr = self.builder.build_struct_gep(self.lisp_val_type, new_lisp_val_ptr, 0, "tag_ptr")?;
        // create a IntValue with tag:TAG_INT
        let tag = self.ctx.i8_type().const_int(TAG_INT as u64, false);
        // write set tag to 0 on new_lisp_val
        self.builder.build_store(tag_ptr, tag)?;

        //same for actual int value
        let data_ptr = self.builder.build_struct_gep(self.lisp_val_type, new_lisp_val_ptr, 1, "data_ptr")?;

        //but first need to extend i32 to i64, as that's the type of data field 
        // build_int_s_extend - Extends by copying the sign bit (the most significant bit) into all the new high-order bits. This is for signed integers.
        let i64_value = self.builder.build_int_s_extend(value, self.ctx.i64_type(), "extend")?;

        self.builder.build_store(data_ptr, i64_value)?;

        Ok(new_lisp_val_ptr)
    }

    fn box_bool(&mut self, value: IntValue<'ctx>) -> Result<PointerValue<'ctx>> {
        // allocate new value on the heap
        let new_lisp_val_ptr = self.alloc_lisp_val()?;


        // get element pointer (GEP) to the first field of the LispVal struct, i.e. the tag in tagged union
        // build_struct_gep computes the address of the tag field.
        let tag_ptr = self.builder.build_struct_gep(self.lisp_val_type, new_lisp_val_ptr, 0, "tag_ptr")?;
        // create a IntValue with tag:TAG_INT
        let tag = self.ctx.i8_type().const_int(TAG_BOOL as u64, false);
        // write set tag to 0 on new_lisp_val
        self.builder.build_store(tag_ptr, tag)?;

        //same for actual int value
        let data_ptr = self.builder.build_struct_gep(self.lisp_val_type, new_lisp_val_ptr, 1, "data_ptr")?;

        //but first need to extend i32 to i64, as that's the type of data field 
        // build_int_z_extend:        Extends by filling the new high-order bits with zeros. This is for unsigned integers.
        let i64_value = self.builder.build_int_z_extend(value, self.ctx.i64_type(), "extend")?;

        self.builder.build_store(data_ptr, i64_value)?;

        Ok(new_lisp_val_ptr)
    }

    fn box_string(&mut self, value: &str) -> Result<PointerValue<'ctx>> {
        let new_lisp_val_ptr = self.alloc_lisp_val()?;

        // write STRING_TAG 
        let tag_ptr = self.builder.build_struct_gep(self.lisp_val_type, new_lisp_val_ptr, 0, "tag_ptr")?;
        let tag = self.ctx.i8_type().const_int(TAG_STRING as u64, false);
        self.builder.build_store(tag_ptr, tag)?;

        //allocate space for the string itself (null-terminated )
        let str_len = value.len() + 1; //+1 for null
        let str_size = self.ctx.i64_type().const_int(str_len as u64, false);

        let str_ptr = self.builder
            .build_call(self.malloc_fn, &[str_size.into()], "malloc_str")?
            .try_as_basic_value()
            .left()
            .ok_or(anyhow!("Failed to malloc for string value"))?
            .into_pointer_value();

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
        let data_ptr = self.builder.build_struct_gep(
            self.lisp_val_type,
             new_lisp_val_ptr, 1   , "data_ptr")?;

        // cast our str_ptr to i64 so it can be stored in second field
        let ptr_as_int = self
            .builder
            .build_ptr_to_int(str_ptr, self.ctx.i64_type(), "ptr_to_int")?;
        
        self.builder.build_store(data_ptr, ptr_as_int)?;

        Ok(new_lisp_val_ptr)

    }

    fn box_nil(&mut self) -> Result<PointerValue<'ctx>> {
        let new_lisp_val_ptr = self.alloc_lisp_val()?;

        // write STRING_TAG 
        let tag_ptr = self.builder.build_struct_gep(self.lisp_val_type, new_lisp_val_ptr, 0, "tag_ptr")?;
        let tag = self.ctx.i8_type().const_int(TAG_NIL as u64, false);
        self.builder.build_store(tag_ptr, tag)?;


        Ok(new_lisp_val_ptr)
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
            self.malloc_fn,
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

    fn get_tag(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<IntValue<'ctx>> {
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
        let tag= self.get_tag(lisp_val_ptr)?;
        let expected_tag = self.ctx.i8_type().const_int(TAG_INT as u64, false);
        self.builder.build_int_compare(IntPredicate::EQ, tag, expected_tag, "is_int_tag")?;

        //TODO - branch and throw error, leaving for now

        // assuming it was a right LispVal
        //load it using struct GEP
        let int_ptr = self.builder.build_struct_gep(self.lisp_val_type, lisp_val_ptr, 1, "data_ptr")?;
        let int_val = self.builder.build_load(self.ctx.i64_type(), int_ptr, "load_int_val")?
        .into_int_value();

        // we still need to truncate i64 to i32
        let value = self.builder.build_int_truncate(int_val, self.ctx.i32_type(), "trunc_i64_i32")?;

        Ok(value)
    }

    fn unbox_bool(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<IntValue<'ctx>>  {
        let tag= self.get_tag(lisp_val_ptr)?;
        let expected_tag = self.ctx.i8_type().const_int(TAG_BOOL as u64, false);
        self.builder.build_int_compare(IntPredicate::EQ, tag, expected_tag, "is_int_tag")?;

        //TODO - branch and throw error, leaving for now

        // assuming it was a right LispVal
        //load it using struct GEP
        let int_ptr = self.builder.build_struct_gep(self.lisp_val_type, lisp_val_ptr, 1, "data_ptr")?;
        let int_val = self.builder.build_load(self.ctx.i64_type(), int_ptr, "load_int_val")?
        .into_int_value();

        // we still need to truncate i64 to i32
        let value = self.builder.build_int_truncate(int_val, self.ctx.i8_type(), "trunc_i64_i8")?;

        Ok(value)
    }

    fn unbox_string(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        let tag= self.get_tag(lisp_val_ptr)?;
        let expected_tag = self.ctx.i8_type().const_int(TAG_STRING as u64, false);
        self.builder.build_int_compare(IntPredicate::EQ, tag, expected_tag, "is_int_tag")?;

        //TODO - branch and throw error, leaving for now
        let int_ptr = self.builder.build_struct_gep(self.lisp_val_type, lisp_val_ptr, 1, "data_ptr")?;
        let data = self.builder.build_load(self.ctx.i64_type(), int_ptr, "load_string_ptr")?
        .into_int_value();

        // need to cast that int we loaded into data as string_ptr
        let str_ptr = self.builder.build_int_to_ptr(data, self.ctx.ptr_type(AddressSpace::default()), "int_data_to_ptr")?;

        Ok(str_ptr)
    }

    fn unbox_lambda(&mut self, lisp_value_ptr: PointerValue<'ctx>) -> Result<PointerValue<'ctx>> {
        let tag= self.get_tag(lisp_value_ptr)?;
        let expected_tag = self.ctx.i8_type().const_int(TAG_LAMBDA as u64, false);
        self.builder.build_int_compare(IntPredicate::EQ, tag, expected_tag, "is_lambda_tag")?;
        //TODO - branch and throw error, leaving for now

        let int_data_ptr = self.builder.build_struct_gep(self.lisp_val_type, lisp_value_ptr, 1 , "data_ptr")?;
        let data = self.builder.build_load(self.ctx.i64_type(), int_data_ptr, "load_data_ptr")?.into_int_value();

        let lambda_ptr = self.builder.build_int_to_ptr(data, self.ctx.ptr_type(AddressSpace::default()), "int_data_to_ptr")?;

        Ok(lambda_ptr)


    }

// Lisp Cons cell type
// In C, this would be:
// struct LispList {
//     LispValue* head;
//     LispList* tail;
// }

    fn create_list_type(&self) -> StructType<'ctx> {
        let list_type = self.ctx.opaque_struct_type("ListType");
        let lisp_value_ptr = self.ctx.ptr_type(AddressSpace::default());

        list_type.set_body(&[
            lisp_value_ptr.into(), // head (car)
            lisp_value_ptr.into()  // tail (cdr)
        ], false);

        list_type
    }


    fn emit_call(&mut self, items:&[SExpr]) -> Result<PointerValue<'ctx>> {
        let func = &items[0];
        let args = &items[1..];

        if let SExpr::Symbol(op) = func {
            match op.as_str() {
                "+" => return self.emit_add(args),
                "*" => return self.compile_mul(args),
                "-" => return self.compile_sub(args),
                "/" => return self.compile_div(args),
                _ => {}
            }
        }

        // if let SExpr::LambdaExpr(params, body) = func { 
        //     let llvm_function = self.emit_lambda(params, body);
        //     let args: Vec<_> = args.iter().map(|a| self.emit_expr(a)).collect::<Result<Vec<_>>>()?;

        //     // Convert to BasicMetadataValueEnum for call
        //     let args: Vec<_> = args.iter().map(|a| (*a).into()).collect();   

        //     let call_site = self.builder.build_call(llvm_function?, &args, "lambda_call")?;

        //     let result = call_site.try_as_basic_value().left().ok_or(anyhow!("Failed to get lambda_call result"));

        //     let result = result?.into_pointer_value();

        //     return Ok(result)
        // }

        let func = self.emit_expr(func)?;
        let func_ptr = self.unbox_lambda(func)?;
        let ptr_type = self.ctx.ptr_type(AddressSpace::default());

        let args: Vec<_> = args.iter().map(|a| self.emit_expr(a)).collect::<Result<Vec<_>>>()?;

        let args: Vec<_> = args.iter().map(|a| (*a).into()).collect();   
        let param_types:Vec<_> = args
            .iter()
            .map(|_| ptr_type.into())
            .collect();

        let fn_type = ptr_type.fn_type(&param_types, false);

        let call_res = self.builder
            .build_indirect_call(fn_type, func_ptr, &args, "lambda_call")
            .map_err(|e| anyhow!("error making indirect_call: {}", e))?;

        let res = call_res.try_as_basic_value().left().ok_or(anyhow!("error converting indirect_call CallSite to PointerValue"))?;

        Ok(res.into_pointer_value())

        // bail!("FunctionCalls for lisp functions not implemented yet!")

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

        let arg0 = self.unbox_int(arg0)?;
        let arg1 = self.unbox_int(arg1)?;

        let result = self.builder.build_int_add(arg0, arg1, "builtin_addint")?;
        self.box_int(result)
    }

    fn compile_sub(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("- requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;


        let arg0 = self.unbox_int(arg0)?;
        let arg1 = self.unbox_int(arg1)?;


        let result = self.builder.build_int_sub(arg0, arg1, "builtin_subint")?;
        self.box_int(result)
    }


    fn compile_mul(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("* requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;


        let arg0 = self.unbox_int(arg0)?;
        let arg1 = self.unbox_int(arg1)?;


        let result = self.builder.build_int_mul(arg0, arg1, "builtin_mulint")?;
        self.box_int(result)
    }


    fn compile_div(&mut self, args: &[SExpr]) -> Result<PointerValue<'ctx>> {
        if args.len() != 2 {
            bail!("/ requires exactly 2 arguments");
        }
        
        let arg1 = self.emit_expr(&args[1])?;
        let arg0 = self.emit_expr(&args[0])?;

        let arg0 = self.unbox_int(arg0)?;
        let arg1 = self.unbox_int(arg1)?;


        let result = self.builder.build_int_signed_div(arg0, arg1, "builtin_divint")?;
        self.box_int(result)
    }
}


#[cfg(test)]
mod compiler_tests {
    use super::*;


    #[test]
    fn scalar_int_expr() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, true);

        let expr = SExpr::Int(42);

        let result = compiler.compile_and_run(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn scalar_bool_expr() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, true);

        // Test false - in your runtime, bools might return as 0/1
        let expr = SExpr::Bool(false);
        let result = compiler.compile_and_run(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);

        // Test true
        let mut compiler2 = CodeGen::new(&ctx, true);
        let expr = SExpr::Bool(true);
        let result = compiler2.compile_and_run(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn integer_math() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, true);

        let expr = SExpr::List(
            vec![SExpr::Symbol("+".to_string()), SExpr::Int(41), SExpr::Int(1)]
        );

        let result = compiler.compile_and_run(&[expr]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    #[ignore]
    fn scalar_conditional_expr() {
        let ctx = Context::create();
        let mut compiler = CodeGen::new(&ctx, true);
        let res = compiler.emit_expr(&SExpr::Bool(false));

        assert!(res.is_ok());
         
    }

    #[test]
    fn test_jit() {
        // Create context
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, true);
        
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
        let mut compiler2 = CodeGen::new(&context, true);
        let result =  compiler2.compile_and_run(&[expr2]).unwrap();
        assert_eq!(42, result); 
    }

    #[test]
    fn test_repl_global_variables() {
        // Test that global variables persist across REPL evaluations
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, false);

        // First REPL line: (def x 42)
        let def_expr = SExpr::DefExpr(
            "x".to_string(),
            Box::new(SExpr::Int(42))
        );

        let result1 = codegen.repl_compile(&[def_expr]);
        assert!(result1.is_ok(), "Failed to define x: {:?}", result1.err());
        assert_eq!(result1.unwrap(), 42);

        // Second REPL line: x (should return 42)
        let read_expr = SExpr::Symbol("x".to_string());

        let result2 = codegen.repl_compile(&[read_expr]);
        assert!(result2.is_ok(), "Failed to read x: {:?}", result2.err());
        assert_eq!(result2.unwrap(), 42);
    }

    #[test]
    fn test_repl_lambda_with_globals() {
        // Test: (def x 40) (def f (fn [x] (* x 10))) (f (+ 2 x))
        // Expected: 420
        let context = Context::create();
        let mut codegen = CodeGen::new(&context, false);

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
        assert_eq!(result1.unwrap(), 420);

    }
}
