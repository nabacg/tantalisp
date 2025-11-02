use std::{
    collections::HashMap,
    ffi::{CStr, c_char},
    hash::Hash,
    sync::atomic::{self, AtomicI32},
};

use crate::{
    ir::{
        instructions::{self, Function, Namespace, Terminator},
        ir_types::{BlockId, Constant, FunctionId, Operator, SsaId, Type, TypedValue},
    },
    v2_codegen::runtime::{
        alloc_cons_cell, alloc_lisp_val, alloc_string, lisp_val_decref, lisp_val_incref, lisp_val_print_refcount, runtime_get_var, runtime_set_var, TAG_LIST
    },
};
use anyhow::{Result, anyhow, bail};
use inkwell::{
    basic_block::{self, BasicBlock}, builder::Builder, context::Context, execution_engine::{ExecutionEngine, JitFunction}, module::Module, types::{BasicMetadataTypeEnum, BasicTypeEnum, FunctionType, IntType, PointerType, StructType}, values::{BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, InstructionValue, IntValue, PointerValue}, AddressSpace, IntPredicate
};

#[cfg(target_pointer_width = "64")]
use super::runtime::ConsCellLayout;
use super::runtime::{self, LispValLayout, TAG_BOOL, TAG_INT, TAG_LAMBDA, TAG_NIL, TAG_STRING};

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
    retain_fn: FunctionValue<'ctx>,
    release_fn: FunctionValue<'ctx>,
    print_refcount_fn: FunctionValue<'ctx>,
    runtime_get_var_fn: FunctionValue<'ctx>,
    runtime_set_var_fn: FunctionValue<'ctx>,

    memcpy_fn: FunctionValue<'ctx>,

    //variables in current scope, dummy environment
    local_env: HashMap<String, PointerValue<'ctx>>,

    //track function parameters during lambda compilation
    current_function: Option<FunctionValue<'ctx>>,
    // print debug info like LLVM IR
    debug: bool,

    // Runtime symbol table - use raw pointer for stable address
    runtime_env: *mut HashMap<String, *mut LispValLayout>,

    // Mapping IR Ids to LLVM values
    ssa_values: HashMap<SsaId, (Type, BasicValueEnum<'ctx>)>,
    basic_blocks: HashMap<BlockId, BasicBlock<'ctx>>,
    functions: HashMap<FunctionId, FunctionValue<'ctx>>,
}

macro_rules! emit_int_binop {
    ($self:expr, $args_vec:expr, $llvm_op: ident, $label: expr) => {{
        if $args_vec.len() != 2 {
            bail!("{} required exactly 2 arguments, got: {}", $label, $args_vec.len());
        }
        let lhs = $args_vec[0].into_int_value();
        let rhs = $args_vec[1].into_int_value();
        let add_res = $self.builder.$llvm_op(lhs, rhs, $label)?;
        Ok(add_res.into())
    }};
}

macro_rules! emit_int_cmp_binop {
    ($self:expr, $args_vec:expr, $llvm_predicate_type: path, $label: expr) => {{
        if $args_vec.len() != 2 {
            bail!("{} required exactly 2 arguments, got: {}", $label, $args_vec.len());
        }
        let lhs = $args_vec[0].into_int_value();
        let rhs = $args_vec[1].into_int_value();
        let add_res = $self.builder.build_int_compare($llvm_predicate_type, lhs, rhs, $label)?;
        Ok(add_res.into())
    }};
}

impl<'ctx> CodeGen<'ctx> {
    // compile_and_run version for REPL, doesn't create fresh execution engine each time,
    // thus allows defining global vars and re-using them in subsequent executions
    pub fn repl_compile(&mut self, ns: Namespace) -> Result<String> {

        if let Some(old_entry_point_fn)  =  ns.get_entry_fn().and_then(|f| self.module.get_function(&f.name)) {
            unsafe {
                old_entry_point_fn.delete();
            }
        }
        // Compile exprs
        let entry_fn_name = self.emit_ns(ns)?;
        if self.debug {
            // print LLVM IR
            println!("-------- LLVM IR ---------");
            self.module.print_to_stderr();
            println!("--------------------------");
        }

        let engine = self.create_execution_engine_for_repl(&entry_fn_name)?;

        unsafe {
            type MainFunc = unsafe extern "C" fn() -> *mut LispValLayout;
            // type MainFunc = unsafe extern "C" fn() ->  i32;
            //get handle to the main function
            let jit_function: JitFunction<MainFunc> = engine.get_function(&entry_fn_name)?;

            //call it
            let lisp_val_ptr = jit_function.call();
            if lisp_val_ptr.is_null() {
                bail!("JIT returned null pointer");
            }
            // TODO - how to print things if we can't depend on result always being a LispVal ?
            lisp_val_to_string(lisp_val_ptr)
            // Ok("TODO - Learn how to print results".to_string())
            // let res = lisp_val_ptr ;
            // Ok(format!("{}", res))
        }
    }

    pub fn compile_and_run(&mut self, ns: Namespace) -> Result<String> {
        let entry_fn_name = self.emit_ns(ns)?;

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
            let jit_function: JitFunction<MainFunc> = engine.get_function(&entry_fn_name)?;

            //call it
            let lisp_val_ptr = jit_function.call();
            if lisp_val_ptr.is_null() {
                bail!("JIT returned null pointer");
            }
            lisp_val_to_string(lisp_val_ptr)
            // Ok("TODO - Learn how to print results".to_string())
        }
    }

    pub fn new(ctx: &'ctx Context, debug_mode: bool) -> Self {
        let module = ctx.create_module("tantalisp_main");
        let builder = ctx.create_builder();

        // Ptr type
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let lisp_val_type = ctx.opaque_struct_type("LispVal");
        lisp_val_type.set_body(
            &[
                ctx.i8_type().into(),  // tag
                ctx.i64_type().into(), // union { i32, bool, ptr(String), ptr(List)}
                ctx.i32_type().into(), // reference count
            ],
            false,
        );

        let cons_cell_type = ctx.opaque_struct_type("ConsCell");
        cons_cell_type.set_body(&[ptr_type.into(), ptr_type.into()], false);

        //Declare alloc_* fns for LispVal, ConsCell, string etc.
        let alloc_lisp_val_type = ptr_type.fn_type(&[], false);
        let alloc_lisp_val_fn = module.add_function("alloc_lisp_val", alloc_lisp_val_type, None);

        let calloc_cons_cell_type = ptr_type.fn_type(&[], false);
        let alloc_cons_cell_fn =
            module.add_function("alloc_cons_cell", calloc_cons_cell_type, None);

        let alloc_string_type = ptr_type.fn_type(&[ctx.i64_type().into()], false);
        let alloc_string_fn = module.add_function("alloc_string", alloc_string_type, None);

        // Declare memcpy: (i8* dest, i8* src, i64 size) -> i8*
        // void* memcpy(void* dest, const void* src, size_t n)
        let memcpy_fn_type = ptr_type.fn_type(
            &[ptr_type.into(), ptr_type.into(), ctx.i64_type().into()],
            false,
        );
        let memcpy = module.add_function("memcpy", memcpy_fn_type, None);

        let incref_type = ctx.void_type().fn_type(&[ptr_type.into()], false);
        let incref_fn = module.add_function("lisp_val_incref", incref_type, None);

        let decref_type = ctx.void_type().fn_type(&[ptr_type.into()], false);
        let decref_fn = module.add_function("lisp_val_decref", decref_type, None);

        let print_refcount_type = ctx.void_type().fn_type(&[ptr_type.into()], false);
        let print_refcount_fn =
            module.add_function("lisp_val_print_refcount", print_refcount_type, None);

        //declare fn: ptr runtime_get_var(ptr env, ptr name)
        let get_var_type = ptr_type.fn_type(&[ptr_type.into(), ptr_type.into()], false);

        let runtime_get_var_fn = module.add_function("runtime_get_var", get_var_type, None);

        // declare: void runtime_set_var(ptr env, ptr name, ptr value)
        let set_var_type =
            ptr_type.fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);

        let runtime_set_var_fn = module.add_function("runtime_set_var", set_var_type, None);

        Self {
            ctx,
            module,
            builder,
            ptr_type,
            lisp_val_type,
            cons_cell_type,
            alloc_lisp_val_fn,
            alloc_cons_cell_fn,
            alloc_string_fn,
            retain_fn: incref_fn,
            release_fn: decref_fn,
            print_refcount_fn,
            memcpy_fn: memcpy,
            runtime_get_var_fn,
            runtime_set_var_fn,
            local_env: HashMap::new(),
            current_function: None,
            runtime_env: Box::into_raw(Box::new(HashMap::new())),
            debug: debug_mode,
            ssa_values: HashMap::new(),
            basic_blocks: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    fn create_execution_engine(&mut self) -> Result<ExecutionEngine<'ctx>> {
        self.module
            .create_jit_execution_engine(inkwell::OptimizationLevel::Aggressive)
            .map_err(|e| anyhow!("Failed to create JIT execution engine: {}", e))
    }

    fn create_execution_engine_for_repl(&mut self, entry_point_fn_name: &str) -> Result<ExecutionEngine<'ctx>> {
        // Clone the module so the original isn't consumed
        let module_clone = self.module.clone();

        // Create execution engine from the clone
        let engine = module_clone
            .create_jit_execution_engine(inkwell::OptimizationLevel::Aggressive)
            .map_err(|e| anyhow!("Failed to create JIT execution engine: {}", e))?;

        // Register our runtime helper functions with the JIT
        // This tells the JIT where to find these symbols at runtime

        // Map them to the actual Rust function addresses
        engine.add_global_mapping(&self.runtime_get_var_fn, runtime_get_var as usize);

        engine.add_global_mapping(&self.runtime_set_var_fn, runtime_set_var as usize);

        engine.add_global_mapping(&self.retain_fn, lisp_val_incref as usize);

        engine.add_global_mapping(&self.release_fn, lisp_val_decref as usize);

        // Map allocation functions
        engine.add_global_mapping(&self.alloc_lisp_val_fn, alloc_lisp_val as usize);

        engine.add_global_mapping(&self.alloc_cons_cell_fn, alloc_cons_cell as usize);

        engine.add_global_mapping(&self.alloc_string_fn, alloc_string as usize);

        engine.add_global_mapping(&self.print_refcount_fn, lisp_val_print_refcount as usize);

        Ok(engine)
    }

    fn emit_ns(&mut self, ns: Namespace) -> Result<String> {
        for (_id, f) in &ns.functions {
            self.emit_fn(f)?;
        }

        let entry_fn = ns
            .get_entry_fn()
            .ok_or(anyhow!("Ns without entry_fn set"))?;
        Ok(entry_fn.name.clone())
    }

    fn emit_fn(&mut self, f: &Function) -> Result<()> {
        let fn_name = &f.name; // TODO - don't forget to autogenerate those names in IR!

        let fn_type = self.compute_fn_signature(&f.params, &f.return_type)?;

        let fn_val = self.module.add_function(&fn_name, fn_type, None);
        self.current_function = Some(fn_val);

        for bb in &f.blocks {
            self.emit_basic_block(&f, bb)?;
        }

        if !fn_val.verify(true) {
            bail!("Function verification failed");
        }

        Ok(())
    }

    fn compute_fn_signature(&self, params: &[TypedValue], return_type: &Type) -> Result<FunctionType<'ctx>> {
        let param_types: Result<Vec<_>> = params
            .iter()
            .map(|p| self.type_to_metadata_type(&p.ty))
            .collect();
        let param_types = param_types?;

        Ok(
            match return_type {
                Type::Int => self.ctx.i32_type().fn_type(&param_types, false),
                Type::Bool => self.ctx.bool_type().fn_type(&param_types, false),
                Type::String 
                | Type::List 
                | Type::Vector
                | Type::BoxedLispVal         
                | Type::Any => self.ptr_type.fn_type(&param_types, false),
                // TODO check if Type::Function signature checks out with params and return_type passed? 
                Type::Function { params, return_type } => todo!(),
                // should we throw an error here? 
                Type::Union(items) => todo!(),
                Type::Bottom => todo!(), // I'm sure there is a void type  in LLVM
        })
    }

    fn emit_basic_block(&mut self, f: &Function, bb: &crate::ir::instructions::BasicBlock) -> Result<()> {

        let fn_val = self.curren_fn_val()?;
        let entry_bb = self.ctx.append_basic_block(fn_val, &f.name);
        self.builder.position_at_end(entry_bb);

        for i in &bb.instructions {
            self.emit_instruction(bb, i)?;
        }

        self.emit_terminator(bb)?;
        Ok(())
    }

            
    fn emit_instruction(&mut self, bb: &instructions::BasicBlock, i: &instructions::Instruction) -> Result<()> {
        match i {
            crate::ir::instructions::Instruction::Const { dest, value } => {
                        let val = self.emit_constant(value)?;
                        self.ssa_values.insert(dest.id, (dest.ty.clone(), val) );
                        Ok(())
                    },
            instructions::Instruction::PrimOp { dest, op, args } => { 
                        let val = self.emit_prim_op(op, args)?;
                        self.ssa_values.insert(dest.id, (dest.ty.clone(), val));
                        Ok(())
                    },
            instructions::Instruction::DirectCall { dest, func, args } => todo!(),
            instructions::Instruction::Call { dest, func, args } => todo!(),
            instructions::Instruction::MakeClosure { dest, func, captures } => todo!(),
            instructions::Instruction::MakeVector { dest, elements } => todo!(),
            instructions::Instruction::MakeList { dest, elements } => todo!(),
            instructions::Instruction::Retain { value } => todo!(),
            instructions::Instruction::Release { value } => todo!(),
            instructions::Instruction::Phi { dest, incoming } => todo!(),
            instructions::Instruction::Box { dest, value } => {
                let (ty, val) = self.lookup_ssa_id_with_type(value)?;
                let boxed = match ty {
                    Type::Int => {
                        let int_val = val.into_int_value();
                        self.box_int(int_val)?
                    }
                    Type::Bool => {
                        let bool_val = val.into_int_value();
                        self.box_bool(bool_val)?
                    }
                    Type::String | Type::List | Type::Vector | Type::BoxedLispVal => {
                        val.into_pointer_value()
                    }
                    _ => bail!("Cannot box type: {:?}", ty)
                };
                self.ssa_values.insert(dest.id, (dest.ty.clone(), boxed.into()));
                Ok(())

            },
            instructions::Instruction::Unbox { dest, value, expected_type } => {
                let lisp_val_ptr = self.lookup_ssa_id(value)?.into_pointer_value();
                let unboxed = match expected_type {
                    Type::Int => self.unbox_int(lisp_val_ptr)?.into(),
                    Type::Bool => self.unbox_bool(lisp_val_ptr)?.into(),
                    Type::String => self.unbox_string(lisp_val_ptr)?.into(),
                    _ => bail!("Cannot unbox to type: {}", expected_type)
                };
                self.ssa_values.insert(dest.id, (expected_type.clone(), unboxed));
                Ok(())
            },
        }
    }

    fn emit_prim_op(&mut self, op: &Operator, args: &[SsaId]) -> Result<BasicValueEnum<'ctx>> {
        let arg_vals: Result<Vec<_>> = args.iter().map(|ssa_id| self.lookup_ssa_id(ssa_id)).collect();
        let arg_vals = arg_vals?;

        match op {
            Operator::Add => emit_int_binop!(self, arg_vals, build_int_add, "add_int"),
            Operator::Sub => emit_int_binop!(self, arg_vals, build_int_sub, "sub_int"),
            Operator::Mul => emit_int_binop!(self, arg_vals, build_int_mul, "mul_int"),
            Operator::Div => emit_int_binop!(self, arg_vals, build_int_signed_div, "div_int"),
            Operator::Mod => emit_int_binop!(self, arg_vals, build_int_signed_rem, "mod_int"),
            Operator::Lt => emit_int_cmp_binop!(self, arg_vals, IntPredicate::SLT, "lt"),
            Operator::Gt => emit_int_cmp_binop!(self, arg_vals, IntPredicate::SGT, "gt"),
            Operator::Eq => emit_int_cmp_binop!(self, arg_vals, IntPredicate::EQ, "eq"),
            Operator::Le => emit_int_cmp_binop!(self, arg_vals, IntPredicate::SLE, "le"),
            Operator::Ge => emit_int_cmp_binop!(self, arg_vals, IntPredicate::SGE, "ge"),
            Operator::Ne => emit_int_cmp_binop!(self, arg_vals, IntPredicate::NE, "ne"),
            Operator::And => emit_int_binop!(self, arg_vals, build_and, "and"),
            Operator::Or => emit_int_binop!(self, arg_vals, build_or, "or"),
            Operator::Not => {
                let val = arg_vals[0].into_int_value();
                let res = self.builder.build_not(val, "not")?;
                Ok(res.into())
            },
            Operator::ListNew => todo!(),
            Operator::ListHead => todo!(),
            Operator::ListTail => todo!(),
            Operator::ListLen => todo!(),
            Operator::VectorNew => todo!(),
            Operator::VectorGet => todo!(),
            Operator::VectorSet => todo!(),
            Operator::VectorLen => todo!(),
            Operator::StringConcat => todo!(),
            Operator::StringLen => todo!(),
        }
    }

    fn emit_constant(&mut self, c: &Constant ) -> Result<BasicValueEnum<'ctx>> {
        match c {
            Constant::Int(i) => Ok(self.ctx.i32_type().const_int(*i as u64, true).into()),
            Constant::Bool(b) => Ok(self.ctx.bool_type().const_int(*b as u64, false).into()),
            Constant::String(str) => self.box_string(str).map(|r| r.into()),
            Constant::Unit => self.box_nil().map(|p| p.into()), // Could use a null pointer, but really hate the idea and unsure we really have unit types
            Constant::Nil => self.box_nil().map(|p| p.into()),
        }
    }

    fn emit_string_value(&mut self, str: &str) -> Result<PointerValue<'ctx>>{
        //allocate space for the string itself (null-terminated )
        let str_len = str.len() + 1; //+1 for null
        let str_size = self.ctx.i64_type().const_int(str_len as u64, false);

        let str_ptr = self.alloc_string(str_size)?;

        // Create a global constant for the source string, as it will come from literal values in source code (list "a" )
        // This creates a read-only string in the data section
        let src_str = self.builder.build_global_string_ptr(str, "str_literal")?;

        //Copy the string data using memcpy
        //memcpy(dest, src, size)
        self.builder.build_call(self.memcpy_fn, 
            &[
            str_ptr.into(),
            src_str.as_pointer_value().into(),
            str_size.into() 
        ], "memcpy_call")?;

        

        Ok(str_ptr)
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

    fn curren_fn_val(&self) -> Result<FunctionValue<'ctx>> {
        self.current_function.ok_or(anyhow!("current_function is None where it's required!"))
    }


    /// Convert an IR Type to an Inkwell BasicMetadataTypeEnum
    fn type_to_metadata_type(&self, ty: &Type) -> Result<BasicMetadataTypeEnum<'ctx>> {
        Ok(match ty {
            Type::Int => BasicMetadataTypeEnum::IntType(self.ctx.i32_type()),
            Type::Bool => BasicMetadataTypeEnum::IntType(self.ctx.bool_type()),
            Type::String | Type::List | Type::Vector | Type::BoxedLispVal | Type::Any => {
                // All heap types are pointers to LispVal
                BasicMetadataTypeEnum::PointerType(self.ptr_type)
            }
            Type::Function { .. } => {
                // Function pointers are also wrapped in LispVal
                BasicMetadataTypeEnum::PointerType(self.ptr_type)
            }
            Type::Union(_) => {
                // Unions must be boxed to handle multiple types
                BasicMetadataTypeEnum::PointerType(self.ptr_type)
            }
            Type::Bottom => {
                bail!("Cannot convert Bottom type to LLVM type (unreachable code)")
            }
        })
    }

    /// Convert an IR Type to an Inkwell BasicTypeEnum (for return types, locals)
    fn type_to_basic_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>> {
        Ok(match ty {
            Type::Int => BasicTypeEnum::IntType(self.ctx.i32_type()),
            Type::Bool => BasicTypeEnum::IntType(self.ctx.bool_type()),
            Type::String | Type::List | Type::Vector | Type::BoxedLispVal | Type::Any => {
                BasicTypeEnum::PointerType(self.ptr_type)
            }
            Type::Function { .. } => BasicTypeEnum::PointerType(self.ptr_type),
            Type::Union(_) => BasicTypeEnum::PointerType(self.ptr_type),
            Type::Bottom => {
                bail!("Cannot convert Bottom type to LLVM type (unreachable code)")
            }
        })
    }
    
    fn emit_terminator(&self, bb: &crate::ir::instructions::BasicBlock) -> Result<InstructionValue<'ctx>> {
        match bb.terminator  {
            Terminator::Return { value } =>   { 
                let ret_val = self.lookup_ssa_id(&value)?;
                let intr_value = self.builder.build_return(Some(ret_val))?;
                Ok(intr_value)
            },
            Terminator::Jump { target } => {
                let target_bb = self.lookup_block_id(&target)?;
                let instr_value = self.builder.build_unconditional_branch(*target_bb)?;
                Ok(instr_value)
            },
            Terminator::Branch { condition, truthy_block, falsy_block } => {
                let pred_val = self.lookup_ssa_id(&condition)?;
                let pred_val_int = pred_val.into_int_value(); // TODO - Will this always work? what about PointerValue<'ctx> ? we probably need to_truthy helper
                let truthy_bb = self.lookup_block_id(&truthy_block)?;
                let falsy_bb = self.lookup_block_id(&falsy_block)?;


                let instr_val = self.builder
                        .build_conditional_branch(pred_val_int, *truthy_bb, *falsy_bb)?;    
                Ok(instr_val)
            },
            Terminator::Unreachable => {  
                let instr_val = self.builder.build_unreachable()?; 
                Ok(instr_val)
            } ,
        }
    }

    fn lookup_ssa_id(&self, ssa_id: &SsaId) -> Result<&BasicValueEnum<'ctx>> {
        self.ssa_values.get(ssa_id).map(|(_, v)| v).ok_or(anyhow!("Expected SsaId: {} not found!", ssa_id))
    }

    fn lookup_ssa_id_with_type(&self, ssa_id: &SsaId) -> Result<&(Type, BasicValueEnum<'ctx>)> {
        self.ssa_values.get(ssa_id).ok_or(anyhow!("Expected SsaId: {} not found!", ssa_id))
    }

    fn lookup_block_id(&self, block_id: &BlockId) -> Result<&BasicBlock<'ctx>> {
        self.basic_blocks.get(block_id).ok_or(anyhow!("Expected BlockId: {} not found!", block_id))
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



    /// --- Candidates for a separate type (struct) for LispValue related operations, but need access &mut self
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

    fn lisp_val_data(&mut self, lisp_val_ptr: PointerValue<'ctx>) -> Result<IntValue<'ctx>> {
        let int_ptr = self.builder.build_struct_gep(self.lisp_val_type, lisp_val_ptr, 1, "data_ptr")?;
        let data = self.builder.build_load(self.ctx.i64_type(), int_ptr, "load_data_ptr_as_int")?
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
    // ^^ Candidates for a separate type (struct) for LispValue related operations, but need access &mut self

}

#[cfg(target_pointer_width = "64")]
fn cons_cell_ptr_from_data_ptr(addr: i64) -> *const ConsCellLayout {
    use super::runtime::ConsCellLayout;

    let addr: usize = addr as u64 as usize;
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
        }
        2 => {
            let str_ptr = list_val_data_to_str(lisp_val.data)
                .ok_or(anyhow!("Failed to read String out of lisp_val"))?;
            Ok(str_ptr.to_string())
        }
        3 => {
            // TODO write a Rust Iterator impl for this !!
            let cons_cell = cons_cell_ptr_from_data_ptr(lisp_val.data);
            let mut cons_cell_maybe = unsafe { cons_cell.as_ref() };
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
        }
        4 => Ok("nil".to_string()),
        5 => Ok("#<lambda>".to_string()), // Lambda

        _ => bail!(
            "Returning other LispVal types not implemented yet, found: {}",
            lisp_val.tag
        ),
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


#[cfg(test)]
mod codegen_test {
    use crate::{ir::ir_builder::lower_to_ir, lexer::tokenize, parser::parse};

    use super::*;
    
    fn source_to_ir(source: &str) -> Result<Namespace> {
        let tokens = tokenize(source)?;
        let ast = parse(&tokens)?;
        lower_to_ir(&ast)
    }



    #[test]
    fn test_int_math() {

        let mut ctx = Context::create();
        let mut codegen = CodeGen::new(&mut ctx, true);

        let ir = source_to_ir("(+ 2 2)").expect("failed to parse input");
        let res = codegen.repl_compile(ir).expect("failed to compile");
        assert_eq!("4", res);

        let ir = source_to_ir("(+ (* 4 10) 2)").expect("failed to parse input");
        let res = codegen.repl_compile(ir).expect("failed to compile");
        assert_eq!("42", res);
    }
}