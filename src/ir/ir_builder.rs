use std::{collections::{HashMap, HashSet}, slice};

use anyhow::{anyhow, bail, Result};

use crate::{ir::{instructions::Terminator, ir_types::{Constant, SsaId}}, parser::SExpr};

use super::{
    instructions::{BasicBlock, Function, Instruction, Namespace},
    ir_types::{BlockId, FunctionId, Operator, Type, TypedValue},
};

pub fn lower_to_ir(ast: &[SExpr]) -> Result<Namespace> {
    let ctx = IrLoweringContext::new();
    ctx.lower_program(&ast)
}

pub struct FunctionIrBuilder {
    next_ssa_id: u32,
    next_block_id: u32,
    current_block: BlockId,
    blocks: Vec<BasicBlock>,
}
const ENTRY_BLOCK_ID: u32 = 0;
const FIRST_SSA_ID: u32 = 0;

impl FunctionIrBuilder {
    pub fn new() -> Self {
        let entry_bb_id = BlockId(ENTRY_BLOCK_ID);
        Self {
            next_ssa_id: FIRST_SSA_ID,
            next_block_id: ENTRY_BLOCK_ID + 1,
            current_block: entry_bb_id,
            blocks: vec![empty_block(entry_bb_id)],
        }
    }

    pub fn fresh_ssa(&mut self, ty: Type) -> TypedValue {
        let id = SsaId(self.next_ssa_id);
        self.next_ssa_id += 1;

        TypedValue { id, ty }
    }

    pub fn fresh_bloc(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        self.blocks.push(empty_block(id));
        id
    }

    pub fn current_block_id(&self) -> BlockId {
        self.current_block
    }

    pub fn current_block_mut(&mut self) -> Option<&mut BasicBlock> {
        self.blocks
            .iter_mut()
            .find(|bb| bb.id == self.current_block)
    }

    pub fn switch_block(&mut self, bb_id: BlockId) {
        self.current_block = bb_id;
    }

    pub fn emit(&mut self, bb: &mut BasicBlock, i: Instruction) {
        bb.instructions.push(i);
    }

    pub fn terminate(&mut self, bb: &mut BasicBlock, t: Terminator) {
        bb.terminator = t;
    }

    pub fn finish(
        self,
        id: FunctionId,
        name: String,
        params: Vec<TypedValue>,
        return_type: Type,
    ) -> Function {
        Function {
            id,
            name,
            params,
            return_type,
            blocks: self.blocks,
            entry_block: BlockId(ENTRY_BLOCK_ID),
        }
    }
}

// TODO maybe that belongs in BasicBlock ?
fn empty_block(id: BlockId) -> BasicBlock {
    BasicBlock {
        id,
        instructions: Vec::new(),
        terminator: Terminator::Unreachable,
        predecessors: Vec::new(),
    }
}

pub struct IrLoweringContext {
    builder: FunctionIrBuilder,
    env: HashMap<String, SsaId>,
    current_function_id: FunctionId,
    pub namespace: Namespace,
    current_params: Vec<TypedValue>
}

macro_rules! emit_collection {
    ($self:expr, $collection_type:path, $instruction_type:ident, $elements_exprs:expr) => {{
            let elements = $self.lower_sequence($elements_exprs)?;
            let dest = $self.builder.fresh_ssa($collection_type);
            let dest_id = dest.id;
            let bb = $self.current_block()?;
            bb.emit(Instruction::$instruction_type { dest: dest, elements: elements });
            Ok(dest_id)
        
    }};
}

fn needs_boxing(ty: &Type) -> bool {
    matches!(ty, Type::Int | Type::Bool)
}


impl IrLoweringContext {
    pub fn new() -> Self {
        let namespace = Namespace::new();

        // First user function ID comes after runtime functions
        let first_user_function_id = FunctionId(namespace.next_function_id);

        Self {
            builder: FunctionIrBuilder::new(),
            env: HashMap::new(),
            current_function_id: first_user_function_id,
            namespace,
            current_params: Vec::new()
        }
    }

    pub fn lower_program(mut self, exprs: &[SExpr]) -> Result<Namespace> {
        // Create a <toplevel> init function
        let toplevel_id = FunctionId(self.namespace.next_function_id);
        self.namespace.next_function_id += 1;
        self.current_function_id = toplevel_id;

        // Top-level has no parameters
        self.current_params = vec![];

        // Lower all expressions
        let mut last_val = None;
        for expr in exprs {
            last_val = Some(self.lower_expr(expr)?);
        }

        // return last value or Nil
        let mut return_val = last_val.unwrap_or_else(|| {
            self.emit_constant(Type::List, Constant::Nil).unwrap()
        });
        let return_type = self.get_ssa_type(return_val)?;

        // Entry point must always return BoxedLispVal for REPL
        // Box everything except values that are already boxed
        match return_type {
            Type::BoxedLispVal | Type::Any => {
                // Already boxed, no need to box again
            }
            _ => {
                // Box everything else (Int, Bool, String, List, etc.)
                return_val = self.emit_box(return_val)?;
            }
        }

        let bb = self.current_block()?;
        bb.terminate(Terminator::Return { value: return_val });

<<<<<<< HEAD
        // finish the function and it to namespace
<<<<<<< HEAD
        let func = self.builder.finish(toplevel_id, "<toplevel>".to_string(), vec![], Type::Any);
        let func_id = func.id;
        self.namespace.add_function(func);
        self.namespace.set_entry_point(func_id);
=======
        let func = self.builder.finish(toplevel_id,
                 "<toplevel>".to_string(), 
=======
        // finish the function and add it to namespace
        let func = self.builder.finish(
            toplevel_id,
            "<toplevel>".to_string(),
>>>>>>> cfa5c66 (simple lambda return_type inference based on body's last expression)
            vec![],
            Type::BoxedLispVal  // Entry point always returns BoxedLispVal for REPL
        );
        self.namespace.add_function(func);
        self.namespace.set_entry_fn(toplevel_id);
        
>>>>>>> c345a8c (codegen_v2 using our lowered SSA based IR! currently can do simple integer math only, but it does it without constant boxing / unboxing like before! WIP but it's a milestone!)
        Ok(self.namespace)

    }



    pub fn lower_expr(&mut self, expr: &SExpr) -> Result<SsaId> {
        match expr {
            SExpr::Int(i)  =>  self.emit_constant(Type::Int, Constant::Int(*i)),
            SExpr::Bool(b) => self.emit_constant( Type::Bool, Constant::Bool(*b)),
            SExpr::String(s) =>  self.emit_constant( Type::String, Constant::String(s.clone())),
            SExpr::Symbol(sym   ) => self.lower_get_var(sym),
            SExpr::IfExpr(pred, truthy_exprs, falsy_exprs) => self.emit_if(pred, truthy_exprs, falsy_exprs),
            SExpr::LambdaExpr(params, body) => self.emit_lambda(params, body),
            SExpr::DefExpr(sym, sexpr) => self.lower_def(sym, sexpr.as_ref()),
            SExpr::List(args) => self.emit_call(args),
            SExpr::Quoted(sexpr) => match sexpr.as_ref() {
                SExpr::List(xs) if xs.is_empty() => self.emit_nil(), 
                _ =>  emit_collection!(self, Type::List, MakeList, slice::from_ref(sexpr.as_ref()))
            },
            SExpr::Vector(sexprs) => emit_collection!(self, Type::Vector, MakeVector, &sexprs),
            SExpr::BuiltinFn(_, _) => bail!("BuiltinFn is only used in Interpreted mode!"),
        }
    }

    fn lower_get_var(&mut self, sym: &String) -> std::result::Result<SsaId, anyhow::Error> {
        if let Some(local_val) =  self.env.get(sym) {
            return Ok(*local_val);
        }
        // Not local - it's a global
        let sym_id = self.namespace.intern_symbol(sym);

        // try to to find type of this global in the global_env
        let dest_type = self.namespace
            .global_env.get(&sym_id)
            .map(|ty| ty.ty.clone())
            .unwrap_or(Type::Any);

        let var_id = self.emit_constant(Type::String, Constant::String(sym.clone()))?;
        let dest = self.builder.fresh_ssa(dest_type);
        let dest_id = dest.id;

        // Get the runtime function ID before borrowing
        let runtime_get_var = self.namespace.runtime_get_var;

        let bb = self.current_block()?;
        bb.emit(Instruction::DirectCall {
            dest,
            func: runtime_get_var,
            args: vec![var_id]
        });
        Ok(dest_id)
    }

    fn lower_def(&mut self, sym: &str,  val_expr: &SExpr) -> Result<SsaId> {
        let sym_id = self.namespace.intern_symbol(sym);

        let val_id = self.lower_expr(val_expr)?;

        // try to find the type of this value
        let val_type = self.get_ssa_type(val_id)?;

        self.namespace.add_def(sym_id, TypedValue { id:val_id, ty: val_type});

        let sym_name_const = self.emit_constant(Type::String, Constant::String(sym.to_string()))?;
        let dest = self.builder.fresh_ssa(Type::Any);
        let dest_id = dest.id;

        // Get the runtime function ID before borrowing
        let runtime_set_var = self.namespace.runtime_set_var;

        let bb = self.current_block()?;
        bb.emit(Instruction::DirectCall {
            dest,
            func: runtime_set_var,
            args: vec![sym_name_const, val_id]
        });
        Ok(dest_id)
    }
    
    fn current_block(&mut self) -> Result<&mut BasicBlock>  {
        self.builder.current_block_mut().ok_or(anyhow!("IRBuilder current_block not found!"))
    }

    fn emit_constant(&mut self, ty: Type, v: Constant) -> Result<SsaId> {
        let dest = self.builder.fresh_ssa(ty);
        let dest_id = dest.id;
        let bb = self.current_block()?;
        bb.emit(Instruction::Const { dest: dest, value: v });
        Ok(dest_id)
    }

    fn lower_block(&mut self, exprs:&[SExpr]) -> Result<SsaId> {
        let body_exprs = self.lower_sequence(exprs)?;
        Ok(body_exprs.last().copied().expect("lower_sequence guarantees non-empty"))
    }

    fn emit_nil(&mut self) -> Result<SsaId> {
        // TODO - might be that that '() is not Type::Any.. But what instead? List? Bool? 
        // That's a consequential decision, you might want to think about more, but now is not the best time
        self.emit_constant(Type::List, Constant::Unit)
    }

    fn lower_sequence(&mut self, exprs:&[SExpr]) -> Result<Vec<SsaId>> {
        if exprs.is_empty() {
            return    Ok(vec![self.emit_nil()?]); 
        }
        exprs
            .iter()
            .map(|e| self.lower_expr(e))
            .collect()
    }
    
    fn emit_if(&mut self, pred: &SExpr, truthy_exprs: &[SExpr], falsy_exprs: &[SExpr]) -> Result<SsaId> {
        let cond_val = self.lower_expr(pred)?;
        
        //define Basic Blocks for the branch
        let truthy_bb = self.builder.fresh_bloc();
        let falsy_bb = self.builder.fresh_bloc();
        let merge_bb = self.builder.fresh_bloc();

        let entry_bb = self.current_block()?;
        entry_bb.terminate(Terminator::Branch { condition: cond_val, truthy_block: truthy_bb, falsy_block: falsy_bb });

        // truthy (then) block 
        self.builder.switch_block(truthy_bb);
        let t_val = self.lower_block(truthy_exprs)?;
        let t_end_block = self.current_block()?;
        t_end_block.terminate(Terminator::Jump { target: merge_bb });
        let t_end_block = self.builder.current_block;

        // falsy (else) block
        self.builder.switch_block(falsy_bb);
        let f_val = self.lower_block(falsy_exprs)?;
        let f_end_block = self.current_block()?;
        f_end_block.terminate(Terminator::Jump { target: merge_bb });
        let f_end_block = self.builder.current_block;

        // merge with Phi node
        self.builder.switch_block(merge_bb);
        let result = self.builder.fresh_ssa(Type::Any);
        let result_id = result.id;
        let merge_block = self.current_block()?;
        merge_block.emit(Instruction::Phi { dest: result, incoming: vec![
            (t_val, t_end_block),
            (f_val, f_end_block)
        ] });
        
        Ok(result_id)
    }

    fn collect_free_vars(&self, body: &[SExpr], bound_params: &HashSet<String>) -> HashSet<String> {
        let mut free_vars = HashSet::new();
  
        fn visit(expr: &SExpr, bound: &HashSet<String>, free: &mut HashSet<String>) {
            match expr {
                SExpr::Symbol(name) if !bound.contains(name) => {
                    free.insert(name.clone());
                }
                SExpr::LambdaExpr(params, body) => {
                    let mut new_bound = bound.clone();
                    for param in params {
                        if let SExpr::Symbol(p) = param {
                            new_bound.insert(p.clone());
                        }
                    }
                    for e in body {
                        visit(e, &new_bound, free);
                    }
                }
                SExpr::List(exprs) | SExpr::Vector(exprs) => {
                    for e in exprs {
                        visit(e, bound, free);
                    }
                }
                SExpr::IfExpr(pred, t_exprs, f_exprs) => {
                    visit(pred, bound, free);
                    for e in t_exprs { visit(e, bound, free); }
                    for e in f_exprs { visit(e, bound, free); }
                }
                SExpr::DefExpr(_, val) => {
                    visit(val, bound, free);
                    // 'def' binds in the current scope, so don't mark as free
                }
                SExpr::Quoted(e) => visit(e, bound, free),
                _ => {}
            }
        }
  
        for expr in body {
            visit(expr, bound_params, &mut free_vars);
        }
  
        free_vars
    }
    
    fn emit_lambda(&mut self, params: &[SExpr], body: &[SExpr]) -> Result<SsaId> {
        // need to store current state, as new function needs fresh env and builder
        let old_builder = std::mem::replace(&mut self.builder, FunctionIrBuilder::new());
        let old_env = std::mem::replace(&mut self.env, HashMap::new());
        let old_params = std::mem::replace(&mut self.current_params, vec![]);

        let fun_id = FunctionId(self.namespace.next_function_id);
        self.namespace.next_function_id += 1;
        self.current_function_id = fun_id;

        let typed_params: Vec<TypedValue> = params.iter().map(|p| match p {
            SExpr::Symbol(sym) => {
                let id = self.builder.fresh_ssa(Type::Any);
                self.env.insert(sym.clone(), id.id);
                Ok(id)
            },
            _ => bail!("Parameter expressions must be a symbol!"),
         }).collect::<Result<_>>()?;
         self.current_params = typed_params.clone();

         let mut result = self.lower_block(body)?;
         let result_type = self.get_ssa_type(result)?;

         // Decide the function's return type and box if necessary
         let return_type = match &result_type {
             Type::Int | Type::Bool => {
                 // Scalar types: return unboxed for performance!
                 result_type.clone()
             }
             Type::String | Type::List | Type::Vector => {
                 // Heap types: already pointers, return as-is
                 result_type.clone()
             }
             Type::Any | Type::Union(_) => {
                 // Unknown/mixed types: must box
                 result = self.emit_box(result)?;
                 Type::BoxedLispVal
             }
             _ => {
                 // Conservative: box everything else
                 result = self.emit_box(result)?;
                 Type::BoxedLispVal
             }
         };

         let bb = self.current_block()?;
         bb.terminate(Terminator::Return { value: result });

         // swap builder and env back
         let lambda_builder = std::mem::replace(&mut self.builder, old_builder);
         self.env = old_env;
         self.current_params = old_params;

         // now lambda_builder can be consumed (with it's blocks etc.) to finish function definition
         // Lambda return type is inferred (Int, Bool, BoxedLispVal, etc.)
         let func = lambda_builder.finish(
             fun_id,
             format!("lambda_{}", fun_id.0),  // Auto-generated name
             typed_params.clone(),
             return_type.clone()
         );
         self.namespace.add_function(func);

         let param_names: HashSet<String> = params.iter().filter_map(|p| match p {
             SExpr::Symbol(sym) => Some(sym.clone()),
             _ => None
         })
         .collect();
         // recursively traverse body, keeping track of declared params to find free variables
         // i.e. variables used that are not a parameter
         let free_variables = self.collect_free_vars(body, &param_names);

         // we only want to capture those free_vars that are present in current env
         let captured_env = free_variables
             .iter()
             .filter_map(|v| self.env.get(v).copied())
             .collect();

         // Build proper function type for the closure
         let func_type = Type::Function {
             params: typed_params.iter().map(|p| p.ty.clone()).collect(),
             return_type: Box::new(return_type)
         };

         let dest = self.builder.fresh_ssa(func_type);
         let dest_id = dest.id;
         let bb = self.current_block()?;
         bb.emit(Instruction::MakeClosure { dest, func: fun_id, captures: captured_env });

         Ok(dest_id)

    }
    
    fn emit_call(&mut self, args: &[SExpr]) -> Result<SsaId> {
        if args.is_empty() {
            bail!("Empty list in call position");
        }

        let fun = &args[0];
        let args = &args[1..];

        if let SExpr::Symbol(sym) = fun {
            if let Some(prim_op) = parse_primop(sym) {
                return self.lower_primop(prim_op, args);
            }
        }



        let func_val = self.lower_expr(fun)?;
        let func_type = self.get_ssa_type(func_val)?;
        // Extract return type from function signature
        let return_type = match &func_type {
            Type::Function { return_type, .. } => return_type.as_ref().clone(),
            Type::Any => Type::Any,  // Unknown function, assume boxed
            _ => bail!("Cannot call non-function type")
        };

        let arg_vals: Result<Vec<_>> = args.iter()
                .map(|a| self.lower_expr(a))
                .collect();
        let arg_vals = arg_vals?;
        let dest = self.builder.fresh_ssa(return_type);
        let dest_id = dest.id;
        let bb = self.current_block()?;
        bb.emit(Instruction::Call { dest, func: func_val, args: arg_vals });

        Ok(dest_id)
    }
    
    fn lower_primop(&mut self, prim_op: Operator, args: &[SExpr]) -> Result<SsaId> {
        let arg_vals: Result<Vec<_>> = args.iter()
            .map(|a| self.lower_expr(a))
            .collect();

        let arg_vals = arg_vals?;

        // Infer result type from operator and argument types
        let arg_types: Vec<Type> = arg_vals.iter()
            .map(|id| self.get_ssa_type(*id).unwrap_or(Type::Any))
            .collect();

        let result_type = prim_op.result_type(&arg_types);
        let dest = self.builder.fresh_ssa(result_type);
        let dest_id = dest.id;
        let bb = self.current_block()?;
        bb.emit(Instruction::PrimOp { dest: dest, op: prim_op, args: arg_vals });

        Ok(dest_id)
    }
    
    fn get_ssa_type(&self, val_id: SsaId) -> Result<Type> {
        // first check current function's params 
        for param in &self.current_params {
            if param.id == val_id {
                return Ok(param.ty.clone());
            }
        }

        // search through all blocks in current function
        for bb in &self.builder.blocks {
            for instr in &bb.instructions {
                if let Some(dest) = instr.dest() {
                    if dest.id == val_id {
                        return Ok(dest.ty.clone())
                    }
                }
            }
        }

        // If not found, it might be a forward reference or parameter
        // For now, return Any (type inference will fix it later)
        Ok(Type::Any)
    }
    
    fn emit_box(&mut self, value: SsaId) -> Result<SsaId> {
        let dest = self.builder.fresh_ssa(Type::BoxedLispVal);
        let dest_id = dest.id;
        let bb = self.current_block()?;
        bb.emit(Instruction::Box{dest, value});
        Ok(dest_id)
    }
    

    fn emit_unbox(&mut self, value: SsaId, expected_type: Type) -> Result<SsaId> {
        let dest = self.builder.fresh_ssa(Type::BoxedLispVal);
        let dest_id = dest.id;
        let bb = self.current_block()?;
        bb.emit(Instruction::Unbox {dest, value, expected_type});
        Ok(dest_id)
    }
    
}


  
fn parse_primop(name: &str) -> Option<Operator> {
    match name {
        "+" => Some(Operator::Add),
        "-" => Some(Operator::Sub),
        "*" => Some(Operator::Mul),
        "/" => Some(Operator::Div),
        "mod" => Some(Operator::Mod),
        "<" => Some(Operator::Lt),
        ">" => Some(Operator::Gt),
        "=" => Some(Operator::Eq),
        "<=" => Some(Operator::Le),
        ">=" => Some(Operator::Ge),
        "!=" => Some(Operator::Ne),
        _ => None,
    }
}
