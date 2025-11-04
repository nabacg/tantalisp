use std::collections::{HashMap, HashSet, VecDeque};
use anyhow::{anyhow, bail, Result};
use super::{instructions::{Function, Instruction, Namespace, Terminator}, ir_types::{BlockId, FunctionId, SsaId, SymbolId, Type, TypedValue}};


pub struct TypeInference {
    // Currently known types
    types: HashMap<SsaId, Type>,
    // worklist of Blocks to process
    worklist: VecDeque<BlockId>,
    module_functions: HashMap<FunctionId, Type>,
    dirty_functions: HashSet<FunctionId>,
    closure_to_function: HashMap<SsaId, FunctionId>,
    iterations_done: usize
}

const MAX_ITERATIONS: usize = 1000;

impl TypeInference {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            worklist: VecDeque::new(),
            module_functions: HashMap::new(),
            dirty_functions: HashSet::new(),
            closure_to_function: HashMap::new(),
            iterations_done: 0
        }
    }

    pub fn process(&mut self, ns: &mut Namespace) -> Result<()> {
        //first add return_types of all module functions 
        for (id, f) in &ns.functions {
            self.module_functions.insert(*id, Type::Function { params: f.params.iter()
                                .map(|tv| &tv.ty).cloned().collect(), return_type: Box::new(f.return_type.clone()) });

            /// populate the SsaId -> FunctionId mapping, required for refining Function signature step
            self.track_closures(f);
        }

        // Collect function IDs to avoid borrow checker issues
        let func_ids: Vec<_> = ns.functions.keys().copied().collect();
        for func_id in func_ids {
            let f = ns.functions.get_mut(&func_id).unwrap();

            // Skip runtime stub functions (they have no blocks to analyze)
            if f.blocks.is_empty() {
                continue;
            }

            self.worklist.clear(); // should really be emptied by a previous function, but just to be sure
            self.iterations_done = 0;
            self.process_function(f, ns.runtime_get_var, &ns.global_env)?;
        }

        let mut refinement_iterations = 0;
        loop {
            self.dirty_functions.clear();
            for (_, f) in &ns.functions {
                // Skip runtime stub functions
                if f.blocks.is_empty() {
                    continue;
                }
                self.check_call_sites(f, ns)?;
            }

            if self.dirty_functions.is_empty() {
                break;
            }

            let dirty_fns:Vec<_> = self.dirty_functions.iter().cloned().collect();

            for func_id in dirty_fns{
                if let Some(f) = ns.functions.get_mut(&func_id) {
                    // Apply refined parameter types from module_functions before reprocessing
                    if let Some(Type::Function { params, return_type }) = self.module_functions.get(&func_id) {
                        for (i, param) in f.params.iter_mut().enumerate() {
                            if i < params.len() {
                                param.ty = params[i].clone();
                            }
                        }
                        // Also update return type from refinement
                        f.return_type = (**return_type).clone();
                    }
                    self.process_function(f, ns.runtime_get_var, &ns.global_env)?;
                }
            }

            refinement_iterations += 1;
            if refinement_iterations > 10 {
                bail!("Inter-procedural refinement did not converge");
            }
        }

        Ok(())
    }

    fn track_closures(&mut self, f: &Function) {
        for bb in &f.blocks {
            for instr in &bb.instructions {
                if let Instruction::MakeClosure { dest, func, .. } = instr {
                    self.closure_to_function.insert(dest.id, *func);
                }
            }
        }
    }

    /// Look up the type of a global variable given its name (as an SSA constant)
    /// Used to infer precise types for runtime_get_var calls
    fn lookup_global_type(&self, _name_ssa: &SsaId, global_env: &HashMap<SymbolId, TypedValue>) -> Option<Type> {
        // Find the constant instruction that defines this SSA value
        // We need to search through all functions to find it
        // For now, just check if we have the type already computed

        // The name_ssa should be a String constant. We need to find which symbol it refers to.
        // Then look up that symbol in global_env to find the SSA value stored there.
        // Then return the type of that SSA value.

        // For now, this is a simplified implementation that doesn't trace back to constants.
        // A full implementation would need to track constant values through the IR.

        // Instead, let's iterate through global_env and check if we have types for any of the values
        for (_symbol_id, typed_value) in global_env {
            // Check if we've inferred a type for this global's value
            if let Some(ty) = self.types.get(&typed_value.id) {
                // This is a heuristic: if we're calling runtime_get_var in a function,
                // and there's a global with a Function type, assume that's what we're getting
                if matches!(ty, Type::Function { .. }) {
                    // TODO: This is imprecise - we should match on the actual symbol name
                    // For now, return the first function type we find
                    return Some(ty.clone());
                }
            }
        }

        None
    }

    fn check_call_sites(&mut self, f: &Function, ns: &Namespace) -> Result<()> {
        for bb in &f.blocks {
            for i in &bb.instructions {
                match i {
                    Instruction::Call {  func, args, .. } => {
                        if let (true, new_type) = self.refine_function_type( self.types.get(func).cloned(), args)? {
                            self.types.insert(*func, new_type);
                              // mark function as dirty
                            let callee_function_id = self.closure_to_function.get(func).ok_or(anyhow!("Failed to find FunctionId for SsaId: {}", func))?;
                            self.dirty_functions.insert(*callee_function_id);
                        }



                    },
                    Instruction::DirectCall {  func, args, .. } => {
                        // Skip runtime stub functions - don't try to refine their types
                        if let Some(callee) = ns.functions.get(func) {
                            if callee.blocks.is_empty() {
                                continue;
                            }
                        }

                        if let (true, new_type) = self.refine_function_type(self.module_functions.get(func).cloned(), args)? {
                            self.module_functions.insert(*func, new_type);
                            self.dirty_functions.insert(*func);
                        }
                    },
                    _ => {}
                }
            }
        }
        Ok(())
    }
    
    fn refine_function_type(&mut self, current_fn_type: Option<Type>, args: &[SsaId]) -> Result<(bool, Type)> {
        if let Some(Type::Function { params, return_type }) = current_fn_type {
            let mut type_changed = false;
            let mut new_params = params.clone();
            for (i, arg) in args.iter().enumerate() {
                if let Some(arg_type) = self.types.get(arg) {
                    if i < new_params.len() {
                        let refined_param = new_params[i].meet(arg_type);

                        if matches!(refined_param, Type::Bottom) {
                            // Type error! Inconsistent constraints
                            bail!("Type error: parameter {} incompatible types", i);
                        } else if refined_param != new_params[i] {
                            new_params[i] = refined_param;
                            type_changed = true;
                        }
                    }
                }
            }
            
            if type_changed {
                // if function signature has changed, we need to update it in the module
                let new_fn_type = Type::Function { params: new_params, return_type: return_type.clone() };
                return Ok((type_changed, new_fn_type));
            }
        }
        Ok((false, Type::Any))
        
    }

    fn process_function(&mut self, f: &mut Function, runtime_get_var: FunctionId, global_env: &HashMap<SymbolId, TypedValue>) -> Result<()> {
        // calculate basic block predecessors for each BB
        f.compute_predecessors();

        self.initalize_types(f);

        // init worklist with the entry BB id
        self.worklist.push_back(f.entry_block);

        // Iterate over worklist Blocks until fixed point or max iterations is reached
        while let Some(bb_id) = self.worklist.pop_front() {
            self.process_block(f, bb_id, runtime_get_var, global_env)?;

            self.iterations_done += 1;
            if self.iterations_done > MAX_ITERATIONS {
                bail!("Type inference pass did not converge in {} iterations", MAX_ITERATIONS);
            }
        }

        self.apply_types(f);
        Ok(())
    }

    fn initalize_types(&mut self, f: &Function) {
        self.types.clear();

        for param in &f.params {
            self.types.insert(param.id, param.ty.clone());
        }

        for bb in &f.blocks {
            bb.instructions
                .iter()
                .filter_map(|i| match i {
                    Instruction::Const { dest, value } => 
                        Some((dest.id, value.ty())),
                    _ => None
                }).for_each(|(id, ty)| {
                    self.types.insert(id, ty);
                });
            
        }


    }
    
    fn process_block(&mut self, f: &Function, bb_id: BlockId, runtime_get_var: FunctionId, global_env: &HashMap<SymbolId, TypedValue>) -> Result<()> {
        let bb = f.find_bloc(bb_id)
                                .ok_or(anyhow!("Unknown blockId: {} in FunctionId: {:?}", bb_id, f.id))?;

        let mut type_changed = false;
        for i in &bb.instructions {
            let new_type = self.infer_instruction(i, f, runtime_get_var, global_env)?;
            if let Some(dest) = i.dest() {
                // lookup current type we have to this SSA_ID
                let old_type = self.types.get(&dest.id);

                if old_type != Some(&new_type) {
                    // overwrite the old type
                    self.types.insert(dest.id, new_type);
                    type_changed = true;

                }
            }
        }

        // if any of the type changed, we want to schedule this Blocks successors for another round
        if type_changed {
            bb.terminator
                .successors()
                .iter()
                .for_each(|bb_id|
                     self.worklist.push_back(*bb_id));
        }

        Ok(())
    }
    
    fn infer_instruction(&self, i: &Instruction, _f: &Function, runtime_get_var: FunctionId, global_env: &HashMap<SymbolId, TypedValue>) -> Result<Type> {
        match i {
            Instruction::Const { value, .. } => Ok(value.ty()),
            Instruction::PrimOp { op, args , ..} => {
                let arg_types: Vec<_> = args
                                            .iter()
                                            .map(|a| self.types.get(a).unwrap_or(&Type::Any))
                                            .cloned()
                                            .collect();
                Ok(op.result_type(&arg_types))
            },
            Instruction::DirectCall { func, args, .. } => {
                // Special case: runtime_get_var returns the actual type stored in the global
                if *func == runtime_get_var && args.len() == 1 {
                    // Try to extract the symbol name from the constant string argument
                    if let Some(symbol_type) = self.lookup_global_type(&args[0], global_env) {
                        return Ok(symbol_type);
                    }
                }

                // Default: use the function's return type
                if let Some(Type::Function { return_type, .. }) = self.module_functions.get(func) {
                    Ok((**return_type).clone())
                } else {
                    Ok(Type::Any)
                }
            },
            Instruction::Call { func, args, .. } => {
                // try to infer based on lambda type
                let arg_types:Vec<_> = args.iter().filter_map(|a| self.types.get(a)).cloned().collect();

                if let Some(Type::Function {  return_type, params }) = self.types.get(func) {
                    if &arg_types != params {
                        // TODO update `func` signature in the namespace and schedule it for reprocessing again with this new information
                    }
                    Ok((**return_type).clone())
                } else {
                    Ok(Type::Any)
                }
            },
            Instruction::MakeClosure { func, .. } => {
                if let Some(fn_type) = self.module_functions.get(func) {
                    Ok(fn_type.clone())
                } else {
                    Ok(Type::Any)
                }
            },
            Instruction::MakeVector { .. } => Ok(Type::Vector),
            Instruction::MakeList { .. } => Ok(Type::List),
            Instruction::Retain { .. } | Instruction::Release { .. } => Ok(Type::Any),
            Instruction::Phi { incoming, .. } => {
                let joined_type = incoming
                    .iter()
                    .map(|(id, _)| self.types.get(id).unwrap_or(&Type::Any))
                    .fold(Type::Bottom, |acc, ty| acc.join(ty));
                Ok(joined_type)
            },
            Instruction::Box { dest, value } => Ok(Type::BoxedLispVal),
            Instruction::Unbox { dest, value, expected_type } => Ok(expected_type.clone()),
        }
    }
    
    fn apply_types(&self, f: &mut Function)  {
        for bb in &mut f.blocks {
            for i in &mut bb.instructions {
                if let Some(dest) = i.dest_mut() {
                    if let Some(new_type) = self.types.get(&dest.id) {
                        dest.ty = new_type.clone();
                    }
                }
            }
        }

        let mut return_type = Type::Bottom; 
        for bb in &f.blocks {
            if let Terminator::Return { value } = &bb.terminator {
                if let Some(ty) = self.types.get(value) {
                    return_type = return_type.join(ty); // join all return types from all blocks
                }
            }
        }
        f.return_type = return_type;
    }

}


#[cfg(test)] 
mod type_inference_tests {
    use crate::{ir::ir_builder::IrLoweringContext, lexer::tokenize, parser::parse};

    use super::*;

    fn source_to_ir(source: &str) -> Result<Namespace> {
        let tokens = tokenize(source)?;
        let ast = parse(&tokens[..])?;
        let ir_ctx = IrLoweringContext::new();
        ir_ctx.lower_program(&ast)
    }

    // Helper to find a function by parameter count (to identify lambdas)
    fn find_function_with_params(ns: &Namespace, param_count: usize) -> Option<&Function> {
        ns.functions.values()
            .find(|f| f.params.len() == param_count && !f.blocks.is_empty())
    }

    // ========================================================================
    // SUCCESS CASES - Where type inference works
    // ========================================================================

    #[test]
    fn test_direct_call_refines_parameters() {
        // Call site has concrete Int arguments, should refine function params
        let source = r#"
(def max (fn [a b]
  (if (> a b) a b)))
(max 2 4)"#;

        let mut ir = source_to_ir(source).expect("Failed to get IR");
        println!("\nBEFORE type inference:\n{}", ir);

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Type inference failed");

        println!("\nAFTER type inference:\n{}", ir);

        // Find the max function (has 2 parameters)
        let max_fn = find_function_with_params(&ir, 2).expect("Couldn't find max function");

        // Should refine both parameters to Int from call site (max 2 4)
        assert_eq!(max_fn.params[0].ty, Type::Int,
            "First parameter should be refined to Int from call site");
        assert_eq!(max_fn.params[1].ty, Type::Int,
            "Second parameter should be refined to Int from call site");

        // Return type should be Int (from if expression joining both Int branches)
        assert_eq!(max_fn.return_type, Type::Int,
            "Return type should be Int");
    }

    #[test]
    fn test_arithmetic_operations_infer_int() {
        let source = r#"
(def formula (fn [a b]
  (if (> a b)
      (+ a (* b 10))
      (* a b))))
(formula 6 24)"#;

        let mut ir = source_to_ir(source).expect("Failed to get IR");
        println!("\nBEFORE:\n{}", ir);

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Type inference failed");

        println!("\nAFTER:\n{}", ir);

        let formula_fn = find_function_with_params(&ir, 2).expect("Couldn't find formula function");

        // Parameters refined from call site
        assert_eq!(formula_fn.params[0].ty, Type::Int);
        assert_eq!(formula_fn.params[1].ty, Type::Int);

        // Return type is Int (both branches return Int)
        assert_eq!(formula_fn.return_type, Type::Int);
    }

    #[test]
    fn test_comparison_returns_bool() {
        let source = r#"
(def is_greater (fn [x y]
  (> x y)))
(is_greater 5 3)"#;

        let mut ir = source_to_ir(source).expect("Failed");

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        let func = find_function_with_params(&ir, 2).expect("Couldn't find function");

        // Parameters should be Int (from call site)
        assert_eq!(func.params[0].ty, Type::Int);
        assert_eq!(func.params[1].ty, Type::Int);

        // Return type should be Bool (comparison operator)
        assert_eq!(func.return_type, Type::Bool,
            "Comparison operators should return Bool");
    }

    #[test]
    fn test_constant_folding_and_types() {
        let source = r#"
(def compute (fn []
  (+ (* 2 3) 4)))"#;

        let mut ir = source_to_ir(source).expect("Failed");

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        let func = find_function_with_params(&ir, 0).expect("Couldn't find function");

        // Return type should be Int (all arithmetic with Int constants)
        assert_eq!(func.return_type, Type::Int,
            "Arithmetic with Int constants should return Int");
    }

    #[test]
    fn test_if_expression_joins_branch_types() {
        let source = r#"
(def conditional (fn [flag]
  (if flag
      42
      99)))"#;

        let mut ir = source_to_ir(source).expect("Failed");

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        let func = find_function_with_params(&ir, 1).expect("Couldn't find function");

        // Return type should be Int (both branches are Int)
        assert_eq!(func.return_type, Type::Int,
            "If with Int branches should return Int");
    }

    #[test]
    fn test_nested_calls_propagate_types() {
        let source = r#"
(def add (fn [a b] (+ a b)))
(def double (fn [x] (add x x)))
(double 5)"#;

        let mut ir = source_to_ir(source).expect("Failed");
        println!("\nBEFORE:\n{}", ir);

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        println!("\nAFTER:\n{}", ir);

        // Find both functions
        let double_fn = ir.functions.values()
            .find(|f| f.params.len() == 1 && !f.blocks.is_empty())
            .expect("Couldn't find double function");

        let add_fn = ir.functions.values()
            .find(|f| f.params.len() == 2 && !f.blocks.is_empty())
            .expect("Couldn't find add function");

        // double parameter should be Int from call site (double 5)
        assert_eq!(double_fn.params[0].ty, Type::Int);
        assert_eq!(double_fn.return_type, Type::Int);

        // add parameters should be Int (called with x, x where x is Int)
        assert_eq!(add_fn.params[0].ty, Type::Int);
        assert_eq!(add_fn.params[1].ty, Type::Int);
        assert_eq!(add_fn.return_type, Type::Int);
    }

    // ========================================================================
    // FAILURE CASES - Documenting current limitations
    // ========================================================================

    #[test]
    fn test_no_backward_inference_from_operators() {
        // LIMITATION: Can't infer parameter types from their usage in operators
        // without a call site providing concrete types
        let source = r#"
(def add (fn [a b] (+ a b)))"#;

        let mut ir = source_to_ir(source).expect("Failed");

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        let add_fn = find_function_with_params(&ir, 2).expect("Couldn't find function");

        // EXPECTED FAILURE: Parameters remain Any
        // We can't infer that + requires Int without constraint propagation
        assert_eq!(add_fn.params[0].ty, Type::Any,
            "LIMITATION: No backward inference - parameters stay Any without call site");
        assert_eq!(add_fn.params[1].ty, Type::Any,
            "LIMITATION: No backward inference - parameters stay Any without call site");

        // Return type is also Any (because params are Any)
        // Even though we know + with concrete Ints returns Int
        assert_eq!(add_fn.return_type, Type::Any,
            "LIMITATION: Return type depends on parameter types");
    }

    #[test]
    fn test_higher_order_functions_stay_generic() {
        // LIMITATION: Higher-order functions without concrete instantiation
        let source = r#"
(def apply (fn [f x] (f x)))"#;

        let mut ir = source_to_ir(source).expect("Failed");

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        let apply_fn = find_function_with_params(&ir, 2).expect("Couldn't find function");

        // EXPECTED FAILURE: Can't infer types without seeing how apply is used
        assert_eq!(apply_fn.params[0].ty, Type::Any,
            "LIMITATION: Function parameter type unknown without usage");
        assert_eq!(apply_fn.params[1].ty, Type::Any,
            "LIMITATION: Argument type unknown without usage");
        assert_eq!(apply_fn.return_type, Type::Any,
            "LIMITATION: Return type unknown without concrete function");
    }

    #[test]
    fn test_polymorphic_usage_generalizes() {
        // LIMITATION: Function used with different types gets generalized
        let source = r#"
(def identity (fn [x] x))
(identity 5)
(identity "hello")"#;

        let mut ir = source_to_ir(source).expect("Failed");

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        let identity_fn = find_function_with_params(&ir, 1).expect("Couldn't find function");

        // With join-based refinement, Int.join(String) = Union or Any
        // Current implementation should use join, so type gets generalized
        // NOTE: This depends on implementation - it might end up as Union(Int, String)
        let param_type = &identity_fn.params[0].ty;

        // Either Any or a Union - not a concrete type
        let is_general = matches!(param_type, Type::Any) ||
                         matches!(param_type, Type::Union(_));

        assert!(is_general,
            "LIMITATION: Polymorphic usage generalizes type (got {:?})", param_type);
    }

    #[test]
    fn test_indirect_call_without_closure_tracking() {
        // LIMITATION: Indirect calls through closure values
        let source = r#"
(def make_adder (fn [n]
  (fn [x] (+ x n))))
(def add5 (make_adder 5))
(add5 10)"#;

        let mut ir = source_to_ir(source).expect("Failed");
        println!("\nBEFORE:\n{}", ir);

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        println!("\nAFTER:\n{}", ir);

        // Find the inner closure (1 parameter: x)
        let inner_fn = ir.functions.values()
            .filter(|f| f.params.len() == 1 && !f.blocks.is_empty())
            .find(|f| {
                // Distinguish from other 1-param functions by checking for PrimOp Add
                f.blocks.iter().any(|bb|
                    bb.instructions.iter().any(|i|
                        matches!(i, Instruction::PrimOp { op: super::super::ir_types::Operator::Add, .. })
                    )
                )
            })
            .expect("Couldn't find inner closure");

        // EXPECTED LIMITATION: Without closure-to-function tracking,
        // the indirect call (add5 10) may not propagate types back
        // The parameter might stay Any or get refined depending on implementation
        println!("Inner closure param type: {:?}", inner_fn.params[0].ty);

        // This is a known limitation - document it
        // Depending on your implementation, this might be Any or Int
        let is_inferred = inner_fn.params[0].ty == Type::Int;
        if !is_inferred {
            println!("LIMITATION: Indirect call through closure didn't refine parameter");
        }
    }

    #[test]
    fn test_recursive_function_inference() {
        // SUCCESS CASE: Recursion with concrete call site
        let source = r#"
(def factorial (fn [n]
  (if (<= n 1)
      1
      (* n (factorial (- n 1))))))
(factorial 5)"#;

        let mut ir = source_to_ir(source).expect("Failed");

        let mut type_inference = TypeInference::new();
        type_inference.process(&mut ir).expect("Inference failed");

        let fact_fn = find_function_with_params(&ir, 1).expect("Couldn't find factorial");

        // Should infer Int from call site and usage
        assert_eq!(fact_fn.params[0].ty, Type::Int,
            "Recursive function parameter should be Int from call site");
        assert_eq!(fact_fn.return_type, Type::Int,
            "Factorial return type should be Int");
    }

}