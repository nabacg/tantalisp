use std::{any::Any, collections::{HashMap, VecDeque}, hash::Hash, process::Termination};
use anyhow::{anyhow, bail, Result};
use super::{instructions::{Function, Instruction, Namespace, Terminator}, ir_types::{BlockId, FunctionId, SsaId, Type}};


pub struct TypeInference {
    // Currently known types
    types: HashMap<SsaId, Type>,
    // worklist of Blocks to process
    worklist: VecDeque<BlockId>,
    module_functions: HashMap<FunctionId, Type>,
    iterations_done: usize
}

const MAX_ITERATIONS: usize = 1000;

impl TypeInference {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            worklist: VecDeque::new(),
            module_functions: HashMap::new(),
            iterations_done: 0
        }
    }

    pub fn process(&mut self, ns: &mut Namespace) -> Result<()> {
        //first add return_types of all module functions 
        for (id, f) in &ns.functions {
            // TODO - maybe we should store full function signature, not just return_type?
            self.module_functions.insert(*id, f.return_type.clone());
        }

        for (_, f) in &mut ns.functions {
            self.worklist.clear(); // should really be emptied by a previous function, but just to be sure
            self.process_function(f)?;
        }
        Ok(())
    }
    
    fn process_function(&mut self, f: &mut Function) -> Result<()> {
        // calculate basic block predecessors for each BB
        f.compute_predecessors();

        self.initalize_types(f);

        // init worklist with the entry BB id
        self.worklist.push_back(f.entry_block);

        // Iterate over worklist Blocks until fixed point or max iterations is reached
        while let Some(bb_id) = self.worklist.pop_front() {
            self.process_block(f, bb_id)?;

            self.iterations_done += 1;
            if self.iterations_done > MAX_ITERATIONS {
                bail!("Type inference pass did not converege in {} iterations", MAX_ITERATIONS);
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
    
    fn process_block(&mut self, f: &Function, bb_id: BlockId) -> Result<()> {
        let bb = f.find_bloc(bb_id)
                                .ok_or(anyhow!("Unknown blockId: {} in FunctionId: {:?}", bb_id, f.id))?;

        let mut type_changed = false;
        for i in &bb.instructions {
            let new_type = self.infer_instruction(i, f)?;
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
    
    fn infer_instruction(&self, i: &Instruction, f: &Function) -> Result<Type> {
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
            Instruction::DirectCall { func, .. } => {
                if let Some(return_type) = self.module_functions.get(func) {
                    Ok(return_type.clone())
                } else {
                    Ok(Type::Any)
                }
            },
            Instruction::Call { func, .. } => {
                // try to infer based on lambda type
                if let Some(Type::Function {  return_type, .. }) = self.types.get(func) {
                    Ok((**return_type).clone())
                } else {
                    Ok(Type::Any)
                }
            },
            Instruction::MakeClosure { func, .. } => {
                if let Some(return_type) = self.module_functions.get(func) {
                    Ok(return_type.clone())
                } else {
                    Ok(Type::Any)
                }
            },
            Instruction::MakeVector { .. } => Ok(Type::Vector),
            Instruction::MakeList { .. } => Ok(Type::List),
            Instruction::Retain { .. } => Ok(Type::Any),
            Instruction::Release { .. } => Ok(Type::Any),
            Instruction::Phi { incoming, .. } => {
                let joined_type = incoming
                    .iter()
                    .map(|(id, _)| self.types.get(id).unwrap_or(&Type::Any))
                    .fold(Type::Bottom, |acc, ty| acc.join(ty));
                Ok(joined_type)
            },
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

    #[test]
    fn test_max_fn() {
        let source = r#"
(def max (fn [a b]
  (if (> a b)
      a
      b)))
(max 2 4)"#;

      let mut ir = source_to_ir(source).expect("Failed to get IR");
      println!("BEFORE IR: {}", ir);

      let mut type_inference = TypeInference::new();
      type_inference.process(&mut ir);

      println!("AFTER IR: {}", ir);

    }



    #[test]
    fn test_integer_math() {
        let source = r#"
(def formula (fn [a b]
  (if (> a b)
      (+ a (* b 10))
      (* a b))))
    (formula (+ 2 4) 24)"#;

      let mut ir = source_to_ir(source).expect("Failed to get IR");
      println!("BEFORE IR: {}", ir);

      let mut type_inference = TypeInference::new();
      type_inference.process(&mut ir);

      println!("AFTER IR: {}", ir);

    }

}