use std::collections::HashMap;

use super::ir_types::{BlockId, Constant, FunctionId, PrimOp, SsaId, Type, TypedValue};


#[derive(Debug, Clone)]
pub enum Instruction {
    // LOAD constant value 
    Const {
        dest: TypedValue,
        value: Constant
    },
    // Primitive (BuiltIn) operations (arithmetic, comparison)
    PrimOp {
        dest: TypedValue,
        op: PrimOp,
        args: Vec<SsaId>
    },
    // Call a statically known function (call direct, enables inlining)
    DirectCall {
        dest: TypedValue,
        func: FunctionId, // Statically known function ID
        args: Vec<SsaId>
    },
    // Call an arbitrary function (indirect, via pointer)
    Call {
        dest: TypedValue,
        func: SsaId,
        args: Vec<SsaId>
    },
    // Create a vector literal 
    MakeVector {
        dest: TypedValue,
        elements: Vec<SsaId>
    },
    // Make list
    MakeList {
        dest: TypedValue,
        elements: Vec<SsaId>
    },
    // Increment reference count 
    Retain {
        value: SsaId
    },
    // Decrement reference count (may free)
    Release {
        value: SsaId
    },
    // Phi node
    Phi {
        dest: TypedValue,
        incoming: Vec<(SsaId, BlockId)>
    }
}

impl Instruction {
    pub fn dest(&self) -> Option<&TypedValue> {
        match self {
            Instruction::Const { dest, .. } 
            | Instruction::PrimOp { dest , ..} 
            | Instruction::DirectCall { dest, ..} 
            | Instruction::Call { dest, ..} 
            | Instruction::MakeVector { dest, .. } 
            | Instruction::MakeList { dest, ..} 
            | Instruction::Phi { dest, ..} => Some(dest),
            Instruction::Retain { .. } | Instruction::Release { .. } => None,
        }
    }

    pub fn dest_mut(&mut self) -> Option<&mut TypedValue> {
        match self {
            Instruction::Const { dest, .. } 
            | Instruction::PrimOp { dest , ..} 
            | Instruction::DirectCall { dest, ..} 
            | Instruction::Call { dest, ..} 
            | Instruction::MakeVector { dest, .. } 
            | Instruction::MakeList { dest, ..} 
            | Instruction::Phi { dest, ..} => Some(dest),
            Instruction::Retain { .. } | Instruction::Release { .. } => None,
        }
    }

    /// Gett all SSA variables used by this instruction
    /// very useful for variable lifetime tracking, like during reference counting
    pub fn uses(&self) -> Vec<SsaId> {
        match self {
            Instruction::Const { .. } => vec![],
            Instruction::PrimOp { args, .. } => args.clone(),
            Instruction::DirectCall { args, .. } => args.clone(),
            Instruction::Call {  args, .. } => args.clone(),
            Instruction::MakeVector { elements, .. } => elements.clone(),
            Instruction::MakeList { elements, .. } => elements.clone(),
            Instruction::Retain { value } => vec![*value],
            Instruction::Release { value } => vec![*value],
            Instruction::Phi { incoming, .. } => incoming
                                                                            .iter()
                                                                            .map(|(val, _)| *val)
                                                                            .collect(),
        }
    }
}

/// Terminator instructions (ones that end each Basic Block)
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Function return
    Return {
        value: SsaId,
    },
    /// Unconditional jump
    Jump {
        target: BlockId
    },
    /// Conditional jump
    Branch {
        condition: SsaId,
        truthy_block: BlockId,
        falsy_block: BlockId,
    },
    /// Unreachable code (panic, infinite loop ets)
    Unreachable
}

impl Terminator {
    /// Successor blocks
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Jump { target } => vec![*target],
            Terminator::Branch { truthy_block,  falsy_block, .. } => vec![*truthy_block, *falsy_block],
            Terminator::Unreachable | Terminator::Return { .. } => vec![]
        }
    }    
}


/// Basic Block 
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
    /// for caching predecessor blocks, so they don't need to be repeatedly recallculated for various CFG algorithms
    pub predecessors: Vec<BlockId>
}

/// Function in IR 
#[derive(Debug, Clone)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub params: Vec<TypedValue>,  // TypedValue contains (SsId, Type) - perfect for Param
    pub return_type: Type,
    pub blocks: Vec<BasicBlock>,
    pub entry_block: BlockId,
}

impl Function {
    pub fn find_bloc(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.iter().find(|bb| bb.id == id)
    }

    pub fn find_block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.iter_mut().find(|bb| bb.id == id)
    }

    /// Populate predecessors for each Basic Block in the function
    pub fn compute_predecessors(&mut self) {
        // Clear existing 
        for bb in &mut self.blocks {
            bb.predecessors.clear();
        }

        let block_ids: Vec<_> = self.blocks.iter().map(|bb| bb.id).collect();
        for id in block_ids {
            let current_bb = self.find_bloc(id).unwrap(); 
            let successors = current_bb.terminator.successors();
            for succ_id in successors {
                if let Some(succ_bb) = self.find_block_mut(succ_id) {
                    succ_bb.predecessors.push(id);
                }
            }
        }
    }
}


/// Namespace (module) containing all functions in current compilation unit
pub struct Namespace {
    pub functions: HashMap<FunctionId, Function>,
    pub next_function_id: u32,
}

impl Namespace {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            next_function_id: 0
        }
    }

    pub fn add_function(&mut self, f: Function) -> FunctionId {
        let id = f.id;
        self.functions.insert(id, f);
        id
    }
}