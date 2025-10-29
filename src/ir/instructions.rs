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
}