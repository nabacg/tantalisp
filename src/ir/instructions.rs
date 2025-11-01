use std::collections::HashMap;

use super::ir_types::{BlockId, Constant, FunctionId, Operator, SsaId, SymbolId, Type, TypedValue};


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
        op: Operator,
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
    // Make closure (create lambda) - captures environment
    MakeClosure {
        dest: TypedValue,
        func: FunctionId,
        captures: Vec<SsaId>,
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
                    | Instruction::MakeClosure { dest, ..}
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
            | Instruction::MakeClosure { dest, ..}
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
            Instruction::Call { func, args, .. } => {
                let mut uses = args.clone();
                uses.push(*func);
                uses
            } ,
            Instruction::MakeClosure { captures, ..} => captures.clone(),
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

impl BasicBlock {
    pub fn emit(&mut self, i: Instruction) {
        self.instructions.push(i);
    }

    pub fn terminate(&mut self, t: Terminator) {
        self.terminator = t;
    }
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
    pub global_env: HashMap<SymbolId, TypedValue>,
    pub symbols: HashMap<String, SymbolId>,
    next_symbol_id: u32,

    // Runtime function IDs (registered during initialization)
    pub runtime_get_var: FunctionId,
    pub runtime_set_var: FunctionId,
}

impl Namespace {
    pub fn new() -> Self {
        let mut ns = Self {
            functions: HashMap::new(),
            global_env: HashMap::new(),
            symbols: HashMap::new(),
            next_function_id: 0,
            next_symbol_id: 0,
            runtime_get_var: FunctionId(0),  // Placeholder
            runtime_set_var: FunctionId(0),  // Placeholder
        };

        // Register runtime functions
        ns.runtime_get_var = ns.register_runtime_function(
            "runtime_get_var",
            vec![Type::String],      // Takes symbol name
            Type::BoxedLispVal       // Returns boxed value
        );

        ns.runtime_set_var = ns.register_runtime_function(
            "runtime_set_var",
            vec![Type::String, Type::BoxedLispVal],  // Takes name + value
            Type::BoxedLispVal       // Returns the value set
        );

        // Now next_function_id is 2, first user function gets ID 2
        ns
    }

    /// Register a runtime function stub (has signature but no implementation blocks)
    fn register_runtime_function(&mut self, name: &str, params: Vec<Type>, return_type: Type) -> FunctionId {
        let func_id = FunctionId(self.next_function_id);
        self.next_function_id += 1;

        // Create parameter TypedValues
        let typed_params: Vec<TypedValue> = params.into_iter()
            .enumerate()
            .map(|(i, ty)| TypedValue { id: SsaId(i as u32), ty })
            .collect();

        // Create a stub function (no actual blocks, just signature)
        let func = Function {
            id: func_id,
            name: name.to_string(),
            params: typed_params,
            return_type,
            blocks: vec![],  // Empty - this is a runtime stub
            entry_block: BlockId(0),
        };

        self.functions.insert(func_id, func);
        func_id
    }

    pub fn add_function(&mut self, f: Function) -> FunctionId {
        let id = f.id;
        self.functions.insert(id, f);
        id
    }

    pub fn intern_symbol(&mut self, var_id: &str) -> SymbolId {
        if let Some(&id) = self.symbols.get(var_id) {
            return id;
        }
        let id = SymbolId(self.next_symbol_id);
        self.next_symbol_id += 1;
        self.symbols.insert(var_id.to_string(), id);
        id
    }

    /// Look up an existing symbol ID (doesn't intern)
    pub fn get_symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbols.get(name).copied()
    }
    
    /// Add a global variable definition
    pub fn add_def(&mut self, sym_id: SymbolId, val_type: TypedValue) {
        self.global_env.insert(sym_id, val_type);
    }

    /// Look up a global variable's type info
    pub fn lookup_global(&self, sym_id: SymbolId) -> Option<&TypedValue> {
        self.global_env.get(&sym_id)
    }

    /// Reverse lookup: SymbolId → String (useful for debugging)
    pub fn get_symbol_name(&self, id: SymbolId) -> Option<&str> {
        self.symbols.iter()
            .find(|(_, sym_id)| *sym_id == &id)
            .map(|(name, _)| name.as_str())
    }

}

// ============================================================================
// Pretty Printing (LLVM-style IR formatting)
// ============================================================================

use std::fmt;

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Const { dest, value } => {
                write!(f, "  {}: {} = const ", dest.id, dest.ty)?;
                match value {
                    Constant::Int(i) => write!(f, "{}", i),
                    Constant::Bool(b) => write!(f, "{}", b),
                    Constant::String(s) => write!(f, "\"{}\"", s.escape_debug()),
                    Constant::Unit => write!(f, "unit"),
                    Constant::Nil => write!(f, "nil"),
                }
            }
            Instruction::PrimOp { dest, op, args } => {
                write!(f, "  {}: {} = {:?}(", dest.id, dest.ty, op)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Instruction::DirectCall { dest, func, args } => {
                write!(f, "  {}: {} = call @f{}(", dest.id, dest.ty, func.0)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Instruction::Call { dest, func, args } => {
                write!(f, "  {}: {} = call {}(", dest.id, dest.ty, func)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Instruction::MakeClosure { dest, func, captures } => {
                write!(f, "  {}: {} = makeclosure @f{}[", dest.id, dest.ty, func.0)?;
                for (i, cap) in captures.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", cap)?;
                }
                write!(f, "]")
            }
            Instruction::MakeVector { dest, elements } => {
                write!(f, "  {}: {} = makevector [", dest.id, dest.ty)?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            Instruction::MakeList { dest, elements } => {
                write!(f, "  {}: {} = makelist (", dest.id, dest.ty)?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", elem)?;
                }
                write!(f, ")")
            }
            Instruction::Retain { value } => {
                write!(f, "  retain {}", value)
            }
            Instruction::Release { value } => {
                write!(f, "  release {}", value)
            }
            Instruction::Phi { dest, incoming } => {
                write!(f, "  {}: {} = phi [", dest.id, dest.ty)?;
                for (i, (val, block)) in incoming.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "({}, {})", val, block)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Return { value } => {
                write!(f, "  return {}", value)
            }
            Terminator::Jump { target } => {
                write!(f, "  jump {}", target)
            }
            Terminator::Branch { condition, truthy_block, falsy_block } => {
                write!(f, "  branch {}, {}, {}", condition, truthy_block, falsy_block)
            }
            Terminator::Unreachable => {
                write!(f, "  unreachable")
            }
        }
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}:", self.id)?;
        for instr in &self.instructions {
            writeln!(f, "{}", instr)?;
        }
        writeln!(f, "{}", self.terminator)
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Function signature
        write!(f, "function @f{}(", self.id.0)?;
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}: {}", param.id, param.ty)?;
        }
        writeln!(f, ") -> {} {{", self.return_type)?;

        // Body
        if self.blocks.is_empty() {
            writeln!(f, "  ; runtime stub")?;
        } else {
            for block in &self.blocks {
                write!(f, "{}", block)?;
            }
        }

        writeln!(f, "}}")
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "; Namespace IR")?;
        writeln!(f, "; {} functions, {} symbols, {} globals\n",
            self.functions.len(),
            self.symbols.len(),
            self.global_env.len())?;

        // Print symbol table
        if !self.symbols.is_empty() {
            writeln!(f, "; Symbol Table:")?;
            let mut syms: Vec<_> = self.symbols.iter().collect();
            syms.sort_by_key(|(_, id)| id.0);
            for (name, id) in syms {
                writeln!(f, ";   {} = SymbolId({})", name, id.0)?;
            }
            writeln!(f)?;
        }

        // Print global environment
        if !self.global_env.is_empty() {
            writeln!(f, "; Global Environment:")?;
            for (sym_id, typed_val) in &self.global_env {
                let name = self.get_symbol_name(*sym_id).unwrap_or("?");
                writeln!(f, ";   {} = {}: {}", name, typed_val.id, typed_val.ty)?;
            }
            writeln!(f)?;
        }

        // Print functions in order
        let mut func_ids: Vec<_> = self.functions.keys().copied().collect();
        func_ids.sort_by_key(|id| id.0);

        for func_id in func_ids {
            let func = &self.functions[&func_id];
            writeln!(f, "{}", func)?;
        }

        Ok(())
    }
}


#[cfg(test)]
mod instructions_tests {
    use super::*;
    use Instruction::*;


    fn int_val(id: SsaId) -> TypedValue {
        TypedValue { id, ty: Type::Int }
    }

    #[test]
    fn test_compute_predecessors() {
        let entry_block_id = BlockId(0);
        let truthy_block_id = BlockId(1);
        let falsy_block_id = BlockId(2);
        let merge_block_id = BlockId(3);

        let arg_0 =  SsaId(0);
        let var_x  = SsaId(1); 
        let var_cmp =  SsaId(2);
        let var_t =  SsaId(3);
        let var_f =  SsaId(4);
        let ret_var = SsaId(5);

        let merge_bb = BasicBlock {
            id: merge_block_id,
            instructions: vec![ 
                Phi { dest: int_val(ret_var), incoming: vec![
                    (var_t, truthy_block_id),
                    (var_f, falsy_block_id)
                ] }
            ],
            terminator: Terminator::Return { value: ret_var },
            predecessors: vec![]
        };

        let truthy_bb = BasicBlock { 
                id: truthy_block_id, 
                instructions: vec![
                    Instruction::Const { dest: int_val(var_t), value: Constant::Int(42) }
                ], 
                terminator: Terminator::Jump { target: merge_block_id }, 
                predecessors: vec![] };

        let falsy_bb = BasicBlock { 
            id: falsy_block_id, 
            instructions: vec![
                Instruction::Const { dest: int_val(var_t), value: Constant::Int(-177) }
            ], 
            terminator: Terminator::Jump { target: merge_block_id }, 
            predecessors: vec![] };
    

        let entry_bb = BasicBlock{
            id: entry_block_id,
            instructions: vec![
                Instruction::Const { dest: int_val(var_x), value: Constant::Int(10) },
                Instruction::PrimOp { dest: int_val(var_cmp), op: Operator::Eq, args: vec![arg_0, var_x] }
            ],
            terminator: Terminator::Branch { condition: var_cmp, truthy_block: truthy_block_id, falsy_block: falsy_block_id },
            predecessors: vec![],
        };

        let mut f1 = Function {
            blocks: vec![entry_bb, truthy_bb, falsy_bb, merge_bb],
            id: FunctionId(0),
            name: "f".to_string(),
            params: vec![ int_val(arg_0) ],
            return_type: Type::Int,
            entry_block: entry_block_id,
        };

        f1.compute_predecessors();


        let entry_bb =  f1.find_bloc(entry_block_id).unwrap();
        assert_eq!(vec![truthy_block_id, falsy_block_id], 
            entry_bb.terminator.successors());

        assert!(entry_bb.predecessors.is_empty());

        let truthy_bb = f1.find_bloc(truthy_block_id).unwrap();
        assert_eq!(vec![merge_block_id], truthy_bb.terminator.successors());
        assert_eq!(vec![entry_block_id], truthy_bb.predecessors);

        let falsy_bb = f1.find_bloc(falsy_block_id).unwrap();
        assert_eq!(vec![merge_block_id], falsy_bb.terminator.successors());
        assert_eq!(vec![entry_block_id], falsy_bb.predecessors);

        let merge_bb = f1.find_bloc(merge_block_id).unwrap();
        assert!(merge_bb.terminator.successors().is_empty()); // function returns here 
        assert_eq!(vec![truthy_block_id, falsy_block_id], merge_bb.predecessors);


    }
}