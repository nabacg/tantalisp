use std::collections::HashMap;

use super::{instructions::{Function, Instruction, Namespace}, ir_types::{BlockId, SsaId, TypedValue}};
use anyhow::Result;

pub struct RefCountOptimizer {
    // keep track of the last use each SSA value has in each block (like a simple lifetime?)
    last_uses: HashMap<(BlockId, SsaId), usize>
}

impl RefCountOptimizer {
    pub fn new() -> Self {
        Self {
            last_uses: HashMap::new()
        }
    }

    pub fn process(&mut self, ns: &mut Namespace) -> Result<()> {
        for (_, f) in &mut ns.functions { 
            self.process_function(f)?;
        }

        Ok(())
    }
    
    fn process_function(&mut self, f: &mut Function) -> Result<()> {
        self.compute_last_uses(f);

        self.insert_ref_count_instr(f);

        self.optimize_rc_pairs(f);

        Ok(())
    }

    fn compute_last_uses(&mut self, f: &Function) {
        for bb in &f.blocks{
            let mut uses: HashMap<SsaId, usize> = HashMap::new();
            for (idx, i) in bb.instructions.iter().enumerate()  {
                for used in i.uses() {
                    // storing the last idx where each SsaId is used, by overwriting it in the HashMap
                    uses.insert(used, idx);
                }
            }
            // store last_idx for this block
            for (ssa_id, last_idx) in uses {
                self.last_uses.insert((bb.id, ssa_id), last_idx);
            }
        }
    }

    fn insert_ref_count_instr(&mut self, f: &mut Function) {
        for bb in &mut f.blocks {
            let mut new_instr = Vec::new();
            for (idx, i) in bb.instructions.iter().enumerate() {
                new_instr.push(i.clone());
                for used in i.uses() {
                    if let Some(TypedValue {ty, ..}) = i.dest() {
                        // if current i dest is heap allocated 
                        if ty.needs_rc() {
                            if self.last_uses.get(&(bb.id, used)) == Some(&idx) {
                                // this is the last use of this SSA id and we can insert release
                                new_instr.push(Instruction::Release { value: used });
                            }
                        }
                    }
                }
            }

            bb.instructions = new_instr;
        }
    }


    fn optimize_rc_pairs(&mut self, f: &mut Function) { 
        for bb in &mut f.blocks {
            let mut i = 0;
            while i + 1 < bb.instructions.len() {
                let is_redundant = match (&bb.instructions[i], &bb.instructions[i+1]) {
                    (
                        Instruction::Retain { value: v1 },
                        Instruction::Release { value: v2 }
                    ) => v1 == v2,
                    _ => false 
                };

                if is_redundant {
                    bb.instructions.remove(i);
                    bb.instructions.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }
}



#[cfg(test)]
mod ref_count_optimizer_tests {
    use super::*;
    use crate::{ir::ir_builder::lower_to_ir, lexer::tokenize, parser::parse};

    fn source_to_ir(source: &str) -> Result<Namespace> {
        let tokens = tokenize(source)?;
        let ast = parse(&tokens)?;
        lower_to_ir(&ast)
    }

    #[test]
    fn test_expect_release_inserted() {
        let mut ir = source_to_ir(r#"
            (def f (fn [x] (map (fn [y] (+ y 1)) (range x))))
            (f 3)
        "#).expect("Input failed to parse or lower ");

        println!("BEFORE: {}", ir);

        let mut opt = RefCountOptimizer::new();
        opt.process(&mut ir);

        println!("AFTER: {}", ir);

    }
}