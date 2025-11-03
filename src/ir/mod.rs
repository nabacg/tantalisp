use instructions::Namespace;
use ir_builder::IrLoweringContext;
use ref_count_optimizer::RefCountOptimizer;

use crate::parser::SExpr;
use anyhow::Result;

pub mod ir_types;
pub mod instructions;
pub mod ir_builder;
mod type_inference;
mod ref_count_optimizer;

#[cfg(test)]
mod ir_builder_tests;



pub fn lower_to_ir(ast: &[SExpr]) -> Result<Namespace> {
    let ctx = IrLoweringContext::new();

    let mut ir = ctx.lower_program(&ast)?;

    let mut ref_count_opto = RefCountOptimizer::new();
    ref_count_opto.process(&mut ir)?;

    Ok(ir)
}