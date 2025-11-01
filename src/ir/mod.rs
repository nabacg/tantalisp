pub mod ir_types;
pub mod instructions;
pub mod ir_builder;
mod type_inference;
mod ref_count_optimizer;

#[cfg(test)]
mod ir_builder_tests;