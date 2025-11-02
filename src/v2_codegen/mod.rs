mod codegen;
mod runtime;

pub use runtime::garbage_collector::{set_gc_debug_mode, start_gc_monitor};
pub use codegen::CodeGen;