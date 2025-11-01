use crate::parser::SExpr;
use crate::ir::ir_builder::IrLoweringContext;
use crate::ir::instructions::{Namespace, Function, Instruction, Terminator};
use crate::ir::ir_types::{FunctionId, Type, Operator, Constant};
use anyhow::Result;

/// Helper to lower a single expression and get the resulting namespace
fn lower_expr_to_namespace(expr: &SExpr) -> Result<Namespace> {
    let ctx = IrLoweringContext::new();
    ctx.lower_program(&[expr.clone()])
}

/// Helper to lower multiple expressions and get the resulting namespace
fn lower_exprs_to_namespace(exprs: &[SExpr]) -> Result<Namespace> {
    let ctx = IrLoweringContext::new();
    ctx.lower_program(exprs)
}

/// Helper to find the toplevel function in a namespace
fn get_toplevel_function(ns: &Namespace) -> Option<&Function> {
    // The toplevel function is added after runtime functions
    // Runtime functions are IDs 0 and 1, so toplevel should be ID 2
    ns.functions.get(&FunctionId(2))
}

/// Helper to count instructions of a specific type in a function
fn count_instructions<F>(func: &Function, predicate: F) -> usize
where
    F: Fn(&Instruction) -> bool
{
    func.blocks.iter()
        .flat_map(|bb| &bb.instructions)
        .filter(|inst| predicate(inst))
        .count()
}

#[cfg(test)]
mod ir_lowering_tests {
    use super::*;

    #[test]
    fn test_lower_int_constant() {
        let expr = SExpr::Int(42);
        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");

        // Should have runtime functions + toplevel
        assert!(ns.functions.len() >= 3);

        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");
        assert_eq!(toplevel.name, "<toplevel>");
        assert!(toplevel.params.is_empty());

        // Should have at least one const instruction
        let const_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Const { value: Constant::Int(42), .. })
        });
        assert_eq!(const_count, 1, "Should have exactly one Const(42) instruction");
    }

    #[test]
    fn test_lower_bool_constants() {
        // Test true
        let expr_true = SExpr::Bool(true);
        let ns = lower_expr_to_namespace(&expr_true).expect("Failed to lower true");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        let true_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Const { value: Constant::Bool(true), .. })
        });
        assert_eq!(true_count, 1);

        // Test false
        let expr_false = SExpr::Bool(false);
        let ns = lower_expr_to_namespace(&expr_false).expect("Failed to lower false");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        let false_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Const { value: Constant::Bool(false), .. })
        });
        assert_eq!(false_count, 1);
    }

    #[test]
    fn test_lower_string_constant() {
        let expr = SExpr::String("hello".to_string());
        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        let string_count = count_instructions(toplevel, |inst| {
            matches!(inst,
                Instruction::Const { value: Constant::String(s), .. }
                if s == "hello"
            )
        });
        assert_eq!(string_count, 1);
    }

    #[test]
    fn test_lower_addition() {
        // (+ 40 2)
        let expr = SExpr::List(vec![
            SExpr::Symbol("+".to_string()),
            SExpr::Int(40),
            SExpr::Int(2),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have PrimOp Add instruction
        let add_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::PrimOp { op: Operator::Add, .. })
        });
        assert_eq!(add_count, 1);

        // Should have two Const Int instructions (40 and 2)
        let const_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Const { value: Constant::Int(_), .. })
        });
        assert_eq!(const_count, 2);
    }

    #[test]
    fn test_lower_arithmetic_operations() {
        let ops = vec![
            ("+", Operator::Add),
            ("-", Operator::Sub),
            ("*", Operator::Mul),
            ("/", Operator::Div),
        ];

        for (op_str, expected_op) in ops {
            let expr = SExpr::List(vec![
                SExpr::Symbol(op_str.to_string()),
                SExpr::Int(10),
                SExpr::Int(5),
            ]);

            let ns = lower_expr_to_namespace(&expr)
                .expect(&format!("Failed to lower {}", op_str));
            let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

            let op_count = count_instructions(toplevel, |inst| {
                matches!(inst, Instruction::PrimOp { op, .. } if op == &expected_op)
            });
            assert_eq!(op_count, 1, "Should have exactly one {} operation", op_str);
        }
    }

    #[test]
    fn test_lower_comparison_operations() {
        let ops = vec![
            ("=", Operator::Eq),
            ("!=", Operator::Ne),
            ("<", Operator::Lt),
            (">", Operator::Gt),
            ("<=", Operator::Le),
            (">=", Operator::Ge),
        ];

        for (op_str, expected_op) in ops {
            let expr = SExpr::List(vec![
                SExpr::Symbol(op_str.to_string()),
                SExpr::Int(1),
                SExpr::Int(2),
            ]);

            let ns = lower_expr_to_namespace(&expr)
                .expect(&format!("Failed to lower {}", op_str));
            let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

            let op_count = count_instructions(toplevel, |inst| {
                matches!(inst, Instruction::PrimOp { op, .. } if op == &expected_op)
            });
            assert_eq!(op_count, 1, "Should have exactly one {} operation", op_str);
        }
    }

    #[test]
    fn test_lower_if_expression() {
        // (if (= 1 1) 42 99)
        let expr = SExpr::IfExpr(
            Box::new(SExpr::List(vec![
                SExpr::Symbol("=".to_string()),
                SExpr::Int(1),
                SExpr::Int(1),
            ])),
            vec![SExpr::Int(42)],
            vec![SExpr::Int(99)],
        );

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have at least 4 blocks: entry, then, else, merge
        assert!(toplevel.blocks.len() >= 4, "Expected at least 4 basic blocks");

        // Should have a Branch terminator
        let has_branch = toplevel.blocks.iter().any(|bb| {
            matches!(bb.terminator, Terminator::Branch { .. })
        });
        assert!(has_branch, "Should have a Branch terminator");

        // Should have a Phi instruction in merge block
        let phi_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Phi { .. })
        });
        assert_eq!(phi_count, 1, "Should have exactly one Phi instruction");
    }

    #[test]
    fn test_lower_def_expression() {
        // (def x 42)
        let expr = SExpr::DefExpr(
            "x".to_string(),
            Box::new(SExpr::Int(42)),
        );

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");

        // Check that symbol was interned
        assert!(ns.symbols.contains_key("x"), "Symbol 'x' should be interned");

        let x_sym_id = ns.symbols.get("x").unwrap();

        // Check that global_env has the entry
        assert!(ns.global_env.contains_key(x_sym_id), "global_env should have entry for 'x'");

        // Check that it has the right type
        let x_typed_val = ns.global_env.get(x_sym_id).unwrap();
        assert_eq!(x_typed_val.ty, Type::Int, "Variable 'x' should have type Int");

        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have DirectCall to runtime_set_var
        let set_var_count = count_instructions(toplevel, |inst| {
            matches!(inst,
                Instruction::DirectCall { func, .. }
                if func == &ns.runtime_set_var
            )
        });
        assert_eq!(set_var_count, 1, "Should have one runtime_set_var call");
    }

    #[test]
    fn test_lower_global_variable_lookup() {
        // Two expressions: (def x 42) and then x
        let exprs = vec![
            SExpr::DefExpr("x".to_string(), Box::new(SExpr::Int(42))),
            SExpr::Symbol("x".to_string()),
        ];

        let ns = lower_exprs_to_namespace(&exprs).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have DirectCall to runtime_get_var
        let get_var_count = count_instructions(toplevel, |inst| {
            matches!(inst,
                Instruction::DirectCall { func, .. }
                if func == &ns.runtime_get_var
            )
        });
        assert_eq!(get_var_count, 1, "Should have one runtime_get_var call");
    }

    #[test]
    fn test_lower_lambda_expression() {
        // (fn [x y] (+ x y))
        let expr = SExpr::LambdaExpr(
            vec![
                SExpr::Symbol("x".to_string()),
                SExpr::Symbol("y".to_string()),
            ],
            vec![SExpr::List(vec![
                SExpr::Symbol("+".to_string()),
                SExpr::Symbol("x".to_string()),
                SExpr::Symbol("y".to_string()),
            ])],
        );

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");

        // Should have at least 3 functions: runtime_get_var, runtime_set_var, toplevel, lambda
        assert!(ns.functions.len() >= 4, "Should have at least 4 functions");

        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have MakeClosure instruction
        let closure_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::MakeClosure { .. })
        });
        assert_eq!(closure_count, 1, "Should have one MakeClosure instruction");

        // Find the lambda function (should be FunctionId(3) - after runtime funcs and toplevel)
        let lambda_func = ns.functions.get(&FunctionId(3))
            .expect("Lambda function should exist");

        // Check parameters
        assert_eq!(lambda_func.params.len(), 2, "Lambda should have 2 parameters");

        // Lambda should have PrimOp Add
        let add_count = count_instructions(lambda_func, |inst| {
            matches!(inst, Instruction::PrimOp { op: Operator::Add, .. })
        });
        assert_eq!(add_count, 1, "Lambda should have one Add operation");
    }

    #[test]
    fn test_lower_lambda_call() {
        // ((fn [x] (* x 2)) 21)
        let expr = SExpr::List(vec![
            SExpr::LambdaExpr(
                vec![SExpr::Symbol("x".to_string())],
                vec![SExpr::List(vec![
                    SExpr::Symbol("*".to_string()),
                    SExpr::Symbol("x".to_string()),
                    SExpr::Int(2),
                ])],
            ),
            SExpr::Int(21),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have Call instruction (indirect call via closure)
        let call_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Call { .. })
        });
        assert_eq!(call_count, 1, "Should have one Call instruction");

        // Should have MakeClosure instruction
        let closure_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::MakeClosure { .. })
        });
        assert_eq!(closure_count, 1, "Should have one MakeClosure instruction");
    }

    #[test]
    fn test_lower_nested_expressions() {
        // (+ (* 2 3) (- 10 4))
        let expr = SExpr::List(vec![
            SExpr::Symbol("+".to_string()),
            SExpr::List(vec![
                SExpr::Symbol("*".to_string()),
                SExpr::Int(2),
                SExpr::Int(3),
            ]),
            SExpr::List(vec![
                SExpr::Symbol("-".to_string()),
                SExpr::Int(10),
                SExpr::Int(4),
            ]),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have Add, Mul, and Sub operations
        let add_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::PrimOp { op: Operator::Add, .. })
        });
        let mul_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::PrimOp { op: Operator::Mul, .. })
        });
        let sub_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::PrimOp { op: Operator::Sub, .. })
        });

        assert_eq!(add_count, 1, "Should have one Add");
        assert_eq!(mul_count, 1, "Should have one Mul");
        assert_eq!(sub_count, 1, "Should have one Sub");
    }

    #[test]
    fn test_lower_lambda_with_closure() {
        // (def x 10)
        // (fn [y] (+ x y))  -- captures x
        let exprs = vec![
            SExpr::DefExpr("x".to_string(), Box::new(SExpr::Int(10))),
            SExpr::LambdaExpr(
                vec![SExpr::Symbol("y".to_string())],
                vec![SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::Symbol("x".to_string()),
                    SExpr::Symbol("y".to_string()),
                ])],
            ),
        ];

        let ns = lower_exprs_to_namespace(&exprs).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Find MakeClosure instruction
        let mut closure_captures = None;
        for bb in &toplevel.blocks {
            for inst in &bb.instructions {
                if let Instruction::MakeClosure { captures, .. } = inst {
                    closure_captures = Some(captures.clone());
                    break;
                }
            }
        }

        let captures = closure_captures.expect("Should have MakeClosure instruction");

        // x is global, so it won't be captured (it's accessed via runtime_get_var)
        // Only local variables are captured
        // So in this case, captures should be empty
        assert_eq!(captures.len(), 0, "Global variable 'x' should not be captured");

        // The lambda function should have a runtime_get_var call for x
        let lambda_func = ns.functions.get(&FunctionId(3))
            .expect("Lambda function should exist");

        let get_var_count = count_instructions(lambda_func, |inst| {
            matches!(inst,
                Instruction::DirectCall { func, .. }
                if func == &ns.runtime_get_var
            )
        });
        assert_eq!(get_var_count, 1, "Lambda should call runtime_get_var for 'x'");
    }

    #[test]
    fn test_lower_lambda_with_local_capture() {
        // ((fn [x] (fn [y] (+ x y))) 10)
        // Inner lambda should capture x (a local parameter)
        let expr = SExpr::List(vec![
            SExpr::LambdaExpr(
                vec![SExpr::Symbol("x".to_string())],
                vec![SExpr::LambdaExpr(
                    vec![SExpr::Symbol("y".to_string())],
                    vec![SExpr::List(vec![
                        SExpr::Symbol("+".to_string()),
                        SExpr::Symbol("x".to_string()),
                        SExpr::Symbol("y".to_string()),
                    ])],
                )],
            ),
            SExpr::Int(10),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");

        // Should have multiple functions: runtime funcs, toplevel, outer lambda, inner lambda
        assert!(ns.functions.len() >= 5, "Should have at least 5 functions");

        // Find the inner lambda (FunctionId 4)
        let inner_lambda = ns.functions.get(&FunctionId(4))
            .expect("Inner lambda should exist");

        // Inner lambda should have 1 parameter (y)
        assert_eq!(inner_lambda.params.len(), 1, "Inner lambda should have 1 param");

        // Find the MakeClosure for inner lambda in outer lambda
        let outer_lambda = ns.functions.get(&FunctionId(3))
            .expect("Outer lambda should exist");

        let mut inner_closure_captures = None;
        for bb in &outer_lambda.blocks {
            for inst in &bb.instructions {
                if let Instruction::MakeClosure { captures, .. } = inst {
                    inner_closure_captures = Some(captures.clone());
                    break;
                }
            }
        }

        let captures = inner_closure_captures.expect("Should have MakeClosure in outer lambda");

        // Inner lambda should capture x (the parameter of outer lambda)
        assert_eq!(captures.len(), 1, "Inner lambda should capture 'x' from outer lambda");
    }

    #[test]
    fn test_lower_quoted_list() {
        // '(1 2 3)
        let expr = SExpr::Quoted(Box::new(SExpr::List(vec![
            SExpr::Int(1),
            SExpr::Int(2),
            SExpr::Int(3),
        ])));

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have MakeList instruction
        let list_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::MakeList { .. })
        });
        assert_eq!(list_count, 1, "Should have one MakeList instruction");
    }

    #[test]
    fn test_lower_quoted_empty_list() {
        // '()
        let expr = SExpr::Quoted(Box::new(SExpr::List(vec![])));

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Empty quoted list should emit nil (Unit constant)
        let unit_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Const { value: Constant::Unit, .. })
        });
        assert_eq!(unit_count, 1, "Empty quoted list should be Unit/nil");
    }

    #[test]
    fn test_lower_vector() {
        // [1 2 3]
        let expr = SExpr::Vector(vec![
            SExpr::Int(1),
            SExpr::Int(2),
            SExpr::Int(3),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have MakeVector instruction
        let vec_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::MakeVector { .. })
        });
        assert_eq!(vec_count, 1, "Should have one MakeVector instruction");
    }

    // ============================================================================
    // Type Checking Tests
    // ============================================================================

    #[test]
    fn test_types_int_constant() {
        let expr = SExpr::Int(42);
        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Find the Const instruction and check its type
        for bb in &toplevel.blocks {
            for inst in &bb.instructions {
                if let Instruction::Const { dest, value: Constant::Int(_) } = inst {
                    assert_eq!(dest.ty, Type::Int, "Int constant should have Type::Int");
                }
            }
        }
    }

    #[test]
    fn test_types_bool_constant() {
        let expr = SExpr::Bool(true);
        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        for bb in &toplevel.blocks {
            for inst in &bb.instructions {
                if let Instruction::Const { dest, value: Constant::Bool(_) } = inst {
                    assert_eq!(dest.ty, Type::Bool, "Bool constant should have Type::Bool");
                }
            }
        }
    }

    #[test]
    fn test_types_string_constant() {
        let expr = SExpr::String("test".to_string());
        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        for bb in &toplevel.blocks {
            for inst in &bb.instructions {
                if let Instruction::Const { dest, value: Constant::String(_) } = inst {
                    assert_eq!(dest.ty, Type::String, "String constant should have Type::String");
                }
            }
        }
    }

    #[test]
    fn test_types_arithmetic_inference() {
        // (+ 1 2) should infer Type::Int
        let expr = SExpr::List(vec![
            SExpr::Symbol("+".to_string()),
            SExpr::Int(1),
            SExpr::Int(2),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Find PrimOp Add and check result type
        for bb in &toplevel.blocks {
            for inst in &bb.instructions {
                if let Instruction::PrimOp { dest, op: Operator::Add, .. } = inst {
                    assert_eq!(dest.ty, Type::Int, "Add with Int args should produce Type::Int");
                }
            }
        }
    }

    #[test]
    fn test_types_comparison_inference() {
        // (= 42 99) should infer Type::Bool
        let expr = SExpr::List(vec![
            SExpr::Symbol("=".to_string()),
            SExpr::Int(42),
            SExpr::Int(99),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Find PrimOp Eq and check result type
        for bb in &toplevel.blocks {
            for inst in &bb.instructions {
                if let Instruction::PrimOp { dest, op: Operator::Eq, .. } = inst {
                    assert_eq!(dest.ty, Type::Bool, "Eq should always produce Type::Bool");
                }
            }
        }
    }

    #[test]
    fn test_types_nested_arithmetic_inference() {
        // (+ (* 2 3) (- 10 4))
        // All operations should infer Int
        let expr = SExpr::List(vec![
            SExpr::Symbol("+".to_string()),
            SExpr::List(vec![
                SExpr::Symbol("*".to_string()),
                SExpr::Int(2),
                SExpr::Int(3),
            ]),
            SExpr::List(vec![
                SExpr::Symbol("-".to_string()),
                SExpr::Int(10),
                SExpr::Int(4),
            ]),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // All arithmetic ops should be Type::Int
        for bb in &toplevel.blocks {
            for inst in &bb.instructions {
                if let Instruction::PrimOp { dest, op, .. } = inst {
                    match op {
                        Operator::Add | Operator::Mul | Operator::Sub => {
                            assert_eq!(dest.ty, Type::Int, "{:?} should produce Type::Int", op);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    #[test]
    fn test_types_all_comparisons_return_bool() {
        let ops = vec![
            ("=", Operator::Eq),
            ("!=", Operator::Ne),
            ("<", Operator::Lt),
            (">", Operator::Gt),
            ("<=", Operator::Le),
            (">=", Operator::Ge),
        ];

        for (op_str, expected_op) in ops {
            let expr = SExpr::List(vec![
                SExpr::Symbol(op_str.to_string()),
                SExpr::Int(1),
                SExpr::Int(2),
            ]);

            let ns = lower_expr_to_namespace(&expr)
                .expect(&format!("Failed to lower {}", op_str));
            let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

            // Find the comparison op
            for bb in &toplevel.blocks {
                for inst in &bb.instructions {
                    if let Instruction::PrimOp { dest, op, .. } = inst {
                        if op == &expected_op {
                            assert_eq!(
                                dest.ty,
                                Type::Bool,
                                "{} should always produce Type::Bool",
                                op_str
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_types_all_arithmetic_ops_infer_int() {
        let ops = vec![
            ("+", Operator::Add),
            ("-", Operator::Sub),
            ("*", Operator::Mul),
            ("/", Operator::Div),
        ];

        for (op_str, expected_op) in ops {
            let expr = SExpr::List(vec![
                SExpr::Symbol(op_str.to_string()),
                SExpr::Int(10),
                SExpr::Int(5),
            ]);

            let ns = lower_expr_to_namespace(&expr)
                .expect(&format!("Failed to lower {}", op_str));
            let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

            // Find the arithmetic op
            for bb in &toplevel.blocks {
                for inst in &bb.instructions {
                    if let Instruction::PrimOp { dest, op, .. } = inst {
                        if op == &expected_op {
                            assert_eq!(
                                dest.ty,
                                Type::Int,
                                "{} with Int args should produce Type::Int",
                                op_str
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_types_global_var_tracking() {
        let exprs = vec![
            SExpr::DefExpr("x".to_string(), Box::new(SExpr::Int(42))),
            SExpr::DefExpr("y".to_string(), Box::new(SExpr::Bool(true))),
        ];

        let ns = lower_exprs_to_namespace(&exprs).expect("Failed to lower");

        // Check global_env has correct types
        let x_sym = ns.symbols.get("x").expect("x should be interned");
        let y_sym = ns.symbols.get("y").expect("y should be interned");

        let x_type = &ns.global_env.get(x_sym).expect("x should be in global_env").ty;
        let y_type = &ns.global_env.get(y_sym).expect("y should be in global_env").ty;

        assert_eq!(x_type, &Type::Int, "x should have Type::Int");
        assert_eq!(y_type, &Type::Bool, "y should have Type::Bool");
    }

    // ============================================================================
    // SSA Validity Tests
    // ============================================================================

    #[test]
    fn test_ssa_single_assignment() {
        // Each SSA ID should be assigned exactly once
        let expr = SExpr::List(vec![
            SExpr::Symbol("+".to_string()),
            SExpr::List(vec![
                SExpr::Symbol("*".to_string()),
                SExpr::Int(2),
                SExpr::Int(3),
            ]),
            SExpr::Int(4),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        use std::collections::HashSet;
        let mut assigned_ids = HashSet::new();

        for bb in &toplevel.blocks {
            for inst in &bb.instructions {
                if let Some(dest) = inst.dest() {
                    let ssa_id = dest.id;
                    assert!(
                        !assigned_ids.contains(&ssa_id),
                        "SSA ID {:?} assigned multiple times!",
                        ssa_id
                    );
                    assigned_ids.insert(ssa_id);
                }
            }
        }
    }

    #[test]
    fn test_ssa_phi_node_has_correct_predecessors() {
        // (if (= 1 1) 42 99)
        let expr = SExpr::IfExpr(
            Box::new(SExpr::List(vec![
                SExpr::Symbol("=".to_string()),
                SExpr::Int(1),
                SExpr::Int(1),
            ])),
            vec![SExpr::Int(42)],
            vec![SExpr::Int(99)],
        );

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let mut func = get_toplevel_function(&ns).expect("No toplevel function").clone();

        // Compute predecessors
        func.compute_predecessors();

        // Find the phi node
        for bb in &func.blocks {
            for inst in &bb.instructions {
                if let Instruction::Phi { incoming, .. } = inst {
                    // Phi should have exactly 2 incoming values (then and else branches)
                    assert_eq!(incoming.len(), 2, "Phi should have 2 incoming values");

                    // The block IDs in incoming should be in the predecessor list
                    let pred_blocks: Vec<_> = incoming.iter().map(|(_, blk)| *blk).collect();

                    for pred in &pred_blocks {
                        assert!(
                            bb.predecessors.contains(pred),
                            "Phi incoming block {:?} should be in predecessors {:?}",
                            pred,
                            bb.predecessors
                        );
                    }
                }
            }
        }
    }

    // ============================================================================
    // Edge Case Tests
    // ============================================================================

    #[test]
    fn test_empty_program() {
        let ns = lower_exprs_to_namespace(&[]).expect("Failed to lower empty program");

        // Should still have runtime functions + toplevel
        assert!(ns.functions.len() >= 3);

        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should return Nil (empty list)
        let nil_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Const { value: Constant::Nil, .. })
        });
        assert_eq!(nil_count, 1, "Empty program should return Nil");
    }

    #[test]
    fn test_deeply_nested_arithmetic() {
        // ((+ 1 2) + (3 + (4 + 5)))
        let expr = SExpr::List(vec![
            SExpr::Symbol("+".to_string()),
            SExpr::List(vec![
                SExpr::Symbol("+".to_string()),
                SExpr::Int(1),
                SExpr::Int(2),
            ]),
            SExpr::List(vec![
                SExpr::Symbol("+".to_string()),
                SExpr::Int(3),
                SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::Int(4),
                    SExpr::Int(5),
                ]),
            ]),
        ]);

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have 4 Add operations
        let add_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::PrimOp { op: Operator::Add, .. })
        });
        assert_eq!(add_count, 4, "Should have 4 Add operations");
    }

    #[test]
    fn test_multiple_defs_and_lookups() {
        let exprs = vec![
            SExpr::DefExpr("a".to_string(), Box::new(SExpr::Int(1))),
            SExpr::DefExpr("b".to_string(), Box::new(SExpr::Int(2))),
            SExpr::DefExpr("c".to_string(), Box::new(SExpr::Int(3))),
            SExpr::List(vec![
                SExpr::Symbol("+".to_string()),
                SExpr::Symbol("a".to_string()),
                SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::Symbol("b".to_string()),
                    SExpr::Symbol("c".to_string()),
                ]),
            ]),
        ];

        let ns = lower_exprs_to_namespace(&exprs).expect("Failed to lower");

        // All three variables should be in global_env
        assert_eq!(ns.symbols.len(), 3, "Should have 3 interned symbols");
        assert!(ns.symbols.contains_key("a"));
        assert!(ns.symbols.contains_key("b"));
        assert!(ns.symbols.contains_key("c"));

        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have 3 runtime_set_var calls
        let set_count = count_instructions(toplevel, |inst| {
            matches!(inst,
                Instruction::DirectCall { func, .. }
                if func == &ns.runtime_set_var
            )
        });
        assert_eq!(set_count, 3, "Should have 3 runtime_set_var calls");

        // Should have 3 runtime_get_var calls (for a, b, c lookups)
        let get_count = count_instructions(toplevel, |inst| {
            matches!(inst,
                Instruction::DirectCall { func, .. }
                if func == &ns.runtime_get_var
            )
        });
        assert_eq!(get_count, 3, "Should have 3 runtime_get_var calls");
    }

    #[test]
    fn test_lambda_with_multiple_params() {
        // (fn [a b c] (+ a (+ b c)))
        let expr = SExpr::LambdaExpr(
            vec![
                SExpr::Symbol("a".to_string()),
                SExpr::Symbol("b".to_string()),
                SExpr::Symbol("c".to_string()),
            ],
            vec![SExpr::List(vec![
                SExpr::Symbol("+".to_string()),
                SExpr::Symbol("a".to_string()),
                SExpr::List(vec![
                    SExpr::Symbol("+".to_string()),
                    SExpr::Symbol("b".to_string()),
                    SExpr::Symbol("c".to_string()),
                ]),
            ])],
        );

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let lambda_func = ns.functions.get(&FunctionId(3))
            .expect("Lambda function should exist");

        // Check that lambda has 3 parameters
        assert_eq!(lambda_func.params.len(), 3, "Lambda should have 3 parameters");

        // All params should initially be Type::Any
        for param in &lambda_func.params {
            assert_eq!(param.ty, Type::Any, "Parameters should have Type::Any initially");
        }
    }

    #[test]
    fn test_nested_if_expressions() {
        // (if cond1 (if cond2 1 2) 3)
        let expr = SExpr::IfExpr(
            Box::new(SExpr::Bool(true)),
            vec![SExpr::IfExpr(
                Box::new(SExpr::Bool(false)),
                vec![SExpr::Int(1)],
                vec![SExpr::Int(2)],
            )],
            vec![SExpr::Int(3)],
        );

        let ns = lower_expr_to_namespace(&expr).expect("Failed to lower");
        let toplevel = get_toplevel_function(&ns).expect("No toplevel function");

        // Should have 2 Phi instructions (one for each if)
        let phi_count = count_instructions(toplevel, |inst| {
            matches!(inst, Instruction::Phi { .. })
        });
        assert_eq!(phi_count, 2, "Should have 2 Phi instructions for nested ifs");

        // Should have 2 Branch terminators
        let branch_count = toplevel.blocks.iter().filter(|bb| {
            matches!(bb.terminator, Terminator::Branch { .. })
        }).count();
        assert_eq!(branch_count, 2, "Should have 2 Branch terminators");
    }

    #[test]
    fn test_runtime_function_stubs() {
        let ns = Namespace::new();

        // Runtime functions should be registered
        assert!(ns.functions.contains_key(&ns.runtime_get_var));
        assert!(ns.functions.contains_key(&ns.runtime_set_var));

        // Check signatures
        let get_var = ns.functions.get(&ns.runtime_get_var).unwrap();
        assert_eq!(get_var.params.len(), 1, "runtime_get_var should take 1 param");
        assert_eq!(get_var.params[0].ty, Type::String, "First param should be String");
        assert_eq!(get_var.return_type, Type::BoxedLispVal, "Should return BoxedLispVal");

        let set_var = ns.functions.get(&ns.runtime_set_var).unwrap();
        assert_eq!(set_var.params.len(), 2, "runtime_set_var should take 2 params");
        assert_eq!(set_var.params[0].ty, Type::String, "First param should be String");
        assert_eq!(set_var.params[1].ty, Type::BoxedLispVal, "Second param should be BoxedLispVal");
        assert_eq!(set_var.return_type, Type::BoxedLispVal, "Should return BoxedLispVal");
    }

    #[test]
    fn test_pretty_print_ir() {
        // This test demonstrates the LLVM-style pretty printing
        let expr = SExpr::DefExpr(
            "factorial".to_string(),
            Box::new(SExpr::LambdaExpr(
                vec![SExpr::Symbol("n".to_string())],
                vec![
                    SExpr::IfExpr(
                        Box::new(SExpr::List(vec![
                            SExpr::Symbol("<=".to_string()),
                            SExpr::Symbol("n".to_string()),
                            SExpr::Int(1),
                        ])),
                        vec![SExpr::Int(1)],
                        vec![SExpr::List(vec![
                            SExpr::Symbol("*".to_string()),
                            SExpr::Symbol("n".to_string()),
                            SExpr::List(vec![
                                SExpr::Symbol("factorial".to_string()),
                                SExpr::List(vec![
                                    SExpr::Symbol("-".to_string()),
                                    SExpr::Symbol("n".to_string()),
                                    SExpr::Int(1),
                                ]),
                            ]),
                        ])],
                    ),
                ],
            )),
        );

        let ns = lower_expr_to_namespace(&expr).unwrap();

        // Print to stdout for manual inspection (only when running with --nocapture)
        println!("\n====== Pretty Printed IR ======");
        println!("{}", ns);
        println!("==============================\n");

        // Basic validation
        assert!(ns.functions.len() >= 2); // At least toplevel + factorial

        // Find the factorial function
        let factorial_fn = ns.functions.values()
            .find(|f| f.params.len() == 1 && f.blocks.len() > 1)
            .expect("Should have factorial function");

        // Should have multiple basic blocks (if-then-else structure)
        assert!(factorial_fn.blocks.len() >= 3);
    }
}
