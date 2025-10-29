use std::{cmp::Ordering, collections::btree_set::Union, fmt::Display};

use inkwell::types;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaId(pub u32);

impl Display for SsaId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // let's do LLVM IR stype %1, %2 for now
        write!(f, "%{}", self.0)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}



#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

// type system 
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub enum Type {
    /// Int (i32)
    Int,
    // Boolean (i1)
    Bool,
    // String (heap-allocated)
    String,
    // Lisp classic linked list of Cons cells  (heap-allocated )
    List,
    // Vector (heap-allocated collection)
    Vector,
    // LispVal struct wrapping dynamic type (heap-allocated)
    Box, 
    Function {
        params: Vec<Type>,
        return_type: Box<Type>
    },
    // Placeholder for the Type that needs to be inferred
    Any,
    // Union of types (Int | List)
    Union(Vec<Type>),
    // Bottom type ( never returns - infinite loop, panic)
    Bottom,
}

impl Type {
    pub fn is_concrete(&self) -> bool {
        !matches!(self, Type::Any | Type::Bottom)
    } 

    pub fn needs_rc(&self) -> bool {
        matches!(self, Type::String | Type::List 
                | Type::Vector | Type::Function{..})
    }

    pub fn is_subtype_of(&self, other: &Type) -> bool {
        use Type::*;

        if self == other {
            return true;
        }

        match (self, other) {
            (_, Any) => true, // everything is a subtype of Any, it's the Top type like Object in Java
            (Bottom, _) => true, // Bottom is a subtype of every type
            (Int | Bool | String | List | Vector, Box) => true, // concrete types are subtypes of Box
            //function subtyping (contravariant in params, covariant in return_type) 
            (Function { params: p1, return_type: r1},
                Function {params: p2, return_type: r2}) => {
                p1.len() == p2.len() // need to have same signature 
                    && p1.iter()
                        .zip(p2)
                        .all(|(t1, t2)| t2.is_subtype_of(t1)) // contravariant in params!
                    && r1.is_subtype_of(r2)
            },
            // For ty to be subtype of a Union, at least one of the types need to be a super type 
            (ty, Union(types)) => types.iter().any(|t| ty.is_subtype_of(t)),
            // For a Union to be a subtype of ty, EACH of union types must be subtype of ty
            (Union(types), ty) => types.iter().all(|t| t.is_subtype_of(ty)),
            _ => false
        }
    }


    /// Join two types in a type lattice (least upper bound or LUB)
    /// LUB - means most precise type we can infer based on current info
    /// necessary for phi node type inference
    /// Example 
    // if condition {
    //     x = 42;        // x : Int
    // } else {
    //     x = 3.14;      // x : Float  
    // }
    // // phi node: x = φ(Int, Float)
    // // What type is x here? Need join(Int, Float)!
    // ```

    // The phi node needs a type that can represent *either* value, so you compute `join(Int, Float)`.

    // ## Typical Type Lattice Structure

    //               Any/Top (⊤)
    //              /    |    \
    //           Int   Float  String
    //            \     |     /
    //               Bottom (⊥)
    pub fn join(&self, other: &Type) -> Type {
        use Type::*;

        if self == other {
            return self.clone();
        }

        match (self, other) {
            // Any absorbs other type and LUB must be the top
            (Any, _) | (_, Any) => Any, 
            // if Bottom is involved, then LUB is the other type (identity)
            (Bottom, ty) | (ty, Bottom) => ty.clone(), 
            // if both are the same type, just return it
            (ty1, ty2) if ty1 == ty2 => ty1.clone(),
            // for 2 different concrete types, create a Union
            (ty1, ty2) if ty1.is_concrete() && ty1.is_concrete() => {
                let mut types = vec![ty1.clone(), ty2.clone()];
                // Without sorting, these are "different" types:
                //      Union(Int, String)
                //      Union(String, Int)
                types.sort();
                types.dedup();
                Union(types)
            },
            // Simplify unions
            (Union(types1), Union(types2)) => {
                let mut all_types = types1.clone();
                all_types.extend(types2.clone());
                all_types.sort();
                all_types.dedup();
                if all_types.len() == 1 {
                    // managed to reduce 2 unions to single type
                    all_types[0].clone()
                } else {
                    Union(all_types)
                }
            },
            (Union(types), ty) | (ty, Union(types)) => {
                let mut all_types = types.clone();
                if !all_types.contains(ty) {
                    all_types.push(ty.clone());
                    all_types.sort();
                    all_types.dedup();
                }
                Union(all_types)
            },
            // finally, need to default to Any if we can't join more precisely 
            _ => Any,

        }
    }
}


impl Ord for Type {
    fn cmp(&self, other: &Self) -> Ordering {
        use Type::*;
        
        // First, compare by discriminant (variant)
        match (self, other) {
            // Same variant - compare contents
            (Int, Int) => Ordering::Equal,
            (Bool, Bool) => Ordering::Equal,
            (String, String) => Ordering::Equal,
            (List, List) => Ordering::Equal,
            (Vector, Vector) => Ordering::Equal,
            (Box, Box) => Ordering::Equal,
            (Any, Any) => Ordering::Equal,
            (Bottom, Bottom) => Ordering::Equal,
            
            // Function - compare params, then return type
            (Function { params: p1, return_type: r1 }, 
             Function { params: p2, return_type: r2 }) => {
                p1.cmp(p2).then_with(|| r1.cmp(r2))
            }
            
            // Union - lexicographic comparison of vectors
            (Union(v1), Union(v2)) => v1.cmp(v2),
            
            // Different variants - order by discriminant
            // (Using the enum declaration order)
            (Int, _) => Ordering::Less,
            (_, Int) => Ordering::Greater,
            
            (Bool, _) => Ordering::Less,
            (_, Bool) => Ordering::Greater,
            
            (String, _) => Ordering::Less,
            (_, String) => Ordering::Greater,
            
            (List, _) => Ordering::Less,
            (_, List) => Ordering::Greater,
            
            (Vector, _) => Ordering::Less,
            (_, Vector) => Ordering::Greater,
            
            (Box, _) => Ordering::Less,
            (_, Box) => Ordering::Greater,
            
            (Function { .. }, _) => Ordering::Less,
            (_, Function { .. }) => Ordering::Greater,
            
            (Any, _) => Ordering::Less,
            (_, Any) => Ordering::Greater,
            
            (Union(_), _) => Ordering::Less,
            (_, Union(_)) => Ordering::Greater,
            
            // Bottom is last
        }
    }
}

/// Primitive Operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimOp {
        // Arithmetic
        Add, Sub, Mul, Div, Mod,
        // Comparisons
        Lt, Gt, Eq, Le, Ge, Ne,
        // Logical
        And, Or, Not,
        // List operations
        ListNew,
        ListHead,
        ListTail,
        ListLen,
        // Vector operations
        VectorNew,
        VectorGet,
        VectorSet,
        VectorLen,
        // String operations 
        StringConcat,
        StringLen,
}

impl PrimOp {
    //compute result type of primitive operation
    pub fn result_type(&self, arg_types: &[Type]) -> Type {
        use PrimOp::*;

        match self {
            Add | Sub | Mul | Div | Mod => {
                        if arg_types.iter().all(|t| matches!(t, Type::Int)) {
                            Type::Int
                        } else {
                            Type::Any
                        }
                    },
            Lt | Gt | Eq | Le | Ge | Ne => Type::Bool,
            And | Or => Type::Bool,
            Not => {
                        if matches!(arg_types.first(), Some(Type::Bool)) {
                            Type::Bool
                        } else {
                            Type::Any
                        }
                    },
            ListHead => Type::Any,
            ListTail => Type::List,
            ListNew => Type::List,
            ListLen => Type::Int,
            VectorLen => Type::Int,
            VectorGet => Type::Any,
            VectorSet => Type::Any,
            VectorNew => Type::Vector,
            StringConcat => Type::String,
            StringLen => Type::Int,
        }
    }
}


/// A value with type annotation
#[derive(Debug, Clone, PartialEq)]
pub struct TypedValue {
    pub id: SsaId,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Int(i32),
    Bool(bool),
    String(String),
    Unit, //  for empty values, should it be Nil ?
}

impl Constant {
    pub fn ty(&self) -> Type {
        match self {
            Constant::Int(_) => Type::Int,
            Constant::Bool(_) => Type::Bool,
            Constant::String(_) => Type::String,
            Constant::Unit => Type::Any,  // technically '() is a list? but also acts as bool 
        }
    }
}