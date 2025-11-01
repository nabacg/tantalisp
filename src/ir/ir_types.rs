use std::{cmp::Ordering, fmt};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaId(pub u32);

impl fmt::Display for SsaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // let's do LLVM IR style %1, %2 for now
        write!(f, "%{}", self.0)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}



#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

// Type system
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
    BoxedLispVal, 
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
        !matches!(self, Type::Any | Type::Bottom | Type::Union(..))
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
            (Int | Bool | String | List | Vector, BoxedLispVal) => true, // concrete types are subtypes of Box
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
            // finally, need to default to Any if we can't join more precisely 
            _ => Any,

        }
    }


      /// Meet operation - find the most specific type satisfying both constraints
      /// Used for constraint-based type narrowing (i.e. more like intersection of types, than union like .join)
      pub fn meet(&self, other: &Type) -> Type {
        use Type::*;

        // If either is Bottom, result is Bottom (inconsistent)
        if matches!(self, Bottom) || matches!(other, Bottom) {
            return Bottom;
        }

        // If either is Any, use the other (narrow down)
        match (self, other) {
            (Any, ty) | (ty, Any) => ty.clone(),  // Intersection not Union of types like for .join

            // Same concrete type - consistent
            (ty1, ty2) if ty1 == ty2 => ty1.clone(),

            // Function types - contravariant params, covariant return
            (Function { params: p1, return_type: r1 },
             Function { params: p2, return_type: r2 }) => {
                if p1.len() != p2.len() {
                    return Bottom;
                }
                let params: Vec<_> = p1.iter()
                    .zip(p2)
                    .map(|(t1, t2)| t1.meet(t2))  // Meet parameters
                    .collect();

                if params.iter().any(|t| matches!(t, Bottom)) {
                    return Bottom;
                }

                Function {
                    params,
                    return_type: Box::new(r1.meet(r2))
                }
            }

            // Union types - intersection
            (Union(types1), Union(types2)) => {
                let intersection: Vec<_> = types1.iter()
                    .filter(|t1| types2.iter().any(|t2| *t1 == t2))
                    .cloned()
                    .collect();

                if intersection.is_empty() {
                    Bottom
                } else if intersection.len() == 1 {
                    intersection[0].clone()
                } else {
                    Union(intersection)
                }
            }

            // Different concrete types - inconsistent!
            _ => Bottom,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => write!(f, "Int"),
            Type::Bool => write!(f, "Bool"),
            Type::String => write!(f, "String"),
            Type::List => write!(f, "List"),
            Type::Vector => write!(f, "Vector"),
            Type::BoxedLispVal => write!(f, "BoxedLispVal"),
            Type::Function { params, return_type } => {
                write!(f, "(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", return_type)
            }
            Type::Any => write!(f, "Any"),
            Type::Union(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 { write!(f, " | ")?; }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            Type::Bottom => write!(f, "Bottom"),
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
            (BoxedLispVal, BoxedLispVal) => Ordering::Equal,
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
            
            (BoxedLispVal, _) => Ordering::Less,
            (_, BoxedLispVal) => Ordering::Greater,
            
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
pub enum Operator {
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

impl Operator {
    //compute result type of primitive operation
    pub fn result_type(&self, arg_types: &[Type]) -> Type {
        use Operator::*;

        match self {
            Add | Sub | Mul | Div | Mod => {
                        if arg_types.iter().all(|t| matches!(t, Type::Int)) {
                            Type::Int
                        } else {
                            Type::Int
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
    Unit, 
    Nil
}

impl Constant {
    pub fn ty(&self) -> Type {
        match self {
            Constant::Int(_) => Type::Int,
            Constant::Bool(_) => Type::Bool,
            Constant::String(_) => Type::String,
            Constant::Unit => Type::Any, 
            Constant::Nil => Type::List // technically '() is a list? but also acts as bool 
        }
    }
}

#[cfg(test)] 
mod ir_types_tests {
    use super::*;

    #[test]
    fn test_type_join(){
        use Type::*;
        assert_eq!(Union(vec![Int, String]), Union(vec![Int, String]).join(&Int));
        assert_eq!(Union(vec![Int, String]), Int.join(&Union(vec![Int, String])));
        assert_eq!(Union(vec![Int, Bool]), Bool.join(&Int));
        assert_eq!(Union(vec![Int, Bool]), Int.join(&Bool));
        
        assert_eq!(Union(vec![Int, Bool, String, Vector, Union(vec![Bool, Vector]) ]), 
            Union(vec![Int, String, Union(vec![Bool, Vector])])
                .join(&Union(vec![Vector, Int, Bool ])));

        // .join with Bottom type acts as identity
        assert_eq!(String, String.join(&Bottom));
        assert_eq!(String, Bottom.join(&String));

        // Any always dominates teh .join 
        assert_eq!(Any, Union(vec![Int, String, Union(vec![Bool, Vector])]).join(&Any));
        assert_eq!(Any, Any.join(&Vector));
    }

    #[test]
    pub fn test_is_subtype_of() {
        use Type::*;
        // everythign is a subtype of Any
        assert!(Int.is_subtype_of(&Any));
        assert!(String.is_subtype_of(&Any));
        assert!(List.is_subtype_of(&Any));
        assert!(Union(vec![Int, String]).is_subtype_of(&Any));
        assert!(Union(vec![Int, String, Union(vec![Bool, Vector])]).is_subtype_of(&Any));
        
        // Bottom is a suptype of very type
        assert!(Bottom.is_subtype_of(&Vector));
        assert!(Bottom.is_subtype_of(&List));
        assert!(Bottom.is_subtype_of(&Union(vec![Int, String])));
        assert!(Bottom.is_subtype_of(&Union(vec![Int, String, Union(vec![Bool, Vector])])));
        

        // Most types are not subtypes of each other
        assert!(!Int.is_subtype_of(&String));
        assert!(!Bool.is_subtype_of(&Vector));
        assert!(!List.is_subtype_of(&Vector));
        assert!(!Union(vec![Int]).is_subtype_of(&Vector));

        // same types are subtypes
        assert!(String.is_subtype_of(&String));
        assert!(Union(vec![Int, String, Union(vec![Bool, Vector])])
            .is_subtype_of(&Union(vec![Int, String, Union(vec![Bool, Vector])])));
        
        // concrete types are subtypes of Box
        assert!(Int.is_subtype_of(&BoxedLispVal));
        assert!(String.is_subtype_of(&BoxedLispVal));
        assert!(Bool.is_subtype_of(&BoxedLispVal));
        assert!(List.is_subtype_of(&BoxedLispVal));
        assert!(Vector.is_subtype_of(&BoxedLispVal));
        assert!(Int.is_subtype_of(&BoxedLispVal));


        // Functions are contravariant in arguments covariant in return types 

        // This is true, because both is true
        //  - all params of f1 (left) are super types of f2 (right) - contravariant
        //  - return_type of f1 (left) is a subtype of f2 return_type - covariant
        // This makes sense because you can use f1 everywhere you could use f2, since
        //  - it's input params are "more general" (super type) than f2 (here BoxedLispVal vs say Int) 
        //  meaning each f2 call site must type check ( you can pass Int where BoxedLispVal is required )
        //  - the f1 return_type is more concrete (subtype) than f2, meaning it can be used as a result in f2 callsite
        //  you can assing result_type: Int where BoxedLispVal was required 
        assert!(Function { 
            params: vec![BoxedLispVal, BoxedLispVal, BoxedLispVal], 
            return_type: Box::new(Int) 
        }.is_subtype_of(&Function { 
            params: vec![Int, String, List], 
            return_type: Box::new(BoxedLispVal) 
        }));

        // The opposite relation ship is not true
        assert!(!Function { params: vec![Int, String, List], return_type: Box::new(BoxedLispVal) }
        .is_subtype_of(&Function { 
            params: vec![BoxedLispVal, BoxedLispVal, BoxedLispVal], 
            return_type: Box::new(Int) 
        }));

        // same if some (all) of the types match exactly
        assert!(Function { 
            params: vec![BoxedLispVal, String, List], 
            return_type: Box::new(Int) 
        }.is_subtype_of(&Function { 
            params: vec![Int, String, List], 
            return_type: Box::new(BoxedLispVal) 
        }));

        // exact match is a subtype
        assert!(Function { 
            params: vec![Int, String, List], 
            return_type: Box::new(Int) 
        }.is_subtype_of(&Function { 
            params: vec![Int, String, List], 
            return_type: Box::new(Int) 
        }));

        // And of course it's false if  param count mismatch
        assert!(!Function { params: vec![Int, String, List], return_type: Box::new(BoxedLispVal) }
        .is_subtype_of(&Function { 
            params: vec![Int], 
            return_type: Box::new(Int) 
        }));

        // or if param types don't match
        assert!(!Function { params: vec![Int, List, List], return_type: Box::new(BoxedLispVal) }
        .is_subtype_of(&Function { 
            params: vec![Int, String, List], 
            return_type: Box::new(Int) 
        }));

        // or if return_type doesn't check
        assert!(!Function { 
            params: vec![Int, String, List], 
            return_type: Box::new(Int) 
        }.is_subtype_of(&Function { 
            params: vec![Int, String, List], 
            return_type: Box::new(String) 
        }));

    }   

}