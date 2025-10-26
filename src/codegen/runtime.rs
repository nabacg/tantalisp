use std::{collections::HashMap, ptr};
mod garbage_collector;

pub const TAG_INT: u8 = 0;
pub const TAG_BOOL: u8 = 1;
pub const TAG_STRING: u8 = 2;
pub const TAG_LIST: u8 = 3;
pub const TAG_NIL: u8 = 4;
pub const TAG_LAMBDA: u8 = 5;

// ┌─────────────────┐
// │ LispVal         │
// ├─────────────────┤
// │ tag: u8         │ ← Type discriminator (0=Int, 1=Bool, 2=String, 3=List, etc.)
// │ data: union {   │
// │   i32           │
// │   bool          │
// │   ptr (String)  │
// │   ptr (List)    │
// │ }               │
// └─────────────────┘

#[repr(C)]
pub struct LispValLayout {
    pub(crate) tag: u8,
    _padding: [u8; 7], // Explicit padding to align the next field to 8 bytes
    pub(crate) data: i64,
    pub refcount: i32,
}

#[repr(C)]
pub struct ConsCellLayout {
    pub(crate) head: *mut LispValLayout,
    pub(crate) tail: *mut ConsCellLayout,
}

#[unsafe(no_mangle)]
pub extern "C" fn runtime_get_var(
    env_ptr: *mut std::ffi::c_void,
    name_ptr: *const std::os::raw::c_char,
) -> *mut LispValLayout {
    unsafe {
        let env = &*(env_ptr as *mut HashMap<String, *mut LispValLayout>);
        let name = std::ffi::CStr::from_ptr(name_ptr).to_str(); //TODO - check errors?
        if let Ok(name) = name {
            env.get(name).copied().unwrap_or(std::ptr::null_mut())
        } else {
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn runtime_set_var(
    env_ptr: *mut std::ffi::c_void,
    name_ptr: *const std::os::raw::c_char,
    value_ptr: *mut LispValLayout,
) {
    unsafe {
        let env = &mut *(env_ptr as *mut HashMap<String, *mut LispValLayout>);
        let name = std::ffi::CStr::from_ptr(name_ptr).to_str(); //TODO - check errors?
        if let Ok(name) = name {
            env.insert(name.to_string(), value_ptr);
        }
    }
}

// Increment reference count on a LispVal
#[unsafe(no_mangle)]
pub extern "C" fn lisp_val_incref(val: *mut LispValLayout) {
    if val.is_null() {
        return;
    }
    unsafe {
        (*val).refcount += 1;
    }
}

// Decrement reference count on LispVal
#[unsafe(no_mangle)]
pub extern "C" fn lisp_val_decref(val: *mut LispValLayout) {
    if val.is_null() {
        return;
    }

    unsafe {
        (*val).refcount -= 1;
        if (*val).refcount <= 0 {
            free_lisp_val(val); // let the val free it's resources
            drop(Box::from_raw(val)); // free the val itself
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc_lisp_val() -> *mut LispValLayout {
    // use our GlobalAllocator
    Box::into_raw(Box::new(LispValLayout {
        tag: TAG_NIL,
        _padding: [0; 7],
        data: 0,
        refcount: 1,
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc_cons_cell() -> *mut ConsCellLayout {
    Box::into_raw(Box::new(ConsCellLayout { head: ptr::null_mut(), tail: ptr::null_mut() }))
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc_string(len: u64) -> *mut u8 {
    let size = len as usize;

    let mut vec = vec![0u8; size];     // 1. Allocate on heap
    let ptr = vec.as_mut_ptr();        // 2. Get raw pointer
    std::mem::forget(vec);                      // 3. Don't drop vec at the end of this function!
    ptr
}

fn free_lisp_val(val: *mut LispValLayout) {
    if val.is_null() {
        return;
    }

    let tag = unsafe { (*val).tag };
    match tag {
        TAG_STRING => {
            let str_ptr = unsafe { (*val).data as *mut u8 };
            if !str_ptr.is_null() {
                let _ = unsafe { std::ffi::CString::from_raw(str_ptr as *mut i8) };
            }
        }
        TAG_LIST => {
            let mut list = unsafe { (*val).data as *mut ConsCellLayout };
            while !list.is_null() {
                let next = unsafe { (*list).tail };

                // decref the head element
                let head = unsafe { (*list).head };
                lisp_val_decref(head);
                // free actual list
                drop(unsafe { Box::from_raw(list) });

                list = next;
            }
        }
        TAG_LAMBDA => {
            // TODO - double check we don't allocate anything in lambdas for now (until we implement lexical closures at least!)
            // free actual LispVal
            drop(unsafe { Box::from_raw(val) });
        }
        TAG_INT | TAG_BOOL | TAG_NIL => {
            // No internal data to free
        }
        _ => {
            eprintln!("Warning: Unknown tag {} in free_lisp_val", tag);
        }
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn lisp_val_print_refcount(val: *const LispValLayout) {
    if val.is_null() {
        println!("NULL");
        return
    }

    unsafe {
        let v = &*val;
        match v.tag {
            TAG_NIL =>  println!("LispVal(tag={}, refcount={}, data=nil)", v.tag, v.refcount),
            TAG_INT | TAG_BOOL  =>  println!("LispVal(tag={}, refcount={}, data={})", v.tag, v.refcount, v.data),
            TAG_LAMBDA | TAG_LIST |  TAG_STRING =>  println!("LispVal(tag={}, refcount={}, data={:#x})", v.tag, v.refcount, v.data as usize),
            _ => {
                eprintln!("Warning: Unknown tag {} in lval_print_refcount", v.tag);
            }
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_alloc_lval() {
        // Test basic allocation
        let val = alloc_lisp_val();
        assert!(!val.is_null(), "alloc_lval should return non-null pointer");

        unsafe {
            assert_eq!((*val).tag, TAG_NIL, "default tag should be TAG_NIL");
            assert_eq!((*val).refcount, 1, "initial refcount should be 1");
            assert_eq!((*val).data, 0, "initial data should be 0");

            // Clean up
            drop(Box::from_raw(val));
        }
    }

    #[test]
    fn test_refcount_incref() {
        let val = alloc_lisp_val();

        unsafe {
            assert_eq!((*val).refcount, 1, "initial refcount should be 1");

            lisp_val_incref(val);
            assert_eq!((*val).refcount, 2, "refcount should be 2 after incref");

            lisp_val_incref(val);
            assert_eq!((*val).refcount, 3, "refcount should be 3 after second incref");

            // Clean up manually (don't use decref to test incref in isolation)
            drop(Box::from_raw(val));
        }
    }

    #[test]
    fn test_refcount_incref_null() {
        // Should not crash on null pointer
        lisp_val_incref(ptr::null_mut());
        // If we get here without crashing, test passes
    }

    #[test]
    fn test_refcount_decref() {
        let val = alloc_lisp_val();

        unsafe {
            (*val).refcount = 3;

            lisp_val_decref(val);
            assert_eq!((*val).refcount, 2, "refcount should be 2 after decref");

            lisp_val_decref(val);
            assert_eq!((*val).refcount, 1, "refcount should be 1 after second decref");

            // This final decref should free the value
            lisp_val_decref(val);
            // Can't test anything after this - val is freed
        }
    }

    #[test]
    fn test_refcount_decref_null() {
        // Should not crash on null pointer
        lisp_val_decref(ptr::null_mut());
        // If we get here without crashing, test passes
    }

    #[test]
    fn test_int_lifecycle() {
        let val = alloc_lisp_val();

        unsafe {
            (*val).tag = TAG_INT;
            (*val).data = 42;

            assert_eq!((*val).tag, TAG_INT);
            assert_eq!((*val).data, 42);
            assert_eq!((*val).refcount, 1);

            // Decref should free it
            lisp_val_decref(val);
        }
    }

    #[test]
    fn test_bool_lifecycle() {
        let val = alloc_lisp_val();

        unsafe {
            (*val).tag = TAG_BOOL;
            (*val).data = 1; // true

            assert_eq!((*val).tag, TAG_BOOL);
            assert_eq!((*val).data, 1);

            lisp_val_decref(val);
        }
    }

    #[test]
    fn test_nil_lifecycle() {
        let val = alloc_lisp_val();

        unsafe {
            (*val).tag = TAG_NIL;
            (*val).data = 0;

            assert_eq!((*val).tag, TAG_NIL);

            lisp_val_decref(val);
        }
    }

    #[test]
    fn test_alloc_cons_cell() {
        let cell = alloc_cons_cell();
        assert!(!cell.is_null(), "alloc_cons_cell should return non-null pointer");

        unsafe {
            assert!((*cell).head.is_null(), "head should be null initially");
            assert!((*cell).tail.is_null(), "tail should be null initially");

            // Clean up
            drop(Box::from_raw(cell));
        }
    }

    #[test]
    fn test_simple_list_lifecycle() {
        // Create a simple list: (42)
        let head_val = alloc_lisp_val();
        unsafe {
            (*head_val).tag = TAG_INT;
            (*head_val).data = 42;
        }

        let cell = alloc_cons_cell();
        unsafe {
            (*cell).head = head_val;
            (*cell).tail = ptr::null_mut();
        }

        let list_val = alloc_lisp_val();
        unsafe {
            (*list_val).tag = TAG_LIST;
            (*list_val).data = cell as i64;

            // Decref should recursively free the list
            lisp_val_decref(list_val);
        }
    }

    #[test]
    fn test_multi_element_list_lifecycle() {
        // Create a list: (1 2 3)

        // Element 1
        let val1 = alloc_lisp_val();
        unsafe {
            (*val1).tag = TAG_INT;
            (*val1).data = 1;
        }

        // Element 2
        let val2 = alloc_lisp_val();
        unsafe {
            (*val2).tag = TAG_INT;
            (*val2).data = 2;
        }

        // Element 3
        let val3 = alloc_lisp_val();
        unsafe {
            (*val3).tag = TAG_INT;
            (*val3).data = 3;
        }

        // Build cons cells: 3 -> 2 -> 1
        let cell3 = alloc_cons_cell();
        unsafe {
            (*cell3).head = val3;
            (*cell3).tail = ptr::null_mut();
        }

        let cell2 = alloc_cons_cell();
        unsafe {
            (*cell2).head = val2;
            (*cell2).tail = cell3;
        }

        let cell1 = alloc_cons_cell();
        unsafe {
            (*cell1).head = val1;
            (*cell1).tail = cell2;
        }

        let list_val = alloc_lisp_val();
        unsafe {
            (*list_val).tag = TAG_LIST;
            (*list_val).data = cell1 as i64;

            // Single decref should clean up entire list
            lisp_val_decref(list_val);
        }
    }

    #[test]
    fn test_alloc_string() {
        let str_ptr = alloc_string(10);
        assert!(!str_ptr.is_null(), "alloc_string should return non-null pointer");

        unsafe {
            // Write some data to verify it works
            for i in 0..9 {
                *str_ptr.add(i) = b'a' + i as u8;
            }
            *str_ptr.add(9) = 0; // null terminator

            // Clean up
            let vec = Vec::from_raw_parts(str_ptr, 10, 10);
            drop(vec);
        }
    }

    #[test]
    fn test_string_lifecycle() {
        // Create a string value
        let c_str = CString::new("hello").unwrap();
        let str_ptr = c_str.into_raw() as *mut u8;

        let val = alloc_lisp_val();
        unsafe {
            (*val).tag = TAG_STRING;
            (*val).data = str_ptr as i64;

            // Decref should free both the LispVal and the string
            lisp_val_decref(val);
        }
    }

    #[test]
    fn test_runtime_get_set_var() {
        let mut env: HashMap<String, *mut LispValLayout> = HashMap::new();
        let env_ptr = &mut env as *mut HashMap<String, *mut LispValLayout> as *mut std::ffi::c_void;

        // Create a value
        let val = alloc_lisp_val();
        unsafe {
            (*val).tag = TAG_INT;
            (*val).data = 42;
        }

        // Set variable
        let name = CString::new("x").unwrap();
        runtime_set_var(env_ptr, name.as_ptr(), val);

        // Get variable
        let retrieved = runtime_get_var(env_ptr, name.as_ptr());
        assert!(!retrieved.is_null(), "retrieved value should not be null");

        unsafe {
            assert_eq!((*retrieved).tag, TAG_INT);
            assert_eq!((*retrieved).data, 42);

            // Clean up
            lisp_val_decref(val);
        }
    }

    #[test]
    fn test_runtime_get_nonexistent_var() {
        let mut env: HashMap<String, *mut LispValLayout> = HashMap::new();
        let env_ptr = &mut env as *mut HashMap<String, *mut LispValLayout> as *mut std::ffi::c_void;

        let name = CString::new("nonexistent").unwrap();
        let result = runtime_get_var(env_ptr, name.as_ptr());

        assert!(result.is_null(), "nonexistent variable should return null");
    }

    #[test]
    fn test_runtime_var_overwrite() {
        let mut env: HashMap<String, *mut LispValLayout> = HashMap::new();
        let env_ptr = &mut env as *mut HashMap<String, *mut LispValLayout> as *mut std::ffi::c_void;

        // Set initial value
        let val1 = alloc_lisp_val();
        unsafe {
            (*val1).tag = TAG_INT;
            (*val1).data = 42;
        }

        let name = CString::new("x").unwrap();
        runtime_set_var(env_ptr, name.as_ptr(), val1);

        // Overwrite with new value
        let val2 = alloc_lisp_val();
        unsafe {
            (*val2).tag = TAG_INT;
            (*val2).data = 100;
        }

        runtime_set_var(env_ptr, name.as_ptr(), val2);

        // Verify new value
        let retrieved = runtime_get_var(env_ptr, name.as_ptr());
        unsafe {
            assert_eq!((*retrieved).data, 100);

            // Clean up both values
            lisp_val_decref(val1);
            lisp_val_decref(val2);
        }
    }

    #[test]
    fn test_shared_ownership_with_incref() {
        // Simulate two references to the same value
        let val = alloc_lisp_val();
        unsafe {
            (*val).tag = TAG_INT;
            (*val).data = 42;
        }

        // First reference (already has refcount=1)
        // Add second reference
        lisp_val_incref(val);

        unsafe {
            assert_eq!((*val).refcount, 2, "refcount should be 2");
        }

        // First reference drops
        lisp_val_decref(val);

        unsafe {
            assert_eq!((*val).refcount, 1, "refcount should be 1 after first decref");

            // Second reference drops (this should free)
            lisp_val_decref(val);
        }
    }

    #[test]
    fn test_nested_list_with_shared_elements() {
        // Create a shared value
        let shared_val = alloc_lisp_val();
        unsafe {
            (*shared_val).tag = TAG_INT;
            (*shared_val).data = 42;
        }

        // Increase refcount since it will be in two lists
        lisp_val_incref(shared_val);

        // Create first list: (42)
        let cell1 = alloc_cons_cell();
        unsafe {
            (*cell1).head = shared_val;
            (*cell1).tail = ptr::null_mut();
        }

        let list1 = alloc_lisp_val();
        unsafe {
            (*list1).tag = TAG_LIST;
            (*list1).data = cell1 as i64;
        }

        // Create second list: (42)
        let cell2 = alloc_cons_cell();
        unsafe {
            (*cell2).head = shared_val;
            (*cell2).tail = ptr::null_mut();
        }

        let list2 = alloc_lisp_val();
        unsafe {
            (*list2).tag = TAG_LIST;
            (*list2).data = cell2 as i64;
        }

        // Free first list (should decref shared_val but not free it)
        lisp_val_decref(list1);

        // Free second list (should decref and free shared_val)
        lisp_val_decref(list2);
    }

    #[test]
    fn test_lval_print_refcount() {
        // Test that print doesn't crash on various types
        let int_val = alloc_lisp_val();
        unsafe {
            (*int_val).tag = TAG_INT;
            (*int_val).data = 42;
        }
        lisp_val_print_refcount(int_val);
        lisp_val_decref(int_val);

        // Test null
        lisp_val_print_refcount(ptr::null());

        // Test nil
        let nil_val = alloc_lisp_val();
        unsafe {
            (*nil_val).tag = TAG_NIL;
        }
        lisp_val_print_refcount(nil_val);
        lisp_val_decref(nil_val);
    }
}
