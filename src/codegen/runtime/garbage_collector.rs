use std::{alloc::{GlobalAlloc, Layout, System}, sync::{atomic::{AtomicUsize, Ordering}, LazyLock}};

pub struct RefCountingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for RefCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        ALLOCATED.fetch_add(size, Ordering::SeqCst);
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);

        let ptr = System.alloc(layout);
        
        if *IS_DEBUG_MODE {
            println!("[ALLOC] {} bytes (total: {})", size, ALLOCATED.load(Ordering::SeqCst));
        };
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        let size = layout.size();

        System.dealloc(ptr, layout);

        DEALLOCATED.fetch_add(size, Ordering::SeqCst);
        if *IS_DEBUG_MODE {
            println!("[DEALLOC] {} bytes (total: {})", size, DEALLOCATED.load(Ordering::SeqCst));
        }
    }
}

static IS_DEBUG_MODE: LazyLock<bool> = std::sync::LazyLock::new(|| {
    #[cfg(not(test))]
    { std::env::var("GC_DEBUG").is_ok() }
    #[cfg(test)]
    { false }
    });

#[global_allocator]
static GC: RefCountingAllocator  = RefCountingAllocator;

pub fn print_memory_stats() {
    let allocated = ALLOCATED.load(Ordering::SeqCst);
    let deallocated = DEALLOCATED.load(Ordering::SeqCst);
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    println!("\n=== Memory Statistics ===");
    println!("Total allocations: {}", count);
    println!("Bytes allocated:   {}", allocated);
    println!("Bytes deallocated: {}", deallocated);
    println!("Bytes leaked:      {}", allocated - deallocated);
    println!("========================\n");
}