use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs::OpenOptions,
    io::Write,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread,
    time::Duration,
};

pub struct RefCountingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static DEALLOCATED: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static IS_DEBUG_MODE: AtomicBool = AtomicBool::new(false);

/// Set GC debug mode - must be called before any allocations if possible
pub fn set_gc_debug_mode(enabled: bool) {
    IS_DEBUG_MODE.store(enabled, Ordering::SeqCst);
}

unsafe impl GlobalAlloc for RefCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        ALLOCATED.fetch_add(size, Ordering::SeqCst);
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);

        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        let size = layout.size();
        unsafe { System.dealloc(ptr, layout) };
        DEALLOCATED.fetch_add(size, Ordering::SeqCst);
    }
}

#[global_allocator]
static GC: RefCountingAllocator = RefCountingAllocator;

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

/// Start a background thread that periodically logs memory stats to a file
pub fn start_gc_monitor(interval_ms: u64, log_file: &str) {
    let log_path = log_file.to_string();

    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_millis(interval_ms));

            let allocated = ALLOCATED.load(Ordering::SeqCst);
            let deallocated = DEALLOCATED.load(Ordering::SeqCst);
            let count = ALLOC_COUNT.load(Ordering::SeqCst);
            let leaked = allocated.saturating_sub(deallocated);

            // Write to log file
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
            {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);

                let _ = writeln!(
                    file,
                    "[{}] allocs={} allocated={} deallocated={} leaked={}",
                    timestamp, count, allocated, deallocated, leaked
                );
            }
        }
    });
}