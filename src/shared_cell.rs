use crate::{Node, Shared, SharedInner};

use core::marker::PhantomData;
use core::ptr::NonNull;

#[cfg(not(all(feature = "loom", test)))]
use core::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
#[cfg(all(feature = "loom", test))]
use loom::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

const WRITER: usize = 1 << (usize::BITS as usize - 1);
const READER_MASK: usize = WRITER - 1;

/// A thread-safe shared mutable memory location that holds a [`Shared<T>`].
///
/// `SharedCell` is designed to be low-overhead for readers at the expense of
/// somewhat higher overhead for writers.
///
/// [`Shared<T>`]: crate::Shared
pub struct SharedCell<T> {
    readers: AtomicUsize,
    node: AtomicPtr<Node<SharedInner<T>>>,
    phantom: PhantomData<Shared<T>>,
}

unsafe impl<T: Send + Sync> Send for SharedCell<T> {}
unsafe impl<T: Send + Sync> Sync for SharedCell<T> {}

impl<T: Send + 'static> SharedCell<T> {
    /// Constructs a new `SharedCell` containing `value`.
    ///
    /// # Examples
    /// ```
    /// use basedrop::{Collector, Shared, SharedCell};
    ///
    /// let collector = Collector::new();
    /// let three = Shared::new(&collector.handle(), 3);
    /// let cell = SharedCell::new(three);
    /// ```
    pub fn new(value: Shared<T>) -> SharedCell<T> {
        let node = value.node.as_ptr();
        core::mem::forget(value);

        SharedCell {
            readers: AtomicUsize::new(0),
            node: AtomicPtr::new(node),
            phantom: PhantomData,
        }
    }
}

impl<T> SharedCell<T> {
    /// Gets a copy of the contained [`Shared<T>`], incrementing its reference
    /// count in the process.
    ///
    /// # Examples
    /// ```
    /// use basedrop::{Collector, Shared, SharedCell};
    ///
    /// let collector = Collector::new();
    /// let x = Shared::new(&collector.handle(), 3);
    /// let cell = SharedCell::new(x);
    ///
    /// let y = cell.get();
    /// ```
    ///
    /// [`Shared<T>`]: crate::Shared
    pub fn get(&self) -> Shared<T> {
        loop {
            let readers = self.readers.load(Ordering::Acquire);
            if readers & WRITER != 0 {
                #[cfg(all(feature = "loom", test))]
                loom::thread::yield_now();
                continue;
            }

            assert!(readers < READER_MASK);
            if self
                .readers
                .compare_exchange_weak(readers, readers + 1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        let shared = Shared {
            node: unsafe { NonNull::new_unchecked(self.node.load(Ordering::SeqCst)) },
            phantom: PhantomData,
        };
        let copy = shared.clone();
        core::mem::forget(shared);

        self.readers.fetch_sub(1, Ordering::AcqRel);

        copy
    }

    /// Replaces the contained [`Shared<T>`], decrementing its reference count
    /// in the process.
    ///
    /// # Examples
    /// ```
    /// use basedrop::{Collector, Shared, SharedCell};
    ///
    /// let collector = Collector::new();
    /// let x = Shared::new(&collector.handle(), 3);
    /// let cell = SharedCell::new(x);
    ///
    /// let y = Shared::new(&collector.handle(), 4);
    /// cell.set(y);
    /// ```
    ///
    /// [`Shared<T>`]: crate::Shared
    pub fn set(&self, value: Shared<T>) {
        let old = self.replace(value);
        core::mem::drop(old);
    }

    /// Replaces the contained [`Shared<T>`] and returns it.
    ///
    /// # Examples
    /// ```
    /// use basedrop::{Collector, Shared, SharedCell};
    ///
    /// let collector = Collector::new();
    /// let x = Shared::new(&collector.handle(), 3);
    /// let cell = SharedCell::new(x);
    ///
    /// let y = Shared::new(&collector.handle(), 4);
    /// let x = cell.replace(y);
    /// ```
    ///
    /// [`Shared<T>`]: crate::Shared
    pub fn replace(&self, value: Shared<T>) -> Shared<T> {
        let node = value.node.as_ptr();
        core::mem::forget(value);

        loop {
            let readers = self.readers.load(Ordering::Acquire);
            if readers & WRITER != 0 {
                #[cfg(all(feature = "loom", test))]
                loom::thread::yield_now();
                continue;
            }

            if self
                .readers
                .compare_exchange_weak(
                    readers,
                    readers | WRITER,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }

        while self.readers.load(Ordering::Acquire) != WRITER {
            #[cfg(all(feature = "loom", test))]
            loom::thread::yield_now();
        }

        let old = self.node.swap(node, Ordering::AcqRel);
        self.readers.store(0, Ordering::Release);

        Shared {
            node: unsafe { NonNull::new_unchecked(old) },
            phantom: PhantomData,
        }
    }

    /// Consumes the `SharedCell` and returns the contained [`Shared<T>`]. This
    /// is safe because we are guaranteed to be the only holder of the
    /// `SharedCell`.
    ///
    /// # Examples
    /// ```
    /// use basedrop::{Collector, Shared, SharedCell};
    ///
    /// let collector = Collector::new();
    /// let x = Shared::new(&collector.handle(), 3);
    /// let cell = SharedCell::new(x);
    ///
    /// let x = cell.into_inner();
    /// ```
    ///
    /// [`Shared<T>`]: crate::Shared
    pub fn into_inner(mut self) -> Shared<T> {
        let node = core::mem::replace(&mut self.node, AtomicPtr::new(core::ptr::null_mut()));
        core::mem::forget(self);
        Shared {
            node: unsafe { NonNull::new_unchecked(node.into_inner()) },
            phantom: PhantomData,
        }
    }
}

impl<T> Drop for SharedCell<T> {
    fn drop(&mut self) {
        let _ = Shared {
            node: unsafe { NonNull::new_unchecked(self.node.load(Ordering::Relaxed)) },
            phantom: PhantomData,
        };
    }
}

#[cfg(all(test, not(feature = "loom")))]
mod tests {
    use crate::{Collector, Shared, SharedCell};

    use core::sync::atomic::AtomicUsize;
    use core::sync::atomic::Ordering;

    #[test]
    fn shared_cell() {
        extern crate alloc;
        use alloc::sync::Arc;

        struct Test(Arc<AtomicUsize>);

        impl Drop for Test {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let counter = Arc::new(AtomicUsize::new(0));

        let mut collector = Collector::new();
        let shared = Shared::new(&collector.handle(), Test(counter.clone()));
        let cell = SharedCell::new(shared);
        collector.collect();

        assert_eq!(counter.load(Ordering::Relaxed), 0);

        let copy = cell.get();
        let copy2 = cell.replace(copy);
        collector.collect();

        assert_eq!(counter.load(Ordering::Relaxed), 0);

        core::mem::drop(cell);
        collector.collect();

        assert_eq!(counter.load(Ordering::Relaxed), 0);

        core::mem::drop(copy2);
        collector.collect();

        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn shared_cell_replace_sees_reader_clone() {
        let collector = Collector::new();
        let handle = collector.handle();

        let old = Shared::new(&handle, 1);
        let cell = SharedCell::new(old);

        let reader = cell.get();
        let new = Shared::new(&handle, 2);
        let mut old = cell.replace(new);

        assert!(Shared::get_mut(&mut old).is_none());
        drop(reader);
    }
}

#[cfg(all(test, feature = "loom"))]
mod tests {
    use crate::{Collector, Shared, SharedCell};

    use loom::sync::{Arc, Condvar, Mutex};
    use loom::thread;

    #[test]
    fn shared_cell_replace_sees_concurrent_get_clone() {
        struct Payload {
            is_old: bool,
        }

        let mut builder = loom::model::Builder::new();
        builder.max_branches = 100_000;
        builder.preemption_bound = Some(3);

        builder.check(|| {
            let mut collector = Collector::new();
            let handle = collector.handle();
            let state = Arc::new((Mutex::new((false, false)), Condvar::new()));

            let old = Shared::new(&handle, Payload { is_old: true });
            let cell = Arc::new(SharedCell::new(old));

            let reader_cell = cell.clone();
            let reader_state = state.clone();
            let reader = thread::spawn(move || {
                let value = reader_cell.get();
                assert!(value.is_old);

                let (lock, cvar) = &*reader_state;
                let mut state = lock.lock().unwrap();
                state.0 = true;
                cvar.notify_one();

                while !state.1 {
                    state = cvar.wait(state).unwrap();
                }
                drop(value);
            });

            let (lock, cvar) = &*state;
            let mut test_state = lock.lock().unwrap();
            while !test_state.0 {
                test_state = cvar.wait(test_state).unwrap();
            }
            drop(test_state);

            let new = Shared::new(&handle, Payload { is_old: false });
            let mut old = cell.replace(new);

            assert!(
                Shared::get_mut(&mut old).is_none(),
                "replace returned an old Shared that appeared unique while a reader still held a clone"
            );

            let mut test_state = lock.lock().unwrap();
            test_state.1 = true;
            cvar.notify_one();
            drop(test_state);
            drop(old);

            reader.join().unwrap();

            drop(cell);
            collector.collect();
        });
    }
}
