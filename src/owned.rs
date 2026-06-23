use crate::{Handle, Node};

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

/// An owned smart pointer with deferred collection, analogous to `Box`.
///
/// When an `Owned<T>` is dropped, its contents are added to the drop queue
/// of the [`Collector`] whose [`Handle`] it was originally allocated with.
/// As the collector may be on another thread, contents are required to be
/// `Send + 'static`.
///
/// [`Collector`]: crate::Collector
/// [`Handle`]: crate::Handle
pub struct Owned<T> {
    node: NonNull<Node<T>>,
    phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for Owned<T> {}
unsafe impl<T: Sync> Sync for Owned<T> {}

impl<T: Send + 'static> Owned<T> {
    /// Constructs a new `Owned<T>`.
    ///
    /// # Examples
    /// ```
    /// use basedrop::{Collector, Owned};
    ///
    /// let collector = Collector::new();
    /// let three = Owned::new(&collector.handle(), 3);
    /// ```
    pub fn new(handle: &Handle, data: T) -> Owned<T> {
        Owned {
            node: unsafe { NonNull::new_unchecked(Node::alloc(handle, data)) },
            phantom: PhantomData,
        }
    }
}

impl<T: Clone + Send + 'static> Clone for Owned<T> {
    fn clone(&self) -> Self {
        let handle = unsafe { Node::handle(self.node.as_ptr()) };
        Owned::new(&handle, self.deref().clone())
    }
}

impl<T> Deref for Owned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &self.node.as_ref().data }
    }
}

impl<T> DerefMut for Owned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut self.node.as_mut().data }
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        unsafe {
            Node::queue_drop(self.node.as_ptr());
        }
    }
}

#[cfg(all(test, not(feature = "loom")))]
mod tests {
    use crate::{Collector, Owned};

    use core::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn owned_drop_is_deferred() {
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
        let owned = Owned::new(&collector.handle(), Test(counter.clone()));

        drop(owned);
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        collector.collect();
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn owned_clone_allocates_independent_value() {
        let mut collector = Collector::new();
        let mut owned = Owned::new(&collector.handle(), 1);
        let mut cloned = owned.clone();

        *owned = 2;
        *cloned = 3;

        assert_eq!(*owned, 2);
        assert_eq!(*cloned, 3);

        drop(owned);
        drop(cloned);
        collector.collect();
    }
}

#[cfg(all(test, feature = "loom"))]
mod tests {
    use crate::{Collector, Owned};

    use loom::thread;

    #[test]
    fn owned_can_move_between_threads() {
        loom::model(|| {
            let mut collector = Collector::new();
            let handle = collector.handle();
            let owned = Owned::new(&handle, 1);

            let thread = thread::spawn(move || {
                let mut owned = owned;
                *owned = 2;
                assert_eq!(*owned, 2);
                drop(owned);
            });

            thread.join().unwrap();
            collector.collect();
        });
    }
}
