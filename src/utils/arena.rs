//! A simple slot-based arena allocator that guarantees **stable, compact IDs** for stored objects.
//!
//! # Responsibility
//! * Provide an _append-only_ (with optional removal) container that returns a monotonically
//!   increasing integer ID (`u32`) each time an element is inserted.
//! * Keep the mapping _ID ↔ value_ stable for the lifetime of the value unless it is explicitly
//!   removed.
//! * Reuse freed slots to avoid unbounded memory growth.
//!
//! # Non-Responsibilities / Limitations
//! * **No built-in thread-safety**: `Arena` is `!Sync` / `!Send` by default. Wrap it in
//!   `std::sync::Mutex` / `RwLock` / `parking_lot` primitives if you need concurrent access.
//! * **No graph semantics** such as topological ordering, inputs/outputs, etc. Those are handled by
//!   higher-level modules (see `graph/`).
//! * Does **not** automatically shrink memory on removals. A manual `shrink_to_fit()` is provided
//!   if you need it.
//! * Not designed for persistent/immutable data-structures. When you clone an `Arena`, the values
//!   and internal state are _deep-copied_.

use std::iter::Enumerate;
use std::slice::{Iter, IterMut};
use std::marker::PhantomData;

/// Slot-based arena that owns a collection of `T` and returns stable `u32` IDs.
#[derive(Debug, Default)]
pub struct Arena<T, ID: ArenaId> {
    items: Vec<Option<T>>, // None = slot is free/reclaimable
    vacant_indices: Vec<u32>, // stack of free indices for O(1) reuse
    _phantom: PhantomData<ID>,
}

impl<T, ID: ArenaId> Arena<T, ID> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            vacant_indices: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            items: Vec::with_capacity(cap),
            vacant_indices: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn alloc(&mut self, value: T) -> ID {
        let idx = match self.vacant_indices.pop() {
            Some(reuse_idx) => {
                self.items[reuse_idx as usize] = Some(value);
                reuse_idx
            }
            None => {
                let idx = self.items.len() as u32;
                self.items.push(Some(value));
                idx
            }
        };

        ID::from_u32(idx)
    }


    pub fn get(&self, id: ID) -> Option<&T> {
        self.items.get(id.into_u32() as usize)?.as_ref()
    }

    pub fn get_mut(&mut self, id: ID) -> Option<&mut T> {
        let idx = id.into_u32() as usize;
        self.items.get_mut(idx)?.as_mut()
    }

    pub fn free(&mut self, id: ID) -> Option<T> {
        let idx = id.into_u32() as usize;

        // Check bounds
        if idx >= self.items.len() {
            return None;
        }

        // Try to take the value if it's not vacant
        match self.items[idx].take() {
            Some(value) => {
                self.vacant_indices.push(id.into_u32()); // mark slot as reusable
                Some(value)
            }
            None => None, // was already vacant
        }
    }

    /// Returns the number of **occupied** slots (not the capacity).
    pub fn len(&self) -> usize {
        let len_items = self.items.len();
        let len_vacant_indices = self.vacant_indices.len();
        let len_occupied = len_items - len_vacant_indices;
        assert!(len_occupied <= self.items.len() );
        len_occupied
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> ArenaIter<'_, T> {
        ArenaIter {
            inner: self.items.iter().enumerate(),
        }
    }

    pub fn iter_mut(&mut self) -> ArenaIterMut<'_, T> {
        ArenaIterMut {
            inner: self.items.iter_mut().enumerate(),
        }
    }
}

pub struct ArenaIter<'a, T> {
    inner: Enumerate<Iter<'a, Option<T>>>,
}

impl<'a, T> Iterator for ArenaIter<'a, T> {
    type Item = (u32, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        for (idx, slot) in &mut self.inner {
            if let Some(val) = slot.as_ref() {
                return Some((idx as u32, val));
            }
        }
        None
    }
}

pub struct ArenaIterMut<'a, T> {
    inner: Enumerate<IterMut<'a, Option<T>>>,
}

impl<'a, T> Iterator for ArenaIterMut<'a, T> {
    type Item = (u32, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        for (idx, slot) in &mut self.inner {
            if let Some(val) = slot.as_mut() {
                return Some((idx as u32, val));
            }
        }
        None
    }
}


pub trait ArenaId: Copy {
    fn from_u32(raw: u32) -> Self;
    fn into_u32(self) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct MockId(u32);
    #[derive(Debug, PartialEq)]
    struct MockNode {
        value: i32,
    }
    impl MockNode {
        fn new(val: i32) -> Self {
            MockNode { value: val }
        }
    }

    impl ArenaId for MockId {
        fn from_u32(id: u32) -> Self {
            MockId(id)
        }

        fn into_u32(self) -> u32 {
            self.0
        }
    }

    #[test]
    fn test_arena_alloc_and_get() {
        let mut arena = Arena::<MockNode, MockId>::new();

        let id = arena.alloc(MockNode::new(42));
        let node = arena.get(id).unwrap();

        assert_eq!(node.value, 42);
    }

    #[test]
    fn test_arena_reuse_vacant_slot() {
        let mut arena = Arena::<MockNode, MockId>::new();

        let id1 = arena.alloc(MockNode::new(1));
        let id2 = arena.alloc(MockNode::new(2));

        arena.items[id1.into_u32() as usize] = None;
        arena.vacant_indices.push(id1.into_u32());

        let id3 = arena.alloc(MockNode::new(3));

        // id3 should reuse id1's index
        assert_eq!(id3.into_u32(), id1.into_u32());
        let node3 = arena.get(id3).unwrap();
        assert_eq!(node3.value, 3);
    }

    #[test]
    fn test_free_arena() {
        let mut arena = Arena::<MockNode, MockId>::new();

        let id = arena.alloc(MockNode::new(42));
        assert_eq!(arena.len(), 1);

        // free occupied slot
        let value = arena.free(id).unwrap();
        assert_eq!(value.value, 42);
        assert_eq!(arena.len(), 0);

        let value = arena.free(id);
        assert!(value.is_none());
    }
}

