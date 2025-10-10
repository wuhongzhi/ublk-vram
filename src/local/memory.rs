use anyhow::{Ok, Result, bail};
use std::{
    alloc::{self, Layout},
    ptr::NonNull,
    sync::RwLock,
};

use crate::VBuffer;

pub struct LOBuffer {
    buffer: RwLock<NonNull<u8>>,
    offset: u64,
    size: usize,
}

impl LOBuffer {
    /// Create a new local memory buffer with the specified configuration
    pub fn new(size: usize) -> Result<Self> {
        let buffer;
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, size_of::<usize>());
            buffer = NonNull::new_unchecked(alloc::alloc_zeroed(layout));
        }
        log::debug!("Created buffer of size {} bytes on vmm", size);
        Ok(Self {
            buffer: RwLock::new(buffer),
            offset: 0,
            size,
        })
    }

    // check offset in this vram
    #[inline]
    fn within(&self, offset: u64) -> bool {
        offset >= self.offset && offset < self.offset + self.size as u64
    }
}

unsafe impl Send for LOBuffer {}
unsafe impl Sync for LOBuffer {}

impl VBuffer for LOBuffer {
    fn remaining(&self, offset: u64) -> Option<usize> {
        if self.within(offset) {
            Some((self.size as u64 + self.offset - offset) as usize)
        } else {
            None
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    fn offset(&mut self, offset: u64) {
        self.offset = offset;
    }

    fn read(&self, offset: u64, data: &mut [u8]) -> Result<()> {
        if !self.within(offset) {
            bail!("Attempted to read out of buffer");
        }
        let local_offset = (offset - self.offset) as usize;
        let length = data.len();
        if local_offset + length > self.size {
            bail!("Attempted to read past end of buffer");
        }
        let buffer_guard = self
            .buffer
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to lock buffer RwLock for read"))
            .unwrap();
        unsafe {
            buffer_guard
                .as_ptr()
                .add(local_offset)
                .copy_to_nonoverlapping(data.as_mut_ptr(), length);
        }
        Ok(())
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<()> {
        if !self.within(offset) {
            bail!("Attempted to write out of buffer");
        }
        let local_offset = (offset - self.offset) as usize;
        let length = data.len();
        if local_offset + length > self.size {
            bail!("Attempted to write past end of buffer");
        }
        let buffer_guard = self
            .buffer
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to lock buffer RwLock for write"))
            .unwrap();
        unsafe {
            buffer_guard
                .as_ptr()
                .add(local_offset)
                .copy_from_nonoverlapping(data.as_ptr(), length);
        }
        Ok(())
    }
}

impl Drop for LOBuffer {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.size, size_of::<u64>());
            alloc::dealloc(self.buffer.get_mut().unwrap().as_ptr(), layout);
        }
        log::debug!("Freeing memory buffer");
    }
}
