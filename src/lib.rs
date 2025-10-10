pub mod local;
pub mod opencl;
#[path = "ublk/server.rs"]
mod server;

pub use server::start_ublk_server;

use anyhow::Result;
pub trait VBuffer: Send + Sync {
    /// read data from buffer
    fn read(&self, offset: u64, data: &mut [u8]) -> Result<()>;
    /// write data to buffer
    fn write(&self, offset: u64, data: &[u8]) -> Result<()>;
    /// check remaining parts
    fn remaining(&self, offset: u64) -> Option<usize>;
    /// set offset in global area
    fn offset(&mut self, offset: u64);
    /// get size of this buffer
    fn size(&self) -> usize;
}
pub struct VMemory<T> {
    vrams: Vec<T>,
    size: u64,
}

unsafe impl<T: VBuffer> Send for VMemory<T> {}
unsafe impl<T: VBuffer> Sync for VMemory<T> {}

impl<T: VBuffer> VMemory<T> {
    pub fn new(mut vrams: Vec<T>) -> Self {
        let mut size: u64 = 0;
        for i in vrams.iter_mut() {
            i.offset(size);
            size += i.size() as u64;
        }
        Self { vrams, size }
    }

    /// # Safety
    /// data must a validate ptr
    pub unsafe fn read(&self, offset: u64, length: usize, data: *mut u8) -> i32 {
        let mut local_offset = 0;
        let mut global_offset = offset;
        let mut global_remaining = length;
        for (i, vram) in self.vrams.iter().enumerate() {
            let local_remaining = vram.remaining(global_offset);
            if local_remaining.is_none() {
                continue;
            }
            // compute local length to read/write
            let local_length = global_remaining.min(local_remaining.unwrap());

            let array = unsafe {
                std::slice::from_raw_parts_mut(data.add(local_offset), local_length)
            };
            if let Err(e) = vram.read(global_offset, array) {
                log::error!(
                    "Read error, device vram-{} offset {} size {}, code {}",
                    i,
                    global_offset,
                    local_length,
                    e
                );
                return -libc::EIO;
            }

            // re-compute rest to read/write
            global_remaining -= local_length;
            if global_remaining == 0 {
                break;
            }
            local_offset += local_length;
            global_offset += local_length as u64;
        }
        if global_remaining > 0 {
            log::error!(
                "Read error, offset {} size {}",
                global_offset,
                global_remaining
            );
            return -libc::EIO;
        }
        length as i32
    }

    /// # Safety
    /// data must a validate ptr
    pub unsafe fn write(&self, offset: u64, length: usize, data: *const u8) -> i32 {
        let mut local_offset = 0;
        let mut global_offset = offset;
        let mut global_remaining = length;
        for (i, vram) in self.vrams.iter().enumerate() {
            let local_remaining = vram.remaining(global_offset);
            if local_remaining.is_none() {
                continue;
            }
            // compute local length to read/write
            let local_length = global_remaining.min(local_remaining.unwrap());

            let array = unsafe { std::slice::from_raw_parts(data.add(local_offset), local_length) };
            if let Err(e) = vram.write(global_offset, array) {
                log::error!(
                    "Write error, device vram-{} offset {} size {}, code {}",
                    i,
                    global_offset,
                    local_length,
                    e
                );
                return -libc::EIO;
            }

            // re-compute rest to read/write
            global_remaining -= local_length;
            if global_remaining == 0 {
                break;
            }
            local_offset += local_length;
            global_offset += local_length as u64;
        }
        if global_remaining > 0 {
            log::error!(
                "Read error, offset {} size {}",
                global_offset,
                global_remaining
            );
            return -libc::EIO;
        }
        length as i32
    }

    pub fn size(&self) -> u64 {
        self.size
    }
    pub fn blocks(&self) -> usize {
        self.vrams.len()
    }
}

impl<T: VBuffer> From<Vec<T>> for VMemory<T> {
    fn from(vrams: Vec<T>) -> Self {
        VMemory::new(vrams)
    }
}
