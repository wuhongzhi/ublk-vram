//! GPU memory management via OpenCL
//!
//! This module provides functionality to allocate and manage
//! GPU memory buffers that will be exposed as block devices.

use anyhow::{Context, Result, bail};
use opencl3::{
    command_queue::{self as cl_command_queue, CommandQueue},
    context::Context as ClContext,
    device::{self as cl_device, Device},
    memory::{self as cl_memory, Buffer},
    platform::{self as cl_platform},
    types,
};
// Use std::sync::RwLock for thread-safe interior mutability
use std::ptr;
use std::sync::RwLock;

/// Configuration for a GPU memory buffer
#[derive(Debug, Clone)]
pub struct VRamBufferConfig {
    /// Size of the buffer in bytes
    pub size: usize,
    /// GPU device index to use (0 for first GPU)
    pub device_index: usize,
    /// Optional platform index (defaults to 0)
    pub platform_index: usize,
}

impl Default for VRamBufferConfig {
    fn default() -> Self {
        Self {
            size: 2048 * 1024 * 1024, // 2 GB default size
            device_index: 0,
            platform_index: 0,
        }
    }
}

/// A buffer allocated in GPU VRAM via OpenCL
// Make VRamBuffer Send + Sync by using RwLock for the buffer
pub struct VRamBuffer {
    queue: CommandQueue,
    // Use RwLock instead of RefCell
    buffer: RwLock<Buffer<u8>>,
    size: usize,
    device: Device,
}

impl VRamBuffer {
    /// Create a new GPU memory buffer with the specified configuration
    pub fn new(config: &VRamBufferConfig) -> Result<Self> {
        let platforms = cl_platform::get_platforms().context("Failed to get OpenCL platforms")?;

        if platforms.is_empty() {
            bail!("No OpenCL platforms available");
        }

        if config.platform_index >= platforms.len() {
            bail!(
                "Platform index {} is out of bounds (max: {})",
                config.platform_index,
                platforms.len() - 1
            );
        }
        let platform = &platforms[config.platform_index];

        let device_ids = platform
            .get_devices(cl_device::CL_DEVICE_TYPE_GPU)
            .context("Failed to get device list")?;

        if device_ids.is_empty() {
            bail!(
                "No GPU devices found for platform {}",
                config.platform_index
            );
        }

        if config.device_index >= device_ids.len() {
            bail!(
                "Device index {} is out of bounds (max: {})",
                config.device_index,
                device_ids.len() - 1
            );
        }
        let device_id = device_ids[config.device_index];
        let device = Device::new(device_id);
        let context = ClContext::from_device(&device).context("Failed to create OpenCL context")?;

        let queue = unsafe {
            CommandQueue::create_with_properties(
                &context,
                device.id(),
                cl_command_queue::CL_QUEUE_PROFILING_ENABLE,
                0,
            )
            .context("Failed to create command queue")?
        };

        let buffer = unsafe {
            Buffer::<u8>::create(
                &context,
                cl_memory::CL_MEM_READ_WRITE,
                config.size,
                ptr::null_mut(),
            )
            .context("Failed to allocate GPU memory")?
        };

        log::info!(
            "Created OpenCL buffer of size {} bytes on device: {}",
            config.size,
            device
                .name()
                .unwrap_or_else(|_| "Unknown device".to_string())
        );

        Ok(Self {
            queue,
            buffer: RwLock::new(buffer),
            size: config.size,
            device,
        })
    }

    /// Get the buffer size in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Read data from the GPU buffer
    pub fn read(&self, offset: usize, data: &mut [u8]) -> Result<()> {
        if offset + data.len() > self.size {
            bail!("Attempted to read past end of buffer");
        }

        let buffer_guard = self
            .buffer
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to lock buffer mutex for read"))?;

        unsafe {
            self.queue
                .enqueue_read_buffer(&*buffer_guard, types::CL_TRUE, offset, data, &[])
                .context("Failed to enqueue blocking read from buffer")?
                .wait()
                .context("Failed waiting for blocking read event")?;
        }

        Ok(())
    }

    /// Write data to the GPU buffer
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.size {
            bail!("Attempted to write past end of buffer");
        }

        let mut buffer_guard = self
            .buffer
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to lock buffer mutex for write"))?;

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut *buffer_guard, types::CL_TRUE, offset, data, &[])
                .context("Failed to enqueue blocking write to buffer")?
                .wait()
                .context("Failed waiting for blocking write event")?;
        }

        Ok(())
    }

    /// Get the device name
    pub fn device_name(&self) -> String {
        self.device
            .name()
            .unwrap_or_else(|_| "Unknown device".to_string())
    }
}

impl Drop for VRamBuffer {
    fn drop(&mut self) {
        log::debug!("Freeing GPU memory buffer");
    }
}
