//! OpenCL module for GPU memory allocation and management
//!
//! This module handles interaction with the GPU via OpenCL,
//! including device selection, memory allocation, and data transfer.

mod memory;
mod device;

pub use memory::{VRamBuffer, VRamBufferConfig};
pub use device::list_opencl_devices;
