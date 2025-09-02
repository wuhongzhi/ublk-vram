//! OpenCL module for OCL memory allocation and management
//!
//! This module handles interaction with the OCL via OpenCL,
//! including device selection, memory allocation, and data transfer.

mod memory;
mod device;

pub use memory::{VRamBuffer, VRamBufferConfig};
pub use device::list_opencl_devices;
