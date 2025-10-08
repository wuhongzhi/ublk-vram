//! OpenCL module for OCL memory allocation and management
//!
//! This module handles interaction with the OCL via OpenCL,
//! including device selection, memory allocation, and data transfer.

mod device;
mod memory;

pub use device::{CLDevice, list_opencl_devices};
pub use memory::{CLBuffer, CLBufferConfig};
