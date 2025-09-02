//! OpenCL module for OCL memory allocation and management
//!
//! This module handles interaction with the OCL via OpenCL,
//! including device selection, memory allocation, and data transfer.

mod ublk;

pub use ublk::start_ublk_server;