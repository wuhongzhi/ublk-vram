use anyhow::{Context, Result};
use opencl3::{
    device::{self as cl_device, Device, get_device_ids},
    platform::get_platforms,
};

/// Lists available OpenCL devices.
pub fn list_opencl_devices() -> Result<()> {
    println!("Available OpenCL Platforms and Devices:");
    let platforms = get_platforms().context("Failed to get OpenCL platforms")?;
    if platforms.is_empty() {
        println!("  No OpenCL platforms found.");
        return Ok(());
    }

    for (plat_idx, platform) in platforms.iter().enumerate() {
        let plat_name = platform
            .name()
            .unwrap_or_else(|_| "Unknown Platform".to_string());
        println!("\nPlatform {}: {}", plat_idx, plat_name);

        match get_device_ids(
            platform.id(),
            cl_device::CL_DEVICE_TYPE_GPU | cl_device::CL_DEVICE_TYPE_ACCELERATOR,
        ) {
            Ok(device_ids) => {
                if device_ids.is_empty() {
                    println!("  No OCL devices found on this platform.");
                } else {
                    for (dev_idx, device_id) in device_ids.iter().enumerate() {
                        let device = Device::new(*device_id);
                        let dev_name = device
                            .name()
                            .unwrap_or_else(|_| "Unknown Device".to_string());
                        let dev_vendor = device
                            .vendor()
                            .unwrap_or_else(|_| "Unknown Vendor".to_string());
                        let dev_mem = device.global_mem_size().unwrap_or(0);
                        println!(
                            "  Device {}: {} ({}) - Memory: {} MB",
                            dev_idx,
                            dev_name,
                            dev_vendor,
                            dev_mem / (1024 * 1024)
                        );
                    }
                }
            }
            Err(e) => {
                println!("  Error getting devices for this platform: {}", e);
            }
        }
    }
    Ok(())
}
