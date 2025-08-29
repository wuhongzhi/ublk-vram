mod opencl;
mod ublk;

use crate::{
    opencl::{VRamBuffer, VRamBufferConfig, list_opencl_devices},
    ublk::start_ublk_server,
};
use anyhow::{Context, Result, bail};
use clap::Parser;
use env_logger::{Builder, Env};
use nix::sys::mman::{MlockAllFlags, mlockall};

/// Command line arguments for the VRAM Block Device
#[derive(Parser, Debug)]
#[clap(
    name = "ublk-vram",
    about = "Expose GPU memory as a block device using a UBLK. Locks memory using mlockall.",
    version
)]
struct Args {
    /// Size of the block device (e.g., 512M, 2G, 1024). Defaults to MB if no suffix.
    #[clap(short, long, value_parser = parse_size_string, default_value = "2048M")]
    size: u64, // Store size in bytes

    /// GPU device index to use (0 for first GPU)
    #[clap(short, long, default_value = "0")]
    device: usize,

    /// OpenCL platform index
    #[clap(short, long, default_value = "0")]
    platform: usize,

    /// Enable verbose logging
    #[clap(short, long)]
    verbose: bool,

    /// List available OpenCL platforms and devices and exit
    #[clap(long)]
    list_devices: bool,
}

/// Parses a size string (e.g., "512M", "2G") into bytes.
pub(crate) fn parse_size_string(size_str: &str) -> Result<u64> {
    let size_str = size_str.trim().to_uppercase();
    let (num_part, suffix) = size_str.split_at(
        size_str
            .find(|c: char| !c.is_digit(10))
            .unwrap_or(size_str.len()),
    );

    let num: u64 = num_part.parse().context("Invalid size number")?;

    match suffix {
        "" | "M" | "MB" => Ok(num * 1024 * 1024),
        "G" | "GB" => Ok(num * 1024 * 1024 * 1024),
        _ => bail!("Invalid size suffix: '{}'. Use M/MB or G/GB.", suffix),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.verbose {
        Builder::from_env(Env::default().default_filter_or("debug")).init();
    } else {
        Builder::from_env(Env::default().default_filter_or("info")).init();
    }
    if args.list_devices {
        return list_opencl_devices();
    }

    log::info!("Attempting to lock process memory using mlockall()...");

    // Use correct flag names from the MlockAllFlags type
    match mlockall(MlockAllFlags::MCL_CURRENT | MlockAllFlags::MCL_FUTURE) {
        Ok(_) => log::info!("Successfully locked process memory."),
        Err(e) => {
            log::warn!(
                "Failed to lock process memory (requires root or CAP_IPC_LOCK): {}",
                e
            );
        }
    }
    // Size is already parsed into bytes
    log::info!(
        "Allocating {} bytes ({} MB) on GPU device {} (Platform {})",
        args.size,
        args.size / (1024 * 1024), // Log MB for readability
        args.device,
        args.platform
    );

    let vram_config = VRamBufferConfig {
        size: args.size as usize, // VRamBufferConfig expects usize
        device_index: args.device,
        platform_index: args.platform,
    };
    let vram =
        VRamBuffer::new(&vram_config).context("Failed to allocate GPU memory")?;

    log::info!(
        "Successfully allocated {} bytes ({} MB) on {}",
        args.size,
        args.size / (1024 * 1024), // Log MB for readability
        vram.device_name()
    );

    log::info!("Starting VRAM Block Device (UBLK)");
    let _ = start_ublk_server(vram);
    log::info!("VRAM Block Device has shut down.");

    Ok(())
}
