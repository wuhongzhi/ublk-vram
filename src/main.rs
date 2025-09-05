use anyhow::{Context, Result, bail};
use clap::Parser;
use env_logger::{Builder, Env};
use nix::sys::mman::{MlockAllFlags, mlockall};
use ublk_vram::{
    opencl::{VRamBuffer, VRamBufferConfig, VramDevice, list_opencl_devices},
    start_ublk_server,
};

/// Command line arguments for the VRAM Block Device
#[derive(Parser, Debug)]
#[clap(
    name = "ublk-vram",
    about = "Expose OCL memory as a block device using a UBLK. Locks memory using mlockall.",
    version
)]
struct Args {
    /// Size of the block device (e.g., 512M, 2G, 1024). Defaults to MB if no suffix.
    #[clap(short, long, value_parser = parse_size_string, default_value = "2048M")]
    size: u64, // Store size in bytes

    /// OCL device index to use (0 for first OCL)
    #[clap(short, long, default_value = "0")]
    device: usize,

    /// OpenCL platform index
    #[clap(short, long, default_value = "0")]
    platform: usize,

    /// Enable verbose logging
    #[clap(short, long)]
    verbose: bool,

    /// Read/Write via memory mapping
    #[clap(short, long)]
    mmap: bool,

    /// List available OpenCL platforms and devices and exit
    #[clap(long)]
    list_devices: bool,

    /// How many block buffers
    #[clap(long, default_value = "1")]
    blocks: usize,

    /// CPU device
    #[clap(long)]
    cpu: bool,
}

/// Parses a size string (e.g., "512M", "2G") into bytes.
pub(crate) fn parse_size_string(size_str: &str) -> Result<u64> {
    let size_str = size_str.trim().to_uppercase();
    let (num_part, suffix) = size_str.split_at(
        size_str
            .find(|c: char| !c.is_ascii_digit())
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
    
    let mut config = VRamBufferConfig {
        platform_index: args.platform,
        device_index: args.device,
        size: args.size as usize,
        mmap: args.mmap,
        ..Default::default()
    };
    if args.cpu {
        config.with_cpu();
    }

    if args.list_devices {
        return list_opencl_devices(&config);
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
        "Allocating {} bytes ({} MB) on OCL device {} (Platform {})",
        args.size * args.blocks.max(1) as u64,
        args.size * args.blocks.max(1) as u64 / (1024 * 1024), // Log MB for readability
        config.device_index,
        config.platform_index
    );

    let device = VramDevice::new(&config).context("Failed to allocate OCL Device")?;
    let mut vrams: Vec<VRamBuffer> = Vec::new();
    for _ in 0..args.blocks.max(1) {
        vrams.push(
            VRamBuffer::new(&device, config.size, config.mmap)
                .context("Failed to allocate OCL memory")?,
        );
    }

    log::info!(
        "Successfully allocated {} bytes ({} MB) on {}",
        args.size * args.blocks.max(1) as u64,
        args.size * args.blocks.max(1) as u64 / (1024 * 1024), // Log MB for readability
        device.name()
    );

    log::info!("Starting VRAM Block Device (UBLK)");
    let _ = start_ublk_server(vrams);
    log::info!("VRAM Block Device has shut down.");

    Ok(())
}
