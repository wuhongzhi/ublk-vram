use std::ops::Div;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use env_logger::{Builder, Env};
use nix::sys::mman::{MlockAllFlags, mlockall};
use ublk_vram::{
    VMemory,
    local::LOBuffer,
    opencl::{CLBuffer, CLBufferConfig, CLDevice, list_opencl_devices},
    start_ublk_server,
};

/// Command line arguments for the VRAM Block Device
#[derive(Parser)]
#[clap(
    name = "ublk-vram",
    about = "Expose OCL memory as a block device using a UBLK. Locks memory using mlockall.",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[clap(short, long)]
    verbose: bool,

    /// Size of the block device (e.g., 512M, 2G, 1024). Defaults to MB if no suffix.
    #[clap(short, long, value_parser = parse_size_string, default_value = "2048M")]
    size: u64, // Store size in bytes

    /// How many blocks, max 100
    #[clap(short, long, default_value = "1")]
    blocks: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// OCL devices
    Ocl(CliOCL),
    /// VMM devices
    Vmm,
}

#[derive(Args)]
struct CliOCL {
    /// List available OpenCL platforms and devices and exit
    #[clap(long)]
    list_devices: bool,

    /// OCL device index to use (0 for first OCL)
    #[clap(short, long, default_value = "0")]
    device: usize,

    /// OpenCL platform index
    #[clap(short, long, default_value = "0")]
    platform: usize,

    /// Read/Write via memory mapping
    #[clap(short, long)]
    mmap: bool,

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
    let cli: Cli = Cli::parse();
    if cli.verbose {
        Builder::from_env(Env::default().default_filter_or("debug")).init();
    } else {
        Builder::from_env(Env::default().default_filter_or("info")).init();
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

    let _ = match cli.command {
        Commands::Vmm => start1(cli.size, cli.blocks.max(1).min(100)),
        Commands::Ocl(ocl) => {
            let mut config: CLBufferConfig = CLBufferConfig {
                platform_index: ocl.platform,
                device_index: ocl.device,
                size: cli.size as usize,
                mmap: ocl.mmap,
                ..Default::default()
            };
            if ocl.cpu {
                config.with_cpu();
            }

            if ocl.list_devices {
                return list_opencl_devices(&config);
            }
            start2(cli.size, cli.blocks.max(1).min(100), config)
        }
    };

    log::info!("VRAM Block Device has shut down.");
    Ok(())
}

fn start1(size: u64, blocks: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Size is already parsed into bytes
    log::info!(
        "Allocating {} bytes ({} MB)",
        size,
        size / (1024 * 1024), // Log MB for readability
    );

    let mut vrams: Vec<LOBuffer> = Vec::new();
    for _ in 0..blocks {
        vrams.push(
            LOBuffer::new(size.div(blocks as u64) as usize).context("Failed to allocate memory")?,
        );
    }
    log::info!(
        "Successfully allocated {} bytes ({} MB)",
        size,
        size / (1024 * 1024), // Log MB for readability
    );

    log::info!("Starting VRAM Block Device (UBLK)");
    start_ublk_server(VMemory::new(vrams))
}

fn start2(
    size: u64,
    blocks: usize,
    config: CLBufferConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // Size is already parsed into bytes
    log::info!(
        "Allocating {} bytes ({} MB) on OCL device {} (Platform {})",
        size,
        size / (1024 * 1024), // Log MB for readability
        config.device_index,
        config.platform_index
    );

    let device = CLDevice::new(&config).context("Failed to allocate OCL Device")?;
    let mut vrams: Vec<CLBuffer> = Vec::new();
    for _ in 0..blocks {
        vrams.push(
            CLBuffer::new(&device, size.div(blocks as u64) as usize, config.mmap)
                .context("Failed to allocate OCL memory")?,
        );
    }

    log::info!(
        "Successfully allocated {} bytes ({} MB) on {}",
        size,
        size / (1024 * 1024), // Log MB for readability
        device.name()
    );

    log::info!("Starting VRAM Block Device (UBLK)");
    start_ublk_server(VMemory::new(vrams))
}
