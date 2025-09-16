use crate::opencl::VRamBuffer;
use anyhow::Result;
use libublk::{
    ctrl::UblkCtrlBuilder,
    helpers::IoBuf,
    io::{UblkDev, UblkQueue},
};
use serde_json::json;
use std::{rc::Rc, sync::Arc};

fn handle_io_cmd(q: &UblkQueue<'_>, tag: u16, buf: &IoBuf<u8>, vrams: &Vec<VRamBuffer>) -> i32 {
    let iod = q.get_iod(tag);
    let global_limit = q.dev.tgt.dev_size;
    // compute global position/size
    let mut global_offset = global_limit.min(iod.start_sector << 9);
    let mut global_length = (iod.nr_sectors << 9) as usize;
    if global_offset + global_length as u64 >= global_limit {
        global_length = (global_limit - global_offset) as usize;
    }
    if global_length > 0 {
        let op = iod.op_flags & 0xff;
        match op {
            libublk::sys::UBLK_IO_OP_READ | libublk::sys::UBLK_IO_OP_WRITE => {
                let operate = match op {
                    libublk::sys::UBLK_IO_OP_READ => "Read",
                    _ => "Write",
                };
                let mut local_offset = 0;
                let mut global_remaining = global_length;
                for (i, vram) in vrams.iter().enumerate() {
                    let local_remaining = vram.remaining(global_offset);
                    if local_remaining.is_none() {
                        continue;
                    }
                    // compute local length to read/write
                    let local_length = global_remaining.min(local_remaining.unwrap());

                    if libublk::sys::UBLK_IO_OP_READ == op {
                        let array = unsafe {
                            std::slice::from_raw_parts_mut(
                                buf.as_mut_ptr().add(local_offset),
                                local_length,
                            )
                        };
                        if let Err(e) = vram.read(global_offset, array) {
                            log::error!(
                                "{} error, device vram-{} offset {} size {}, code {}",
                                operate,
                                i,
                                global_offset,
                                local_length,
                                e
                            );
                            return -libc::EIO;
                        }
                    } else {
                        let array = unsafe {
                            std::slice::from_raw_parts(buf.as_ptr().add(local_offset), local_length)
                        };
                        if let Err(e) = vram.write(global_offset, array) {
                            log::error!(
                                "{} error, device vram-{} offset {} size {}, code {}",
                                operate,
                                i,
                                global_offset,
                                local_length,
                                e
                            );
                            return -libc::EIO;
                        }
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
                        "{} error, offset {} size {}",
                        operate,
                        global_offset,
                        global_remaining
                    );
                    return -libc::EIO;
                }
            }
            libublk::sys::UBLK_IO_OP_FLUSH => {}
            _ => {
                return -libc::EINVAL;
            }
        }
    }
    global_length as i32
}

// implement whole ublk IO level protocol
async fn io_task(q: &UblkQueue<'_>, tag: u16, vrams: Arc<Vec<VRamBuffer>>) {
    // IO buffer for exchange data with /dev/ublkbN
    let buf_bytes = q.dev.dev_info.max_io_buf_bytes as usize;
    let buf = libublk::helpers::IoBuf::<u8>::new(buf_bytes);
    let mut cmd_op = libublk::sys::UBLK_U_IO_FETCH_REQ;
    let mut res = 0;

    // Register IO buffer, so that buffer pages can be discarded
    // when queue becomes idle
    q.register_io_buf(tag, &buf);
    loop {
        // Complete previous command with result and re-submit
        // IO command for fetching new IO request from /dev/ublkbN
        res = q.submit_io_cmd(tag, cmd_op, buf.as_mut_ptr(), res).await;
        if res == libublk::sys::UBLK_IO_RES_ABORT {
            break;
        }

        // Handle this incoming IO command
        res = handle_io_cmd(q, tag, &buf, &vrams);
        cmd_op = libublk::sys::UBLK_U_IO_COMMIT_AND_FETCH_REQ;
    }
}

fn q_fn(qid: u16, dev: &UblkDev, vrams: Arc<Vec<VRamBuffer>>) {
    let q_rc = Rc::new(UblkQueue::new(qid, dev).unwrap());
    let exe = smol::LocalExecutor::new();
    let mut f_vec = Vec::new();

    for tag in 0..dev.dev_info.queue_depth {
        let q = q_rc.clone();
        let use_vram = vrams.clone();
        f_vec.push(exe.spawn(async move { io_task(&q, tag, use_vram).await }));
    }

    // Drive smol executor, won't exit until queue is dead
    libublk::uring_async::ublk_wait_and_handle_ios(&exe, &q_rc);
    smol::block_on(async { futures::future::join_all(f_vec).await });
}
pub fn start_ublk_server(mut vrams: Vec<VRamBuffer>) -> Result<(), Box<dyn std::error::Error>> {
    // Create ublk device
    let workers = num_cpus::get().max(2) as u16;
    let ctrl = Arc::new(
        UblkCtrlBuilder::default()
            .name("ublk-vram")
            .depth(workers * 64_u16)
            .io_buf_bytes(1024 * 1024)
            .nr_queues(workers)
            .dev_flags(libublk::UblkFlags::UBLK_DEV_F_ADD_DEV)
            .build()?,
    );
    // Kill ublk device by handling "Ctrl + C"
    let ctrl_sig = ctrl.clone();
    let _ = ctrlc::set_handler(move || {
        ctrl_sig.kill_dev().unwrap();
    });

    // compute vram sets
    let mut dev_size: u64 = 0;
    for v in vrams.iter_mut() {
        v.offset(dev_size);
        dev_size += v.size() as u64;
    }
    let dev_blocks = vrams.len();
    let use_vram = Arc::new(vrams);
    // Now start this ublk target
    ctrl.run_target(
        // target initialization
        |dev| {
            dev.set_default_params(dev_size);
            dev.set_target_json(json!({
                "blocks": dev_blocks,
            }));
            Ok(())
        },
        // queue IO logic
        |tag, dev| q_fn(tag, dev, use_vram),
        // dump device after it is started
        |dev| dev.dump(),
    )?;

    // Usually device is deleted automatically when `ctrl` drops, but
    // here `ctrl` is leaked by the global sig handler closure actually,
    // so we have to delete it explicitly
    ctrl.del_dev()?;
    Ok(())
}
