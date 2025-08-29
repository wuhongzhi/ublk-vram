use crate::opencl::VRamBuffer;
use anyhow::Result;
use libublk::{ctrl::UblkCtrlBuilder, io::UblkDev, io::UblkQueue};
use std::{sync::Arc, rc::Rc};

fn handle_io_cmd(q: &UblkQueue<'_>, tag: u16, vram: &Arc<VRamBuffer>, buf: *mut u8) -> i32 {
    let iod = q.get_iod(tag);
    let limit = vram.size() as usize;
    let offset = limit.min((iod.start_sector << 9) as usize);
    let mut length = (iod.nr_sectors << 9) as usize;
    if offset + length >= limit {
        length = limit - offset;
    }
    if length > 0 {
        let op = iod.op_flags & 0xff;
        match op {
            libublk::sys::UBLK_IO_OP_READ => unsafe {
                let mut array = std::slice::from_raw_parts_mut(buf, length);
                let _ = vram.read(offset, &mut array);
            },
            libublk::sys::UBLK_IO_OP_WRITE => unsafe {
                let array = std::slice::from_raw_parts(buf, length);
                let _ = vram.write(offset, &array);
            },
            libublk::sys::UBLK_IO_OP_FLUSH => {}
            _ => {
                return -libc::EINVAL;
            }
        }
    }
    length as i32
}

// implement whole ublk IO level protocol
async fn io_task(q: &UblkQueue<'_>, tag: u16, vram: Arc<VRamBuffer>) {
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
        res = handle_io_cmd(&q, tag, &vram, buf.as_mut_ptr());
        cmd_op = libublk::sys::UBLK_U_IO_COMMIT_AND_FETCH_REQ;
    }
}

fn q_fn(qid: u16, dev: &UblkDev, vram: Arc<VRamBuffer>) {
    let q_rc = Rc::new(UblkQueue::new(qid as u16, &dev).unwrap());
    let exe = smol::LocalExecutor::new();
    let mut f_vec = Vec::new();

    for tag in 0..dev.dev_info.queue_depth {
        let q = q_rc.clone();
        let use_vram = vram.clone();
        f_vec.push(exe.spawn(async move { io_task(&q, tag, use_vram).await }));
    }

    // Drive smol executor, won't exit until queue is dead
    libublk::uring_async::ublk_wait_and_handle_ios(&exe, &q_rc);
    smol::block_on(async { futures::future::join_all(f_vec).await });
}
pub fn start_ublk_server(vram: VRamBuffer) -> Result<(), Box<dyn std::error::Error>> {
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

    // Now start this ublk target
    let dev_size = vram.size() as u64;
    let use_vram = Arc::new(vram);
    ctrl.run_target(
        // target initialization
        |dev| {
            dev.set_default_params(dev_size);
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
