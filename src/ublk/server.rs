use crate::{VBuffer, VMemory};
use anyhow::Result;
use libublk::{
    ctrl::UblkCtrlBuilder,
    helpers::IoBuf,
    io::{UblkDev, UblkQueue},
    sys,
};
use serde_json::json;
use std::{rc::Rc, sync::Arc};

fn handle_io_cmd<T: VBuffer>(
    q: &UblkQueue<'_>,
    tag: u16,
    buf: &IoBuf<u8>,
    vrams: &Arc<VMemory<T>>,
) -> i32 {
    let iod = q.get_iod(tag);
    let limit = q.dev.tgt.dev_size;
    // compute global position/size
    let offset = limit.min(iod.start_sector << 9);
    let mut length = (iod.nr_sectors << 9) as usize;
    if offset + length as u64 >= limit {
        length = (limit - offset) as usize;
    }
    if length == 0 {
        return length as i32;
    }
    match iod.op_flags & 0xff {
        sys::UBLK_IO_OP_READ => unsafe { vrams.read(offset, length, buf.as_mut_ptr()) },
        sys::UBLK_IO_OP_WRITE => unsafe { vrams.write(offset, length, buf.as_ptr()) },
        sys::UBLK_IO_OP_FLUSH | sys::UBLK_IO_OP_DISCARD => length as i32,
        _ => -libc::EINVAL,
    }
}

// implement whole ublk IO level protocol
async fn io_task<T: VBuffer>(q: &UblkQueue<'_>, tag: u16, vrams: Arc<VMemory<T>>) {
    // IO buffer for exchange data with /dev/ublkbN
    let buf_bytes = q.dev.dev_info.max_io_buf_bytes as usize;
    let buf = libublk::helpers::IoBuf::<u8>::new(buf_bytes);
    let mut cmd_op = sys::UBLK_U_IO_FETCH_REQ;
    let mut res = 0;

    // Register IO buffer, so that buffer pages can be discarded
    // when queue becomes idle
    q.register_io_buf(tag, &buf);
    loop {
        // Complete previous command with result and re-submit
        // IO command for fetching new IO request from /dev/ublkbN
        res = q.submit_io_cmd(tag, cmd_op, buf.as_mut_ptr(), res).await;
        if res == sys::UBLK_IO_RES_ABORT {
            break;
        }

        // Handle this incoming IO command
        res = handle_io_cmd(q, tag, &buf, &vrams);
        cmd_op = sys::UBLK_U_IO_COMMIT_AND_FETCH_REQ;
    }
}

fn q_fn<T: VBuffer>(qid: u16, dev: &UblkDev, vrams: Arc<VMemory<T>>) {
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
pub fn start_ublk_server<T>(vrams: VMemory<T>) -> Result<(), Box<dyn std::error::Error>>
where
    T: VBuffer + 'static,
{
    // Create ublk device
    let workers = num_cpus::get().max(2) as u16;
    let ctrl = Arc::new(
        UblkCtrlBuilder::default()
            .name("ublk-vram")
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
    let dev_size: u64 = vrams.size();
    let dev_blocks = vrams.blocks();
    let use_vram = Arc::new(vrams);
    // Now start this ublk target
    ctrl.run_target(
        // target initialization
        |dev| {
            dev.set_default_params(dev_size);
            dev.set_target_json(json!({
                "blocks": dev_blocks
            }));
            Ok(())
        },
        // queue IO logic
        |tag, dev| q_fn(tag, dev, use_vram),
        // dump device after it is started
        |dev| {
            dev.dump();
            log::info!("Press CTRL+C to exit.");
        },
    )?;

    // Usually device is deleted automatically when `ctrl` drops, but
    // here `ctrl` is leaked by the global sig handler closure actually,
    // so we have to delete it explicitly
    ctrl.del_dev()?;
    Ok(())
}
