use crate::{VBuffer, VMemory};
use anyhow::Result;
use libublk::{
    BufDesc,
    ctrl::{UblkCtrl, UblkCtrlBuilder},
    helpers::IoBuf,
    io::{UblkDev, UblkQueue},
    sys,
};
use serde_json::json;
use std::sync::Arc;

// async/.await IO handling
async fn handle_io_cmd<T: VBuffer>(
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
async fn io_task<T: VBuffer>(
    q: &UblkQueue<'_>,
    tag: u16,
    vrams: Arc<VMemory<T>>,
) -> Result<(), libublk::UblkError> {
    // IO buffer for exchange data with /dev/ublkbN
    let buf_bytes = q.dev.dev_info.max_io_buf_bytes as usize;
    let buf = libublk::helpers::IoBuf::<u8>::new(buf_bytes);

    // Submit initial prep command for setup IO forward
    q.submit_io_prep_cmd(tag, BufDesc::Slice(buf.as_slice()), 0, Some(&buf))
        .await?;

    loop {
        // Handle this incoming IO command, whole IO logic
        let res = handle_io_cmd(&q, tag, &buf, &vrams).await;

        // Commit result and fetch next IO request
        q.submit_io_commit_cmd(tag, BufDesc::Slice(buf.as_slice()), res)
            .await?;
    }
}

fn q_fn<T: VBuffer>(qid: u16, dev: &UblkDev, vrams: Arc<VMemory<T>>) {
    let q_rc = std::rc::Rc::new(UblkQueue::new(qid as u16, &dev).unwrap());
    let exe_rc = std::rc::Rc::new(smol::LocalExecutor::new());
    let exe = exe_rc.clone();
    let mut f_vec = Vec::new();

    for tag in 0..dev.dev_info.queue_depth {
        let q = q_rc.clone();
        let use_vram = vrams.clone();
        f_vec.push(exe.spawn(async move { io_task(&q, tag, use_vram).await }));
    }

    // Drive smol executor, won't exit until queue is dead
    smol::block_on(exe_rc.run(async move {
        let run_ops = || while exe.try_tick() {};
        let done = || f_vec.iter().all(|task| task.is_finished());

        if let Err(e) = libublk::wait_and_handle_io_events(&q_rc, Some(20), run_ops, done).await {
            log::error!("handle_uring_events failed: {}", e);
        }
    }));
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
        let id = ctrl_sig.dev_info().dev_id;
        if let Ok(ctrl) = UblkCtrl::new_simple(id as i32) {
            let _ = ctrl.kill_dev();
        }
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
