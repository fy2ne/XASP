// Copyright (c) 2026 fy2ne
// SPDX-License-Identifier: Apache-2.0

use crate::stripe_mux::NUM_STREAMS;
use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;

#[cfg(target_os = "linux")]
mod linux_sock {
    use std::ffi::c_void;
    use std::net::TcpStream;
    use std::os::fd::AsRawFd;

    const SOL_SOCKET: i32 = 1;
    const IPPROTO_TCP: i32 = 6;
    const TCP_NODELAY: i32 = 1;
    const TCP_QUICKACK: i32 = 12;
    const SO_SNDBUF: i32 = 7;
    const SO_RCVBUF: i32 = 8;
    const SO_PRIORITY: i32 = 12;
    const SO_RCVLOWAT: i32 = 18;

    extern "C" {
        fn setsockopt(
            socket: i32,
            level: i32,
            option_name: i32,
            option_value: *const c_void,
            option_len: u32,
        ) -> i32;
    }

    pub fn tune(stream: &TcpStream, is_stratum0_vip: bool) {
        let fd = stream.as_raw_fd();
        unsafe {
            let one: i32 = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one as *const _ as *const c_void, 4);
            setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one as *const _ as *const c_void, 4);

            let lowat: i32 = 64;
            setsockopt(fd, SOL_SOCKET, SO_RCVLOWAT, &lowat as *const _ as *const c_void, 4);

            let buf_sz: i32 = 16 * 1024 * 1024;
            setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf_sz as *const _ as *const c_void, 4);
            setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf_sz as *const _ as *const c_void, 4);

            if is_stratum0_vip {
                let prio: i32 = 6;
                setsockopt(fd, SOL_SOCKET, SO_PRIORITY, &prio as *const _ as *const c_void, 4);
            }
        }
    }

    pub fn rearm_quickack(stream: &TcpStream) {
        let fd = stream.as_raw_fd();
        unsafe {
            let one: i32 = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one as *const _ as *const c_void, 4);
        }
    }
}

pub fn configure_xasp_tcp_stream(stream: &TcpStream, is_stratum0_vip: bool) -> io::Result<()> {
    stream.set_nodelay(true)?;
    #[cfg(target_os = "linux")]
    linux_sock::tune(stream, is_stratum0_vip);
    #[cfg(not(target_os = "linux"))]
    let _ = is_stratum0_vip;
    Ok(())
}

#[inline]
pub fn rearm_quickack(stream: &TcpStream) {
    #[cfg(target_os = "linux")]
    linux_sock::rearm_quickack(stream);
    #[cfg(not(target_os = "linux"))]
    let _ = stream;
}

pub struct StripedTcpTxPool {
    pub streams: Vec<TcpStream>,
}

pub struct StripedTcpRxPool {
    pub streams: Vec<TcpStream>,
}

pub fn bind_loopback_pair(base_port: u16) -> io::Result<(StripedTcpTxPool, StripedTcpRxPool)> {
    let mut listeners = Vec::with_capacity(NUM_STREAMS);
    let mut addrs: Vec<SocketAddr> = Vec::with_capacity(NUM_STREAMS);

    for idx in 0..NUM_STREAMS {
        let port = if base_port == 0 {
            0
        } else {
            base_port + (idx as u16)
        };
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        addrs.push(listener.local_addr()?);
        listeners.push(listener);
    }

    let handle = thread::spawn(move || -> io::Result<Vec<TcpStream>> {
        let mut tx = Vec::with_capacity(NUM_STREAMS);
        for (idx, addr) in addrs.iter().enumerate() {
            let conn = TcpStream::connect(addr)?;
            configure_xasp_tcp_stream(&conn, idx < 2)?;
            tx.push(conn);
        }
        Ok(tx)
    });

    let mut rx = Vec::with_capacity(NUM_STREAMS);
    for (idx, listener) in listeners.into_iter().enumerate() {
        let (conn, _) = listener.accept()?;
        configure_xasp_tcp_stream(&conn, idx < 2)?;
        conn.set_nonblocking(true)?;
        rx.push(conn);
    }

    let tx = handle
        .join()
        .map_err(|_| io::Error::new(io::ErrorKind::Other, "connect thread panicked"))??;

    Ok((
        StripedTcpTxPool { streams: tx },
        StripedTcpRxPool { streams: rx },
    ))
}
