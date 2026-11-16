//! The relay in front of Rerun's data server. Rerun's server sends each message to each
//! page and waits for the slowest: a page that stops reading, as a browser tab left in
//! the background does, holds the data back from every page, and from a page that
//! opens later, for as long as it stays connected. So each page's connection passes
//! through this relay, byte for byte, and the relay closes a connection whose page takes
//! no byte for `stall_timeout` while bytes wait for it. Rerun's server then drops that
//! page, and the others go on. The page that was dropped shows what it last read, until
//! it is loaded again.

use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

/// How many bytes the relay reads at a time.
const BUFFER_LEN: usize = 64 * 1024;

/// How a copy to a page ended.
#[derive(Debug, PartialEq)]
pub enum CopyEnd {
    /// The source closed: every byte it gave was written.
    Finished,
    /// The page took no byte for the stall timeout while bytes waited for it.
    Stalled,
}

/// Copies the bytes of `from` to `to` until `from` closes. Ends with
/// [`CopyEnd::Stalled`] when a write to `to` takes no byte for `stall_timeout`; a wait
/// for the next bytes of `from` is not a stall, however long.
pub async fn copy_until_stalled(
    mut from: impl AsyncRead + Unpin,
    mut to: impl AsyncWrite + Unpin,
    stall_timeout: Duration,
) -> io::Result<CopyEnd> {
    let mut buffer = vec![0; BUFFER_LEN];
    loop {
        let read = from.read(&mut buffer).await?;
        if read == 0 {
            return Ok(CopyEnd::Finished);
        }
        let mut written = 0;
        while written < read {
            let write = to.write(&buffer[written..read]);
            match tokio::time::timeout(stall_timeout, write).await {
                Ok(Ok(0)) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(Ok(taken)) => written += taken,
                Ok(Err(e)) => return Err(e),
                Err(_) => return Ok(CopyEnd::Stalled),
            }
        }
    }
}

/// Serves the pages that connect to `listener` from the data server at `server`, until
/// the listener fails: each connection is copied both ways, and closed when its page
/// stalls for `stall_timeout` (`copy_until_stalled`).
pub async fn relay_pages(listener: TcpListener, server: SocketAddr, stall_timeout: Duration) {
    loop {
        let (page, address) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(e) => {
                warn!("the viewer's data port takes no more pages: {e}");
                return;
            }
        };
        tokio::spawn(async move {
            match relay_page(page, server, stall_timeout).await {
                Ok(CopyEnd::Stalled) => info!(
                    "the page at {address} took no data for {stall_timeout:?}: its connection is \
                     closed so that the other pages go on; load the page again to follow the stack"
                ),
                Ok(CopyEnd::Finished) => {}
                // A page that closes its tab ends its connection with an error or without.
                Err(_) => {}
            }
        });
    }
}

/// Copies one page's connection to the data server and back, until either side closes
/// or the page stalls. Both connections close when it returns.
async fn relay_page(
    page: TcpStream,
    server: SocketAddr,
    stall_timeout: Duration,
) -> io::Result<CopyEnd> {
    let server = TcpStream::connect(server).await?;
    // The relay adds no wait of its own to a small message.
    page.set_nodelay(true)?;
    server.set_nodelay(true)?;
    let (page_reads, page_writes) = page.into_split();
    let (server_reads, mut server_writes) = server.into_split();
    let mut page_reads = page_reads;
    tokio::select! {
        to_page = copy_until_stalled(server_reads, page_writes, stall_timeout) => to_page,
        to_server = tokio::io::copy(&mut page_reads, &mut server_writes) => {
            to_server.map(|_| CopyEnd::Finished)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STALL: Duration = Duration::from_secs(15);

    /// The clock of these tests is tokio's paused clock: a wait ends as soon as every
    /// task waits, so no test takes the time it names.
    #[tokio::test(start_paused = true)]
    async fn every_byte_reaches_a_page_that_reads() {
        let (mut source, relay_reads) = tokio::io::duplex(8);
        let (relay_writes, mut page) = tokio::io::duplex(8);
        let copy = tokio::spawn(copy_until_stalled(relay_reads, relay_writes, STALL));
        // 40 bytes through pipes of 8: the page reads as the source writes.
        let sent: Vec<u8> = (0..40).collect();
        let write = tokio::spawn(async move {
            source.write_all(&sent).await.unwrap();
            // Dropping the source closes it.
        });
        let mut received = vec![0; 40];
        page.read_exact(&mut received).await.unwrap();
        write.await.unwrap();
        assert_eq!(received, (0..40).collect::<Vec<u8>>());
        assert_eq!(copy.await.unwrap().unwrap(), CopyEnd::Finished);
    }

    #[tokio::test]
    async fn a_page_reads_the_server_and_the_server_reads_the_page_through_the_relay() {
        let server = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_address = server.local_addr().unwrap();
        let relay = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let relay_address = relay.local_addr().unwrap();
        tokio::spawn(relay_pages(relay, server_address, STALL));
        let mut page = TcpStream::connect(relay_address).await.unwrap();
        let (mut at_server, _) = server.accept().await.unwrap();
        page.write_all(b"read_messages").await.unwrap();
        let mut asked = [0; 13];
        at_server.read_exact(&mut asked).await.unwrap();
        at_server.write_all(b"the stack now").await.unwrap();
        let mut answered = [0; 13];
        page.read_exact(&mut answered).await.unwrap();
        assert_eq!((&asked, &answered), (b"read_messages", b"the stack now"));
        // The page closes its tab: the relay closes the server's side too.
        drop(page);
        let mut rest = Vec::new();
        at_server.read_to_end(&mut rest).await.unwrap();
        assert!(rest.is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn a_page_that_takes_no_byte_for_the_stall_timeout_ends_the_copy() {
        let (mut source, relay_reads) = tokio::io::duplex(64);
        // A page with room for 8 bytes, which it never reads.
        let (relay_writes, _page) = tokio::io::duplex(8);
        let copy = tokio::spawn(copy_until_stalled(relay_reads, relay_writes, STALL));
        source.write_all(&[7; 32]).await.unwrap();
        let started = tokio::time::Instant::now();
        assert_eq!(copy.await.unwrap().unwrap(), CopyEnd::Stalled);
        assert!(started.elapsed() >= STALL, "not before the stall timeout");
    }

    #[tokio::test(start_paused = true)]
    async fn a_source_that_is_silent_for_long_is_no_stall() {
        let (mut source, relay_reads) = tokio::io::duplex(64);
        let (relay_writes, mut page) = tokio::io::duplex(64);
        let copy = tokio::spawn(copy_until_stalled(relay_reads, relay_writes, STALL));
        source.write_all(&[1, 2, 3]).await.unwrap();
        let mut first = [0; 3];
        page.read_exact(&mut first).await.unwrap();
        // Ten stall timeouts with nothing to send, then more bytes.
        tokio::time::sleep(10 * STALL).await;
        source.write_all(&[4, 5]).await.unwrap();
        let mut second = [0; 2];
        page.read_exact(&mut second).await.unwrap();
        drop(source);
        assert_eq!((first, second), ([1, 2, 3], [4, 5]));
        assert_eq!(copy.await.unwrap().unwrap(), CopyEnd::Finished);
    }

    #[tokio::test(start_paused = true)]
    async fn a_page_that_reads_again_within_the_stall_timeout_is_kept() {
        let (mut source, relay_reads) = tokio::io::duplex(64);
        let (relay_writes, mut page) = tokio::io::duplex(8);
        let copy = tokio::spawn(copy_until_stalled(relay_reads, relay_writes, STALL));
        source.write_all(&[9; 24]).await.unwrap();
        drop(source);
        // The page reads nothing for most of the stall timeout, then reads everything.
        tokio::time::sleep(STALL - Duration::from_secs(1)).await;
        let mut received = Vec::new();
        page.read_to_end(&mut received).await.unwrap();
        assert_eq!(received, [9; 24]);
        assert_eq!(copy.await.unwrap().unwrap(), CopyEnd::Finished);
    }
}
