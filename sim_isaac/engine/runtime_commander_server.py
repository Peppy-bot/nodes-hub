#!/usr/bin/env python3

import json
import logging
import queue
import socket
import threading

logger = logging.getLogger(__name__)


class RuntimeCommanderServer:
    def __init__(self, host="0.0.0.0", port=5556):
        self.host = host
        self.port = port
        self.commands = queue.Queue()
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self._server_loop,
            daemon=True,
        )

    def start(self):
        logger.info(
            "Starting runtime commander on %s:%d",
            self.host,
            self.port,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _server_loop(self):
        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        sock.bind((self.host, self.port))
        sock.listen(5)
        sock.settimeout(0.5)

        logger.info(
            "Runtime commander listening on port %d",
            self.port,
        )

        while not self.stop_event.is_set():
            try:
                conn, addr = sock.accept()

            except socket.timeout:
                continue

            except Exception:
                logger.exception(
                    "Commander socket error"
                )
                continue

            with conn:
                try:
                    data = b""

                    while True:
                        chunk = conn.recv(65536)

                        if not chunk:
                            break

                        data += chunk

                        if b"\n" in data:
                            break

                    command = json.loads(
                        data.decode("utf-8").strip()
                    )

                    self.commands.put(command)

                    conn.sendall(
                        b'{"status":"queued"}\n'
                    )

                except Exception as exc:
                    logger.exception(
                        "Invalid runtime command"
                    )

                    reply = {
                        "status": "error",
                        "error": str(exc),
                    }

                    conn.sendall(
                        (
                            json.dumps(reply) + "\n"
                        ).encode("utf-8")
                    )

        sock.close()

    def process_pending(self, sim_launcher):
        while True:
            try:
                command = self.commands.get_nowait()

            except queue.Empty:
                break

            try:
                sim_launcher.execute_runtime_command(
                    command
                )

            except Exception:
                logger.exception(
                    "Runtime command failed: %s",
                    command,
                )
