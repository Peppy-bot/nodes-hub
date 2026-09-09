#!/usr/bin/env python3
"""Print an available localhost TCP port for the isolated CI daemon."""

import socket


with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
