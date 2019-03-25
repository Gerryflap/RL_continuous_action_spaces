import socket
from queue import Queue
import numpy as np
import base64

from competition_system import util


def decode_command(command_bytes):
    command = command_bytes.decode()
    return command.split()


class Player(object):
    def __init__(self, info_string, pid=None, port=None):
        self.sock = None
        self.info_string = info_string
        self.byte_queue = Queue()
        self.thread = None
        self.pid = pid
        self.port = port

    def connect(self, addr, port):
        self.sock = socket.create_connection((addr, port))
        self.thread = util.start_recv_thread(self.sock, self.byte_queue)
        self.send_hello()
        self.send_registration()

        while True:
            command = decode_command(self.byte_queue.get())
            if command[0] == "REG_ACCEPT":
                self.pid = int(command[1])
                break

    def send_hello(self):
        self.send("CONNECT %s" % self.info_string)

    def send_registration(self):
        self.send("REGISTER%s" % ("" if self.pid is None else " %d" % self.pid))

    def join_queue(self):
        self.send("JOIN_QUEUE")

    def reset(self):
        while self.sock is None:
            self.connect("127.0.0.1", 1337 if self.port is None else self.port)

        self.join_queue()
        while True:
            command = decode_command(self.byte_queue.get())
            if command[0] == "SRD":
                state = command[1]
                state = state.encode()
                state = base64.b64decode(state)
                state = np.frombuffer(state, dtype=np.float32)
                return state

    def step(self, actions):
        actions = base64.b64encode(actions.astype(np.float32).tobytes()).decode()
        self.send("ACT %s" % actions)
        while True:
            command = decode_command(self.byte_queue.get())
            if command[0] == "SRD":
                state = command[1]
                state = state.encode()
                state = base64.b64decode(state)
                state = np.frombuffer(state, dtype=np.float32)

                r = float(command[2])
                done = len(command) == 4

                return state, r, done, None

    def send(self, command_str):
        command_bytes = command_str.encode()
        size = len(command_bytes)
        size = size.to_bytes(4, 'big')
        self.sock.sendall(size + command_bytes)


if __name__ == "__main__":
    player = Player("test_client")
    player.connect("localhost", 23415)
    print(player.pid)