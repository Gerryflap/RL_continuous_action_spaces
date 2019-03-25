"""
    The competition server manages the matchmaking and game starting parts of the system
"""
import socket
from queue import Queue
from threading import Lock, Thread
import numpy as np

from competition_system import util


class CompetitionServer(object):
    def __init__(self, addr="127.0.0.1", port=1337):
        self.addr = addr
        self.port = port
        self.max_pid = 0
        self.unregistered_clients = set()
        self.clients = dict()
        self.queue = set()

    def _client_accept_thread(self, sock, sock_queue):
        while True:
            client, info = sock.accept()
            print("Client %s connected" % str(info))
            sock_queue.put(client)

    def run(self):
        sock = socket.socket()
        sock.bind((self.addr, self.port))
        sock.listen(5)
        socket_queue = Queue()
        Thread(target=lambda: self._client_accept_thread(sock, socket_queue)).start()

        while True:
            while not socket_queue.empty():
                client_sock = socket_queue.get(False)
                self.unregistered_clients.add(Client(self, client_sock))
            for client in self.unregistered_clients:
                client.process_commands()

            for client in self.clients.values():
                client.process_commands()

            # Get matches, start games, deregister clients from queue

    def unregister_client(self, pid):
        self.clients.pop(pid)

    def register_client(self, client, pid=None):
        if pid is None:
            # Assign new player identifier
            pid = self.max_pid
            self.max_pid += 1

        self.clients[pid] = client
        self.unregistered_clients.remove(client)

        client.accept_registration(pid)

    def add_to_queue(self, pid):
        self.queue.add(pid)


class Client(object):
    def __init__(self, server: CompetitionServer, socket):
        self.server = server
        self.socket = socket
        self.pid = None
        self.byte_queue = Queue()
        self.act_queue = Queue()
        self.thread = util.start_recv_thread(socket, self.byte_queue)
        self.info_str = ""

    def process_commands(self):
        while not self.byte_queue.empty():
            command = self.byte_queue.get(False)
            command = command.decode()
            command = command.split(" ")
            if len(command) == 0:
                continue

            cstr = command[0]

            # Handle commands
            if cstr == "EXIT":
                self.socket.close()
                self.server.unregister_client(self.pid)

            elif cstr == "CONNECT":
                if len(command) > 1:
                    self.info_str = " ".join(command[1:])

            elif cstr == "REGISTER":
                pid = None
                if len(command) > 1:
                    pid = int(command[1])
                self.server.register_client(self, pid)
            elif cstr == "ACT":
                act_str = command[1]
                act = act_str.encode('base64')
                self.act_queue.put(act)
            elif cstr == "JOIN_QUEUE":
                self.server.add_to_queue(self.pid)

    def accept_registration(self, pid):
        self.pid = pid
        self.send("REG_ACCEPT %d" % pid)

    def send_srd(self, state: np.ndarray, reward, done):
        state = state.tobytes()
        state = state.decode('base64')
        self.send("SRD %s %f%s" % (state, reward, " DONE" if done else ""))

    def send(self, command: str):
        command = command.encode()
        self.socket.sendall(command)

