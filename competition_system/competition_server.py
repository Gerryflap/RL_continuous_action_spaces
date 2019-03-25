"""
    The competition server manages the matchmaking and game starting parts of the system
"""
import base64
import random
import socket
import time
from queue import Queue
from threading import Lock, Thread
import numpy as np
import multiprocessing as mp
import competition_system.matchmaking_systems as ms

from competition_system import util

LOG_TIME = 30

class CompetitionServer(object):
    def __init__(self, play_match_function, matchmaking: ms.MatchmakingSystem, addr="127.0.0.1", port=1337):
        self.addr = addr
        self.port = port
        self.max_pid = 0
        self.unregistered_clients = set()
        self.clients = dict()
        self.queue = set()
        self.match_result_queue = mp.Queue()
        self.deregister_queue = mp.Queue()
        self.join_queue_queue = mp.Queue()
        self.play_match = play_match_function
        self.matchmaking = matchmaking

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

        ticks = 0
        last_log = 0

        while True:
            while not socket_queue.empty():
                client_sock = socket_queue.get(False)
                self.unregistered_clients.add(Client(self, client_sock))

            clients = list(self.clients.values()) + list(self.unregistered_clients)
            for client in clients:
                client.process_commands()

            if len(self.queue) >= 2 and len(self.queue) >= 0.2*len(self.clients):
                matches = self.matchmaking.get_matches(self.queue)

                for match in matches:
                    print("Starting match ", match)
                    self.queue -= set(match)
                    client1, client2 = self.clients[match[0]], self.clients[match[1]]
                    process = mp.Process(target=self.play_match, args=(client1, client2, self.match_result_queue))
                    process.start()
            # Get matches, start games, deregister clients from queue
            while not self.match_result_queue.empty():
                pid1, pid2, outcome = self.match_result_queue.get()
                print("Match finished ", (pid1, pid2, outcome))
                self.matchmaking.report_outcome(pid1, pid2, outcome)
            t2 = time.time()
            ticks += 1
            if t2 - last_log > LOG_TIME:
                print("Server log: ")
                print("Average tick length: %.6f seconds" % ((t2 - last_log)/ticks))
                print("Standings: ")
                for pid in self.clients.keys():
                    print(pid, self.clients[pid].info_str, self.matchmaking.get_rating(pid))

                last_log = time.time()
                ticks = 0

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
        self.act_queue = mp.Queue()
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
                act = act_str.encode()
                act = base64.b64decode(act)
                act = np.frombuffer(act, dtype=np.float32)
                self.act_queue.put(act)
            elif cstr == "JOIN_QUEUE":
                self.server.add_to_queue(self.pid)

    def accept_registration(self, pid):
        self.pid = pid
        self.send("REG_ACCEPT %d" % pid)

    def send_srd(self, state: np.ndarray, reward=None, done=False):
        state = state.astype(np.float32).tobytes()
        state = base64.b64encode(state)
        state = state.decode()
        self.send("SRD %s%s%s" % (state, "" if reward is None else (" %f" % reward), " DONE" if done else ""))

    def send(self, command: str):
        command_bytes = command.encode()
        size = len(command_bytes)
        size = size.to_bytes(4, 'big')
        self.socket.sendall(size + command_bytes)

