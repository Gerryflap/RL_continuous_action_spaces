import socket
from queue import Queue
from threading import Thread


def data_receiver(sock: socket.SocketType, queue: Queue, buff_size=1024):
    buffer = b""

    while True:
        if len(buffer) > 4:
            size = buffer[:4]
            size = int.from_bytes(size, 'big')

            if len(buffer) >= 4 + size:
                queue.put(buffer[4:4+size])
                buffer = buffer[4+size:]
                continue

        received = sock.recv(buff_size)
        if len(received) == 0:
            print("Received 0 bytes, exiting client thread...")
            queue.put("EXIT".encode())
            break
        buffer += received


def start_recv_thread(sock, queue: Queue, buff_size=1024):
    thread = Thread(target=lambda: data_receiver(sock, queue, buff_size))
    thread.start()
    return thread


def send_data(sock, data: bytes):
    size = len(data)
    size = size.to_bytes(4, 'big')

    sock.sendall(size + data)
