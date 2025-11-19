"""
abb_robot_comm.py

Provides the RobotComm class for TCP/IP socket communication with an ABB robot client.
Supports connecting, sending/receiving messages, and managing connections for industrial automation scenarios.
"""

import socket
import logging

logger = logging.getLogger(__name__)


class RobotComm:
    """Handles TCP communication with a robot client."""

    def __init__(self, ip: str, port: int = 5000):
        """
        Initialize RobotComm with IP and port.

        Args:
            ip (str): IP address to bind.
            port (int): Port to bind (default: 5000).
        """
        self.ip = ip
        self.port = port
        self.socket = None
        self.client_socket = None
        self.client_ip = None

    def connect(self) -> None:
        """
        Set up server socket and wait for client connection.
        Logs connection status and errors.
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.ip, self.port))
            self.socket.listen()
            logger.info("Looking for TCP client")
            (self.client_socket, self.client_ip) = self.socket.accept()
            logger.info(f"Robot at address {self.client_ip} connected.")
        except socket.error as e:
            logger.error(f"Socket error on connect: {e}")

    def disconnect(self) -> None:
        """
        Close client and server sockets.
        Logs disconnection status and errors.
        """
        try:
            if self.client_socket is not None:
                self.client_socket.close()
                self.client_socket = None
            self.socket = None
            logger.info("Disconnected from client.")
        except socket.error as e:
            logger.error(f"Socket error on disconnect: {e}")

    def send_message(self, message: str) -> bool:
        """
        Send a message to the client.

        Args:
            message (str): Message to send.

        Returns:
            bool: True if send succeeded, else False.
        """
        try:
            self.client_socket.send(message.encode("UTF-8"))
            logger.info(f"Message sent to client: {message}")
            return True
        except (socket.error, OSError) as e:
            logger.error(f"Error sending message: {e}")
            return False

    def receive_message(self) -> str | None:
        """
        Receive a message from the client. Will wait until a message comes in.

        Returns:
            str or None: Received message decoded, or None if error/none.
        """
        try:
            if self.client_socket:
                data = None  # Critical buffer reset to avoid bug
                data = self.client_socket.recv(4096)
                if len(data) == 0:
                    logger.info("Client closed connection")
                    self.disconnect()
                    return None
                message = data.decode("latin-1")
                logger.info(f"Received message from client: {message}")
                return message
            else:
                logger.error("Receive failed: No client socket connected.")
                return None
        except (socket.error, OSError) as e:
            logger.error(f"Error receiving message: {e}")
            return None

    def communicate(self, message: str) -> str | None:
        """
        Send a message and receive a response from the client.

        Args:
            message (str): Message to send.

        Returns:
            str or None: Received message or None on communication error.
        """
        try:
            if self.send_message(message):
                return self.receive_message()
        except Exception as e:
            logger.error(f"Communication failure: {e}")
            return None
