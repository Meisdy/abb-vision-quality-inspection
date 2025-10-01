# abb_robot_comm.py

import socket
import logging

# Configure logging to output to terminal with timestamp and log level
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class RobotComm:
    """Handles TCP communication with a robot client."""

    def __init__(self, ip, port=5000):
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

    def connect(self):
        """
        Set up server socket and wait for client connection.
        Logs connection status and errors.
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.ip, self.port))
            self.socket.listen()
            logging.info("Looking for TCP client")

            # Accept incoming client connection
            (self.client_socket, self.client_ip) = self.socket.accept()
            logging.info(f"Robot at address {self.client_ip} connected.")
            logging.info("If you would like to end the program, enter 'quit'.")

        except socket.error as e:
            logging.error(f"Socket error on connect: {e}")

    def disconnect(self):
        """
        Close client and server sockets.
        Logs disconnection status and errors.
        """
        try:
            if self.client_socket is not None:
                self.client_socket.close()
                self.client_socket = None
                self.socket = None
                logging.info("Disconnected from client.")

        except socket.error as e:
            logging.error(f"Socket error on disconnect: {e}")

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
            logging.info(f"Message sent to client: {message}")
            return True

        except (socket.error, OSError) as e:
            logging.error(f"Error sending message: {e}")
            return False

    def receive_message(self) -> str | None:
        """
        Receive a message from the client. Will wait until a message comes in.

        Returns:
            str or None: Received message decoded, or None if error/none.
        """
        try:
            if self.client_socket:
                data = self.client_socket.recv(4096)
                message = data.decode("latin-1")
                logging.info(f"Received message from client: {message}")
                return message
            else:
                logging.error("Receive failed: No client socket connected.")
                return None

        except (socket.error, OSError) as e:
            logging.error(f"Error receiving message: {e}")
            return None

    def communicate(self, message=None):
        """
        Combined send and receive.

        Args:
            message (str, optional): Message to send. If None, prompts user.

        Returns:
            str or None: Received message or None on communication error.
        """
        try:
            while True:
                if message is None:
                    user_message = input("Please type your message: ")
                else:
                    user_message = message

                if user_message.lower() == "quit":
                    if self.send_message(user_message):
                        logging.info("Goodbye!")
                        return self.receive_message()
                    break
                else:
                    if self.send_message(user_message):
                        return self.receive_message()
        except Exception as e:
            logging.error(f"Communication failure: {e}")
            return None



