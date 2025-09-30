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

    def communicate(self, message=None):
        """
        Send and receive messages with the client.

        Args:
            message (str, optional): Message to send. If None, prompts user input.

        Returns:
            str or None: Received message from client, or None on error.
        """
        try:
            while True:
                # Get message from argument or user input
                if message is None:
                    user_input = input("Please type your message: ")
                else:
                    user_input = message

                # Check for quit command
                if user_input.lower() == "quit":
                    # Send quit command and receive final message
                    self.client_socket.send(user_input.encode("UTF-8"))
                    logging.info("Goodbye!")
                    if self.client_socket:
                        client_message = self.client_socket.recv(4094)
                        client_message = client_message.decode("latin-1")
                        logging.info(f"The received message is: {client_message}")
                        return client_message
                    break

                else:
                    # Send regular message and receive response
                    self.client_socket.send(user_input.encode("UTF-8"))
                    logging.info(f"Message sent to client: {user_input}")
                    if self.client_socket:
                        client_message = self.client_socket.recv(4094)
                        client_message = client_message.decode("latin-1")
                        logging.info(f"The received message is: {client_message}")
                        return client_message

        except (socket.error, OSError) as e:
            logging.error(f"Communication error: {e}")
            return None
