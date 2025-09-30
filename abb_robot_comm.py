import socket


class RobotComm:
    def __init__(self, ip, port=5000):
        self.ip = ip
        self.port = port
        self.socket = None
        self.client_socket = None
        self.client_ip = None

    def connect(self):
        try:
            # Set up the server
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.ip, self.port))
            self.socket.listen()
            print("Looking for client")

            # accept and store incoming socket connection
            (self.client_socket, self.client_ip) = self.socket.accept()
            print(f"Robot at address {self.client_ip} connected.")
            print("If you would like to end the program, enter 'quit'.")

        # Handle exceptions
        except socket.error as e:
            print(f"Socket error on connect: {e}")

    def disconnect(self):
        try:
            if self.client_socket is not None:
                self.client_socket.close()

        # Handle exceptions
        except socket.error as e:
            print(f"Socket error on disconnect: {e}")

    def communicate(self, message=None):
        try:
            while True:
                if message is None:
                    user_input = input("Please type your message: ")
                else:
                    user_input = message

                if user_input.lower() == "quit":
                    self.client_socket.send(user_input.encode("UTF-8"))
                    print("Goodbye!")
                    # wait for answer and print in terminal
                    if self.client_socket:
                        client_message = self.client_socket.recv(4094)
                        client_message = client_message.decode("latin-1")
                        print("The received message is:", client_message)
                        return client_message

                    break
                else:
                    self.client_socket.send(user_input.encode("UTF-8"))
                    print(f"Message sent to client: {user_input}")
                    # Wait for answer and print in terminal
                    if self.client_socket:
                        print("The received message is:")
                        client_message = self.client_socket.recv(4094)
                        client_message = client_message.decode("latin-1")
                        print("!!", client_message)
                        return client_message

        except (socket.error, OSError) as e:
            print(f"Communication error: {e}")
            return None