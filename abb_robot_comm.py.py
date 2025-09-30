import socket


class RobotComm:
    def __init__(self, ip, port=5000):
        self.ip = ip
        self.port = port
        self.socket = None
        self.client_socket = None
        self.client_ip = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen()
        print("Looking for client")

        (self.client_socket, self.client_ip) = self.socket.accept()
        print(f"Robot at address {self.client_ip} connected.")
        print("If you would like to end the program, enter 'quit'.")

    def disconnect(self):
        if self.client_socket is not None:
            self.client_socket.close()

    def communicate(self, message=None):
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


def connect_rob(ip):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Set socket listening port. To find the robot address use cmd and type ipconfig in the terminal, there you see
    # the ip of the robot. you may have to open the port in the firewall as well. Note that the ip address is
    # 127.0.0.1 if you run via RobotStudio
    server_socket.bind((ip, 5000))

    # Set up the server
    # listen to incoming client connection
    server_socket.listen()
    print("Looking for client")

    # accept and store incoming socket connection
    (client_socket, client_ip) = server_socket.accept()
    print(f"Robot at address {client_ip} connected.")
    print("If you would like to end the program, enter 'quit'.")
    return client_socket


def disconnect_rob(client_socket):
    client_socket.close()


def communicate_rob(message, client_socket):
    while True:
        if message is None:
            UserInput = input("Please type your message: ")
        else:
            UserInput = message

        if UserInput.lower() == "quit":
            server_message = UserInput
            client_socket.send(server_message.encode("UTF-8"))
            print("Goodbye!")
            # wait for answer and print in terminal
            if client_socket.recv:
                client_message = client_socket.recv(4094)
                client_message = client_message.decode("latin-1")
                print("The received message is:", client_message)
                return client_message

            break

        else:
            client_socket.send(UserInput.encode("UTF-8"))
            print(f"Message sent to client: {message}")
            # Wait for answer and print in terminal
            if client_socket.recv:
                print("The received message is:")
                client_message = client_socket.recv(4094)
                client_message = client_message.decode("latin-1")
                print("!!", client_message)
                return client_message
