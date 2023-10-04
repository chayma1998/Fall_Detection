"""FL server"""
import flwr as fl
import numpy as np
import os
import datetime as dt
import socket
from chayma import CustomFLStrategy
import flwr as fl
import requests


IP = socket.gethostbyname(socket.gethostname()) 
PORT = 5001
URL = f"{IP}:{PORT}"
NUM_ROUNDS = 5
GRPC_LENGTH = 1024*1024*1024
MIN_CLIENTS = 15

def start_fl_session():

    initial_weights = None

    strategy = CustomFLStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,
        num_rounds=NUM_ROUNDS,
        initial_parameters=initial_weights  
    )

    fl.server.start_server(
    server_address=URL,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    grpc_max_message_length=GRPC_LENGTH,
    strategy=strategy,
    )

  
if __name__ == '__main__':
    start_fl_session()




