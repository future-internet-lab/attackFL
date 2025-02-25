import os
import pika
import pickle
import argparse
import sys
import yaml

import torch
import requests
import random

import src.Validation
import src.Log
import src.Model

from tqdm import tqdm
from collections import OrderedDict
from requests.auth import HTTPBasicAuth

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

parser.add_argument('--device', type=str, required=False, help='Device of server')

args = parser.parse_args()

device = None

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

total_clients = config["server"]["clients"]
model_name = config["server"]["model"]
data_name = config["server"]["data-name"]
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

num_round = config["server"]["num-round"]
load_parameters = config["server"]["parameters"]["load"]
validation = config["server"]["validation"]
genuine_rate = config["server"]["genuine-rate"]
random_seed = config["server"]["random-seed"]

data_distribution = config["server"]["data-distribution"]
server_mode = config["server"]["mode"]
data_range = data_distribution["num-data-range"]

# Clients
epoch = config["learning"]["epoch"]
batch_size = config["learning"]["batch-size"]
lr = config["learning"]["learning-rate"]
hyper_lr = config["learning"]["hyper-lr"]
momentum = config["learning"]["momentum"]

log_path = config["log_path"]

if data_name == "CIFAR10" or data_name == "MNIST":
    num_labels = 10
else:
    num_labels = 0

if random_seed:
    random.seed(random_seed)


class Server:
    def __init__(self):
        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        self.channel = self.connection.channel()
        self.num_round = num_round
        self.round = self.num_round

        self.channel.queue_declare(queue='rpc_queue')

        self.total_clients = total_clients
        self.current_clients = 0
        self.updated_clients = 0
        self.responses = {}  # Save response
        self.list_clients = []
        self.list_attack_clients = []
        self.all_model_parameters = []
        self.all_genuine_parameters = []
        self.final_state_dict = None
        self.round_result = True

        self.selected_client = []

        # Initial hyper training model
        self.net = None
        self.hnet = None
        if server_mode == "hyper":
            if not self.net and not self.hnet:
                if hasattr(src.Model, model_name):
                    self.net = getattr(src.Model, model_name)().to(device)
                else:
                    raise ValueError(f"Model name '{model_name}' is not valid.")

                if load_parameters:
                    filepath_hyper = f'{model_name}_hyper_{total_clients}.pth'
                    filepath = f'{model_name}.pth'
                    if os.path.exists(filepath_hyper):
                        src.Log.print_with_color(f"Load state dict from hyper model: {filepath_hyper}", "yellow")
                        state_dict = torch.load(filepath_hyper, weights_only=True)
                        self.load_new_hyper()
                        self.hnet.load_state_dict(state_dict)
                    elif os.path.exists(filepath):
                        src.Log.print_with_color(f"Load state dict from original model: {filepath}", "yellow")
                        state_dict = torch.load(filepath, weights_only=True)
                        self.net.load_state_dict(state_dict)
                        self.load_new_hyper()
                    else:
                        self.load_new_hyper()
                else:
                    self.load_new_hyper()

            self.optimizer = torch.optim.Adam(self.hnet.parameters(), lr=hyper_lr)

        self.logger = src.Log.Logger(f"{log_path}/app.log")
        self.validation = src.Validation.Validation(model_name, data_name, self.logger)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        self.logger.log_info("### Application start ###\n")
        src.Log.print_with_color(f"Server is waiting for {self.total_clients} clients.", "green")

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        """
        Response message to clients
        :param client_id: client ID
        :param message: message
        :return:
        """
        reply_channel = self.channel
        reply_queue_name = f'reply_{client_id}'
        reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def on_request(self, ch, method, props, body):
        """
        Handler request from clients
        :param ch: channel
        :param method:
        :param props:
        :param body: message body
        :return:
        """
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            attack = message["attack"]

            if str(client_id) not in self.list_clients:
                if attack:
                    self.list_attack_clients.append(str(client_id))

                self.list_clients.append(str(client_id))
                src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")

            # If consumed all clients - Register for first time
            if len(self.list_clients) == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.client_selection()
                src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
                self.notify_clients()
        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from client: {data_message}", "blue")
            self.updated_clients += 1
            # Save client's model parameters
            if not result:
                self.round_result = False

            if self.round_result:
                model_state_dict = message["parameters"]
                client_size = message["size"]
                self.all_model_parameters.append({'client_id': client_id, 'weight': model_state_dict,
                                                  'size': client_size})
                if str(client_id) not in self.list_attack_clients:
                    self.all_genuine_parameters.append(model_state_dict)

            # If consumed all client's parameters
            if self.updated_clients == len(self.selected_client):
                self.process_consumer()

        # Ack the message
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_consumer(self):
        """
        After collect all training clients, start validation and make decision for the next training round
        :return:
        """
        self.updated_clients = 0
        src.Log.print_with_color("Collected all parameters.", "yellow")
        # TODO: detect model poisoning with self.all_model_parameters at here
        if self.round_result:
            if server_mode == "fedavg":
                self.avg_all_parameters()
            elif server_mode == "hyper":
                src.Log.print_with_color(f"Start training hyper model!", "yellow")
                self.train_hyper()
            self.all_model_parameters = []
        # Server validation
        if validation and self.round_result:
            self.round_result = self.validation.test(self.final_state_dict, device)

        if not self.round_result:
            src.Log.print_with_color(f"Training failed!", "yellow")
        else:
            # Save to files
            if server_mode == "fedavg":
                torch.save(self.final_state_dict, f'{model_name}.pth')
            elif server_mode == "hyper":
                torch.save(self.hnet.state_dict(), f'{model_name}_hyper_{total_clients}.pth')
            self.round -= 1
        self.round_result = True

        if self.round > 0:
            # Start a new training round
            src.Log.print_with_color(f"Start training round {self.num_round - self.round + 1}", "yellow")
            self.client_selection()
            self.notify_clients()
        else:
            # Stop training
            self.notify_clients(start=False)
            sys.exit()

    def notify_clients(self, start=True):
        """
        Control message to clients
        :param start: If True (default), request clients to start. Else if False, stop training
        :return:
        """
        # Send message to clients when consumed all clients
        if start:
            # Read parameters file
            state_dict = None
            if server_mode == "fedavg":
                if load_parameters:
                    filepath = f'{model_name}.pth'
                    if os.path.exists(filepath):
                        state_dict = torch.load(filepath, weights_only=True)
            for i in self.selected_client:
                if server_mode == "hyper":
                    state_dict = self.hnet(torch.tensor([i], dtype=torch.long).to(device))
                # convert client indices to client id
                client_id = self.list_clients[i]
                # Request clients to start training
                src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")

                genuine_models = None
                if client_id in self.list_attack_clients:
                    if len(self.all_genuine_parameters) > 0:
                        genuine_models = random.sample(self.all_genuine_parameters,
                                                       max(int(genuine_rate * len(self.all_genuine_parameters)), 1))

                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "model_name": model_name,
                            "data_name": data_name,
                            "parameters": state_dict,
                            "data_ranges": data_range,
                            "epoch": epoch,
                            "batch_size": batch_size,
                            "lr": lr,
                            "momentum": momentum,
                            "genuine_models": genuine_models}
                self.send_to_response(client_id, pickle.dumps(response))
            # clear all genuine models
            self.all_genuine_parameters = []
        else:
            for client_id in self.list_clients:
                # Request clients to stop process
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))

    def client_selection(self):
        """
        Select the specific clients
        :return: The list contain index of active clients: `self.selected_client`.
        E.g. `self.selected_client = [2,3,5]` means client 2, 3 and 5 will train this current round
        """
        self.selected_client = [i for i in range(len(self.list_clients))]

        # From client selected, calculate and log training time
        self.logger.log_info(f"Active with {len(self.selected_client)} client: {self.selected_client}")

    def train_hyper(self):
        """
        Consuming all client's weight from `self.all_model_parameters` and start training hyper model
        :return: Global weight on `self.final_state_dict`
        """
        for node_id in tqdm(self.selected_client):
            self.hnet.train()
            weights = self.hnet(torch.tensor([node_id], dtype=torch.long).to(device))

            # Instead of loading weights directly, update net's parameters using the generated weights
            for name, param in self.net.named_parameters():
                param.data = weights[name]

            self.optimizer.zero_grad()

            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})
            final_state = self.find_weight(node_id)         # get client's weight
            final_state = {k: v.to(device) for k, v in final_state.items()}
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            # Calculate gradients with respect to hypernetwork parameters
            hnet_grads = torch.autograd.grad(
                # Get the outputs from the hypernetwork (weights)
                outputs=list(weights.values()),
                # Get the inputs that need gradients (hypernetwork parameters)
                inputs=self.hnet.parameters(),
                # Specify the gradient with respect to the outputs
                grad_outputs=list(delta_theta.values()),
                # Allows for unused parameters
                allow_unused=True
            )

            # Update the hypernetwork parameters
            for p, g in zip(self.hnet.parameters(), hnet_grads):
                # Check if gradient is not None
                # if g is not None:
                p.grad = g

            # torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), 50)
            self.optimizer.step()

        # Update new clients' weight
        src.Log.print_with_color(f"Update new clients' weight", "yellow")
        for i in range(len(self.all_model_parameters)):
            c = self.list_clients.index(str(self.all_model_parameters[i]["client_id"]))
            print(f"Get client on index {c}")
            model_dict = self.hnet(torch.tensor([c], dtype=torch.long).to(device))
            self.all_model_parameters[i]["weight"] = model_dict

        self.avg_all_parameters()

    def find_weight(self, node_id):
        for w in self.all_model_parameters:
            if str(w["client_id"]) == self.list_clients[node_id]:
                return w["weight"]
        src.Log.print_with_color(f"[Warning] Cannot find weight of node id {node_id}!", "yellow")

    def avg_all_parameters(self):
        """
        Consuming all client's weight from `self.all_model_parameters` - a list contain all client's weight
        :return: Global weight on `self.final_state_dict`
        """
        # Average all client parameters
        num_models = len(self.all_model_parameters)
        src.Log.print_with_color(f"Number of models' parameters = {num_models}", "yellow")

        if num_models == 0:
            return

        self.final_state_dict = self.all_model_parameters[0]['weight']
        all_client_sizes = [item['size'] for item in self.all_model_parameters]

        for key in self.final_state_dict.keys():
            if self.final_state_dict[key].dtype != torch.long:
                self.final_state_dict[key] = sum(self.all_model_parameters[i]['weight'][key] * all_client_sizes[i]
                                                 for i in range(num_models)) / sum(all_client_sizes)
            else:
                self.final_state_dict[key] = sum(self.all_model_parameters[i]['weight'][key] * all_client_sizes[i]
                                                 for i in range(num_models)) // sum(all_client_sizes)

        if not self.final_state_dict:
            src.Log.print_with_color(f"[Warning] Final state dict is None!", "yellow")

    def load_new_hyper(self):
        self.hnet = src.Model.HyperNetwork(self.net, self.total_clients, 3, 100, False, 1).to(device)

def delete_old_queues():
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, '/', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue"):
                try:
                    http_channel.queue_delete(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' deleted.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to delete queue '{queue_name}': {e}", "yellow")
            else:
                try:
                    http_channel.queue_purge(queue=queue_name)
                    src.Log.print_with_color(f"Queue '{queue_name}' purged.", "green")
                except Exception as e:
                    src.Log.print_with_color(f"Failed to purge queue '{queue_name}': {e}", "yellow")

        connection.close()
        return True
    else:
        src.Log.print_with_color(
            f"Failed to fetch queues from RabbitMQ Management API. Status code: {response.status_code}", "yellow")
        return False


if __name__ == "__main__":
    delete_old_queues()
    server = Server()
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
