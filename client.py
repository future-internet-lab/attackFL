import pika
import uuid
import argparse
import yaml
import sys
from tqdm import tqdm

import torch
import torch.optim as optim

import src.Log
from src.RpcClient import RpcClient

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--attack', type=bool, required=False,
                    default=False, help='Set to True to enable attack mode, False otherwise.')
parser.add_argument('--attack_mode', type=str, choices=['MPAF', 'Min-Max', 'Min-Sum'],
                    help="Mode of operation when attack is enabled (e.g., MPAF or Min-Max, Min-Sum ...).")
parser.add_argument('--attack_round', type=int,
                    help="Client attack at round.")
parser.add_argument('--attack_args', type=float, nargs='+', required=False,
                    help="A list of attack args. Use space to separate values (e.g., 0.1 0.2 0.5).")

args = parser.parse_args()

# Validate condition
if args.attack and not args.attack_mode:
    print("Error: --attack_mode is required when --attack is True.")
    sys.exit()
if args.attack and not args.attack_round:
    print("Error: --attack_round is required when --attack is True.")
    sys.exit()

print(f"Attack: {args.attack}, Mode: {args.attack_mode}")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]

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


credentials = pika.PlainCredentials(username, password)


def train_on_device(model, lr, momentum, trainloader, criterion):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()
    for (training_data, label) in tqdm(trainloader):
        if training_data.size(0) == 1:
            continue
        training_data = training_data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(training_data)
        loss = criterion(output, label)

        if torch.isnan(loss).any():
            src.Log.print_with_color("NaN detected in loss, stop training", "yellow")
            return False

        loss.backward()
        optimizer.step()

    return True


if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "message": "Hello from Client!", "attack": args.attack}
    client = RpcClient(client_id, address, username, password, train_on_device, device)
    if args.attack:
        client.set_attack_config(args.attack_mode, args.attack_round, args.attack_args)

    client.send_to_server(data)
    client.wait_response()