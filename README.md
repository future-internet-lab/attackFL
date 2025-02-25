# Federated Learning Poisoning Attack

## Required Packages
```
pika
torch
torchvision
numpy
requests
tqdm
pyyaml
```

Set up a RabbitMQ server for message communication over the network environment. `docker-compose.yaml` file:

```yaml
version: '3'

services:
  rabbitmq:
    image: rabbitmq:management
    container_name: rabbitmq
    ports:
      - "5672:5672"   # RabbitMQ main port
      - "15672:15672" # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: user
      RABBITMQ_DEFAULT_PASS: password
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
volumes:
  rabbitmq_data:
    driver: local
```

Then run the RabbitMQ container

```commandline
docker-compose up -d
```

## Configuration

Application configuration is in the `config.yaml` file:

```yaml
name: Federated Learning poisoning attack testbed
server:   # server configuration
  num-round: 2  # number of training rounds
  clients: 3    # number of FL clients
  mode: hyper   # server mode: `fedavg` or `hyper`
  model: RNNModel     # class name of DNN model
  data-name: ICU      # training data
  parameters:
    load: True     # allow to load parameters file
  validation: True  # allow to validate on server-side
  ### algorithm
  data-distribution:      # data distribution config
    num-data-range:       # minimum and maximum number of label's data
      - 0
      - 500
  genuine-rate: 0.5       # genuine clients rate and send to malicious clients 
  random-seed: 1

rabbit:   # RabbitMQ connection configuration
  address: 127.0.0.1    # address
  username: admin
  password: admin

log_path: .   # logging directory

learning:
  epoch: 5          # client local epochs
  learning-rate: 0.01
  hyper-lr: 0.0005      # learning rate of hyper model
  momentum: 0.5
  batch-size: 256
```

This configuration is use for server and all clients.

### List of DNN model

#### For ICU
```
CNNModel
RNNModel
TransformerModel
```

#### For CIFAR10
```
CNNTarget
```

## How to Run

Alter your configuration, you need to run the server to listen and control the request from clients.

### Server

```commandline
python server.py
```

### Client

```commandline
python client.py
```

If using a specific device configuration for the training process, declare it with the `--device` argument when running the command line:

```commandline
python client.py --device cpu
```

### Client attack

#### Random

```commandline
python client.py --attack True --attack_mode Random --attack_round 30 --attack_args 0.5
```

#### LIE

```commandline
python client.py --attack True --attack_mode LIE --attack_round 1 --attack_args 0.74
```

#### Min-Max

```commandline
python client.py --attack True --attack_mode Min-Max --attack_round 2
```

#### Min-Sum

```commandline
python client.py --attack True --attack_mode Min-Sum --attack_round 2
```

#### Opt-Fang

```commandline
python client.py --attack True --attack_mode Opt-Fang --attack_round 2
```

## Parameter Files

On the server, the `*.pth` files are saved in the main execution directory of `server.py` after completing one training round.

If the `*.pth` file exists, the server will read the file and send the parameters to the clients. Otherwise, if the file does not exist, a new DNN model will be created with fresh parameters. Therefore, if you want to reset the training process, you should delete the `*.pth` files.
