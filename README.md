# Federated Learning Poisoning Attack

## Federated Learning and Poisoning Attack

...

## Deployment Model

...

## Required Packages
```
pika
torch
torchvision
numpy
requests
tqdm
pyyaml
scikit-learn
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
name: Coordinated Federated Learning
server:   # server configuration
  num-round: 2  # number of training rounds
  clients: 3    # number of FL clients
  model: ResNet50     # class name of DNN model
  data-name: CIFAR10  # training data: MNIST, CIFAR10
  parameters:
    load: False     # allow to load parameters file
    save: False     # allow to save parameters file
                    # if turn on, server will be averaging all parameters
  validation: True  # allow to validate on server-side
  ### algorithm
  data-mode: even         # data distribution `even` or `uneven`
  data-distribution:      # data distribution config
    num-data-range:       # minimum and maximum number of label's data
      - 0
      - 500
    non-iid-rate: 0.5     # non-IID rate, range (0, 1]
    refresh-each-round: True  # if set True, non-IID on label will be reset on each round
  random-seed: 1

rabbit:   # RabbitMQ connection configuration
  address: 127.0.0.1    # address
  username: admin
  password: admin

log_path: .   # logging directory

learning:
  learning-rate: 0.01
  momentum: 1
  batch-size: 256
```

This configuration is use for server and all clients.

### List of DNN model

#### For MNIST
```
SimpleCNN
LeNet_MNIST
```

#### For CIFAR10
```
LeNet_CIFAR10
MobileNetV2
ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
VGG16, VGG19
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

#### Min-Max

```commandline
python client.py --attack True --attack_mode Min-Max --attack_round 2 --attack_args 0.00001
```

#### Min-Sum

```commandline
python client.py --attack True --attack_mode Min-Sum --attack_round 2 --attack_args 0.00001
```

## Parameter Files

On the server, the `*.pth` files are saved in the main execution directory of `server.py` after completing one training round.

If the `*.pth` file exists, the server will read the file and send the parameters to the clients. Otherwise, if the file does not exist, a new DNN model will be created with fresh parameters. Therefore, if you want to reset the training process, you should delete the `*.pth` files.
