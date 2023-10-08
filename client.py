
import argparse
import sys
from copy import deepcopy
from tqdm import tqdm

from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.contrib.algorithm.basic_client import SerialClientTrainer as SerialTrainer
from fedlab.contrib.algorithm.basic_client import ClientTrainer as Trainer
from fedlab.core.network import DistNetwork
from fedlab.utils import Logger, MessageCode, SerializationTool
sys.path.append("../../")
from fedlab.core.client import SERIAL_TRAINER, ORDINARY_TRAINER
from fedlab.core.client.manager import PassiveClientManager as Manager
import net
import torch
from model import SimpleDataset, BasicModel as Model

parser = argparse.ArgumentParser(description="Network connection checker")
parser.add_argument("--ip", type=str, default='localhost')
parser.add_argument("--port", type=str, default=1234)
parser.add_argument("--world_size", type=int, default=3)
parser.add_argument("--rank", type=int, default=1)
parser.add_argument("--ethernet", type=str, default='Wi-Fi')
args = parser.parse_args()

network = net.Network(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=args.rank,
    ethernet=args.ethernet,
)

client_model = Model()

class ClientTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, cuda: bool = False, device: str = None, logger: Logger = None) -> None:
        super().__init__(model, cuda, device)
        self._LOGGER = Logger() if logger is None else logger
        self.type = ORDINARY_TRAINER

    def setup(self):
        self.setup_dataset()
        self.setup_optim()

    def setup_dataset(self):
        self.dataset = SimpleDataset()
    
    def setup_optim(self):
        self.epochs = 1
        self.batch_size=1
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr = 0.01)
    
    @property
    def uplink_package(self):
        return [self.model_parameters]
    
    def local_process(self, payload, id):
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)
        
    def train(self, model_parameters, train_loader) -> None:
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                output = self._model(data)
                loss = self.criterion(output, target.to(torch.float32))
                print("Target:",target,"Ouput:", output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]
    
    def evaluate(self, output, target):
        correct = (output == target).sum().item()
        total = target.size(0)
        multi_accuracy = correct / total
        print('Accuracy:', multi_accuracy)


class SerialClientTrainer(SerialTrainer):
    def __init__(self, model: torch.nn.Module, num_clients: int, cuda: bool = False, device: str = None, logger: Logger = None, personal: bool = False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.type = SERIAL_TRAINER
        self.cache = []
    
    def setup(self):
        self.setup_dataset()
        self.setup_optim()

    def setup_dataset(self):
        self.dataset = SimpleDataset()
    
    def setup_optim(self):
        self.epochs = 1
        self.batch_size=1
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr = 0.01)
    
    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package
    
    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            print("Model _ Params" , model_parameters , model_parameters.dtype)
            model_parameters = model_parameters.to(torch.float32)
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)
        
    def train(self, model_parameters, train_loader) -> None:
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                output = self._model(data)
                loss = self.criterion(output, target.to(torch.float32))
                print("Target:",target,"Ouput:", output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]
    
    def evaluate(self, output, target):
        correct = (output == target).sum().item()
        total = target.size(0)
        multi_accuracy = correct / total
        print('Accuracy:', multi_accuracy)


class ClientManager(Manager):
    def __init__(self, network: net.Network, trainer: ModelMaintainer, logger: Logger = None):
        super().__init__(network, trainer, logger)
        self._trainer = trainer
        self._network = network
        self._LOGGER = Logger() if logger is None else logger
    
    def run(self):
        self.setup()
        self.main_loop()
        self.shutdown()
    
    def setup(self):
        self._network.init_network_connection()
        self._trainer.setup()
        tensor = torch.tensor([self._trainer.num_clients])
        self._network.send(content=tensor, message_code=MessageCode.SetUp, dst=0)
    
    def main_loop(self):
        """Actions to perform when receiving a new message, including local training.

        Main procedure of each client:
            1. client waits for data from server (PASSIVELY).
            2. after receiving data, client start local model training procedure.
            3. client synchronizes with server actively.
        """
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)
            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:
                id_list, payload = payload[0].to(
                    torch.int32).tolist(), payload[1:]
                # check the trainer type
                if self._trainer.type == SERIAL_TRAINER:
                    self._trainer.local_process(payload=payload, id_list=id_list)

                elif self._trainer.type == ORDINARY_TRAINER:
                    assert len(id_list) == 1
                    self._trainer.local_process(payload=payload, id=id_list[0])

                self.synchronize()

            else:
                raise ValueError("Invalid MessageCode {}. Please check MessageCode list.".format(message_code))
        
    def synchronize(self):
        """Synchronize with server."""
        self._LOGGER.info("Uploading information to server.")

        if self._trainer.type == SERIAL_TRAINER:
            payloads = self._trainer.uplink_package
            print('Payloads: ', payloads)
            for elem in payloads:
                self._network.send(content=elem, message_code=MessageCode.ParameterUpdate, dst=0)
                            
        if self._trainer.type == ORDINARY_TRAINER:
            self._network.send(content=self._trainer.uplink_package, message_code=MessageCode.ParameterUpdate, dst=0)
    
    def shutdown(self):
        self._network.close_network_connection()
    
# clientTrainer = SerialClientTrainer(model=client_model, num_clients=3)
clientTrainer = ClientTrainer(model=client_model)
clientManager = ClientManager(network=network, trainer=clientTrainer)
clientManager.run()