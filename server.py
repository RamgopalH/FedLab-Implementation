import argparse
import sys
from fedlab.core.network import DistNetwork
from fedlab.core.server.handler import ServerHandler
from fedlab.utils import Logger, Aggregators, SerializationTool
sys.path.append("../../")
import torch
import net
from copy import deepcopy
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.utils.message_code import MessageCode
from fedlab.contrib.algorithm.basic_server import SyncServerHandler as Handler
from fedlab.core.coordinator import Coordinator
import threading
from model import BasicModel as Model, SimpleDataset as Dataset
from fedlab.utils.functional import evaluate

parser = argparse.ArgumentParser(description="Network connection checker")

parser.add_argument("--ip", type=str, default='localhost')
parser.add_argument("--port", type=str, default=1234)
parser.add_argument("--world_size", type=int, default=3)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--ethernet", type=str, default='Wi-Fi')
args = parser.parse_args()

network = net.Network(
    address=(args.ip, args.port),
    world_size=args.world_size,
    rank=args.rank,
    ethernet=args.ethernet,
)

model = Model()

class ServerHandler(Handler):
    def __init__(self, model: torch.nn.Module, num_clients: int = 1, sample_ratio: float = 1, global_round: int = 3, cuda: bool = False, device: str = None, ) -> None:
        super().__init__(model, global_round, sample_ratio, cuda, device)

        self.client_buffer_cache = []
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.global_round = global_round
        self.round = 0

        self.round_clients = max(1, int(self.sample_ratio * self.num_clients)) 
    
    @property
    def if_stop(self):
        return self.round>=self.global_round
    
    @property
    def downlink_package(self):
        """Property for manager layer. Server manager will call this property when activates clients."""
        """The model parameters passed down to each client on activation, or whatever you want to pass to each client before activation"""
        return [self.model_parameters]
    
    def setup_optim(self):
        self.criterion = torch.nn.MSELoss(size_average = False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01)

    def load(self, payload , sender_rank):
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`model_parameters_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            payload (list[torch.Tensor]): A list of tensors passed by manager layer.
        """
        assert len(payload) > 0
        self.client_buffer_cache.append(deepcopy(payload))

        assert len(self.client_buffer_cache) <= self.num_clients_per_round

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1

            # reset cache
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False
        
    # Aggregator Function goes Here
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)
    
    def setup_dataset(self, dataset):
        self.dataset = dataset

    def evaluate(self):
        self._model.eval()
        test_loader = self.dataset.get_dataloader(type="test", batch_size=128)
        loss_, acc_ = evaluate(self._model, torch.nn.CrossEntropyLoss(), test_loader)
        self._LOGGER.info(
            f"Round [{self.round - 1}/{self.global_round}] test performance on server: \t Loss: {loss_:.5f} \t Acc: {100*acc_:.3f}%"
        )

        return loss_, acc_
        

class ServerManager(SynchronousServerManager):
    def __init__(self, network: net.Network, handler: ServerHandler, mode: str = "LOCAL", logger: Logger = None):
        super().__init__(network, handler, mode, logger)
        self._network = network
        self._handler = handler
        self.coordinator = None
    
    def run(self):
        self.setup()
        self.main_loop()
        self.shutdown()

    def setup(self):
        """Initialization Stage.

        - Server accept local client num report from client manager.
        - Init a coordinator for client_id -> rank mapping.
        """
        self._network.init_network_connection()
        rank_client_id_map = {}
        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map, self.mode)
        if self._handler is not None:
            self._handler.num_clients = self.coordinator.total
    
    def main_loop(self):
        """Actions to perform in server when receiving a package from one client.

        Server transmits received package to backend computation handler for aggregation or others
        manipulations.

        Loop:
            1. activate clients for current training round.
            2. listen for message from clients -> transmit received parameters to server handler.

        Note:
            Communication agreements related: user can overwrite this function to customize
            communication agreements. This method is key component connecting behaviors of
            :class:`ServerHandler` and :class:`NetworkManager`.

        Raises:
            Exception: Unexpected :class:`MessageCode`.
        """
        while self._handler.if_stop is not True:
            activator = threading.Thread(target=self.activate_clients)
            activator.start()

            while True:
                sender_rank, message_code, payload = self._network.recv()
                if message_code == MessageCode.ParameterUpdate:
                    if self._handler.load(payload, sender_rank):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))
    
    def shutdown(self):
        self.shutdown_clients()
        # self._network.close_network_connection()
    
    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The id of clients are obtained from :meth:`handler.sample_clients`. And their communication ranks are are obtained via coordinator.
        """
        self._LOGGER.info("Client activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            content = downlink_package + [id_list]
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.ParameterUpdate,
                               dst=rank)

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit`.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        client_list = range(self._handler.num_clients)
        rank_dict = self.coordinator.map_id_list(client_list)

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.Exit,
                               dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(src=self._network.world_size -
                                                1)
        assert message_code == MessageCode.Exit
        self._network.close_network_connection()

serverhandler = ServerHandler(model=model, cuda=False, num_clients=2)
serverManager = ServerManager(network=network, handler=serverhandler)
serverManager.run()