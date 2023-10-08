import sys
sys.path.append("../../")
from fedlab.core.network import DistNetwork
import package
import torch.distributed as dist
import os
import datetime
import torch


type2byte = {
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.float16: 2,
    torch.float32: 4,
    torch.float64: 8
}

class Network(DistNetwork):
    def init_network_connection(self):
        self._LOGGER.info(self.__str__())

        if self.ethernet is not None:
            os.environ["GLOO_SOCKET_IFNAME"] = self.ethernet

        dist.init_process_group(
            backend=self.dist_backend,
            init_method="tcp://{}:{}".format(self.address[0], self.address[1]),
            rank=self.rank,
            world_size=self.world_size,
            timeout=datetime.timedelta(seconds=10)
        )

    def send(self, content=None, message_code=None, dst=0, count=True):
        """Send tensor to process rank=dst"""
        pack = package.Package(message_code=message_code, content=content)
        package.PackageProcessorA.send_package(pack, dst=dst)
        if pack.content is not None and count is True:
            self.send_volume_intotal += pack.content.numel() * type2byte[
                pack.dtype]

        self._LOGGER.info(
            "Sent package to destination {}, message code {}, content length {}"
            .format(dst, message_code,
                    'Empty' if pack.content is None else pack.content))
    
    def recv(self, src=None, count=True):
        """Receive tensor from process rank=src"""
        sender_rank, message_code, content = package.PackageProcessorA.recv_package(
            src=src)

        if content is not None and count is True:
            volumn = sum([data.numel() for data in content])

            # content from server to client, the first content is id_list.
            # remove the size of id_list in the count.
            if self.rank != 0:
                volumn -= content[0].numel()

            self.recv_volume_intotal += volumn * type2byte[content[0].dtype]

        self._LOGGER.info(
            "Received package from source {}, message code {}, content {}"
            .format(sender_rank, message_code,
                    'Empty' if content is None else content))
        return sender_rank, message_code, content

