from fedlab.core.communicator.processor import Package, PackageProcessor
import torch.distributed as dist
import torch
import numpy as np
from fedlab.core.communicator import HEADER_DATA_TYPE_IDX, HEADER_SIZE, HEADER_RECEIVER_RANK_IDX, HEADER_SLICE_SIZE_IDX, dtype_flab2torch, dtype_torch2flab

class PackageProcessorA(PackageProcessor):
    def send_package(package, dst):
        """Three-segment tensor communication pattern based on ``torch.distributed``

        Pattern is shown as follows:
            1.1 sender: send a header tensor containing ``slice_size`` to receiver

            1.2 receiver: receive the header, and get the value of ``slice_size`` and create a buffer for incoming slices of content

            2.1 sender: send a list of slices indicating the size of every content size.

            2.2 receiver: receive the slices list.

            3.1 sender: send a content tensor composed of a list of tensors.

            3.2 receiver: receive the content tensor, and parse it to obtain slices list using parser function
        """
        def send_header(header, dst):
            header[HEADER_RECEIVER_RANK_IDX] = dst
            dist.send(header, dst=dst, tag=1)

        def send_slices(slices, dst):
            np_slices = np.array(slices, dtype=np.int32)
            tensor_slices = torch.from_numpy(np_slices)
            dist.send(tensor_slices, dst=dst, tag=2)

        def send_content(content, dst):
            dist.send(content, dst=dst, tag=3)

        # body
        if package.dtype is not None:
            package.header[HEADER_DATA_TYPE_IDX] = dtype_torch2flab(
                package.dtype)

        # sender header firstly
        send_header(header=package.header, dst=dst)

        # if package got content, then send remain parts
        if package.header[HEADER_SLICE_SIZE_IDX] > 0:
            send_slices(slices=package.slices, dst=dst)
            send_content(content=package.content, dst=dst)
        print('Sent Content:', package.content)

    def recv_package(src=None):
        """Three-segment tensor communication pattern based on ``torch.distributed``

        Pattern is shown as follows:
            1.1 sender: send a header tensor containing ``slice_size`` to receiver

            1.2 receiver: receive the header, and get the value of ``slice_size`` and create a buffer for incoming slices of content

            2.1 sender: send a list of slices indicating the size of every content size.

            2.2 receiver: receive the slices list.

            3.1 sender: send a content tensor composed of a list of tensors.

            3.2 receiver: receive the content tensor, and parse it to obtain slices list using parser function
        """

        def recv_header(src=src, parse=True):
            buffer = torch.zeros(size=(HEADER_SIZE, ), dtype=torch.int32)
            dist.recv(buffer, src=src, tag=1)
            if parse is True:
                return Package.parse_header(buffer)
            else:
                return buffer

        def recv_slices(slices_size, src):
            buffer_slices = torch.zeros(size=(slices_size, ),
                                        dtype=torch.int32)
            dist.recv(buffer_slices, src=src, tag=2)
            slices = [slc.item() for slc in buffer_slices]
            return slices

        def recv_content(slices, data_type, src):
            sum = 0 
            i=0
            while i<len(slices):
                sum+=slices[i]
                i += slices[i+1] + 2
            content_size = sum
            dtype = dtype_flab2torch(data_type)
            buffer = torch.zeros(size=(content_size, ), dtype=dtype)
            dist.recv(buffer, src=src, tag=3)
            return Package.parse_content(slices, buffer)

        # body
        
        sender_rank, _, slices_size, message_code, data_type = recv_header(
            src=src)
        if slices_size > 0:
            slices = recv_slices(slices_size=slices_size, src=sender_rank)
            content = recv_content(slices, data_type, src=sender_rank)
        else:
            content = None
        print('Recieved Content:', content)
        return sender_rank, message_code, content