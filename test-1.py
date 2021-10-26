import torch
from torch import nn
from torch._C import dtype
import sys
import os
from DataLoader import FakeDataLoader
from initialize import initialize_model_parallel
_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}
# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None


from utils import *
def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)

def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP
def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())
# 无parallel的最初始的MLP
class MLP(nn.Module):
    def __init__(self, dim, mult=16):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult, bias=False)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.linear3 = nn.Linear(dim, dim * mult, bias=False)
        self.linear4 = nn.Linear(dim * mult, dim)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        return output






class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, gather_output=True, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False, world_size = 2):
        super(ColumnParallelLinear, self).__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # 将模型的权重以最后一个dimension进行分割
        
        self.output_size_per_partition = output_size//world_size # 强制切
        self.skip_bias_add = skip_bias_add
        # 初始化模型的权重
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,device=torch.cuda.current_device(),
                                                dtype=torch.float))

        #添加bias参数
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,device=torch.cuda.current_device(), dtype=torch.float))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input_):
        "不考虑async的all reduce"
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        '''执行forward操作'''
        output_parallel = nn.functional.linear(input_parallel, self.weight, self.bias)
        
        if self.gather_output:
            # All-gather across the partitions.
            print(output_parallel)
            print(torch.distributed.get_world_size())
            output = OutputAdapter.apply(output_parallel)
        else:
            os.system("pause")
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias



def train():
    batch_size = 64
    dim = 1024

    model = ColumnParallelLinear(dim, dim*batch_size, gather_output=True)
    model = model.cuda()

    dataloader = FakeDataLoader((batch_size, dim))

    def train_iter(model, dataloader):
        for _ in range(4):
            data = next(dataloader)
            output = model(data)

            output_list = list(output)
            output =output_list[0]

            output = torch.as_tensor(output, dtype=float, device=None)
            loss = torch.sum(output) / 1000
            print(f'loss={loss.item()}')
            loss.backward()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()



if __name__ == '__main__':
    torch.distributed.init_process_group(backend='nccl')
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    initialize_model_parallel(2)
    train()