import torch
from torch import nn
from torch._C import dtype
from utils import *
import os

# Original MLP without any Parallel
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





# Column Parallel Linear layer
class ColumnParallelLinear(torch.nn.Module):
    # Initialize function
    def __init__(self, input_size, output_size, bias=True, gather_output=True, skip_bias_add=False, world_size = 2):
        super(ColumnParallelLinear, self).__init__()
        # Get input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Cut the model's weight from its last dimension
        self.output_size_per_partition = output_size//world_size
        self.skip_bias_add = skip_bias_add
        # Initialize the original weight of the model
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,device=torch.cuda.current_device(),
                                                dtype=torch.float))
        # Add the bias parameter
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.output_size_per_partition,device=torch.cuda.current_device(), dtype=torch.float))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
    # Forward functions
    def forward(self, input_):
        "not consider about the async all reduce"
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        '''conduct linear computation'''
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

class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False, 
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = 2
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.weight = nn.Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=torch.float))
        if bias:
            self.bias = nn.Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=torch.float))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)



    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = nn.functional.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias