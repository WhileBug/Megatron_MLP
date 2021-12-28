import torch
import os
from lib.DataLoader import FakeDataLoader
from lib.initialize import initialize_model_parallel
from lib.utils import *
from lib.layers import ColumnParallelLinear
from numpy import mean
import time
import argparse

PHASE_1_FORWARD_TIME = True
PHASE_1_BACKWARD_TIME = False # not executed when there is only one layer
PHASE_2_FORWARD_TIME = True
PHASE_2_BACKWARD_TIME = True
PHASE_3_FORWARD_TIME = True
PHASE_3_BACKWARD_TIME = True
TIME_MEASURE = True


time_record_list = []
compute_time = 0

def train(input_size, hidden_size, output_size):
    input_size_m = input_size
    input_size_k = hidden_size
    output_size_n = output_size

    model = ColumnParallelLinear(input_size_k, output_size_n, gather_output=True, phase_2_forward_time=PHASE_2_FORWARD_TIME, phase_3_forward_time=PHASE_3_FORWARD_TIME, phase_1_forward_time=PHASE_1_FORWARD_TIME)
    model = model.cuda()

    dataloader = FakeDataLoader((input_size_m, input_size_k))


    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    backward_time_list = []
    for epoch in range(10):
        data = next(dataloader)
        output = model(data)

        output_list = list(output)
        output =output_list[0]

        output = torch.as_tensor(output, dtype=float, device=None)
        loss = torch.sum(output) / 1000
        if(PHASE_2_BACKWARD_TIME or PHASE_3_BACKWARD_TIME):
            torch.cuda.synchronize()
            time_before = time.time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if(PHASE_2_BACKWARD_TIME or PHASE_3_BACKWARD_TIME):
            torch.cuda.synchronize()
            time_after = time.time()
            backward_time_list.append(time_after-time_before)

    if(TIME_MEASURE):
        compute_time = mean(model.phase_2_forward_time_list[1:])
        return compute_time


def compute_time_profile(input_size, hidden_size, output_size, partition_size):
    torch.distributed.init_process_group(backend='nccl')
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    initialize_model_parallel(2)
    compute_T = train(input_size, hidden_size, output_size)
    return compute_T

'''
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--nproc_per_node', type=int, default = 2)
parser.add_argument('--nnodes', type=int, default=1)
args = parser.parse_args()
'''
if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='manual to this script')
    #parser.add_argument('--nproc_per_node', type=int, default = 2)
    #parser.add_argument('--nnodes', type=int, default=1)
    #args = parser.parse_args()
    result_T = compute_time_profile(1024, 1024, 1024, 2)
    print("compute time",result_T)
    return result_T