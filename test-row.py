import torch
import os
from DataLoader import FakeDataLoader
from initialize import initialize_model_parallel
from numpy import mean

from utils import *
from layers import RowParallelLinear

COMPUTE_TIME_RECORD = True
COMMUNICATE_TIME_RECORD = True


def train():
    input_size_m = 512
    input_size_k = 512
    output_size_n =1024
    model = RowParallelLinear(input_size_k, output_size_n, compute_time_record=COMPUTE_TIME_RECORD, communicate_time_record=COMMUNICATE_TIME_RECORD)
    model = model.cuda()

    dataloader = FakeDataLoader((input_size_m, input_size_k))

    def train_iter(model, dataloader):
        for _ in range(4):
            data = next(dataloader)
            output = model(data)

            output_list = list(output)
            output =output_list[0]

            output = torch.as_tensor(output, dtype=float, device=None)
            loss = torch.sum(output) / 1000
            #print(f'loss={loss.item()}')
            loss.backward()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
    
    if(COMPUTE_TIME_RECORD):
        print(torch.distributed.get_rank()," device's compute time is ",mean(model.compute_time[1:]),"in ",len(model.communicate_time)," rounds")
    if(COMMUNICATE_TIME_RECORD):
        print(torch.distributed.get_rank()," device's communicate time is ",mean(model.communicate_time[1:]),"in ",len(model.communicate_time)," rounds")



if __name__ == '__main__':
    torch.distributed.init_process_group(backend='nccl')
    world_size = int(os.environ['WORLD_SIZE'])
    print("world size is ",world_size)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    initialize_model_parallel(world_size)
    train()