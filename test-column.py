import torch
import os
from DataLoader import FakeDataLoader
from initialize import initialize_model_parallel
from utils import *
from layers import ColumnParallelLinear
from numpy import mean

COMPUTE_TIME_RECORD = True

def train():
    batch_size = 64
    dim = 1024

    model = ColumnParallelLinear(dim, dim*batch_size, gather_output=True, compute_time_record=COMPUTE_TIME_RECORD)
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
    if(COMPUTE_TIME_RECORD):
        print(model.compute_time)
        print(mean(model.compute_time))



if __name__ == '__main__':
    torch.distributed.init_process_group(backend='nccl')
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    initialize_model_parallel(2)
    train()