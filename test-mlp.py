import torch
import os
from DataLoader import FakeDataLoader
from initialize import initialize_model_parallel
from utils import *
from layers import ColumnParallelLinear, RowParallelLinear

def train():
    batch_size = 64
    dim = 1024

    column_parallel_model = ColumnParallelLinear(dim, dim*batch_size, gather_output=False)
    column_parallel_model = column_parallel_model.cuda()

    row_parallel_model = RowParallelLinear(dim*batch_size, dim*batch_size, input_is_parallel=True)
    row_parallel_model = row_parallel_model.cuda()

    dataloader = FakeDataLoader((batch_size, dim))

    def train_iter(column_parallel_model, row_parallel_model, dataloader):
        for _ in range(4):
            data = next(dataloader)
            temp_output = column_parallel_model(data)
            output = row_parallel_model(temp_output)

            output_list = list(output)
            output =output_list[0]

            output = torch.as_tensor(output, dtype=float, device=None)
            loss = torch.sum(output) / 1000
            print(f'loss={loss.item()}')
            loss.backward()
    
    optimizer = torch.optim.SGD(row_parallel_model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10):
        train_iter(column_parallel_model, row_parallel_model, dataloader)
        optimizer.step()
        optimizer.zero_grad()



if __name__ == '__main__':
    torch.distributed.init_process_group(backend='nccl')
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    initialize_model_parallel(2)
    train()