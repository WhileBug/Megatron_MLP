import torch
import os
from DataLoader import FakeDataLoader
from initialize import initialize_model_parallel
from numpy import mean

from utils import *
from layers import RowParallelLinear
import time



PHASE_1_FORWARD_TIME = True
PHASE_1_BACKWARD_TIME = False # not executed when there is only one layer
PHASE_2_FORWARD_TIME = True
PHASE_2_BACKWARD_TIME = True
PHASE_3_FORWARD_TIME = True
PHASE_3_BACKWARD_TIME = True
TIME_MEASURE = True

time_record_list = []

def train(TEST_SIZE):
    input_size_m = TEST_SIZE
    input_size_k = TEST_SIZE
    output_size_n =TEST_SIZE

    model = RowParallelLinear(input_size_k, output_size_n, phase_2_forward_time=PHASE_2_FORWARD_TIME, phase_3_forward_time=PHASE_3_FORWARD_TIME, phase_1_forward_time=PHASE_1_FORWARD_TIME)
    model = model.cuda()

    dataloader = FakeDataLoader((input_size_m, input_size_k))

            
    
    

    backward_time_list = []
    for epoch in range(10):
        data = next(dataloader)
        output = model(data)

        output_list = list(output)
        output =output_list[0]

        output = torch.as_tensor(output, dtype=float, device=None)
        loss = torch.sum(output) / 1000

        # Measure the time of backward function in phase 2
        if(PHASE_2_BACKWARD_TIME or PHASE_3_BACKWARD_TIME):
            torch.cuda.synchronize()
            time_before = time.time()
        loss.backward()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer.step()
        optimizer.zero_grad()
        if(PHASE_2_BACKWARD_TIME or PHASE_3_BACKWARD_TIME):
            torch.cuda.synchronize()
            time_after = time.time()
            backward_time_list.append(time_after-time_before)
        
    
    if(TIME_MEASURE):
        time_record_dict = {}
        time_record_dict["test size"]=TEST_SIZE
        time_record_dict["device number"]=torch.distributed.get_rank()
        time_record_dict["phase 1 forward time"]=mean(model.phase_1_forward_time_list[1:])
        time_record_dict["phase 2 forward time"]=mean(model.phase_2_forward_time_list[1:])
        time_record_dict["phase 3 forward time"]=mean(model.phase_3_forward_time_list[1:])
        time_record_dict["backward time"]=mean(backward_time_list[1:])
        time_record_dict["round number"]=len(model.phase_1_forward_time_list)
        time_record_list.append(time_record_dict)



if __name__ == '__main__':

    torch.distributed.init_process_group(backend='nccl')
    world_size = int(os.environ['WORLD_SIZE'])
    print("world size is ",world_size)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    initialize_model_parallel(world_size)
    train(512)
    #train(1024)
    #train(2048)
    #train(4096)
    #train(8192)
    #train(16384)
    print(time_record_list)