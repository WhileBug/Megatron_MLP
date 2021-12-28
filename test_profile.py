import os

exec_str = "python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 MLP_profiling.py"
os.system(exec_str)