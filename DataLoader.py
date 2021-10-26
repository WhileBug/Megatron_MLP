import torch

class FakeDataLoader:
    def __init__(self, shape, num=640):
        self.shape = shape
        self.length = num
        self.pos = 0
    def __iter__(self):
        self.pos = 0
        return self
    def __next__(self):
        self.pos += 1
        if self.pos == self.length:
            raise StopIteration 
        return torch.randn(self.shape).cuda()