import time
from operator import itemgetter
import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1234)
np.random.seed(3456)
torch.set_printoptions(precision=3, sci_mode=False)


####

data_size = 1000
train_split = 0.8
b_size = 64
state_shape = (3, 80, 80)
n_actions = 5
training_steps = 100
lr = 0.01

####


class DataLoader():

    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets
        self.size = len(inputs)
        self.stream = np.arange(self.size)
        np.random.shuffle(self.stream)
        self.head = 0
        self.active = True
    
    def reset(self):
        np.random.shuffle(self.stream)
        self.head = 0
        self.active = True
    
    def get_batch(self, n):
        if not self.active:
            raise ValueError('DataLoader is not active; try reset.')
        
        b = self.stream[self.head: self.head + n]
        self.head += n
        if self.head >= self.size:
            self.active = False
        
        return torch.stack(itemgetter(*b)(self.inputs)), torch.stack(itemgetter(*b)(self.targets))


class Net(nn.Module):

    def __init__(self, state_shape, n_actions):
        super().__init__()

        self.state_shape = [1] + list(state_shape)  # --> (batch, 3, w, h)
        self.n_actions = n_actions

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 8, 1),
            nn.Conv2d(16, 16, 4, 1)
        )

        self.proj = nn.Sequential(
            nn.Linear(self._n_features(), 64),
            nn.GELU(),
            nn.Linear(64, self.n_actions)
        )

        self.features.apply(self._module_init)
        self.proj.apply(self._module_init)
    
    def _n_features(self):
        return self.features(torch.zeros(self.state_shape)).view(-1).size(0)
    
    def _module_init(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0, std=0.2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.proj(x.view(x.size(0), -1))
        return x


def evaluate(model, data_loader, device):
    data_loader.reset()
    losses = []
    while data_loader.active:
        x, y = data_loader.get_batch(512)
        x, y = x.squeeze(1), y.squeeze(1)
        x, y = x.to(device), y.to(device)
        q = model(x)
        loss = F.smooth_l1_loss(q, y)
        losses.append(loss.item() ** 0.5)
    return np.mean(losses)


inputs = torch.randn(data_size, *state_shape)
inputs = torch.tensor_split(inputs, data_size, dim=0)
targets = torch.ones(data_size, n_actions)
targets = torch.tensor_split(targets, data_size, dim=0)

split = int(data_size * train_split)
inputs_train = inputs[:split]
inputs_val = inputs[split:]
targets_train = targets[:split]
targets_val = targets[split:]

data_loader_train = DataLoader(inputs_train, targets_train)
data_loader_val = DataLoader(inputs_val, targets_val)

model = Net(state_shape, n_actions)


@click.command()
@click.option('--device', '-d', default='cpu', help="'cpu' or 'mps'")
def worker(device):
    model.to(device)
    st = time.time()
    loss = evaluate(model, data_loader_val, device)
    print(f'initial loss: {round(loss, 3)}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for k in range(training_steps):
        if not data_loader_train.active:
            data_loader_train.reset()
        
        x, y = data_loader_train.get_batch(b_size)
        x, y = x.squeeze(1), y.squeeze(1)
        x, y = x.to(device), y.to(device)
        q = model(x)
        loss = F.smooth_l1_loss(q, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if k % 10 == 0:
            loss = evaluate(model, data_loader_val, device)
            print(f'after step: {k} | loss: {round(loss, 3)}')

    loss = evaluate(model, data_loader_val, device)
    print(f'step {training_steps} | loss: {round(loss, 3)}')
    et = time.time()
    process_time = et - st
    print(f"-----\nTime:  {round(process_time, 2)} seconds\n-----")


if __name__ == '__main__':
    worker()
