import torch


class Generator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = torch.nn.Linear(input_size, hidden_size)
        self.map2 = torch.nn.Linear(hidden_size, hidden_size)
        self.map3 = torch.nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = torch.nn.Linear(input_size, hidden_size)
        self.map2 = torch.nn.Linear(hidden_size, hidden_size)
        self.map3 = torch.nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))
