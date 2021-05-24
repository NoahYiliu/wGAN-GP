from torch import nn
from torch.nn.utils import spectral_norm


class AddDimension(nn.Module):
    def forward(self, x):
        # 这里把第2维变成1
        return x.unsqueeze(1)


class SqueezeDimension(nn.Module):
    def forward(self, x):
        # 这里把第2维去掉
        return x.squeeze(1)


def create_generator_architecture():
    return nn.Sequential(nn.Linear(50, 100),
                         nn.LeakyReLU(0.2, inplace=True),
                         # 变成100行一列的数据
                         AddDimension(),
                         spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                         nn.Upsample(200),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Upsample(400),

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Upsample(800),

                         spectral_norm(nn.Conv1d(32, 1, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),

                         SqueezeDimension(),
                         nn.Linear(800, 252)
                         )


def create_critic_architecture():
    # 如果是252长度的输入，通过一次卷积，池化我们维度会缩小一半
    return nn.Sequential(AddDimension(),
                         spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2), # 126

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2), # 63
                         
                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.MaxPool1d(2), # 31

                         spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Flatten(),
                            
                         # 这里如果输入长度100的话，最后flatten完长度是31*32
                         
                         nn.Linear(992, 50),
                         nn.LeakyReLU(0.2, inplace=True),

                         nn.Linear(50, 15),
                         nn.LeakyReLU(0.2, inplace=True),

                         nn.Linear(15, 1)
                         )


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = create_generator_architecture()

    def forward(self, input):
        return self.main(input)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = create_critic_architecture()

    def forward(self, input):
        return self.main(input)
