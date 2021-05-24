import argparse
import os

import torch
from tqdm import tqdm
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from math import pi

from datasets.datasets import Sines, ARMA, Financial_index
from models.wgangp import Generator, Critic

from stats import auto_correlation, acf, plot_leverage_effect, leverage_effect

import numpy as np
import pandas as pd

from datetime import datetime as dt


# 初始化训练器
class Trainer:

    NOISE_LENGTH = 50

    def __init__(self, generator, critic, gen_optimizer, critic_optimizer, real_acf, real_clustering, real_leverage,
                 gp_weight=10, critic_iterations=5, print_every=100, use_cuda=False, checkpoint_frequency=200):

        self.g = generator                                  # 生成器
        self.g_opt = gen_optimizer                          # 生成器的优化器
        self.c = critic                                     # 判别器
        self.c_opt = critic_optimizer                       # 判别器的优化器
        self.losses = {'g': [], 'c': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda                            # 决定是否使用显卡
        self.gp_weight = gp_weight                          # 梯度惩罚系数
        self.critic_iterations = critic_iterations          # 判别器迭代次数
        self.print_every = print_every                      # 打印间隔
        self.checkpoint_frequency = checkpoint_frequency    # 检查频率

        self.real_acf = real_acf                            # 保存预先计算的真实数据的 acf
        self.real_clustering = real_clustering
        self.real_leverage = np.array(real_leverage).flatten()


        if self.use_cuda:       # 选择是否使用显卡加速
            self.g.cuda()
            self.c.cuda()

    def _critic_train_iteration(self, real_data):

        batch_size = real_data.size()[0]
        noise_shape = (batch_size, self.NOISE_LENGTH)           # 创建同样 batch_size 大小的噪声
        generated_data = self.sample_generator(noise_shape)     # 生成虚假序列

        real_data = Variable(real_data)                         # 将真实序列打包成 tensor

        if self.use_cuda:                                       # 选择是否使用显卡加速
            real_data = real_data.cuda()

        # 用 discriminator 处理真实及虚假序列
        c_real = self.c(real_data)                              # 将真实数据放至评价者模型中
        c_generated = self.c(generated_data)                    # 将虚假序列放至评价者模型中


        gradient_penalty = self._gradient_penalty(real_data, generated_data)    # 计算梯度惩罚项    (重点！ WGan-GP 的核心)
        self.losses['GP'].append(gradient_penalty.data.item())

        # Create total loss and optimize
        self.c_opt.zero_grad()                                                  # 预先将优化器中的梯度置零
        d_loss = c_generated.mean() - c_real.mean() + gradient_penalty          # 给出 discriminator 的 loss 计算公式 = 生成的虚假序列的均值+真是序列的均值-梯度惩罚项
        d_loss.backward()                                                       # 梯度反向传播
        self.c_opt.step()                                                       # 更新 Critic

        self.losses['c'].append(d_loss.data.item())

    # 定义生成器 g 的训练迭代过程
    def _generator_train_iteration(self, data):
        self.g_opt.zero_grad()                                      # 将 optimizer 中的梯度置零，以便后续训练
        batch_size = data.size()[0]                                 # batch size 等于 data 的行数，既每条数据以行分布
        latent_shape = (batch_size, self.NOISE_LENGTH)              # 确定噪音的行列形态大小

        generated_data = self.sample_generator(latent_shape)        # 生成虚假数据，1、先生成对应形态大小的噪声数据；2、将噪声放入生成器g中生成虚假的金融数据

        # Calculate loss and optimize

        d_generated = self.c(generated_data)                        # 将生成的虚假数据放到判别器中，判别器输出结果 1则判断为真，0则判断其为假样本（我们希望能将生成的判断为1）
        g_loss = - d_generated.mean()                               # 生成器 g 的 loss 就是判别器的结论，我们希望判别器能将我们生成的数据判断为 1
        g_loss.backward()
        self.g_opt.step()
        self.losses['g'].append(g_loss.data.item())

    # 定义 梯度惩罚项
    def _gradient_penalty(self, real_data, generated_data):

        batch_size = real_data.size()[0]                                            # 获取 batch_size

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)                                           # 生成随机权重
        alpha = alpha.expand_as(real_data)                                          # 扩张至真实数据的维度

        if self.use_cuda:
            alpha = alpha.cuda()

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data   # 将真实数据和虚假数据按比例融合成interpolation
        interpolated = Variable(interpolated, requires_grad=True)                   # 将插值打包成 tensor
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # 将按比例插值过后的数据放入判别器中
        prob_interpolated = self.c(interpolated)  # 将插值放入判别器中，计算插值概率


        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,      # 计算经过判别器之后结果 prob_interpolated 的梯度
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda
                               else torch.ones(prob_interpolated.size()), create_graph=True,
                               retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, series length),
        # here we flatten to take the norm per example for every batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data.item())

        # Derivatives of the gradient close to 0 can cause problems because of the
        # square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)       # 计算梯度惩罚项，为了避免求导后梯度为零，加上一个 epsilon

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()                  # 返回梯度惩罚项

    def _train_epoch(self, data_loader, epoch):
        for i, data in enumerate(data_loader):
            self.num_steps += 1
            self._critic_train_iteration(data.float())

            if self.num_steps % self.critic_iterations == 0:    # 每训练5次判别器后才训练一次生成器
                self._generator_train_iteration(data)

            if i % self.print_every == 0:                       # 每 print_every 记录一次信息
                global_step = i + epoch * len(data_loader.dataset)
                writer = SummaryWriter()
                writer.add_scalar('Losses/Critic', self.losses['c'][-1], global_step)
                writer.add_scalar('Losses/Gradient Penalty', self.losses['GP'][-1], global_step)
                writer.add_scalar('Gradient Norm', self.losses['gradient_norm'][-1], global_step)

                '''
                只有在第一次更新生成器的时候才开始记录
                '''
                if self.num_steps > self.critic_iterations:
                    writer.add_scalar('Losses/Generator', self.losses['g'][-1], global_step)

    def train(self, data_loader, epochs, plot_training_samples=True, checkpoint=None):

        if checkpoint:                                                                  # 决定是否要读取之前的模型结果，可以从之前某个参数开始继续训练
            path = os.path.join('checkpoints', checkpoint)
            state_dicts = torch.load(path, map_location=torch.device('cpu'))
            self.g.load_state_dict(state_dicts['g_state_dict'])                         # 读入生成器的参数
            self.c.load_state_dict(state_dicts['d_state_dict'])                         # 读入判别器的参数
            self.g_opt.load_state_dict(state_dicts['g_opt_state_dict'])                 # 读取生成器的最优参数
            self.c_opt.load_state_dict(state_dicts['d_opt_state_dict'])                 # 读取判别器的最优参数

        # if plot_training_samples:
        #     # Fix latents to see how series generation improves during training
        #     noise_shape = (1, self.NOISE_LENGTH)
        #     fixed_latents = Variable(self.sample_latent(noise_shape))
        #     if self.use_cuda:
        #         fixed_latents = fixed_latents.cuda()

        acf_mean = []
        clustering_mean = []
        leverage_mean = []

        levs_input_dim = []

        for epoch in tqdm(range(epochs)):

            # Sample a different region of the latent distribution to check for mode collapse
            noise_shape = (1, self.NOISE_LENGTH)
            dynamic_latents = Variable(self.sample_latent(noise_shape))
            if self.use_cuda:
                dynamic_latents = dynamic_latents.cuda()

            self._train_epoch(data_loader, epoch + 1)               # 每个epoch都训练判别器

            # 保存模型
            if epoch % self.checkpoint_frequency == 0:
                torch.save({
                    'epoch': epoch,
                    'd_state_dict': self.c.state_dict(),
                    'g_state_dict': self.g.state_dict(),
                    'd_opt_state_dict': self.c_opt.state_dict(),
                    'g_opt_state_dict': self.g_opt.state_dict(),
                }, 'Model_saved/%s/epoch_%i.pkl'% (timestamp, epoch))

            if plot_training_samples and (epoch % self.print_every == 0):
                self.g.eval()

                # 生成虚假序列
                fake_data_dynamic_latents = self.g(dynamic_latents).cpu().data


                if plot_training_samples:                                                               # 画出生成的序列
                    plt.figure(figsize=(10, 8))
                    plt.title('Generated Financial Series', fontsize='xx-large')
                    plt.xlabel('Time',fontsize='xx-large')
                    plt.ylabel('Price',fontsize='xx-large')
                    plt.plot(fake_data_dynamic_latents.numpy()[0].T)
                    plt.savefig('./Results_img/%s/Generated_series/Generated_series_epoch_%i.png'%(timestamp, epoch))
                    plt.savefig('./Results_img/%s/Generated_series/Generated_series_epoch_%i.svg'%(timestamp, epoch))
                    plt.close()

#######################################################################################################################################################################

                '''生成 5488 条虚假数据，并分别计算前10阶的acf，与真实数据作比较'''

                fake_data_record = []       # 用来保存生成的虚假序列

                for i in range(len(self.real_acf)):
                    fake_data_record.append(self.g(torch.randn(noise_shape).cuda()).cpu().data.numpy().flatten())


                fake_data_record = np.array(fake_data_record).T             # 将生成的数据转置成 252 x 5489 的格式，方便后续计算 log return
                fake_data_log_return = np.log(fake_data_record[1:]) - np.log(fake_data_record[:-1])     # 计算 log return
                fake_acf = []       # 用来保存每条生成数据的 log return

                for fake in tqdm(fake_data_log_return.copy().T):
                    fake_acf.append(np.mean(acf(fake, max_lag=10)))


                # for _ in range(len(self.real_acf)):
                #     fake_test = self.g(torch.randn(noise_shape).cuda()).cpu().data  # 获取生成的虚假数据
                #     fake_np = fake_test.cpu().detach().numpy().flatten()            # 转化成一维的numpy
                #     fake_log_return = np.log(fake_np[1:])-np.log(fake_np[:-1])      # 计算 log return
                #     fake_acf.append(np.mean(acf(fake_log_return, max_lag=30)))      # 计算自相关系数 acf


                # 画出生成的 5488 条数据的 acf 与真实数据 acf 之间的对比
                print('现在开始画生成数据与真实数据 自相关系数 之间的对比图 \n')
                plt.figure(figsize=(10, 8))
                plt.title('ACF on Real vs Generated', fontsize='xx-large')
                plt.boxplot([self.real_acf, fake_acf], labels=['Real', 'Generated'])
                plt.savefig('./Results_img/%s/ACF_vs_real/fake_acf_vs_real_epoch_%i'%(timestamp, epoch)+'.png',transparent=False)
                plt.savefig('./Results_img/%s/ACF_vs_real/fake_acf_vs_real_epoch_%i'%(timestamp, epoch)+'.svg',transparent=False)
                plt.close()

                acf_mean.append([epoch, np.mean(fake_acf)])

                # #############################################################################################################################################################################

                fake_cluster = []       # 用来保存每条生成数据的 波动率

                for fake in tqdm(fake_data_log_return.copy().T):
                    fake_cluster.append(np.mean(acf(np.abs(fake), max_lag=10)))

                # 画出生成的 5488 条数据的生成数据与真实数据 波动率 之间的对比
                print('现在开始画生成数据与真实数据 波动率聚集 之间的对比图 \n')
                plt.figure(figsize=(10, 8))
                plt.title('Volatility Clustering on Real vs Generated', fontsize='xx-large')
                plt.boxplot([self.real_clustering, fake_cluster], labels=['Real', 'Generated'])
                plt.savefig('./Results_img/%s/Volatility_vs_real/fake_volatility_clustering_vs_real_epoch_%i'%(timestamp, epoch)+'.png',transparent=False)
                plt.savefig('./Results_img/%s/Volatility_vs_real/fake_volatility_clustering_vs_real_epoch_%i'%(timestamp, epoch)+'.svg',transparent=False)
                plt.close()

                clustering_mean.append([epoch, np.mean(fake_cluster)])

                #############################################################################################################################################################################

                fake_leverage = []       # 用来保存每条生成数据的 杠杆值

                for fake in tqdm(fake_data_log_return.copy().T):
                    fake_leverage.append(np.mean(leverage_effect(fake, min_lag=1, max_lag=10)))

                # 画出生成的 5488 条数据的生成数据与真实数据 波动率 之间的对比
                print('现在开始画生成数据与真实数据 杠杆效应 之间的对比图 \n')

                plt.figure(figsize=(10, 8))
                plt.title('Leverage effect on Real vs Generated', fontsize='xx-large')
                plt.boxplot([self.real_leverage, fake_leverage], labels=['Real', 'Generated'])
                plt.savefig('./Results_img/%s/Leverage_vs_real/fake_Leverage_vs_real_epoch_%i'%(timestamp, epoch)+'.png',transparent=False)
                plt.savefig('./Results_img/%s/Leverage_vs_real/fake_Leverage_vs_real_epoch_%i'%(timestamp, epoch)+'.svg',transparent=False)
                plt.close()

                leverage_mean.append([epoch, np.mean(fake_leverage)])

                #############################################################################################################################################################################

                # 自相关系数 acf
                # 每200个epoch我们计算一次ACF，自相关系数，并且与真实数据对比
                # 首先计算生成序列的log return

                fake_test = fake_data_dynamic_latents                           # 获取生成的虚假数据
                fake_np = fake_test.cpu().detach().numpy().flatten()            # 转化成一维的numpy
                fake_log_return = np.log(fake_np[1:])-np.log(fake_np[:-1])      # 计算 log return
                fake_acf = acf(fake_log_return.copy(), max_lag=101)                    # 计算自相关系数 acf

                # 自相关系数画图函数
                plt.figure(figsize=(10, 8))
                plt.plot(range(len(fake_acf)),fake_acf.copy(),'.')
                plt.ylim(-1.,1.)
                plt.xscale('log')
                plt.yscale('linear')
                plt.title('Linear Unpredictability', fontsize='xx-large')
                plt.xlabel('Lag $k$',fontsize='xx-large')
                plt.ylabel('Auto-correlation',fontsize='xx-large')
                plt.savefig('./Results_img/%s/ACF/fake_acf_%i'%(timestamp, epoch)+'.png',transparent=False)
                plt.savefig('./Results_img/%s/ACF/fake_acf_%i'%(timestamp, epoch)+'.svg',transparent=False)
                plt.close()

                ###############################################################################################################################################################

                # 杠杆效应
                # 同样也是200个epoch计算一次
                # 数据要重新reshape一下
                # 这个函数自动调用了画图函数
                fake_return_lev = np.log(fake_np[1:]) - np.log(fake_np[:-1])

                min_lag = 1
                max_lag = 100
                levs = leverage_effect(fake_return_lev, min_lag=min_lag, max_lag=max_lag)

                # 画出杠杆效应的图
                plot_leverage_effect([i for i in range(min_lag, max_lag)], levs, './Results_img/%s/Leverage/fake_leverage_%i'%(timestamp, epoch)+'.png')
                plot_leverage_effect([i for i in range(min_lag, max_lag)], levs, './Results_img/%s/Leverage/fake_leverage_%i'%(timestamp, epoch)+'.svg')
                levs_input_dim.append(levs)


                # 波动率聚集
                fake_return_cluster = np.log(fake_np[1:]) - np.log(fake_np[:-1])
                fake_acf_2 = acf(np.abs(fake_return_cluster), max_lag=101)
                plt.figure(figsize=(10, 8))
                plt.plot(range(len(fake_acf_2)),fake_acf_2,'.')
                plt.xscale('log')
                plt.yscale('log')
                plt.title('Volatility Clustering', fontsize='xx-large')
                plt.xlabel('Lag $k$',fontsize='xx-large')
                plt.ylabel('Auto-correlation',fontsize='xx-large')
                plt.savefig('./Results_img/%s/Votality_clustering/Votality_clustering_epoch_%i'%(timestamp, epoch)+'.png',transparent=False)
                plt.savefig('./Results_img/%s/Votality_clustering/Votality_clustering_epoch_%i'%(timestamp, epoch)+'.svg',transparent=False)
                plt.close()

                # 要注意把g.train放后面
                self.g.train()              # 训练生成器

                plt.figure(figsize=(10, 8))
                plt.title('Generator Loss', fontsize='xx-large')
                plt.ylabel('Loss', fontsize='xx-large')
                plt.xlabel('Epoch', fontsize='xx-large')
                plt.plot(range(1, len(self.losses['g'])+1), self.losses['g'])
                plt.savefig('./Results_img/%s/Loss/G_Loss_%i'%(timestamp, epoch)+'.png',transparent=False)
                plt.savefig('./Results_img/%s/Loss/G_Loss_%i'%(timestamp, epoch)+'.svg',transparent=False)
                plt.close()

                plt.figure(figsize=(10, 8))
                plt.title('Critic Loss', fontsize='xx-large')
                plt.ylabel('Loss', fontsize='xx-large')
                plt.xlabel('Epoch', fontsize='xx-large')
                plt.plot(range(1, len(self.losses['c'])+1), self.losses['c'])
                plt.savefig('./Results_img/%s/Loss/C_Loss_%i'%(timestamp, epoch)+'.png',transparent=False)
                plt.savefig('./Results_img/%s/Loss/C_Loss_%i'%(timestamp, epoch)+'.svg',transparent=False)
                plt.close()

        np.savetxt('./Results_img/%s/acf_mean.txt'%(timestamp), acf_mean, fmt="%.2f", delimiter='      ')
        np.savetxt('./Results_img/%s/cluster_mean.txt'%(timestamp), clustering_mean, fmt="%.2f", delimiter='       ')
        np.savetxt('./Results_img/%s/leverage_mean.txt'%(timestamp), leverage_mean, fmt="%.2f", delimiter='       ')

        np.savetxt('./Results_img/%s/input_dim.txt'%(timestamp), levs_input_dim, fmt="%.2f", delimiter='    ')


    def sample_generator(self, latent_shape):                           # 利用生成器生成序列
        latent_samples = Variable(self.sample_latent(latent_shape))     # 首先根据传入的噪声维度生成随机噪声，并将其打包成 tensor

        if self.use_cuda:
            latent_samples = latent_samples.cuda()

        return self.g(latent_samples)                                   # 返回生成器根据噪声生成的时间序列

    @staticmethod
    def sample_latent(shape):                                           # 生成对应形态的随机噪声
        return torch.randn(shape)

    # def sample(self, num_samples):
    #     generated_data = self.sample_generator(num_samples)
    #     return generated_data.data.cpu().numpy()




if __name__ == '__main__':


    # 生成时间戳
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    timestamp += '_'
    # timestamp += args.folder_name

    os.makedirs('./Model_saved/%s' % (timestamp))                       # 保存模型

    os.makedirs('./Results_img/%s/Loss' % (timestamp))                   # 自相关系数
    os.makedirs('./Results_img/%s/ACF' % (timestamp))                   # 自相关系数
    os.makedirs('./Results_img/%s/Votality_clustering' % (timestamp))   # 波动率聚集
    os.makedirs('./Results_img/%s/Generated_series' % (timestamp))      # 生成序列
    os.makedirs('./Results_img/%s/Leverage' % (timestamp))              # 杠杆效应

    os.makedirs('./Results_img/%s/ACF_vs_real' % (timestamp))           # 自相关系数与真实值对比
    os.makedirs('./Results_img/%s/Volatility_vs_real' % (timestamp))    # 波动率聚集与真实值对比
    os.makedirs('./Results_img/%s/Leverage_vs_real' % (timestamp))      # 杠杆效应与真实值对比

    parser = argparse.ArgumentParser(prog='GANetano', usage='%(prog)s [options]')
    parser.add_argument('-ds', '--dataset', type=str, dest='dataset', default='fin',
                        help='choose between sines, arma and real_fin data')
    parser.add_argument('-ln', '--logname', type=str, dest='log_name', default=None, required=False,
                        help='tensorboard filename')
    parser.add_argument('-e', '--epochs', type=int, dest='epochs', default=10000, help='number of training epochs')
    parser.add_argument('-bs', '--batches', type=int, dest='batches', default=16,
                        help='number of batches per training iteration')
    parser.add_argument('-cp', '--checkpoint', type=str, dest='checkpoint', default=None,
                        help='checkpoint to use for a warm start')

    args = parser.parse_args(args=[])



    # 初始化生成器
    g = Generator()
    g_opt = torch.optim.RMSprop(g.parameters(), lr=0.00005)

    # 初始化判别器
    d = Critic()
    d_opt = torch.optim.RMSprop(d.parameters(), lr=0.00005)

    # 创建 DataLoader
    if args.dataset == 'fin':
        dataset = Financial_index()
    elif args.dataset == 'sins':
        dataset = Sines(frequency_range=[0, 2 * pi], amplitude_range=[0, 2 * pi], seed=42, n_series=200)

    dataloader = DataLoader(dataset, batch_size=args.batches)

    ################################################################################

    real_data = Financial_index().dataset       # 读取原始数据（已经做了rolling）

    # 计算真实数据的 ACF
    print('正在计算真实数据的自相关系数...\n')

    real_log_return = np.log(real_data.T[1:]) - np.log(real_data.T[:-1])
    real_acf = []

    for real in tqdm(real_log_return.copy().T):
        real_acf.append(np.mean(acf(real, max_lag=10)))

# # ################################################################################

# # 计算真实数据的 波动率聚集系数
    print('正在计算真实数据的波动率聚集...\n')

    real_clustering = []

    for real in tqdm(real_log_return.copy().T):
        real_clustering.append(np.mean(acf(np.abs(real), max_lag=10)))

####################################################################################

# 计算真实数据前 10 阶杠杆系数的值
    print('正在计算真实数据的杠杆系数的值...\n')

    real_leverage = []

    for real in tqdm(real_log_return.copy().T):
        real_leverage.append(leverage_effect(real, min_lag=1, max_lag=10))

####################################################################################

    np.savetxt('acf_real_mean.txt', [np.mean(real_acf)], fmt="%.5f")
    np.savetxt('cluster_real_mean.txt', [np.mean(real_clustering)], fmt="%.5f")
    np.savetxt('leverage_real_mean.txt', [np.mean(real_leverage)], fmt="%.5f")


    ###################################################################################


    # 自相关系数 acf
    real_acf_100 = acf(real_log_return, max_lag=101)             # 计算自相关系数 acf

    # 自相关系数画图函数
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(real_acf_100)), real_acf_100, '.')
    plt.ylim(-1.,1.)
    plt.xscale('log')
    plt.yscale('linear')
    plt.title('Linear Unpredictability', fontsize='xx-large')
    plt.xlabel('Lag $k$',fontsize='xx-large')
    plt.ylabel('Auto-correlation',fontsize='xx-large')
    plt.savefig('./Results_img/%s/real_acf'%(timestamp)+'.png',transparent=False)
    plt.savefig('./Results_img/%s/real_acf'%(timestamp)+'.svg',transparent=False)
    plt.close()

    ###############################################################################################################################################################

    # 杠杆效应
    min_lag = 1
    max_lag = 100
    real_levs_100 = leverage_effect(real_log_return, min_lag=min_lag, max_lag=max_lag)
    plot_leverage_effect([i for i in range(min_lag, max_lag)], real_levs_100, './Results_img/%s/real_leverage'%(timestamp)+'.png')
    plot_leverage_effect([i for i in range(min_lag, max_lag)], real_levs_100, './Results_img/%s/real_leverage'%(timestamp)+'.svg')


    # 波动率聚集
    real_volatility_100 = acf(np.abs(real_log_return), max_lag=101)
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(real_volatility_100)), real_volatility_100, '.')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Volatility Clustering', fontsize='xx-large')
    plt.xlabel('Lag $k$',fontsize='xx-large')
    plt.ylabel('Auto-correlation',fontsize='xx-large')
    plt.savefig('./Results_img/%s/real_Votality_clustering'%(timestamp)+'.png',transparent=False)
    plt.savefig('./Results_img/%s/real_Votality_clustering'%(timestamp)+'.svg',transparent=False)
    plt.close()


    # 新建一个训练器的实例
    trainer = Trainer(g, d, g_opt, d_opt, real_acf, real_clustering, real_leverage, use_cuda=torch.cuda.is_available())
    # Train model
    print('现在开始训练！')
    # Instantiate Tensorboard writer
    tb_logdir = os.path.join('..', 'tensorboard', args.log_name)
    writer = SummaryWriter(log_dir=tb_logdir)

    trainer.train(dataloader, epochs=args.epochs, plot_training_samples=True, checkpoint=args.checkpoint)