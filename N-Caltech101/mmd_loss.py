import torch
import torch.nn as nn


class MMDLoss(nn.Module):

    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def forward(self, source, target):
        return DAN(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                   fix_sigma=self.fix_sigma)


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    k = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = (k[:n, :n].triu(1).sum() + k[n:, n:].triu(1).sum()) / (n * (n - 1) / 2) \
           - (torch.sum(k[:n, n:]) * 2) / (n * n)

    return loss