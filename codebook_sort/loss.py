import torch
import torch.nn as nn


class CorrelationLoss(nn.Module):
    def __init__(self, distance_matrixA, codebook_num):
        super(CorrelationLoss, self).__init__()
        self.distance_matrixA = distance_matrixA
        self.codebook_num = codebook_num

    def forward(self, index_ls, weight_tensor):

        loss_ls = []
        for i in range(1, len(index_ls)):

            Astar = torch.zeros(1, self.codebook_num - 1)

            for j in range(i):
                Astar[0][j] = self.distance_matrixA[index_ls[i]][index_ls[j]]

            loss_ls.append(torch.matmul(Astar[:, :i], torch.flip(weight_tensor[:, :i], dims=(1,)).t()))

        correlation_loss_tensor = torch.stack(loss_ls, dim=0)
        correlation_loss = -torch.mean(correlation_loss_tensor)

        weight_tensor = weight_tensor[:, :self.codebook_num - 2] - weight_tensor[:, 1:]
        weight_tensor[weight_tensor > 0] = 0
        weight_tensor *= 5
        # print(weight_tensor)
        weight_loss = -torch.sum(weight_tensor)

        return correlation_loss, weight_loss

