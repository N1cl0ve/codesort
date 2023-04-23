import torch
import torch.nn as nn
import random


class MLP(nn.Module):
    def __init__(self, distance_matrixA, codebook_num,):
        super(MLP, self).__init__()
        self.input_layer = MyLinear(distance_matrixA, codebook_num)
        self.Linear = MyLinear(distance_matrixA, codebook_num)
        self.hidden_layers = nn.ModuleList([MyLinear(distance_matrixA, codebook_num) for _ in range(codebook_num - 3)])
        self.output_layer = MyLinear(distance_matrixA, codebook_num)

    def forward(self, index_ls):
        index_ls = self.input_layer(index_ls)
        weight_tensor = next(self.input_layer.parameters())
        # print(weight_tensor, weight_tensor.shape)

        for hidden_layer in self.hidden_layers:
            hidden_layer.update_weight(weight_tensor)
            index_ls = hidden_layer(index_ls)
            weight_tensor = next(hidden_layer.parameters())

        self.output_layer.update_weight(weight_tensor)

        out_index = self.output_layer(index_ls)
        weight_tensor = next(self.output_layer.parameters())

        return out_index, weight_tensor


class MyLinear(nn.Module):
    def __init__(self, distance_matrixA, codebook_num):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.rand(1, codebook_num - 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.distance_matrixA = distance_matrixA
        self.codebook_num = codebook_num

    def update_weight(self, weight_tensor):

        self.weight = weight_tensor

        return

    def get_Astar(self, index_ls):

        hsize = self.codebook_num - len(index_ls)
        wsize = len(index_ls)
        Astar = torch.zeros(hsize, wsize)
        Astar_index = []

        for i in range(self.codebook_num):
            if i in index_ls:
                continue
            Astar_index.append(i)

        for i in range(hsize):
            for j in range(wsize):
                Astar[i][j] = self.distance_matrixA[Astar_index[i]][index_ls[j]]

        return Astar, Astar_index

    def forward(self, index_ls):

        if len(index_ls) == self.codebook_num:

            return index_ls

        else:
            Astar, Astar_index = self.get_Astar(index_ls)
            # print(Astar.shape, self.weight.shape)

            correlation = torch.matmul(Astar, torch.flip(self.weight[:, :len(index_ls)], dims=(1,)).t()) + self.bias
            # if len(index_ls) == self.codebook_num - 1:
            #     print(correlation.shape, correlation)
            correlation = torch.relu(correlation)

            # print(correlation.shape)

            correlation[correlation == 0] = float("inf")
            out = torch.argmin(correlation)

            index_ls.append(Astar_index[out])

            return index_ls
