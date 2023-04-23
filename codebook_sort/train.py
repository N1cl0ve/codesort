import torch
import torch.nn as nn
import random
from scipy.spatial.distance import cdist
from network import MLP
from loss import CorrelationLoss

pth_file_path = r'.\codebook\256_codebook.pth'  # 码书路径
codebook = torch.load(pth_file_path)
codebook_num = codebook.shape[0]
codebook = codebook.numpy()
distance_matrixA = cdist(codebook, codebook)  # 距离矩阵A
distance_matrixA = torch.from_numpy(distance_matrixA)

model = MLP(distance_matrixA, codebook_num)
loss_fn = CorrelationLoss(distance_matrixA, codebook_num)
optimzier = torch.optim.SGD(model.parameters(), lr=0.02)

num_epoch = 20

for epoch in range(num_epoch):

    index_ls = [247]
    #  random.seed(epoch)
    #  index_ls = [random.randint(0, codebook_num - 1)]
    #  print('Now start with random index: {}'.format(index_ls[-1]))

    output_index, weight_tensor = model(index_ls)

    correlation_loss, weight_loss = loss_fn(index_ls=output_index, weight_tensor=weight_tensor)
    print(-correlation_loss, weight_loss)

    loss = correlation_loss + weight_loss
    loss.backward()

    optimzier.step()

    model.input_layer.update_weight(weight_tensor)

    optimzier.zero_grad()


weight_tensor = next(model.output_layer.parameters())
print(weight_tensor)

