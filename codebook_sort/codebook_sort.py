import torch
from scipy.spatial.distance import cdist
import numpy as np


def dp_loss_version(distance_matrix_A, start=0):  # 排序函数

    index_ls = []
    index_ls.append(start)  # 随机初始点，由start确定
    loss_for_all = 0.0
    ####################### 优化有点问题
    # average_array = calculate_average_distance_for_sort(distance_matrix_A)

    for k in range(1, codebook.shape[0]):

        loss_for_k = []
        # S_for_k = []
        # loss_with_S = []

        for index, distance in enumerate(distance_matrix_A[index_ls[-1]]):

            if index in index_ls:
                continue
            # if distance >= average_array[index_ls[-1]]:
            #     S_for_k.append(index)
            #     continue

            # print('index {} start calculate loss'.format(index))
            loss = sum([1/((k-i)**2)*distance_matrix_A[index][index_ls[i]] for i in range(k)])
            loss_for_k.append((index, loss))

        # if loss_for_k is None:
        #     for index in S_for_k:
        #         loss = sum([1/((k-i)**2)*distance_matrix_A[index][index_ls[i]] for i in range(k)])
        #         loss_with_S.append((index, loss))

        # loss_for_k = loss_with_S

        argmin_index = np.array(loss_for_k).argmin(axis=0)[1]
        index_ls.append(loss_for_k[argmin_index][0])
        loss_for_all += loss_for_k[argmin_index][1]

    index_ls.append(loss_for_all)  # 这里排好序后会把总Loss加入列表末尾

    return index_ls


def calculate_average_distance_for_sort(distance_matrix_A):

    len = distance_matrix_A.shape[1] - 1
    average_array = np.sum(distance_matrix_A, axis=1)
    average_array = average_array/len
    # print(average_array)

    return average_array


def calc_loss(ls, distance_matrix_A):
    loss = sum([1/(i-j)*distance_matrix_A[ls[i]][ls[j]] for i in range(len[ls]) for j in range(i)])
    return loss

pth_file_path = r'C:\Users\NiclovePC\Desktop\256_codebook.pth'  # 码书路径
codebook = torch.load(pth_file_path)
codebook = codebook.numpy()
distance_matrix_A = cdist(codebook, codebook)  # 距离矩阵A
# print(distance_matrix_A)

ls = []

for i in range(codebook.shape[0]):  # 把码书中所有的点都当成初始点一次，结果存在一个列表里
    ls.append(dp_loss_version(distance_matrix_A, i))
    print('finish epoch:{}'.format(i))

# ls.append(dp_loss_version(distance_matrix_A, 226))

best_sort_index = np.array(ls).argmin(axis=0)[-1]
print(ls[best_sort_index][:-1])  # 从不同初始点情况中找到一个Loss最小的序列，返回序列，不返回Loss
print(ls[best_sort_index][-1])
