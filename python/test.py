import numpy as np

# 构建无向带权图
W = np.array([[0, 1, 1, 0, 0, 0],
              [1, 0, 1, 1, 0, 0],
              [1, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 1],
              [0, 0, 0, 1, 0, 1],
              [0, 0, 0, 1, 1, 0]])
n = len(W)

# 计算度数矩阵
D = np.diag(np.sum(W, axis=1))

# 计算邻接矩阵
A = W + W.T

# 计算模块度
k = np.sum(W)
q = np.sum(A) / (2 * k)
S = np.outer(np.sum(W, axis=1), np.sum(W, axis=1)) / (2 * k)
Q = np.sum((W - S) * A) / (2 * k)

# Louvain算法
C = np.arange(n)  # 初始时每个节点为一个社区
while True:
    C_old = np.copy(C)
    for i in range(n):
        dQ = np.zeros(n)
        for j in range(n):
            if C[i] == C[j]:
                dQ[j] = (W[i, j] - S[i, j]) / k - 2 * A[i, j] / (2 * k)**2
            else:
                dQ[j] = (W[i, j] - S[i, j]) / k - A[i, j] / (2 * k)**2
        C[i] = np.argmax(dQ)
    if np.array_equal(C, C_old):
        break

# 输出结果
print('Final partition:', C)


import networkx as nx
import community


community.best_partition()