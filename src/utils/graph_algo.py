import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

def normalize_adj_mx(adj_mx, adj_type, return_type='dense'):
    if adj_type == 'normlap':           # 归一化拉普拉斯矩阵 L = I - D^(-1/2) * A * D^(-1/2)
        adj = [calculate_normalized_laplacian(adj_mx)]
    elif adj_type == 'scalap':          # 缩放拉普拉斯矩阵 缩放使其范围为[-1,1] (最大特征值约为2)    L' = 2 * D^(-1/2) * A * D^(-1/2) - I
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adj_type == 'symadj':          #  对称归一化邻接矩阵   A' = D^(-1/2) * A * D^(-1/2)
        adj = [calculate_sym_adj(adj_mx)]
    elif adj_type == 'transition':          # 转移矩阵  P = D^(-1) * A
        adj = [calculate_asym_adj(adj_mx)]
    elif adj_type == 'doubletransition':    # 双向转移矩阵  P = D^(-1) * A, P' = D^(-1) * A^T
        adj = [calculate_asym_adj(adj_mx), calculate_asym_adj(np.transpose(adj_mx))]
    elif adj_type == 'identity':            # 单位矩阵 I
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        return []
    
    if return_type == 'dense':  # 转换为稠密矩阵
        adj = [a.astype(np.float32).todense() for a in adj]
    elif return_type == 'coo':  # 转换为coo稀疏矩阵 (row_index, col_index, value)
        adj = [a.tocoo() for a in adj]
    return adj


def calculate_normalized_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = sp.eye(adj_mx.shape[0]) - d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt).tocoo()
    return res


def calculate_scaled_laplacian(adj_mx, lambda_max=None, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)    # 转换为压缩稀疏行矩阵
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    res = (2 / lambda_max * L) - I
    return res


def calculate_sym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = d_mat_inv_sqrt.dot(adj_mx).dot(d_mat_inv_sqrt)
    return res


def calculate_asym_adj(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    rowsum = np.array(adj_mx.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    res = d_mat_inv.dot(adj_mx)
    return res


def calculate_cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L.copy()]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[i - 1]) - LL[i - 2])
    return np.asarray(LL)