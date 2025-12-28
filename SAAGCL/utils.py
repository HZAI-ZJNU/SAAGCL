import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import dgl


def sparse_to_tuple_rep(sparse_matrix, add_batch_dim=False):
    """Convert sparse matrix to tuple representation (coords, values, shape)."""
    """Set add_batch_dim=True if you want to insert a batch dimension."""

    def matrix_to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if add_batch_dim:
            coordinates = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values_data = mx.data
            matrix_shape = (1,) + mx.shape
        else:
            coordinates = np.vstack((mx.row, mx.col)).transpose()
            values_data = mx.data
            matrix_shape = mx.shape
        return coordinates, values_data, matrix_shape

    if isinstance(sparse_matrix, list):
        for i in range(len(sparse_matrix)):
            sparse_matrix[i] = matrix_to_tuple(sparse_matrix[i])
    else:
        sparse_matrix = matrix_to_tuple(sparse_matrix)

    return sparse_matrix


def preprocess_features(input_features):
    """Row-normalize feature matrix and convert to tuple representation"""
    row_sum = np.array(input_features.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    norm_features = r_mat_inv.dot(input_features)
    return norm_features.todense(), sparse_to_tuple_rep(norm_features)


def normalize_adj(adj_matrix):
    """Symmetrically normalize adjacency matrix."""
    adj_matrix = sp.coo_matrix(adj_matrix)
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(label_vector, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = label_vector.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + label_vector.ravel()] = 1
    return labels_one_hot


def load_mat(dataset_name, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data_dict = sio.loadmat("./Data/{}.mat".format(dataset_name))
    class_labels = data_dict['Label'] if ('Label' in data_dict) else data_dict['gnd']
    features_attr = data_dict['Attributes'] if ('Attributes' in data_dict) else data_dict['X']
    network_data = data_dict['Network'] if ('Network' in data_dict) else data_dict['A']

    adj_mat = sp.csr_matrix(network_data)
    feat_mat = sp.lil_matrix(features_attr)

    anomaly_status = np.squeeze(np.array(class_labels))

    return adj_mat, feat_mat, anomaly_status


def adj_to_dgl_graph(adj_matrix):
    """Convert adjacency matrix to dgl format."""
    if hasattr(nx, 'from_scipy_sparse_array'):
        nx_graph = nx.from_scipy_sparse_array(adj_matrix)
    else:
        nx_graph = nx.from_scipy_sparse_matrix(adj_matrix)
    dgl_g = dgl.DGLGraph(nx_graph)
    return dgl_g


def generate_rwr_subgraph(dgl_graph, required_size):
    """Generate subgraph with RWR algorithm."""
    all_indices = list(range(dgl_graph.number_of_nodes()))
    reduced_size = required_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_indices, restart_prob=1,
                                                           max_nodes_per_seed=required_size * 3)

    subgraph_nodes_list = []

    for i, trace in enumerate(traces):
        unique_nodes = torch.unique(torch.cat(trace), sorted=False).tolist()
        subgraph_nodes_list.append(unique_nodes)

        retry_counter = 0
        while len(subgraph_nodes_list[i]) < reduced_size:
            current_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                          max_nodes_per_seed=required_size * 5)

            subgraph_nodes_list[i] = torch.unique(torch.cat(current_trace[0]), sorted=False).tolist()
            retry_counter += 1
            if (len(subgraph_nodes_list[i]) <= 2) and (retry_counter > 10):
                subgraph_nodes_list[i] = (subgraph_nodes_list[i] * reduced_size)

        subgraph_nodes_list[i] = subgraph_nodes_list[i][:reduced_size]
        subgraph_nodes_list[i].append(i)

    return subgraph_nodes_list
