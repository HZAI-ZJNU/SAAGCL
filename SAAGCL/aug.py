import torch as th
import torch.nn.functional as F
from torch_scatter import scatter_add
from utils import *


def drop_node_features(feature_tensor, drop_prob):
    drop_mask = th.empty((feature_tensor.size(1),),
                         dtype=th.float32).uniform_(0, 1) < drop_prob

    new_feat = feature_tensor.clone()
    new_feat[:, drop_mask] = 0
    return new_feat


def degree_based_edge_masking(adj_tensor, node_indices, sim_matrix, max_degree_val, node_degrees, mask_probability):
    aug_degree = (node_degrees * (1 - mask_probability)).long()

    sim_dist = sim_matrix[node_indices] * adj_tensor[node_indices]

    new_targets = th.multinomial(sim_dist + 1e-12, int(max_degree_val))

    target_indices = th.arange(max_degree_val).unsqueeze(dim=0)

    new_col_indices = new_targets[(target_indices - aug_degree.unsqueeze(dim=1) < 0)]
    new_row_indices = node_indices.repeat_interleave(aug_degree)

    return new_row_indices, new_col_indices


def degree_masking_with_threshold(adj_tensor, node_indices, sim_matrix, max_degree_val, node_degrees, fix_threshold):
    fixed_aug_degree = th.full_like(node_degrees, fill_value=fix_threshold)

    sim_dist = sim_matrix[node_indices] * adj_tensor[node_indices]

    new_targets = th.multinomial(sim_dist + 1e-12, int(max_degree_val))
    target_indices = th.arange(max_degree_val).unsqueeze(dim=0)

    new_col_indices = new_targets[(target_indices - fixed_aug_degree.unsqueeze(dim=1) < 0)]
    new_row_indices = node_indices.repeat_interleave(fixed_aug_degree)

    return new_row_indices, new_col_indices


def get_feature_similarity(embed_set1, embed_set2):
    embed_set1 = F.normalize(embed_set1)
    embed_set2 = F.normalize(embed_set2)
    similarity = th.mm(embed_set1, embed_set2.t())
    return similarity


def neighbor_mixup(source_indices, dest_indices, adj_dist, sim_matrix,
                   max_degree_val, augmentation_degree, device_id):
    phi_coeff = sim_matrix[source_indices, dest_indices].unsqueeze(dim=1).to(device_id)
    phi_coeff = th.clamp(phi_coeff, 0, 0.5)

    mix_dist = sim_matrix[dest_indices] * adj_dist[dest_indices] * phi_coeff + \
               sim_matrix[source_indices] * adj_dist[source_indices] * (1 - phi_coeff)

    new_targets = th.multinomial(mix_dist + 1e-12, int(max_degree_val))

    target_indices = th.arange(max_degree_val).unsqueeze(dim=0)
    new_col_indices = new_targets[(target_indices - augmentation_degree.unsqueeze(dim=1) < 0)]

    new_row_indices = source_indices.repeat_interleave(augmentation_degree)

    return new_row_indices, new_col_indices


def redundancy_pruning(dgl_graph, adj_dist, sim_matrix, features_input, node_degrees, feat_drop_prob_1,
                       feat_drop_prob_2, degree_threshold):
    feat_v1 = drop_node_features(features_input, feat_drop_prob_1)
    feat_v2 = drop_node_features(features_input, feat_drop_prob_2)

    max_degree_val = np.max(node_degrees)

    low_deg_idx = th.LongTensor(np.argwhere(node_degrees < degree_threshold).flatten())
    high_deg_idx = th.LongTensor(np.argwhere(node_degrees >= degree_threshold).flatten())

    mix_node_degree = node_degrees[(node_degrees <= degree_threshold) & (node_degrees > 2)]
    mask_node_degree = node_degrees[node_degrees >= degree_threshold]

    sim_matrix = get_feature_similarity(features_input, features_input)
    sim_matrix = th.clamp(sim_matrix, 0, 1)
    sim_matrix = sim_matrix - th.diag_embed(th.diag(sim_matrix))

    mix_node_degree = th.LongTensor(mix_node_degree)
    mask_node_degree = th.LongTensor(mask_node_degree)

    degree_dist_counts = scatter_add(th.ones(mix_node_degree.size()), mix_node_degree)

    deg_prob = degree_dist_counts.unsqueeze(dim=0).repeat(low_deg_idx.size(0), 1)

    aug_deg_mix = th.multinomial(deg_prob, 1).flatten()

    src_nodes, dst_nodes = dgl_graph.edges()
    src_edge_indices = np.where(np.isin(src_nodes.numpy(), low_deg_idx))[0]
    dst_edge_indices = np.where(np.isin(dst_nodes.numpy(), low_deg_idx))[0]

    combined_indices = np.unique(np.concatenate((src_edge_indices, dst_edge_indices)))

    mix_rows, mix_cols = src_nodes[combined_indices], dst_nodes[combined_indices]

    mask_rows, mask_cols = degree_masking_with_threshold(adj_dist, high_deg_idx, sim_matrix, max_degree_val,
                                                         mask_node_degree, degree_threshold)

    new_src_nodes = np.concatenate((mix_rows, mask_rows))
    new_dst_nodes = np.concatenate((mix_cols, mask_cols.cpu()))

    new_dgl_graph = dgl.graph((new_src_nodes, new_dst_nodes), num_nodes=dgl_graph.number_of_nodes())
    new_dgl_graph = dgl.transform.remove_self_loop(new_dgl_graph)

    new_dgl_graph = dgl.transform.add_self_loop(new_dgl_graph)

    new_adj_dense = new_dgl_graph.adjacency_matrix().to_dense()
    new_adj_sparse = sp.csr_matrix(new_adj_dense)
    new_adj_norm = normalize_adj(new_adj_sparse)
    new_adj_matrix = (new_adj_norm + sp.eye(new_adj_norm.shape[0])).todense()
    new_adj_tensor = torch.FloatTensor(new_adj_matrix[np.newaxis])

    return new_dgl_graph, feat_v1.unsqueeze(0), feat_v2.unsqueeze(0), new_adj_tensor


def neighbor_completion(dgl_graph, adj_dist, sim_matrix, ano_sim_matrix, features_input, node_degrees,
                        feat_drop_prob_1, edge_mask_rate_1, feat_drop_prob_2, edge_mask_rate_2,
                        degree_threshold, device_id):
    feat_v1 = drop_node_features(features_input, feat_drop_prob_1)
    feat_v2 = drop_node_features(features_input, feat_drop_prob_2)

    max_degree_val = np.max(node_degrees)

    low_deg_idx = th.LongTensor(np.argwhere(node_degrees < degree_threshold).flatten())
    high_deg_idx = th.LongTensor(np.argwhere(node_degrees >= degree_threshold).flatten())

    mix_node_degree = node_degrees[(node_degrees <= degree_threshold) & (node_degrees > 2)]
    mask_node_degree = node_degrees[node_degrees >= degree_threshold]

    sim_matrix = get_feature_similarity(features_input, features_input)
    sim_matrix = th.clamp(sim_matrix, 0, 1)
    sim_matrix = sim_matrix - th.diag_embed(th.diag(sim_matrix))

    sim_matrix = sim_matrix * ano_sim_matrix
    sim_matrix = th.clamp(sim_matrix, 0, 1)
    low_deg_sim = sim_matrix[low_deg_idx]

    dest_mix_idx = th.multinomial(low_deg_sim + 1e-12, 1).flatten()

    mix_node_degree = th.LongTensor(mix_node_degree)
    mask_node_degree = th.LongTensor(mask_node_degree)

    degree_dist_counts = scatter_add(th.ones(mix_node_degree.size()), mix_node_degree)

    deg_prob = degree_dist_counts.unsqueeze(dim=0).repeat(low_deg_idx.size(0), 1)
    aug_deg_mix = th.multinomial(deg_prob, 1).flatten()

    mix_rows_1, mix_cols_1 = neighbor_mixup(low_deg_idx, dest_mix_idx, adj_dist, sim_matrix,
                                            max_degree_val, aug_deg_mix, device_id)
    mask_rows_1, mask_cols_1 = degree_based_edge_masking(adj_dist, high_deg_idx, sim_matrix, max_degree_val,
                                                         mask_node_degree, edge_mask_rate_1)

    src_nodes_1 = th.cat((mix_rows_1, mask_rows_1)).cpu()
    dst_nodes_1 = th.cat((mix_cols_1, mask_cols_1)).cpu()

    new_dgl_graph_1 = dgl.graph((src_nodes_1, dst_nodes_1), num_nodes=dgl_graph.number_of_nodes())
    new_dgl_graph_1 = dgl.transform.remove_self_loop(new_dgl_graph_1)

    new_dgl_graph_1 = dgl.transform.add_self_loop(new_dgl_graph_1)


    mix_rows_2, mix_cols_2 = neighbor_mixup(low_deg_idx, dest_mix_idx, adj_dist, sim_matrix,
                                            max_degree_val, aug_deg_mix, device_id)
    mask_rows_2, mask_cols_2 = degree_based_edge_masking(adj_dist, high_deg_idx, sim_matrix, max_degree_val,
                                                         mask_node_degree, edge_mask_rate_2)

    src_nodes_2 = th.cat((mix_rows_2, mask_rows_2)).cpu()
    dst_nodes_2 = th.cat((mix_cols_2, mask_cols_2)).cpu()

    new_dgl_graph_2 = dgl.graph((src_nodes_2, dst_nodes_2), num_nodes=dgl_graph.number_of_nodes())
    new_dgl_graph_2 = dgl.transform.remove_self_loop(new_dgl_graph_2)

    new_dgl_graph_2 = dgl.transform.add_self_loop(new_dgl_graph_2)

    adj_dense_1 = new_dgl_graph_1.adjacency_matrix().to_dense()
    adj_sparse_1 = sp.csr_matrix(adj_dense_1)
    adj_norm_1 = normalize_adj(adj_sparse_1)
    adj_matrix_1 = (adj_norm_1 + sp.eye(adj_norm_1.shape[0])).todense()
    adj_tensor_1 = torch.FloatTensor(adj_matrix_1[np.newaxis])

    adj_dense_2 = new_dgl_graph_2.adjacency_matrix().to_dense()
    adj_sparse_2 = sp.csr_matrix(adj_dense_2)
    adj_norm_2 = normalize_adj(adj_sparse_2)
    adj_matrix_2 = (adj_norm_2 + sp.eye(adj_norm_2.shape[0])).todense()
    adj_tensor_2 = torch.FloatTensor(adj_matrix_2[np.newaxis])

    return new_dgl_graph_1, new_dgl_graph_2, feat_v1.unsqueeze(0), feat_v2.unsqueeze(0), adj_tensor_1, adj_tensor_2
