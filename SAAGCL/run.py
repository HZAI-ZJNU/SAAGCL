import torch.nn as nn

from model import Model, Gene, Disc
from utils import *

from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
from tqdm import tqdm
from aug import redundancy_pruning, neighbor_completion
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='SAA-GCL')
parser.add_argument('--dataset', type=str, default='citeseer')
parser.add_argument('--lr', type=float)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--train_epoch', type=int)
parser.add_argument('--test_rounds', type=int, default=256)
parser.add_argument('--threshold', type=int, default=8)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--tau', type=float, default=0.07)
parser.add_argument('--degree', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  # max min avg weighted_sum
parser.add_argument('--neg_sam_rat', type=int, default=1)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ', args.dataset)

dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dev_id = args.gpu_id
adj_matrix, feat_data, anomaly_labels = load_mat(args.dataset)

feat_data, _ = preprocess_features(feat_data)

dgl_g = adj_to_dgl_graph(adj_matrix)
dgl_g = dgl_g.to(dev_id)

num_nodes = feat_data.shape[0]
feature_size = feat_data.shape[1]

adj_tens = dgl_g.adjacency_matrix().to_dense().clone().detach().requires_grad_(True)
node_degrees = adj_tens.sum(0).detach().numpy().squeeze()

adj_tens = adj_tens + torch.eye(adj_tens.size(0))
adj_tens = adj_tens.to(dev_id)

adj_matrix = normalize_adj(adj_matrix)
adj_matrix = (adj_matrix + sp.eye(adj_matrix.shape[0])).todense()

feat_data = torch.FloatTensor(feat_data[np.newaxis]).to(dev_id)
adj_matrix = torch.FloatTensor(adj_matrix[np.newaxis]).to(dev_id)

# Initialize model and optimiser
gcl_model = Model(feature_size, args.embedding_dim, 'prelu', args.neg_sam_rat, args.readout).to(dev_id)
optimizer = torch.optim.Adam(gcl_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

generator = Gene(num_nodes, feature_size, 128)
discriminator = Disc(128, 1)

bce_loss_func = nn.BCELoss().to(dev_id)

best_gen_model = copy.deepcopy(generator)

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.neg_sam_rat]).to(dev_id))

cross_ent_loss = nn.CrossEntropyLoss().to(dev_id)
alter_count = 0
wait_count = 0
best_loss = 1e9
best_ep = 0
num_batches = num_nodes // batch_size + 1
ano_sim_matrix, sim_matrix = None, None

# These tensors are for padding subgraphs to a uniform size (subgraph_size + 1)
pad_adj_row = torch.zeros((num_nodes, 1, subgraph_size)).to(dev_id)
pad_adj_col = torch.zeros((num_nodes, subgraph_size + 1, 1)).to(dev_id)
pad_adj_col[:, -1, :] = 1.
pad_feat_row = torch.zeros((num_nodes, 1, feature_size)).to(dev_id)

# Train model
with tqdm(total=args.train_epoch) as prog_bar:
    prog_bar.set_description('Training')
    loss_hist_list = []

    for epoch_num in range(args.train_epoch):

        full_batch_loss = torch.zeros((num_nodes, 1)).to(dev_id)

        gcl_model.train()

        all_indices = list(range(num_nodes))
        random.shuffle(all_indices)

        total_epoch_loss = 0.
        all_node_features = []
        cur_loss_matrix = torch.zeros((num_nodes, 2)).to(dev_id)
        feature_diff = torch.zeros((num_nodes, args.embedding_dim)).to(dev_id)

        if epoch_num < args.train_epoch // 2:
            graph_v1, adj_v1 = dgl_g, adj_matrix
            graph_v2, feature_v1, feature_v2, adj_v2 = redundancy_pruning(dgl_g, adj_tens, sim_matrix,
                                                                          feat_data.squeeze(), node_degrees, 0.2, 0.2,
                                                                          args.threshold)
        else:
            graph_v1, graph_v2, feature_v1, feature_v2, adj_v1, adj_v2 = neighbor_completion(dgl_g, adj_tens,
                                                                                             sim_matrix, ano_sim_matrix,
                                                                                             feat_data.squeeze(),
                                                                                             node_degrees,
                                                                                             0.2, 0.2, 0.2, 0.2,
                                                                                             args.threshold, dev_id)

        subg_set1 = generate_rwr_subgraph(graph_v1, subgraph_size)
        subg_set2 = generate_rwr_subgraph(graph_v2, subgraph_size)

        for batch_num_idx in range(num_batches):

            optimizer.zero_grad()

            is_last_batch = (batch_num_idx == (num_batches - 1))

            if not is_last_batch:
                indices = all_indices[batch_num_idx * batch_size: (batch_num_idx + 1) * batch_size]
            else:
                indices = all_indices[batch_num_idx * batch_size:]

            current_batch_size = len(indices)

            batch_labels = torch.unsqueeze(
                torch.cat((torch.ones(current_batch_size), torch.zeros(current_batch_size * args.neg_sam_rat))), 1).to(
                dev_id)

            batch_adj_list = []
            batch_feat_list = []
            pad_adj_row_b = torch.zeros((current_batch_size, 1, subgraph_size)).to(dev_id)
            pad_adj_col_b = torch.zeros((current_batch_size, subgraph_size + 1, 1)).to(dev_id)
            pad_adj_col_b[:, -1, :] = 1.
            pad_feat_row_b = torch.zeros((current_batch_size, 1, feature_size)).to(dev_id)

            for i in indices:
                current_adj = adj_v1[:, subg_set1[i], :][:, :, subg_set1[i]]
                current_feat = feature_v1[:, subg_set1[i], :]

                batch_adj_list.append(current_adj)
                batch_feat_list.append(current_feat)

            b_adj_v1 = torch.cat(batch_adj_list).to(dev_id)
            b_adj_v1 = torch.cat((b_adj_v1, pad_adj_row_b), dim=1)
            b_adj_v1 = torch.cat((b_adj_v1, pad_adj_col_b), dim=2)
            b_feat_v1 = torch.cat(batch_feat_list).to(dev_id)
            b_feat_v1 = torch.cat((b_feat_v1[:, :-1, :], pad_feat_row_b, b_feat_v1[:, -1:, :]), dim=1)

            logits_v1, h_v1, g_v1 = gcl_model(b_feat_v1, b_adj_v1)

            batch_adj_list = []
            batch_feat_list = []
            pad_adj_row_b = torch.zeros((current_batch_size, 1, subgraph_size)).to(dev_id)
            pad_adj_col_b = torch.zeros((current_batch_size, subgraph_size + 1, 1)).to(dev_id)
            pad_adj_col_b[:, -1, :] = 1.
            pad_feat_row_b = torch.zeros((current_batch_size, 1, feature_size)).to(dev_id)
            for i in indices:
                current_adj = adj_v2[:, subg_set2[i], :][:, :, subg_set2[i]]
                current_feat = feature_v2[:, subg_set2[i], :]

                batch_adj_list.append(current_adj)
                batch_feat_list.append(current_feat)

            b_adj_v2 = torch.cat(batch_adj_list).to(dev_id)
            b_adj_v2 = torch.cat((b_adj_v2, pad_adj_row_b), dim=1)
            b_adj_v2 = torch.cat((b_adj_v2, pad_adj_col_b), dim=2)
            b_feat_v2 = torch.cat(batch_feat_list).to(dev_id)
            b_feat_v2 = torch.cat((b_feat_v2[:, :-1, :], pad_feat_row_b, b_feat_v2[:, -1:, :]), dim=1)

            logits_v2, h_v2, g_v2 = gcl_model(b_feat_v2, b_adj_v2)

            # InfoMax loss terms: cross-view similarities
            pred_h_12, pred_h_21 = torch.mm(h_v1, h_v2.T), torch.mm(h_v2, h_v1.T)
            pred_g_12, pred_g_21 = torch.mm(logits_v1, logits_v2.T), torch.mm(logits_v2, logits_v1.T)

            # Labels for CL objectives (main diagonal)
            cl_labels_h = torch.arange(pred_h_12.shape[0]).to(dev_id)
            cl_labels_g = torch.arange(pred_g_12.shape[0]).to(dev_id)

            loss_cl_terms = (cross_ent_loss(pred_h_12 / args.tau, cl_labels_h) + cross_ent_loss(pred_h_21 / args.tau,
                                                                                                cl_labels_h) +
                             cross_ent_loss(pred_g_12 / args.tau, cl_labels_g) + cross_ent_loss(pred_g_21 / args.tau,
                                                                                                cl_labels_g)) / 4

            # Anomaly scoring (Node-Context discrimination)
            loss_anomaly_detection = b_xent(logits_v1, batch_labels) + b_xent(logits_v2, batch_labels)

            cur_loss_matrix[indices] = torch.cat(
                (loss_anomaly_detection[:current_batch_size], loss_anomaly_detection[current_batch_size:]), dim=1)
            feature_diff[indices] = ((h_v1 - g_v1) + (h_v2 - g_v2)) / 2  # Unused but kept for structure

            total_batch_loss = torch.mean(loss_anomaly_detection) + loss_cl_terms * args.alpha

            total_batch_loss.backward()
            optimizer.step()

            batch_loss_val = total_batch_loss.detach().cpu().numpy()
            full_batch_loss[indices] = loss_anomaly_detection[: current_batch_size].detach()

            if not is_last_batch:
                total_epoch_loss += batch_loss_val

        mean_epoch_loss = (total_epoch_loss * batch_size + batch_loss_val * current_batch_size) / num_nodes

        loss_hist_list.append(cur_loss_matrix)
        window_size = 5
        if len(loss_hist_list) >= window_size:
            sim_calc_matrix = torch.cat(loss_hist_list[-window_size:], dim=1)

            mean_pos, var_pos = torch.mean(sim_calc_matrix[:, 0::2], dim=1), torch.var(sim_calc_matrix[:, 0::2], dim=1)
            mean_neg, var_neg = torch.mean(sim_calc_matrix[:, 1::2], dim=1), torch.var(sim_calc_matrix[:, 1::2], dim=1)

            sim_calc_matrix = torch.cat(
                [sim_calc_matrix, mean_pos.unsqueeze(1), var_pos.unsqueeze(1), mean_neg.unsqueeze(1),
                 var_neg.unsqueeze(1)], dim=1)

            ano_sim_matrix = torch.sigmoid(torch.mm(sim_calc_matrix, sim_calc_matrix.t()) * 0.07)

            loss_hist_list = loss_hist_list[-window_size:]

        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            best_ep = epoch_num
            wait_count = 0
            torch.save(gcl_model.state_dict(), './SAAGCL/results/best_model_{}.pkl'.format(args.dataset))
        else:
            wait_count += 1

        prog_bar.set_postfix(loss=mean_epoch_loss)
        prog_bar.update(1)

# Test model
print('Loading {}th epoch'.format(best_ep))
gcl_model.load_state_dict(torch.load('./SAAGCL/results/best_model_{}.pkl'.format(args.dataset)))

multi_round_scores = np.zeros((args.test_rounds, num_nodes))

with tqdm(total=args.test_rounds) as prog_bar_test:
    prog_bar_test.set_description('Testing')
    for round_idx in range(args.test_rounds):

        all_indices = list(range(num_nodes))
        random.shuffle(all_indices)

        subg_set = generate_rwr_subgraph(dgl_g, subgraph_size)

        for batch_num_idx in range(num_batches):

            optimizer.zero_grad()

            is_last_batch = (batch_num_idx == (num_batches - 1))

            if not is_last_batch:
                indices = all_indices[batch_num_idx * batch_size: (batch_num_idx + 1) * batch_size]
            else:
                indices = all_indices[batch_num_idx * batch_size:]

            current_batch_size = len(indices)

            batch_adj_list = []
            batch_feat_list = []
            pad_adj_row_b = torch.zeros((current_batch_size, 1, subgraph_size)).to(dev_id)
            pad_adj_col_b = torch.zeros((current_batch_size, subgraph_size + 1, 1)).to(dev_id)
            pad_adj_col_b[:, -1, :] = 1.
            pad_feat_row_b = torch.zeros((current_batch_size, 1, feature_size)).to(dev_id)

            for i in indices:
                current_adj = adj_matrix[:, subg_set[i], :][:, :, subg_set[i]]
                current_feat = feat_data[:, subg_set[i], :]
                batch_adj_list.append(current_adj)
                batch_feat_list.append(current_feat)

            b_adj = torch.cat(batch_adj_list).to(dev_id)
            b_adj = torch.cat((b_adj, pad_adj_row_b), dim=1)
            b_adj = torch.cat((b_adj, pad_adj_col_b), dim=2)
            b_feat = torch.cat(batch_feat_list).to(dev_id)
            b_feat = torch.cat((b_feat[:, :-1, :], pad_feat_row_b, b_feat[:, -1:, :]), dim=1)

            with torch.no_grad():
                logits_out, h_mv_out, _ = gcl_model(b_feat, b_adj)
                logits_out = torch.squeeze(logits_out)
                logits_out = torch.sigmoid(logits_out)

            # Anomaly score calculation
            score_batch = - (logits_out[:current_batch_size] - logits_out[current_batch_size:]).cpu().numpy()
            multi_round_scores[round_idx, indices] = score_batch

        prog_bar_test.update(1)

final_anomaly_scores = np.mean(multi_round_scores, axis=0)
auc_result = roc_auc_score(anomaly_labels, final_anomaly_scores)

print('AUC:{:.4f}'.format(auc_result))