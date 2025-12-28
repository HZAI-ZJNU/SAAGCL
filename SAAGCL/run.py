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
parser.add_argument('--readout', type=str, default='avg')  #max min avg weighted_sum
parser.add_argument('--neg_sam_rat', type=int, default=1)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--gpu_id', type=int, default=0)


args = parser.parse_args()

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset: ',args.dataset)


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

device = args.gpu_id
adj, features, ano_label = load_mat(args.dataset)


features, _ = preprocess_features(features)

dgl_graph = adj_to_dgl_graph(adj)
dgl_graph = dgl_graph.to(device)

nb_nodes = features.shape[0]
ft_size = features.shape[1]

adj_tensor = dgl_graph.adjacency_matrix().to_dense().clone().detach().requires_grad_(True)
degree = adj_tensor.sum(0).detach().numpy().squeeze()

adj_tensor = adj_tensor + torch.eye(adj_tensor.size(0))
adj_tensor = adj_tensor.to(device)

adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis]).to(device)
adj = torch.FloatTensor(adj[np.newaxis]).to(device)

# Initialize model and optimiser
model = Model(ft_size, args.embedding_dim, 'prelu', args.neg_sam_rat, args.readout).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

gene = Gene(nb_nodes, ft_size, 128)
disc = Disc(128, 1)

bce_loss = nn.BCELoss().to(device)

best_gene = copy.deepcopy(gene)

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.neg_sam_rat]).to(device))

xent = nn.CrossEntropyLoss().to(device)
alter = 0
cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size + 1
ano_sim, sim = None, None

added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size)).to(device)
added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1)).to(device)
added_adj_zero_col[:,-1,:] = 1.
added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size)).to(device)

# Train model
with tqdm(total=args.train_epoch) as pbar:
    pbar.set_description('Training')
    loss_matrix_list = []

    for epoch in range(args.train_epoch):

        loss_full_batch = torch.zeros((nb_nodes,1)).to(device)

        model.train()

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        total_loss = 0.
        all_node_features = []
        loss_matrix = torch.zeros((nb_nodes,2)).to(device)
        diff_feat = torch.zeros((nb_nodes,args.embedding_dim)).to(device)
      

        if epoch < args.train_epoch // 2:
            graph1, adj1 = dgl_graph, adj
            graph2, feat1, feat2, adj2 = redundancy_pruning(dgl_graph, adj_tensor, sim, features.squeeze(), degree, 0.2, 0.2, args.threshold)
        else:
            graph1, graph2, feat1, feat2, adj1, adj2 = neighbor_completion(dgl_graph, adj_tensor, sim, ano_sim, features.squeeze(), degree, 
            0.2, 0.2, 0.2, 0.2, args.threshold, device)

      
        subgraphs1 = generate_rwr_subgraph(graph1, subgraph_size)
        subgraphs2 = generate_rwr_subgraph(graph2, subgraph_size)

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.neg_sam_rat))), 1).to(device)

            ba = []
            bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

            for i in idx:
                cur_adj = adj1[:, subgraphs1[i], :][:, :, subgraphs1[i]]
                cur_feat = feat1[:, subgraphs1[i], :]

                ba.append(cur_adj)
                bf.append(cur_feat)

            ba = torch.cat(ba).to(device)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf).to(device)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)

            logits1, h1, g1 = model(bf, ba)
            
            ba = []
            bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
            for i in idx:
                cur_adj = adj2[:, subgraphs2[i], :][:, :, subgraphs2[i]]
                cur_feat = feat2[:, subgraphs2[i], :]

                ba.append(cur_adj)
                bf.append(cur_feat)

            ba = torch.cat(ba).to(device)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf).to(device)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)

            logits2, h2, g2 = model(bf, ba)
      
            pred1, pred2 = torch.mm(h1, h2.T), torch.mm(h2, h1.T)
            pred3, pred4 = torch.mm(logits1, logits2.T), torch.mm(logits2, logits1.T)

            labels = torch.arange(pred1.shape[0]).to(device)
            labels1 = torch.arange(pred3.shape[0]).to(device)

            loss_fn = (xent(pred1 / args.tau, labels) + xent(pred2 / args.tau, labels) + xent(pred3 / args.tau, labels1) + xent(pred4 / args.tau, labels1)) / 4

            loss_all = b_xent(logits1, lbl) + b_xent(logits2, lbl)
            
            loss_matrix[idx] = torch.cat((loss_all[:cur_batch_size], loss_all[cur_batch_size:]), dim=1)
            diff_feat[idx] = ((h1 -g1) + (h2 - g2)) / 2

            loss = torch.mean(loss_all) + loss_fn * args.alpha

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()
            loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

            if not is_final_batch:
                total_loss += loss

        mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes
      
        loss_matrix_list.append(loss_matrix)
        w = 5 
        if len(loss_matrix_list) >= w:
            s_matrix = torch.cat(loss_matrix_list[-w:], dim=1)
    
            mean_p, var_p = torch.mean(s_matrix[:, 0::2], dim=1), torch.var(s_matrix[:, 0::2], dim=1)
            mean_n, var_n = torch.mean(s_matrix[:, 1::2], dim=1), torch.var(s_matrix[:, 1::2], dim=1)
            
            s_matrix = torch.cat([s_matrix, mean_p.unsqueeze(1), var_p.unsqueeze(1), mean_n.unsqueeze(1), var_n.unsqueeze(1)], dim=1)

            ano_sim = torch.sigmoid(torch.mm(s_matrix, s_matrix.t()) * 0.07)

            loss_matrix_list = loss_matrix_list[-w:]

        if mean_loss < best:
            best = mean_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), './SAAGCL/results/best_model_{}.pkl'.format(args.dataset))
        else:
            cnt_wait += 1
        

        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)


# Test model
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('./SAAGCL/results/best_model_{}.pkl'.format(args.dataset)))

multi_round_ano_score = np.zeros((args.test_rounds, nb_nodes))
multi_round_ano_score_p = np.zeros((args.test_rounds, nb_nodes))
multi_round_ano_score_n = np.zeros((args.test_rounds, nb_nodes))

with tqdm(total=args.test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.test_rounds):

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            ba = []
            bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                ba.append(cur_adj)
                bf.append(cur_feat)

            ba = torch.cat(ba).to(device)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf).to(device)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            with torch.no_grad():
                logits, h_mv, _ = model(bf, ba)
                logits = torch.squeeze(logits)
                logits = torch.sigmoid(logits)

            ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
            multi_round_ano_score[round, idx] = ano_score

        pbar_test.update(1)

ano_score_final = np.mean(multi_round_ano_score, axis=0)
auc = roc_auc_score(ano_label, ano_score_final)

print('AUC:{:.4f}'.format(auc))
