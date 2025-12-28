import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act_func, bias_flag=True):
        super(GCN, self).__init__()
        self.linear_map = nn.Linear(in_dim, out_dim, bias=False)
        self.activation = nn.PReLU() if act_func == 'prelu' else act_func

        if bias_flag:
            self.bias_param = nn.Parameter(torch.FloatTensor(out_dim))
            self.bias_param.data.fill_(0.0)
        else:
            self.register_parameter('bias_param', None)

        for mod in self.modules():
            self.init_weights(mod)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, input_seq, adjacency, sparse_flag=False):
        mapped_fts = self.linear_map(input_seq)
        if sparse_flag:
            output_fts = torch.unsqueeze(torch.spmm(adjacency, torch.squeeze(mapped_fts, 0)), 0)
        else:
            output_fts = torch.bmm(adjacency, mapped_fts)
        if self.bias_param is not None:
            output_fts += self.bias_param

        return self.activation(output_fts)


class AveragePoolingReadout(nn.Module):
    def __init__(self):
        super(AveragePoolingReadout, self).__init__()

    def forward(self, input_seq):
        return torch.mean(input_seq, 1)


class MaxPoolingReadout(nn.Module):
    def __init__(self):
        super(MaxPoolingReadout, self).__init__()

    def forward(self, input_seq):
        return torch.max(input_seq, 1).values


class MinPoolingReadout(nn.Module):
    def __init__(self):
        super(MinPoolingReadout, self).__init__()

    def forward(self, input_seq):
        return torch.min(input_seq, 1).values


class WeightedSumReadout(nn.Module):
    def __init__(self):
        super(WeightedSumReadout, self).__init__()

    def forward(self, input_seq, attention_query):
        query_t = attention_query.permute(0, 2, 1)
        similarity = torch.matmul(input_seq, query_t)
        similarity = F.softmax(similarity, dim=1)
        # Assuming embedding size is 64 based on run.py args
        similarity = similarity.repeat(1, 1, 64)
        weighted_out = torch.mul(input_seq, similarity)
        weighted_out = torch.sum(weighted_out, 1)
        return weighted_out


class NodeDiscriminator(nn.Module):
    def __init__(self, hidden_dim, neg_samples):
        super(NodeDiscriminator, self).__init__()
        self.bilinear_layer = nn.Bilinear(hidden_dim, hidden_dim, 1)

        for mod in self.modules():
            self.init_weights(mod)

        self.negsamp_num = neg_samples

    def init_weights(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, context_embed, node_embed):
        scores_list = []
        # positive pair: (node_embed, context_embed)
        scores_list.append(self.bilinear_layer(node_embed, context_embed))

        # negative pairs by shifting context_embed
        context_shifted = context_embed
        for _ in range(self.negsamp_num):
            context_shifted = torch.cat((context_shifted[-2:-1, :], context_shifted[:-1, :]), 0)
            scores_list.append(self.bilinear_layer(node_embed, context_shifted))

        output_logits = torch.cat(tuple(scores_list))

        return output_logits


class Model(nn.Module):
    def __init__(self, in_dim, hid_dim, act_func, neg_samples, pooling_type):
        super(Model, self).__init__()
        self.readout_mode = pooling_type
        self.gcn_layer = GCN(in_dim, hid_dim, act_func)

        if pooling_type == 'max':
            self.readout_op = MaxPoolingReadout()
        elif pooling_type == 'min':
            self.readout_op = MinPoolingReadout()
        elif pooling_type == 'avg':
            self.readout_op = AveragePoolingReadout()
        elif pooling_type == 'weighted_sum':
            self.readout_op = WeightedSumReadout()

        self.disc_model = NodeDiscriminator(hid_dim, neg_samples)

    def forward(self, feature_seq, adj_matrix, sparse_flag=False):
        h_all = self.gcn_layer(feature_seq, adj_matrix, sparse_flag)

        node_interest_h = h_all[:, -1, :]

        if self.readout_mode != 'weighted_sum':
            context_c = self.readout_op(h_all[:, : -1, :])
        else:
            context_c = self.readout_op(h_all[:, : -1, :], h_all[:, -2: -1, :])

        logits_out = self.disc_model(context_c, node_interest_h)

        return logits_out, node_interest_h, context_c


class Gene(nn.Module):
    def __init__(self, nb_nodes, hid_dim, out_dim):
        super(Gene, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.layer1 = GraphConv(hid_dim, hid_dim)
        self.layer2 = GraphConv(hid_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

        self.batchnorm = nn.BatchNorm1d(out_dim)

        self.epsilon = torch.nn.Parameter(torch.Tensor(nb_nodes))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.epsilon, 0.5)
        
    def forward(self, g, x, adj):
        h1 = self.fc(x)
        h2 = F.relu(self.layer1(g, x))
        h2 = self.layer2(g, h2)

        h = (1 - self.epsilon.view(-1,1)) * h1 + self.epsilon.view(-1,1) * h2

        ret = (torch.mm(h, h.t()) + torch.mm(x, x.t())) / 2

        h = self.batchnorm(h)

        return ret, h


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(self, dgl_graph, features):
        dgl_graph = dgl_graph.local_var()
        dgl_graph.ndata['h_feat'] = features
        # Message passing: copy source features to edges, sum features at destination nodes
        dgl_graph.update_all(message_func=dgl.function.copy_src(src='h_feat', out='msg_feat'),
                             reduce_func=dgl.function.sum(msg='msg_feat', out='h_neigh'))

        neighbor_sum_h = dgl_graph.ndata['h_neigh']
        return self.output_linear(neighbor_sum_h)


class Disc(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Disc, self).__init__()
        self.bilinear_comp = nn.Bilinear(hidden_dim, hidden_dim, output_dim)
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, embed1, embed2):
        logits_val = self.bilinear_comp(embed1, embed2)
        logits_val = self.sigmoid_act(logits_val)

        return logits_val