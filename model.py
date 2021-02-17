import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim, steps=2):
        super(GraphLayer, self).__init__()

        self.steps = steps

        self.encode = nn.Linear(input_dim, output_dim, bias=False)

        self.z0 = nn.Linear(output_dim, output_dim, bias=False)
        self.z1 = nn.Linear(output_dim, output_dim, bias=False)

        self.r0 = nn.Linear(output_dim, output_dim, bias=False)
        self.r1 = nn.Linear(output_dim, output_dim, bias=False)

        self.h0 = nn.Linear(output_dim, output_dim, bias=False)
        self.h1 = nn.Linear(output_dim, output_dim, bias=False)

        torch.nn.init.xavier_uniform_(self.encode.weight)
        torch.nn.init.xavier_uniform_(self.z0.weight)
        torch.nn.init.xavier_uniform_(self.z1.weight)
        torch.nn.init.xavier_uniform_(self.r0.weight)
        torch.nn.init.xavier_uniform_(self.r1.weight)
        torch.nn.init.xavier_uniform_(self.h0.weight)
        torch.nn.init.xavier_uniform_(self.h1.weight)

    def forward(self, inputs, adj_matrix, mask):
        # TODO: Add Dropout from line 219 of layers (Original Code)

        x = self.encode(inputs)
        x = mask * F.relu(x)

        for _ in range(self.steps):
            # TODO Dropout : L56 layers.py
            a = torch.matmul(adj_matrix, x)
            # update gate
            z0 = self.z0(a)
            z1 = self.z1(x)
            z = torch.sigmoid(z0 + z1)
            # reset gate
            r0 = self.r0(a)
            r1 = self.r1(x)
            r = torch.sigmoid(r0 + r1)
            # update embeddings
            h0 = self.h0(a)
            h1 = self.h1(x * r)
            h = F.relu(mask * (h0 + h1))
            # Update x for next iteration
            x = h * z + x * (1 - z)

        return x


class ReadoutLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReadoutLayer, self).__init__()

        self.att = nn.Linear(input_dim, 1, bias=False)

        self.emb = nn.Linear(input_dim, input_dim, bias=False)
        self.mlp = nn.Linear(input_dim, output_dim, bias=False)

        torch.nn.init.xavier_uniform_(self.att.weight)
        torch.nn.init.xavier_uniform_(self.emb.weight)
        torch.nn.init.xavier_uniform_(self.mlp.weight)

    def forward(self, inputs, mask):
        x = inputs
        att = torch.sigmoid(self.att(x))
        emb = torch.relu(self.emb(x))
        n = torch.sum(mask, dim=1)
        m = (mask - 1) * 1e9

        # Graph Summation
        g = mask * att * emb
        g = (torch.sum(g, dim=1) / n) + torch.max(g + m, dim=1).values

        # TODO Add Dropout
        # g = torch.nn.dropout(g, 1-self.dropout)

        output = self.mlp(g)
        return output


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.graph = GraphLayer(input_dim=input_dim, output_dim=hidden_dim)
        self.readout = ReadoutLayer(input_dim=hidden_dim, output_dim=output_dim)

    def forward(self, inputs, adj_matrix, mask):
        graph = self.graph(inputs, adj_matrix, mask)
        output = self.readout(graph, mask)
        return output


# my_nn = GraphLayer(input_dim=300, output_dim=96, steps=2)
# my_readout = ReadoutLayer(input_dim=96, output_dim=4)

# net = GNN(input_dim=300, hidden_dim=96, output_dim=4)
#
# train_adj = torch.rand(20, 38, 38)
# train_mask = torch.rand(20, 38, 1)
# train_feature = torch.rand(20, 38, 300)
#
# # print(my_nn)
#
# out = net.forward(train_feature, train_adj, train_mask)
# # out = my_readout(out, train_mask)
#
# print(out.shape)
