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


my_nn = GraphLayer(input_dim=300, output_dim=96, steps=2)
train_adj = torch.rand(20, 38, 38)
train_mask = torch.rand(20, 38, 1)
train_feature = torch.rand(20, 38, 300)

print(my_nn)

out = my_nn.forward(train_feature, train_adj, train_mask)

print(out.shape)
