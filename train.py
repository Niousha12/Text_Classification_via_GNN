import argparse
from model import GNN
import torch.optim as optim
import torch

parser = argparse.ArgumentParser(description='Pytorch TextIGN Training')
parser.add_argument('--dataset', default='mr', help='Training dataset')  # 'mr','ohsumed','R8','R52'
parser.add_argument('--learning_rate', default=0.005, help='Initial learning rate.')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=4096, type=int, help='Size of batches per epoch.')
parser.add_argument('--input_dim', default=300, type=int, help='Dimension of input.')
parser.add_argument('--hidden', default=96, type=int, help='Number of units in hidden layer.')  # 32, 64, 96, 128
parser.add_argument('--steps', default=2, type=int, help='Number of graph layers.')
parser.add_argument('--dropout', default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', default=0, help='Weight for L2 loss on embedding matrix.')  # 5e-4
parser.add_argument('--max_degree', default=3, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--early_stopping', default=-1, help='Tolerance for early stopping (# of epochs).')

args = parser.parse_args()


def train():
    net = GNN(input_dim=300, hidden_dim=96, output_dim=4)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,
                           weight_decay=1e-5)  # weight_decay=1e-5 if for L2 Reg
    criterion = torch.nn.CrossEntropyLoss()
    epochs = args.epochs

    for epoch in range(epochs):
        # Call Dataloader here
        print(epoch)


if __name__ == '__main__':
    train()
