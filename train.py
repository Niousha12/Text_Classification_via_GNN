import argparse
from model import GNN
import torch.optim as optim
import torch
from dataloader import TextIGNGraphDataset
from torch.utils.data import DataLoader

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
    log = open("./log.txt", "w", encoding="utf-8")
    net = GNN(input_dim=300, hidden_dim=96, output_dim=4)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate,
                           weight_decay=1e-5)  # weight_decay=1e-5 if for L2 Reg
    criterion = torch.nn.CrossEntropyLoss()
    epochs = args.epochs

    dataset = TextIGNGraphDataset(dataset=args.dataset, root_dir='dataloader', name='train')

    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=False, num_workers=0)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for data in dataloader:
            adj, mask, emb, y = data
            adj = adj.float().cuda()
            mask = mask.float().cuda()
            emb = emb.float().cuda()
            y = y.float().cuda()
            optimizer.zero_grad()
            # train_feature, train_adj, train_mask

            y = torch.argmax(y, dim=1)

            output = net(emb, adj, mask)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        log.write("Loss in epoch {} : {}".format(epoch, epoch_loss))
        log.write("\n")
        print(f"Loss in epoch {epoch} : {epoch_loss}")

        if epoch % 50 == 0:
            torch.save(net.state_dict(), './checkpoint/net_epoch_{}.pth'.format(epoch))
            torch.save(optimizer.state_dict(), './checkpoint/opt_epoch{}.pth'.format(epoch))
            # test(net, dataloader, test_dataloader, epoch)
    log.close()


if __name__ == '__main__':
    train()
