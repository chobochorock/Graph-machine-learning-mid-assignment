import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--log', action='store_true', help='Enable logging')
parser.add_argument('--no-log', dest='log', action='store_false', help='Disable logging')
parser.add_argument('--use_original', type=bool, default=True)
parser.add_argument('--problem', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

if args.problem == 1 or args.problem == 2: 
    from torch_geometric.utils import degree
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    median = deg.median()
    median_node = torch.argmin(torch.abs(deg - median))
    if args.problem == 1: 
        print(f'degree of nodes: {deg}')
        print(f'median degree: {median}')
        print(f'median degree node index: {median_node}')
        exit()

if args.use_original: 
    class GAT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
            # On the Pubmed dataset, use `heads` output heads in `conv2`.
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                                concat=False, dropout=0.6)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return x


    model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
                args.heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
else: 
    # this is for assignment C.
    class GAT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
            # On the Pubmed dataset, use `heads` output heads in `conv2`.
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                                concat=False, dropout=0.6)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return x


    model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
                args.heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


times = []
best_val_acc = final_test_acc = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    if args.log : 
        print(args.log)
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)

if args.problem == 2:
    # median_edge = data.edge_index[:, data.edge_index[0] == median_node]
    model.eval()
    with torch.no_grad():
        out_features, (edge_index, alpha) = model.conv2(
                F.elu(model.conv1(data.x, data.edge_index)), #data.x, 
                data.edge_index, 
                return_attention_weights=True
            )
    median_node = torch.tensor(median_node, device=data.edge_index.device)
    median_edge_mask = (edge_index[0] == median_node)
    
    median_edge = edge_index[:, median_edge_mask]
    attention_coefficient = alpha[median_edge_mask]
    print(f'median edges: {median_edge[1]}')
    print(f'attention coefficient: {attention_coefficient.flatten()}')

# print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
