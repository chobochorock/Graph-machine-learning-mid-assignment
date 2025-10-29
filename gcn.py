import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--log', type=bool, default=True)
parser.add_argument('--use_original', type=bool, default=True)
parser.add_argument('--problem', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

device = torch_geometric.device('auto')

init_wandb(
    name=f'GCN-{args.dataset}',
    lr=args.lr,
    epochs=args.epochs,
    hidden_channels=args.hidden_channels,
    device=device,
)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

if args.problem == 4: 
    print(f'training data: {data.train_mask.count_nonzero()}') 
    print(f'val data: {data.val_mask.count_nonzero()}')
    print(f'test data: {data.test_mask.count_nonzero()}')
    exit()

if args.problem == 5:
    from torch_geometric.utils import is_undirected
    is_symmetric = is_undirected(data.edge_index)
    print(f'is undirected: {is_symmetric}')
    exit()

if args.problem == 1:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='col',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
else: 
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
data = transform(data)


if args.use_original:
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            if args.problem == 3:
                self.conv1 = GCNConv(in_channels, hidden_channels,
                                    normalize=False)
                self.conv2 = GCNConv(hidden_channels, out_channels,
                                    normalize=False) #True
            elif args.problem == 6:
                self.conv1 = GCNConv(in_channels, hidden_channels,
                                    normalize=not args.use_gdc)
                self.conv2 = GCNConv(hidden_channels, out_channels,
                                    normalize=not args.use_gdc)
            elif args.problem == 7:
                self.conv1 = GCNConv(in_channels, hidden_channels,
                                    normalize=not args.use_gdc)
                self.conv2 = GCNConv(hidden_channels, out_channels,
                                    normalize=not args.use_gdc)
                # need to add new layer
            else:
                self.conv1 = GCNConv(in_channels, hidden_channels,
                                    normalize=not args.use_gdc)
                self.conv2 = GCNConv(hidden_channels, out_channels,
                                    normalize=not args.use_gdc)

        def forward(self, x, edge_index, edge_weight=None):
            if args.problem == 6:
                x = F.dropout(x, p=0.9, training=self.training)
                x = self.conv1(x, edge_index, edge_weight).relu()
                x = F.dropout(x, p=0.9, training=self.training)
                x = self.conv2(x, edge_index, edge_weight)
            elif args.problem == 7:
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv1(x, edge_index, edge_weight).relu()
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index, edge_weight)
            else:
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv1(x, edge_index, edge_weight).relu()
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index, edge_weight)
            return x

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
    ).to(device)
    print(f'use normalization: {model.conv1.normalize}')
else: 
    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels,
                                normalize=not args.use_gdc)
            self.conv2 = GCNConv(hidden_channels, out_channels,
                                normalize=not args.use_gdc)

        def forward(self, x, edge_index, edge_weight=None):
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
    ).to(device)


optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    if args.log : log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
# print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
