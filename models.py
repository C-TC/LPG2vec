import math
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch.nn.functional import normalize
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (Adj, NoneType, OptPairTensor, OptTensor, Size)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.dense import Linear
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from dataset import LPGdataset


class MyGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, encoded_mappings=None, concat=True, negative_slope=0.2,
                 dropout=0.0,
                 _add_self_loops=True, edge_dim=None, fill_value='mean', bias=True):
        super().__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self._add_self_loops = _add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.encoded_mappings = encoded_mappings

        # assume vertex mapping and edge mapping coexist
        self.lin_dst = self.lin_src = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')

        if self.encoded_mappings is None:
            self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
            self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.num_v_features = encoded_mappings[0].shape[0]
            self.att_src = Parameter(torch.Tensor(1, self.num_v_features))
            self.att_dst = Parameter(torch.Tensor(1, self.num_v_features))

        if edge_dim is not None:
            if self.encoded_mappings is None:
                self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
                self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
            else:
                self.lin_edge = None
                self.num_e_features = encoded_mappings[1].shape[0]
                self.att_edge = Parameter(torch.Tensor(1, self.num_e_features))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        if edge_attr is None:
            assert self.edge_dim is None

        H, C = self.heads, self.out_channels
        x_src = x_dst = self.lin_src(x).view(-1, H, C)
        x_new = (x_src, x_dst)

        if self.encoded_mappings is None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
            alpha = (alpha_src, alpha_dst)
        else:
            alpha_src = (torch.mm(x, torch.transpose(self.encoded_mappings[0], 0, 1)) * self.att_src).sum(dim=-1)
            alpha_dst = (torch.mm(x, torch.transpose(self.encoded_mappings[0], 0, 1)) * self.att_dst).sum(dim=-1)
            alpha = (alpha_src, alpha_dst)

        '''
        if self._add_self_loops:
            num_nodes = x_src.size(0)
            if x_dst is not None:
                num_nodes = min(num_nodes, x_dst.size(0))
            num_nodes = min(size) if size is not None else num_nodes
            edge_index, edge_attr = remove_self_loops(
                edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value,
                                                   num_nodes=num_nodes)
        '''

        out = self.propagate(edge_index, x=x_new, alpha=alpha, edge_attr=edge_attr, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            if self.encoded_mappings is None:
                edge_attr = self.lin_edge(edge_attr)
                edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
                alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
                alpha = alpha + alpha_edge
            else:
                alpha_edge = (torch.mm(edge_attr, torch.transpose(self.encoded_mappings[1], 0, 1)) * self.att_edge).sum(
                    dim=-1)
                alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if self.encoded_mappings is None:
            return x_j * alpha.unsqueeze(-1)
        else:
            return x_j * alpha.view(-1, 1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class MyGAT(torch.nn.Module):
    def __init__(self, data, encoded_mappings=None, use_edge_features=True):
        super(MyGAT, self).__init__()
        self.hid = 16
        self.in_head = 8
        self.out_head = 1
        self.use_edge_features = True

        if encoded_mappings is not None:
            assert use_edge_features

        num_features = data.x.shape[1]
        num_classes = len(data.y.unique())
        if use_edge_features:
            edge_dim = data.edge_attr.shape[1]
            if encoded_mappings is None:
                self.conv1 = MyGATConv(num_features, self.hid, edge_dim=edge_dim, heads=self.in_head, dropout=0.6)
            else:
                self.conv1 = MyGATConv(num_features, self.hid, encoded_mappings=encoded_mappings, edge_dim=edge_dim,
                                       heads=self.in_head, dropout=0.6)
            self.conv2 = MyGATConv(self.hid * self.in_head, num_classes, edge_dim=edge_dim, concat=False,
                                   heads=self.out_head, dropout=0.6)
        else:
            self.conv1 = MyGATConv(num_features, self.hid, heads=self.in_head, dropout=0.6)
            self.conv2 = MyGATConv(self.hid * self.in_head, num_classes, concat=False, heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr if self.use_edge_features else None)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr if self.use_edge_features else None)

        return F.log_softmax(x, dim=1)


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr,
                             size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GAT(torch.nn.Module):
    def __init__(self, dataset, use_edge_attr=False):
        super(GAT, self).__init__()
        self.hid = 16
        self.in_head = 8
        self.out_head = 1
        self.use_edge_attr = use_edge_attr

        edge_dim = dataset[0].edge_attr.shape[1] if self.use_edge_attr else None
        self.conv1 = GATConv(dataset.num_features, self.hid, edge_dim=edge_dim, heads=self.in_head, dropout=0.6,
                             add_self_loops=False)
        self.conv2 = GATConv(self.hid * self.in_head, dataset.num_classes, edge_dim=edge_dim, concat=False,
                             heads=self.out_head, dropout=0.6, add_self_loops=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if self.use_edge_attr else None
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, dataset):
        super(MLP, self).__init__()

        self.input_dim = dataset[0].x.shape[1]
        self.output_dim = dataset.num_classes
        self.hidden_dim = int(math.sqrt(self.input_dim * self.output_dim))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin3 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, data):
        x = data.x
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x) + x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin3(x)

        return F.log_softmax(x, dim=1)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()

        self.hidden_dim = int(math.sqrt(input_dim * output_dim))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_dim, output_dim),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_dim, input_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)


class weighted_MSELoss(_Loss):
    def __init__(self, mappings, size_average=None, reduce=None, reduction: str = 'mean'):
        super(weighted_MSELoss, self).__init__(size_average, reduce, reduction)
        self.reduction = reduction
        assert reduction in ['mean', 'sum']
        self.weight = (torch.sum(mappings / torch.sum(mappings, dim=1, keepdim=True), dim=0)).t()

    def forward(self, input, target):
        sum = torch.matmul((input - target) ** 2, self.weight)
        if self.reduction == 'sum':
            return torch.sum(sum)
        else:
            return torch.sum(sum) / input.shape[0]


def dimension_reduction(tensor_data, output_dim, mappings=None, Epoch=2, Batch_size=64, LR=0.002, device='cpu'):
    data = tensor_data.to(device)
    dataset = TensorDataset(data, data)
    [num_samples, input_dim] = tensor_data.shape
    train_ratio = 0.8
    num_train_samples = int(num_samples * train_ratio)
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Batch_size, shuffle=True)
    ae = AutoEncoder(input_dim=input_dim, output_dim=output_dim).to(device)

    print(ae)

    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)

    if mappings is None:
        loss_func = nn.MSELoss(reduction='sum')
    else:
        mappings = mappings.to(device)
        loss_func = weighted_MSELoss(mappings, reduction='sum')

    for epoch in range(Epoch):
        ae.train()
        train_loss = 0
        for step, (x, _) in enumerate(train_loader):
            reconstructed = ae(x)
            loss = loss_func(reconstructed, x)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('====> Epoch: {} Average train loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

        ae.eval()
        val_loss = 0
        with torch.no_grad():
            for step, (x, _) in enumerate(val_loader):
                reconstructed = ae(x)
                val_loss += loss_func(reconstructed, x).item()

        print('====> Epoch: {} Average validation loss: {:.4f}'.format(
            epoch, val_loss / len(val_loader.dataset)))

    print('AutoEncoder training complele.')
    with torch.no_grad():
        encoded_data = ae.encode(data)
        if mappings is not None:
            encoded_mapping = ae.encode(mappings)
            return encoded_data, normalize(encoded_mapping, p=2, dim=1)
        else:
            return encoded_data


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    dataset = LPGdataset('./recommendation')
    mappings = dataset.get_mappings()
    encoded_x, encoded_v_mapping = dimension_reduction(dataset[0].x, output_dim=96, mappings=mappings[0], device=device)
    encoded_e, encoded_e_mapping = dimension_reduction(dataset[0].edge_attr, output_dim=32, mappings=mappings[1],
                                                       device=device)
    data = dataset[0].to(device=device)
    data.x, data.edge_attr = encoded_x, encoded_e
    # model = MyGAT(data, use_edge_features=True, encoded_mappings=[encoded_v_mapping, encoded_e_mapping]).to(device)
    model = MyGAT(data, use_edge_features=True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    model.train()
    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        if epoch % 50 == 0:
            print(loss)

        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))


if __name__ == '__main__':
    train()
