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
from pytorchtools import EarlyStopping
from dataset import LPGdataset


class GAT_ATTLOC_CONV(MessagePassing):
    """
    Graph convolution layer based on GATConv and integrated with attention locator.
    """
    def __init__(self, in_channels, out_channels, v_mapping, heads=1, e_mapping=None, concat=True,
                 negative_slope=0.2, dropout=0.0, edge_dim=None, bias=True, restrict_W=False):
        """
        :param in_channels:
        :param out_channels:
        :param v_mapping: vertex feature mapping matrix
        :param heads: heads in attention mechanism
        :param e_mapping: edge feature mapping matrix
        :param concat: concatenate multi-head hidden embedding or averaging them.
        :param negative_slope: slope of leaky relu
        :param dropout: drop out probability
        :param edge_dim: edge feature dimension
        :param bias:
        :param restrict_W: use restricted attention score function or not
        """
        super().__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.v_encoded_mapping = v_mapping
        self.e_encoded_mapping = e_mapping

        self.lin_dst = self.lin_src = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')

        self.num_v_features = v_mapping.shape[0]
        self.att_src = Parameter(torch.Tensor(1, self.num_v_features))
        self.att_dst = Parameter(torch.Tensor(1, self.num_v_features))

        self.restrict_W = restrict_W
        if self.restrict_W:
            self.W_r = Parameter(torch.Tensor(self.num_v_features, v_mapping.shape[1]))

        if edge_dim is not None:
            if self.e_encoded_mapping is None:
                self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
                self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
            else:
                self.lin_edge = None
                self.num_e_features = e_mapping.shape[0]
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
        if self.restrict_W:
            glorot(self.W_r)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        if edge_attr is None:
            assert self.edge_dim is None

        H, C = self.heads, self.out_channels
        x_src = x_dst = self.lin_src(x).view(-1, H, C)
        x_new = (x_src, x_dst)

        if self.restrict_W:
            alpha_src = (torch.mm(x, normalize(self.v_encoded_mapping.t() * torch.sigmoid(self.W_r).t(), dim=0)) * normalize(self.att_src)).sum(dim=-1)
            alpha_dst = (torch.mm(x, normalize(self.v_encoded_mapping.t() * torch.sigmoid(self.W_r).t(), dim=0)) * normalize(self.att_dst)).sum(dim=-1)
        else:
            alpha_src = (torch.mm(x, torch.transpose(self.v_encoded_mapping, 0, 1)) * normalize(self.att_src)).sum(dim=-1)
            alpha_dst = (torch.mm(x, torch.transpose(self.v_encoded_mapping, 0, 1)) * normalize(self.att_dst)).sum(dim=-1)

        alpha = (alpha_src, alpha_dst)

        out = x_src + self.propagate(edge_index, x=x_new, alpha=alpha, edge_attr=edge_attr, size=size)

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
            if self.e_encoded_mapping is None:
                edge_attr = self.lin_edge(edge_attr)
                edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
                alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
                alpha = alpha + alpha_edge
            else:
                alpha_edge = (torch.mm(edge_attr, torch.transpose(self.e_encoded_mapping, 0, 1)) * self.att_edge).sum(
                    dim=-1)
                alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, 1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

    def get_att_vectors(self):
        return self.att_src.data, self.att_dst.data


class GAT_ATTLOC(torch.nn.Module):
    """
    Graph Neural Network integrated with attention locator.

    sturcture:
        first layer(conv1)      GATConv with attention locator

        second layer(conv1)     GATConv
    """
    def __init__(self, data, v_mapping, e_mapping=None, use_edge_features=False, restrict_W=False):
        super(GAT_ATTLOC, self).__init__()
        self.hid = 16
        self.in_head = 8
        self.out_head = 1
        self.use_edge_features = use_edge_features

        if e_mapping is not None:
            assert use_edge_features

        num_features = data.x.shape[1]
        num_classes = len(data.y.unique())
        if use_edge_features:
            edge_dim = data.edge_attr.shape[1]
            if e_mapping is None:
                self.conv1 = GAT_ATTLOC_CONV(num_features, self.hid, v_mapping=v_mapping, edge_dim=edge_dim, heads=self.in_head, dropout=0.6, restrict_W=restrict_W)
            else:
                self.conv1 = GAT_ATTLOC_CONV(num_features, self.hid, v_mapping=v_mapping, e_mapping=e_mapping, edge_dim=edge_dim,
                                             heads=self.in_head, dropout=0.6, restrict_W=restrict_W)
            self.conv2 = GATConv(self.hid * self.in_head, num_classes, edge_dim=edge_dim, concat=False,
                                   heads=self.out_head, dropout=0.6)
        else:
            self.conv1 = GAT_ATTLOC_CONV(num_features, self.hid, v_mapping=v_mapping, heads=self.in_head, dropout=0.6, restrict_W=restrict_W)
            self.conv2 = GATConv(self.hid * self.in_head, num_classes, concat=False, heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr if self.use_edge_features else None)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr if self.use_edge_features else None)

        return F.log_softmax(x, dim=1)

    def get_att_vectors(self):
        return self.conv1.get_att_vectors()


class GATConv(MessagePassing):
    """
    Similar to PYG GATConv.
    """
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = False,
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
        out = x_src + self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr,
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
    """
    GAT network based on GATConv.
    """
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
    """
    MLP classifier on node classification.
    """
    def __init__(self, dataset, hidden_dim=None):
        super(MLP, self).__init__()

        self.input_dim = dataset[0].x.shape[1]
        self.output_dim = dataset.num_classes
        self.hidden_dim = int(math.sqrt(self.input_dim * self.output_dim)) if hidden_dim is None else hidden_dim

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

    def encode(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x) + x
        return x.detach().clone()

    def train_model(self, data, train_mask, val_mask, num_epochs=200, use_early_stopping=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)
        early_stopping = EarlyStopping(verbose=False, patience=50, activate=use_early_stopping)

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            out = self(data)
            loss = F.nll_loss(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                val_out = self(data)
                val_loss = F.nll_loss(val_out[val_mask], data.y[val_mask])

            early_stopping(val_loss.item(), self)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if epoch % 40 == 0:
                _, pred = out.max(dim=1)
                correct = float(pred[train_mask].eq(data.y[train_mask]).sum().item())
                acc = correct / train_mask.sum().item()
                print(f"Train Accuracy: {acc:.4f}, Train loss: {loss.item():.4f}")

                _, val_pred = val_out.max(dim=1)
                val_correct = float(val_pred[val_mask].eq(data.y[val_mask]).sum().item())
                val_acc = val_correct / val_mask.sum().item()
                print(f"Validation Accuracy: {val_acc:.4f}, Validation loss: {val_loss.item():.4f}")

        if use_early_stopping:
            self.load_state_dict(torch.load('checkpoint.pt'))


class AutoEncoder(nn.Module):
    """
    AutoEncoder used to compress feature vectors.
    """
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
    """
    A modified MSE loss that is weighted by feature length.
    """
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


def dimension_reduction(tensor_data, output_dim, train_mask, val_mask, mappings=None, Epoch=50, Batch_size=64, LR=0.002,
                        device='cpu'):
    """
    Use autoencoder to compress the feature vector.
    """
    data = tensor_data.to(device)
    [num_samples, input_dim] = tensor_data.shape
    train_set, val_set = TensorDataset(data[train_mask], data[train_mask]), TensorDataset(data[val_mask],
                                                                                          data[val_mask])
    train_loader = DataLoader(train_set, batch_size=Batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Batch_size, shuffle=True)
    ae = AutoEncoder(input_dim=input_dim, output_dim=output_dim).to(device)

    # print(ae)

    optimizer = torch.optim.Adam(ae.parameters(), lr=LR)

    if mappings is None:
        loss_func = nn.MSELoss(reduction='sum')
    else:
        mappings = mappings.to(device)
        loss_func = weighted_MSELoss(mappings, reduction='sum')

    early_stopping = EarlyStopping(verbose=False, patience=7, activate=True)

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

        ae.eval()
        val_loss = 0
        with torch.no_grad():
            for step, (x, _) in enumerate(val_loader):
                reconstructed = ae(x)
                val_loss += loss_func(reconstructed, x).item()
        if epoch % 10 == 0:
            print('====> Epoch: {} Average train loss: {:.4f}'.format(
                epoch, train_loss / len(train_loader.dataset)))
            print('====> Epoch: {} Average validation loss: {:.4f}'.format(
                epoch, val_loss / len(val_loader.dataset)))

        early_stopping(val_loss / len(val_loader.dataset), ae)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if early_stopping.activate:
        ae.load_state_dict(torch.load('checkpoint.pt'))

    print('AutoEncoder training complele.')
    with torch.no_grad():
        encoded_data = ae.encode(data)
        if mappings is not None:
            encoded_mapping = ae.encode(mappings)
            return encoded_data, normalize(encoded_mapping, p=2, dim=1)
        else:
            return encoded_data

