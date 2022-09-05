from typing import Union, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset, InMemoryDataset
import os
from torch_geometric.utils import coalesce, remove_self_loops, stochastic_blockmodel_graph
from util import train_test_split
from typing import Callable, List, Optional, Union
import numpy as np


class LPGdataset(Dataset):
    """
    Returns a LPG dataset in the PYG format.
    """
    def __init__(self, root, transform=None, pre_transform=None, use_v_label=True, use_v_property=True,
                 use_v_str_property=True, use_e_label=True, use_e_property=True, use_e_str_property=True):
        # self.raw_dir = os.path.join(root,'raw')
        self.use_v_label = use_v_label
        self.use_v_property = use_v_property
        self.use_v_str_property = use_v_str_property
        self.use_e_label = use_e_label
        self.use_e_property = use_e_property
        self.use_e_str_property = use_e_str_property
        self.num_classes = None

        super(LPGdataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['edge_index.pt', 'vertex_features.pt', 'edge_features.pt', 'target.pt']

    @property
    def processed_file_names(self):
        return 'ALLWAYSPROCESS.pt'

    def download(self):
        pass

    def process(self):
        edge_index = torch.load(os.path.join(self.raw_dir, 'edge_index.pt'))
        vertex_features = torch.load(os.path.join(self.raw_dir, 'vertex_features.pt'))
        edge_features = torch.load(os.path.join(self.raw_dir, 'edge_features.pt'))
        target = torch.load(os.path.join(self.raw_dir, 'target.pt'))

        v_feat, v_str_property = vertex_features[0], vertex_features[1]
        e_feat, e_str_property = edge_features[0], edge_features[1]
        v_label, v_property = v_feat[0], v_feat[1:]
        e_label, e_property = e_feat[0], e_feat[1:]

        num_vertex = v_label.shape[0]
        num_edge = e_label.shape[0]

        def create_property_map(property_list):
            num_property = len(property_list)
            property_length = [property_list[i].shape[1] for i in range(num_property)]
            property_map = torch.zeros(num_property, sum(property_length))
            start = 0
            for i in range(num_property):
                property_map[i, start:start + property_length[i]] = 1
                start += property_length[i]
            return property_map

        x_list = []
        v_label_map, v_property_map, v_str_property_map, v_map = None, None, None, None
        if self.use_v_label:
            x_list += [v_label]
            v_label_map = torch.eye(v_label.shape[1])
        if self.use_v_property:
            x_list += v_property
            v_property_map = create_property_map(v_property)
        if self.use_v_str_property:
            x_list += v_str_property
            v_str_property_map = create_property_map(v_str_property)
        if not x_list:
            # only structure info, use random vertex features.
            x = torch.rand(num_vertex, 50)
            v_map = torch.eye(50)
        else:
            x = torch.cat(x_list, dim=1)
            v_maplist = [v_label_map, v_property_map, v_str_property_map]
            v_map = torch.block_diag(*[map for map in v_maplist if map is not None])

        edge_attr_list = []
        e_label_map, e_property_map, e_str_property_map, e_map = None, None, None, None
        if self.use_e_label:
            edge_attr_list += [e_label]
            e_label_map = torch.eye(e_label.shape[1])
        if self.use_e_property:
            edge_attr_list += e_property
            e_property_map = create_property_map(e_property)
        if self.use_e_str_property:
            edge_attr_list += e_str_property
            e_str_property_map = create_property_map(e_str_property)
        if not edge_attr_list:
            edge_attr = torch.rand(num_edge, 10)
            e_map = torch.eye(10)
        else:
            edge_attr = torch.cat(edge_attr_list, dim=1)
            e_maplist = [e_label_map, e_property_map, e_str_property_map]
            e_map = torch.block_diag(*[map for map in e_maplist if map is not None])

        print(
            f"vertices:{num_vertex}, edges:{num_edge}, v_label_dim:{v_label.shape[1]}, e_label_dim:{e_label.shape[1]},"
            f" v_prop_dim:{x.shape[1] - v_label.shape[1]}, e_prop_dim:{edge_attr.shape[1] - e_label.shape[1]}")

        train_mask, test_mask = train_test_split(x.shape[0], 0.7)

        data = Data(x=x.float(), edge_index=edge_index.long(), edge_attr=edge_attr.float(), y=target.long(),
                    train_mask=train_mask, test_mask=test_mask)
        torch.save(data, os.path.join(self.processed_dir, 'data.pt'))
        torch.save([v_map, e_map], os.path.join(self.processed_dir, 'mappings.pt'))

        self.num_classes = torch.unique(target).shape[0]

    def len(self):
        # only one graph in dataset
        return 1

    def get(self, idx: int):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data

    def get_mappings(self):
        mappings = torch.load(os.path.join(self.processed_dir, 'mappings.pt'))
        return mappings


def get_edge_index(num_src_nodes, num_dst_nodes, avg_degree):
    num_edges = num_src_nodes * avg_degree
    row = torch.randint(num_src_nodes, (num_edges,), dtype=torch.int64)
    col = torch.randint(num_dst_nodes, (num_edges,), dtype=torch.int64)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index=edge_index)
    num_nodes = max(num_src_nodes, num_dst_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index


class ArtificialDataset(InMemoryDataset):
    """
    The artificial dataset that mimics PYG FakeDataset.
    """
    def __init__(self, num_nodes=1000, avg_degree=10, num_classes=10,
                 num_v_label=20, num_v_property=200, num_e_label=5, num_e_property=50,
                 label_prob=0.3, property_prob=0.6, fuse_y_to_x=True, fuse_y_to_edge_attr=True):
        super().__init__('.')
        self.num_nodes = max(num_nodes, avg_degree)
        self.avg_degree = max(avg_degree, 1)
        self._num_classes = num_classes
        self.num_v_label = num_v_label
        self.num_v_property = num_v_property
        self.num_e_label = num_e_label
        self.num_e_property = num_e_property
        self.label_prob = label_prob
        self.property_prob = property_prob
        self.fuse_y_to_x = fuse_y_to_x
        self.fuse_y_to_edge_attr = fuse_y_to_edge_attr

        data_list = [self.generate_data()]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self):
        y = torch.randint(self._num_classes, (self.num_nodes,))
        edge_index = get_edge_index(self.num_nodes, self.num_nodes, self.avg_degree)

        x_label_mask = torch.rand(self.num_nodes, self.num_v_label)
        x_label_mask = x_label_mask < self.label_prob
        x_label = torch.rand(self.num_nodes, self.num_v_label)
        x_label[x_label_mask] = 1
        x_label[~x_label_mask] = 0
        x_property = torch.rand((self.num_nodes, self.num_v_property))
        x = torch.cat([x_label, x_property], dim=1)
        if self.fuse_y_to_x:
            x = x + y.view(-1, 1)
        if self.num_v_label == 0 and self.num_v_property == 0:
            x = torch.rand(self.num_nodes, 50)

        e_label_mask = torch.rand(edge_index.shape[1], self.num_e_label)
        e_label_mask = e_label_mask < self.label_prob
        e_label = torch.rand(edge_index.shape[1], self.num_e_label)
        e_label[e_label_mask] = 1
        e_label[~e_label_mask] = 0
        e_property = torch.rand((edge_index.shape[1], self.num_e_property))
        edge_attr = torch.cat([e_label, e_property], dim=1)
        if self.num_e_label == 0 and self.num_e_property == 0:
            edge_attr = torch.rand(edge_index.shape[1], 20)
        if self.fuse_y_to_edge_attr:
            edge_attr = edge_attr + y[edge_index[0]].view(-1, 1) - y[edge_index[1]].view(-1, 1)
        train_mask, test_mask = train_test_split(y.shape[0], 0.7)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, test_mask=test_mask)
        return data


class StochasticBlockModelDataset(InMemoryDataset):
    """
    Artificial Dataset similar to PYG StochasticBlockModelDataset.
    """
    def __init__(
            self,
            block_sizes: Union[List[int], Tensor],
            edge_probs: Union[List[List[float]], Tensor],
            num_channels: Optional[int] = None,
            is_undirected: bool = True,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            flip_target=0,
            **kwargs,
    ):
        super().__init__('.')
        if not isinstance(block_sizes, torch.Tensor):
            block_sizes = torch.tensor(block_sizes, dtype=torch.long)
        if not isinstance(edge_probs, torch.Tensor):
            edge_probs = torch.tensor(edge_probs, dtype=torch.float)

        self.block_sizes = block_sizes
        self.edge_probs = edge_probs
        self.num_channels = num_channels
        self.is_undirected = is_undirected
        self.flip_target = flip_target

        self.kwargs = {
            'n_informative': n_informative,
            'n_redundant': n_redundant,
            'n_repeated': n_repeated,
            'flip_y': 0.0,
            'shuffle': False,
        }
        self.kwargs.update(kwargs)

        data_list = [self.generate_data()]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self):
        from sklearn.datasets import make_classification

        edge_index = stochastic_blockmodel_graph(
            self.block_sizes, self.edge_probs, directed=not self.is_undirected)

        num_samples = int(self.block_sizes.sum())
        num_classes = self.block_sizes.size(0)

        x = None
        if self.num_channels is not None:
            x, _ = make_classification(
                n_samples=num_samples,
                n_features=self.num_channels,
                n_classes=num_classes,
                weights=self.block_sizes / num_samples,
                **self.kwargs,
            )
            x = torch.from_numpy(x).to(torch.float)
        y = torch.arange(num_classes).repeat_interleave(self.block_sizes)
        train_mask, test_mask = train_test_split(num_samples, 0.7)

        if self.flip_target > 0:
            flip_mask = torch.rand(num_samples) < self.flip_target
            y[flip_mask] = torch.randint(num_classes, size=(torch.sum(flip_mask).item(),))

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

        return data


class TempDataset(InMemoryDataset):
    """
    Utility dataset to generate new dataset using modified data. (Bypassing some restrictions in PYG.)
    """
    def __init__(self, oldData, new_X=None, new_edge_attr=None):
        super().__init__('.')
        self.oldData = oldData
        self.new_X = new_X
        self.new_edge_attr = new_edge_attr

        data_list = [self.generate_data()]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self):
        data = Data(x=self.oldData.x.clone().detach() if self.new_X is None else self.new_X,
                    edge_index=self.oldData.edge_index.clone().detach(),
                    edge_attr=self.oldData.edge_attr.clone().detach() if self.new_edge_attr is None and self.oldData.edge_attr is not None else self.new_edge_attr,
                    y=self.oldData.y, train_mask=self.oldData.train_mask, test_mask=self.oldData.test_mask)
        return data


def get_dataset_info():
    datasets_name = ["recommendation", "citations", "fraud-detection", "fincen"]
    for name in datasets_name:
        print(name)
        data = LPGdataset(os.path.join('.', name))
