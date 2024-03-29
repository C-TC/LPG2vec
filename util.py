import torch_geometric.data.data

from feature_encoders import *
import os
import torch
import pandas as pd
from torch_geometric.data import Data

def encoding(data, col_info):
    tmp_data = data[col_info.keys()]
    features = []
    features_str = []
    mapping = None
    target = None
    for col in tmp_data.keys():
        if col_info[col] == ColType.INDEX:
            encoder = IndexEncoder()
            mapping = encoder(tmp_data[col])
            continue
        elif col_info[col] == ColType.CLASSIFICATION_TARGET:
            encoder = BinEncoder()
            target = torch.argmax(encoder(tmp_data[col]), 1)
            continue
        elif col_info[col] == ColType.CATEGORY:
            encoder = CategoryEncoder()
        elif col_info[col] == ColType.NUMERICAL:
            encoder = NumericalEncoder()
        elif col_info[col] == ColType.LARGE_NUMBER:
            encoder = LargeNumberEncoder()
        elif col_info[col] == ColType.NUMBER_TO_BIN:
            encoder = BinEncoder()
        elif col_info[col] == ColType.STRING:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            encoder = StringEncoder(device=device)
            feature_str = encoder(tmp_data[col])
            features_str.append(feature_str if feature_str.dim() != 1 else torch.unsqueeze(feature_str, 1))
            continue

        feature = encoder(tmp_data[col])
        features.append(feature if feature.dim() != 1 else torch.unsqueeze(feature, 1))

    # attr = torch.cat(features, dim=1)
    return mapping, [features, features_str], target


def generate_edge_index(edge_data, vertex_mapping):
    start = edge_data["_start"].astype('int64').values
    end = edge_data["_end"].astype('int64').values
    edge_index = [[vertex_mapping[start[i]], vertex_mapping[end[i]]] for i in range(len(start))]
    result = torch.tensor(edge_index, dtype=torch.int32)
    return result.t()


def generate_feature_vector(filename):
    if os.path.basename(filename) == 'recommendation.csv':
        v_columns_info = {"_id": ColType.INDEX, "_labels": ColType.CATEGORY, "budget": ColType.LARGE_NUMBER,
                          "countries": ColType.CATEGORY, "imdbRating": ColType.CLASSIFICATION_TARGET,
                          "imdbVotes": ColType.NUMBER_TO_BIN, "name": ColType.STRING, "plot": ColType.STRING,
                          "revenue": ColType.LARGE_NUMBER, "runtime": ColType.NUMBER_TO_BIN, "title": ColType.STRING,
                          "year": ColType.NUMBER_TO_BIN}
        e_columns_info = {"_type": ColType.CATEGORY, "rating": ColType.NUMBER_TO_BIN,
                          "role": ColType.STRING, "timestamp": ColType.LARGE_NUMBER}
    elif os.path.basename(filename) == 'citations.csv':
        v_columns_info = {"_id": ColType.INDEX, "_labels": ColType.CATEGORY, "abstract": ColType.STRING,
                          "n_citation": ColType.CLASSIFICATION_TARGET, "title": ColType.STRING,
                          "year": ColType.NUMBER_TO_BIN}
        e_columns_info = {"_type": ColType.CATEGORY}
    elif os.path.basename(filename) == 'fraud-detection.csv':
        v_columns_info = {"_id": ColType.INDEX, "_labels": ColType.CATEGORY, "amount": ColType.CLASSIFICATION_TARGET,
                          "name": ColType.STRING}
        e_columns_info = {"_type": ColType.CATEGORY}
    elif os.path.basename(filename) == 'fincen.csv':
        v_columns_info = {"_id": ColType.INDEX, "_labels": ColType.CATEGORY, "amount": ColType.CLASSIFICATION_TARGET,
                          "begin_date": ColType.NUMBER_TO_BIN, "beneficiary_bank": ColType.STRING,
                          "end_date": ColType.NUMBER_TO_BIN,
                          "name": ColType.STRING, "originator_bank": ColType.STRING}
        e_columns_info = {"_type": ColType.CATEGORY}
    else:
        raise NotImplementedError

    raw_data = pd.read_csv(filename, low_memory=False)
    vertex_data = raw_data[raw_data["_start"].isna()].dropna(axis=1, how="all")
    edge_data = raw_data[raw_data["_start"].notna()].dropna(axis=1, how="all")
    vertex_data = vertex_data[v_columns_info.keys()]

    mapping, v_feat, target = encoding(vertex_data, v_columns_info)
    _, e_feat, _ = encoding(edge_data, e_columns_info)
    e_index = generate_edge_index(edge_data, mapping)
    return e_index, v_feat, e_feat, target


def train_test_split(num_vertex, train_ratio):
    train_mask, test_mask = torch.zeros(num_vertex, dtype=torch.bool), torch.zeros(num_vertex, dtype=torch.bool)
    prob = torch.rand(num_vertex)
    train_mask[prob < train_ratio] = True
    test_mask[prob >= train_ratio] = True
    return train_mask, test_mask


def train_validation_split(train_mask, train_ratio, device='cpu'):
    num_vertex = train_mask.shape[0]
    # new_train_mask, val_mask = torch.zeros(num_vertex, dtype=torch.bool), torch.zeros(num_vertex, dtype=torch.bool)
    prob = torch.rand(num_vertex).to(device)
    new_train_mask = torch.where((prob < train_ratio) & (train_mask == True), True, False)
    val_mask = torch.where((prob >= train_ratio) & (train_mask == True), True, False)
    return new_train_mask, val_mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_raw_data(filename, data_loadpath='./raw_data', dataset_savepath='./datasets'):
    data_path = os.path.join(data_loadpath, filename + '.csv')
    e_index, v_feat, e_feat, target = generate_feature_vector(data_path)
    dataset_path = os.path.join(dataset_savepath, filename, 'raw')
    os.makedirs(dataset_path, exist_ok=True)
    torch.save(e_index, os.path.join(dataset_path, 'edge_index.pt'))
    torch.save(v_feat, os.path.join(dataset_path, 'vertex_features.pt'))
    torch.save(e_feat, os.path.join(dataset_path, 'edge_features.pt'))
    torch.save(target, os.path.join(dataset_path, 'target.pt'))

    # train_mask, test_mask = train_test_split(target.shape[0], 0.7)
    # torch.save(train_mask, os.path.join(root, 'train_mask.pt'))
    # torch.save(test_mask, os.path.join(root, 'test_mask.pt'))


def preprocess_LPG_data(datasets_list=None):
    if datasets_list is None:
        datasets_list = ["recommendation", "citations", "fraud-detection", "fincen"]

    for dataset_name in datasets_list:
        generate_raw_data(dataset_name)


def make_inductive_training_data(data: torch_geometric.data.data.Data, train_mask):
    """
    remove edges related to validation and test vertices. only edge_index and edge_attr are new, others are shallow copy.
    """

    def isin(ar1, ar2):
        return (ar1[..., None] == torch.squeeze(ar2)).any(-1)

    def make_inductive_edge_index_attr(edge_index: torch.Tensor, edge_attr, train_mask):
        index_list = edge_index.clone().detach()
        index_to_remove = torch.nonzero(~train_mask)

        attr = edge_attr.clone().detach()

        from_mask = isin(index_list[0, :], index_to_remove)
        to_mask = isin(index_list[1, :], index_to_remove)

        index_list = index_list[:, ~(from_mask | to_mask)]
        attr = attr[~(from_mask | to_mask), :]

        return index_list, attr

    new_edge_index, new_edge_attr = make_inductive_edge_index_attr(data.edge_index, data.edge_attr, train_mask)
    train_data = Data(x=data.x, edge_index=new_edge_index, y=data.y, edge_attr=new_edge_attr, train_mask=data.train_mask, test_mask=data.test_mask)
    return train_data

if __name__ == '__main__':
    preprocess_LPG_data()
