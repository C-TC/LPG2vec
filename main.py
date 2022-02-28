# imports
import os.path

import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from models import GAT, MLP
from dataset import LPGdataset, ArtificialDataset
from util import train_validation_split, count_parameters
from pytorchtools import EarlyStopping


# func: argparser


def main():
    name_data = 'recommendation'
    dataset = LPGdataset('./recommendation', use_v_label=True, use_v_property=True, use_v_str_property=True)
    # dataset.transform = T.NormalizeFeatures()

    print(f"Number of Classes in {name_data}:", dataset.num_classes)
    print(f"Number of Node Features in {name_data}:", dataset.num_node_features)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    model = GAT(dataset).to(device)
    data = dataset[0].to(device)

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


def train():
    return 0


def original(model_name="GAT"):
    num_rounds = 10
    datasets_name = ["recommendation", "citations", "fraud-detection", "fincen", "artificial"]
    # datasets_name = ["artificial"]
    test_conditions = [[False, False, False, False], [True, False, False, False], [True, True, False, False],
                       [True, True, True, True]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    accuracy_data = torch.zeros(num_rounds, len(datasets_name), len(test_conditions))
    dist_data = torch.zeros_like(accuracy_data)
    param_data = torch.zeros(len(datasets_name), len(test_conditions))
    epoch_data = torch.zeros_like(param_data)

    for round in range(num_rounds):
        for dataset_id, dataset_name in enumerate(datasets_name):
            for test_id, test_condition in enumerate(test_conditions):
                if dataset_name == "artificial":
                    dataset = ArtificialDataset(num_nodes=100000, avg_degree=2, num_classes=10,
                                                num_v_label=100, num_v_property=300, num_e_label=20, num_e_property=60,
                                                label_prob=0.8, property_prob=0.8)
                else:
                    dataset = LPGdataset(os.path.join('.', dataset_name), use_v_label=test_condition[0],
                                         use_v_property=test_condition[1],
                                         use_v_str_property=test_condition[1], use_e_label=test_condition[2],
                                         use_e_property=test_condition[3],
                                         use_e_str_property=test_condition[3])
                # dataset.transform = T.NormalizeFeatures()

                print(f"Number of Classes in {dataset_name}:", dataset.num_classes)
                print(f"Number of Node Features in {dataset_name}:", dataset.num_node_features)
                print(
                    f"Test condition: v_label--{test_condition[0]}  v_property--{test_condition[1]}  e_info--{test_condition[2]}")
                print(dataset[0])

                if model_name == "GAT":
                    model = GAT(dataset, use_edge_attr=test_condition[2] or test_condition[3]).to(device)
                elif model_name == "MLP":
                    model = MLP(dataset).to(device)

                num_params = count_parameters(model)
                print(f"Number of parameters: {num_params}")
                param_data[dataset_id, test_id] = num_params

                data = dataset[0].to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

                train_mask, test_mask = data.train_mask, data.test_mask
                train_mask, val_mask = train_validation_split(train_mask, 0.7, device=device)

                early_stopping = EarlyStopping(verbose=False)

                for epoch in range(200):
                    model.train()
                    optimizer.zero_grad()
                    out = model(data)
                    loss = F.nll_loss(out[train_mask], data.y[train_mask])
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_out = model(data)
                        val_loss = F.nll_loss(val_out[val_mask], data.y[val_mask])

                    early_stopping(val_loss.item(), model)

                    if early_stopping.early_stop:
                        print("Early stopping")
                        epoch_data[dataset_id, test_id] = epoch
                        break

                    if epoch % 20 == 0:
                        _, pred = out.max(dim=1)
                        correct = float(pred[train_mask].eq(data.y[train_mask]).sum().item())
                        acc = correct / train_mask.sum().item()
                        print(f"Train Accuracy: {acc:.4f}, Train loss: {loss.item():.4f}")

                        _, val_pred = val_out.max(dim=1)
                        val_correct = float(val_pred[val_mask].eq(data.y[val_mask]).sum().item())
                        val_acc = val_correct / val_mask.sum().item()
                        print(f"Validation Accuracy: {val_acc:.4f}, Validation loss: {val_loss.item():.4f}")

                    epoch_data[dataset_id, test_id] = epoch

                model.load_state_dict(torch.load('checkpoint.pt'))
                model.eval()
                with torch.no_grad():
                    _, pred = model(data).max(dim=1)
                    correct = float(pred[test_mask].eq(data.y[test_mask]).sum().item())
                    acc = correct / test_mask.sum().item()
                    print('Accuracy: {:.4f}'.format(acc))
                    dist = float(
                        torch.mean(torch.abs(pred[test_mask].float() - data.y[test_mask].float())).item())
                    print('Average bin distance: {:.4f}'.format(dist))
                    accuracy_data[round, dataset_id, test_id] = acc
                    dist_data[round, dataset_id, test_id] = dist

    accuracy_mean = torch.mean(accuracy_data, dim=0)
    accuracy_std = torch.std(accuracy_data, dim=0)
    dist_mean = torch.mean(dist_data, dim=0)
    dist_std = torch.std(dist_data, dim=0)
    rootpath = './result/original'
    path = os.path.join(rootpath,model_name)
    os.makedirs(path, exist_ok=True)
    np.savetxt(os.path.join(path, "accuracy_mean.csv"), accuracy_mean.numpy(), delimiter=',')
    np.savetxt(os.path.join(path, "accuracy_std.csv"), accuracy_std.numpy(), delimiter=',')
    np.savetxt(os.path.join(path, "dist_mean.csv"), dist_mean.numpy(), delimiter=',')
    np.savetxt(os.path.join(path, "dist_std.csv"), dist_std.numpy(), delimiter=',')




if __name__ == '__main__':
    #original("MLP")
    original("GAT")
