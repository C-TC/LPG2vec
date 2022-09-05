# imports
import os.path
import torch
import numpy as np
import torch.nn.functional as F
from models import GAT, MLP, dimension_reduction, GAT_ATTLOC
from dataset import LPGdataset, ArtificialDataset, StochasticBlockModelDataset, TempDataset
from util import train_validation_split, count_parameters, preprocess_LPG_data, make_inductive_training_data
from pytorchtools import EarlyStopping


def original(model_name="GAT", num_rounds=5, num_epochs=200, datasets_list=None, root_datasets='./datasets',
             use_early_stopping=True,
             record_test_result=False):
    """
    Testing node classification with models in several test conditions on real datasets.

    :param num_epochs: number of training epochs
    :param model_name: "GAT" or "MLP", the model being tested.
    :param num_rounds: number of rounds of the whole test (to generate the standard deviation result)
    :param datasets_list: datasets to be tested
    :param root_datasets: root path of datasets
    :param use_early_stopping: activate early stopping or not
    :param record_test_result: record the result to csv files or not
    """
    if datasets_list is None:
        datasets_list = ["recommendation", "citations", "fraud-detection", "fincen", "artificial"]

    test_conditions = [[False, False, False, False], [True, False, False, False], [True, True, False, False],
                       [True, True, True, True]]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    accuracy_data = torch.zeros(num_rounds, len(datasets_list), len(test_conditions))
    dist_data = torch.zeros_like(accuracy_data)
    param_data = torch.zeros(len(datasets_list), len(test_conditions))
    epoch_data = torch.zeros_like(param_data)

    for round in range(num_rounds):
        for dataset_id, dataset_name in enumerate(datasets_list):
            for test_id, test_condition in enumerate(test_conditions):
                if dataset_name == "artificial":
                    dataset = ArtificialDataset(num_nodes=100000, avg_degree=2, num_classes=10,
                                                num_v_label=100, num_v_property=300, num_e_label=20, num_e_property=60,
                                                label_prob=0.8, property_prob=0.8)
                else:
                    dataset = LPGdataset(os.path.join(root_datasets, dataset_name), use_v_label=test_condition[0],
                                         use_v_property=test_condition[1],
                                         use_v_str_property=test_condition[1], use_e_label=test_condition[2],
                                         use_e_property=test_condition[3],
                                         use_e_str_property=test_condition[3])

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

                early_stopping = EarlyStopping(verbose=False, patience=50, activate=use_early_stopping)

                for epoch in range(num_epochs):
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

    if record_test_result:
        accuracy_mean = torch.mean(accuracy_data, dim=0)
        accuracy_std = torch.std(accuracy_data, dim=0)
        dist_mean = torch.mean(dist_data, dim=0)
        dist_std = torch.std(dist_data, dim=0)
        rootpath = './result/original'
        path = os.path.join(rootpath, model_name)
        os.makedirs(path, exist_ok=True)
        np.savetxt(os.path.join(path, "accuracy_mean.csv"), accuracy_mean.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_std.csv"), accuracy_std.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "dist_mean.csv"), dist_mean.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "dist_std.csv"), dist_std.numpy(), delimiter=',')


def SBMdata(num_rounds=5, num_epochs=400, model_names=None, channels_list=None, use_early_stopping=False,
            record_test_result=False):
    """
    Testing node classification with models in several test conditions on SBM datasets.
    :param num_rounds: number of rounds of the whole test (to generate the standard deviation result)
    :param num_epochs: number of training epochs
    :param model_names: "GAT" or "MLP", the model being tested.
    :param channels_list: number of channels in SBM dataset
    :param use_early_stopping: activate early stopping or not
    :param record_test_result: record the result to csv files or not
    """
    if channels_list is None:
        channels_list = [50, 100, 200, 400, 800]
    if model_names is None:
        model_names = ["GAT", "MLP"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    accuracy_data = torch.zeros(num_rounds, len(model_names), len(channels_list))

    validation_losses = torch.zeros(len(channels_list), num_epochs)

    for round in range(num_rounds):
        for model_id, model_name in enumerate(model_names):
            for channel_id, channels in enumerate(channels_list):

                num_blocks = 10
                block_size = 1000
                prob_out = 0.02
                prob_degree = 0.05
                block_sizes = [block_size] * num_blocks
                assert prob_out < 1. / num_blocks
                edge_probs = np.ones((num_blocks, num_blocks)) * prob_out * prob_degree + np.eye(num_blocks) * (
                        1 - prob_out * num_blocks) * prob_degree
                dataset = StochasticBlockModelDataset(block_sizes=block_sizes, edge_probs=edge_probs,
                                                      is_undirected=False, num_channels=channels,
                                                      n_informative=channels, n_redundant=0, n_repeated=0,
                                                      flip_target=0.1)

                print(dataset[0])

                if model_name == "GAT":
                    model = GAT(dataset, use_edge_attr=False).to(device)
                elif model_name == "MLP":
                    model = MLP(dataset).to(device)
                print(f"training with model {model_name}.")
                num_params = count_parameters(model)
                print(f"Number of parameters: {num_params}")

                data = dataset[0].to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

                train_mask, test_mask = data.train_mask, data.test_mask
                train_mask, val_mask = train_validation_split(train_mask, 0.7, device=device)

                early_stopping = EarlyStopping(verbose=False, patience=50, activate=use_early_stopping)

                for epoch in range(num_epochs):
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
                        if model_name == "GAT" and round == 0:
                            validation_losses[channel_id, epoch] = val_loss

                    early_stopping(val_loss.item(), model)

                    if early_stopping.early_stop:
                        print("Early stopping")
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

                if early_stopping.activate:
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
                    accuracy_data[round, model_id, channel_id] = acc

    if record_test_result:
        accuracy_mean = torch.mean(accuracy_data, dim=0)
        accuracy_std = torch.std(accuracy_data, dim=0)
        path = './result/SBM'
        os.makedirs(path, exist_ok=True)
        np.savetxt(os.path.join(path, "accuracy_mean.csv"), accuracy_mean.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_std.csv"), accuracy_std.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "validation_losses.csv"), validation_losses.numpy(), delimiter=',')


def autoencoder(num_rounds=5, num_epochs=400, compression_conditions=None, channels_list=None,
                datasets_list=None, root_datasets='./datasets',
                use_early_stopping=False, record_test_result=False):
    """
    Testing node classification with models with autoencoder compressor on several datasets.

    :param num_rounds: number of rounds of the whole test (to generate the standard deviation result)
    :param num_epochs: number of training epochs
    :param channels_list: number of channels in the compressed vector
    :param use_early_stopping: activate early stopping or not
    :param record_test_result: record the result to csv files or not
    :param compression_conditions: compress or not (in a list)
    :param datasets_list: datasets to be tested
    :param root_datasets: root path of datasets
    """
    if datasets_list is None:
        datasets_list = ["SBM", "recommendation", "citations", "fraud-detection", "fincen"]
    if channels_list is None:
        channels_list = [20, 40, 60]
    if compression_conditions is None:
        compression_conditions = [False, True]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    accuracy_data_uncomp = torch.zeros(num_rounds, len(datasets_list), len(channels_list))
    accuracy_data_comp = torch.zeros(num_rounds, len(datasets_list), len(channels_list))

    for round in range(num_rounds):
        for channel_id, channels in enumerate(channels_list):
            for dataset_id, dataset_name in enumerate(datasets_list):
                for comp_id, compression in enumerate(compression_conditions):
                    if compression is False and channel_id != 0:
                        continue
                    if dataset_name == "SBM":
                        num_blocks = 10
                        block_size = 1000
                        prob_out = 0.02
                        prob_degree = 0.05
                        block_sizes = [block_size] * num_blocks
                        assert prob_out < 1. / num_blocks
                        edge_probs = np.ones((num_blocks, num_blocks)) * prob_out * prob_degree + np.eye(num_blocks) * (
                                1 - prob_out * num_blocks) * prob_degree
                        inchannels = 400
                        dataset = StochasticBlockModelDataset(block_sizes=block_sizes, edge_probs=edge_probs,
                                                              is_undirected=False, num_channels=inchannels,
                                                              n_informative=inchannels, n_redundant=0, n_repeated=0,
                                                              flip_target=0.1)
                    else:
                        dataset = LPGdataset(os.path.join(root_datasets, dataset_name),
                                             use_v_label=True,
                                             use_v_property=True,
                                             use_v_str_property=True, use_e_label=False,
                                             use_e_property=False,
                                             use_e_str_property=False)

                    data = dataset[0].to(device)

                    print(f"Dataset: {dataset_name}, Compression: {compression}, Channels if compress: {channels}")
                    train_mask, test_mask = data.train_mask, data.test_mask
                    train_mask, val_mask = train_validation_split(train_mask, 0.7, device=device)

                    if compression:
                        encoded_x = dimension_reduction(data.x, output_dim=channels, Epoch=50, train_mask=train_mask,
                                                        val_mask=val_mask, device=device)
                        dataset = TempDataset(data, new_X=encoded_x)
                        data = dataset[0].to(device)

                    print(data)

                    model = GAT(dataset, use_edge_attr=False).to(device)

                    num_params = count_parameters(model)
                    print(f"Number of parameters: {num_params}")

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

                    early_stopping = EarlyStopping(verbose=False, patience=50, activate=use_early_stopping)

                    for epoch in range(num_epochs):
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

                    if early_stopping.activate:
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
                        if compression:
                            accuracy_data_comp[round, dataset_id, channel_id] = acc
                        else:
                            accuracy_data_uncomp[round, dataset_id, channel_id] = acc

    if record_test_result:
        accuracy_data_mean_uncomp = torch.mean(accuracy_data_uncomp, dim=0)
        accuracy_data_mean_comp = torch.mean(accuracy_data_comp, dim=0)
        accuracy_data_std_uncomp = torch.std(accuracy_data_uncomp, dim=0)
        accuracy_data_std_comp = torch.std(accuracy_data_comp, dim=0)
        path = './result/autoencoder'
        os.makedirs(path, exist_ok=True)
        np.savetxt(os.path.join(path, "accuracy_data_mean_uncomp.csv"), accuracy_data_mean_uncomp.numpy(),
                   delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_data_mean_comp.csv"), accuracy_data_mean_comp.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_data_std_uncomp.csv"), accuracy_data_std_uncomp.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_data_std_comp.csv"), accuracy_data_std_comp.numpy(), delimiter=',')


def MLPtransfer(num_rounds=5, num_epochs=400,
                datasets_list=None, channels_list=None, compression_conditions=None, root_datasets='./datasets',
                use_early_stopping=False, record_test_result=False):
    """

    Testing node classification with models with MLP compressor on several datasets.

    :param num_rounds: number of rounds of the whole test (to generate the standard deviation result)
    :param num_epochs: number of training epochs
    :param channels_list: number of channels in the compressed vector
    :param use_early_stopping: activate early stopping or not
    :param record_test_result: record the result to csv files or not
    :param compression_conditions: compress or not (in a list)
    :param datasets_list: datasets to be tested
    :param root_datasets: root path of datasets
    """
    if compression_conditions is None:
        compression_conditions = [True]
    if channels_list is None:
        channels_list = [20, 40, 60]
    if datasets_list is None:
        datasets_list = ["SBM", "recommendation", "citations", "fraud-detection", "fincen"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    accuracy_data_uncomp = torch.zeros(num_rounds, len(datasets_list), len(channels_list))
    accuracy_data_comp = torch.zeros(num_rounds, len(datasets_list), len(channels_list))

    for round in range(num_rounds):
        for channel_id, channels in enumerate(channels_list):
            for dataset_id, dataset_name in enumerate(datasets_list):
                for comp_id, compression in enumerate(compression_conditions):
                    if compression is False and channel_id != 0:
                        continue
                    if dataset_name == "SBM":
                        num_blocks = 10
                        block_size = 1000
                        prob_out = 0.02
                        prob_degree = 0.05
                        block_sizes = [block_size] * num_blocks
                        assert prob_out < 1. / num_blocks
                        edge_probs = np.ones((num_blocks, num_blocks)) * prob_out * prob_degree + np.eye(num_blocks) * (
                                1 - prob_out * num_blocks) * prob_degree
                        inchannels = 400
                        dataset = StochasticBlockModelDataset(block_sizes=block_sizes, edge_probs=edge_probs,
                                                              is_undirected=False, num_channels=inchannels,
                                                              n_informative=inchannels, n_redundant=0, n_repeated=0,
                                                              flip_target=0.1)
                    else:
                        dataset = LPGdataset(os.path.join(root_datasets, dataset_name),
                                             use_v_label=True,
                                             use_v_property=True,
                                             use_v_str_property=True, use_e_label=False,
                                             use_e_property=False,
                                             use_e_str_property=False)

                    data = dataset[0].to(device)

                    print(f"Dataset: {dataset_name}, Compression: {compression}, Channels if compress: {channels}")
                    train_mask, test_mask = data.train_mask, data.test_mask
                    train_mask, val_mask = train_validation_split(train_mask, 0.7, device=device)

                    if compression:
                        MLPmodel = MLP(dataset, hidden_dim=channels)
                        MLPmodel.to(device)
                        MLPmodel.train_model(data, train_mask=train_mask, val_mask=val_mask)

                        encoded_x = MLPmodel.encode(data.x)
                        dataset = TempDataset(data, new_X=encoded_x)
                        data = dataset[0].to(device)

                    print(data)

                    model = GAT(dataset, use_edge_attr=False).to(device)

                    num_params = count_parameters(model)
                    print(f"Number of parameters: {num_params}")

                    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

                    early_stopping = EarlyStopping(verbose=False, patience=50, activate=use_early_stopping)

                    for epoch in range(num_epochs):
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

                    if early_stopping.activate:
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
                        if compression:
                            accuracy_data_comp[round, dataset_id, channel_id] = acc
                        else:
                            accuracy_data_uncomp[round, dataset_id, channel_id] = acc

    if record_test_result:
        accuracy_data_mean_uncomp = torch.mean(accuracy_data_uncomp, dim=0)
        accuracy_data_mean_comp = torch.mean(accuracy_data_comp, dim=0)
        accuracy_data_std_uncomp = torch.std(accuracy_data_uncomp, dim=0)
        accuracy_data_std_comp = torch.std(accuracy_data_comp, dim=0)
        path = './result/MLPtransfer'
        os.makedirs(path, exist_ok=True)
        np.savetxt(os.path.join(path, "accuracy_data_mean_uncomp.csv"), accuracy_data_mean_uncomp.numpy(),
                   delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_data_mean_comp.csv"), accuracy_data_mean_comp.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_data_std_uncomp.csv"), accuracy_data_std_uncomp.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_data_std_comp.csv"), accuracy_data_std_comp.numpy(), delimiter=',')


def Locator(compression=False, channels_list=None, restrict_W=False, inductive=False, record_att=False, record_att_interval=1,
            num_rounds=5, num_epochs=400, datasets_list=None, root_datasets='./datasets', use_early_stopping=True, record_test_result=False):
    """
    Testing node classification with models with attention locator on several datasets. Notice that we don't consider
    compression with attention locator, but still we provide a compression option.

    :param num_rounds: number of rounds of the whole test (to generate the standard deviation result)
    :param num_epochs: number of training epochs
    :param channels_list: number of channels in the compressed vector
    :param use_early_stopping: activate early stopping or not
    :param record_test_result: record the result to csv files or not
    :param datasets_list: datasets to be tested
    :param root_datasets: root path of datasets
    :param compression: compress or not
    :param restrict_W: use the restricted attention score function or not
    :param record_att: record the attention value to csv file or not
    :param record_att_interval: the interval of two consecutive records
    :param inductive: switch to inductive learning
    """
    if datasets_list is None:
        datasets_list = ["recommendation", "citations", "fraud-detection", "fincen", "artificial"]
    if compression is False:
        channels_list = [-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    accuracy_data = torch.zeros(num_rounds, len(datasets_list), len(channels_list))

    for round in range(num_rounds):
        for channel_id, channels in enumerate(channels_list):
            for dataset_id, dataset_name in enumerate(datasets_list):
                if dataset_name == "SBM":
                    num_blocks = 10
                    block_size = 1000
                    prob_out = 0.02
                    prob_degree = 0.05
                    block_sizes = [block_size] * num_blocks
                    assert prob_out < 1. / num_blocks
                    edge_probs = np.ones((num_blocks, num_blocks)) * prob_out * prob_degree + np.eye(num_blocks) * (
                            1 - prob_out * num_blocks) * prob_degree
                    inchannels = 100
                    dataset = StochasticBlockModelDataset(block_sizes=block_sizes, edge_probs=edge_probs,
                                                          is_undirected=False, num_channels=inchannels,
                                                          n_informative=inchannels, n_redundant=0, n_repeated=0,
                                                          flip_target=0.1)
                else:
                    dataset = LPGdataset(os.path.join(root_datasets, dataset_name),
                                         use_v_label=True,
                                         use_v_property=True,
                                         use_v_str_property=True, use_e_label=False,
                                         use_e_property=False,
                                         use_e_str_property=False)
                    v_mapping, e_mapping = dataset.get_mappings()
                    v_mapping = v_mapping / torch.norm(v_mapping, dim=-1).view(-1, 1)

                data = dataset[0].to(device)

                print(f"Dataset: {dataset_name}, Channels if compress: {channels}")

                train_mask, test_mask = data.train_mask, data.test_mask
                train_mask, val_mask = train_validation_split(train_mask, 0.7, device=device)

                if record_att and round == 0:
                    att_src_data = torch.zeros(int(num_epochs / record_att_interval), v_mapping.shape[0])
                    att_dst_data = torch.zeros(int(num_epochs / record_att_interval), v_mapping.shape[0])

                if compression:
                    MLPmodel = MLP(dataset, hidden_dim=channels)
                    MLPmodel.to(device)
                    MLPmodel.train_model(data, train_mask=train_mask, val_mask=val_mask)
                    encoded_x = MLPmodel.encode(data.x)
                    v_mapping = MLPmodel.encode(v_mapping.to(device))
                    dataset = TempDataset(data, new_X=encoded_x)
                    data = dataset[0].to(device)

                print(data)

                if inductive:
                    train_data = make_inductive_training_data(data, train_mask=train_mask).to(device)
                else:
                    train_data = data

                model = GAT_ATTLOC(data, v_mapping=v_mapping.to(device), restrict_W=restrict_W).to(device)

                num_params = count_parameters(model)
                print(f"Number of parameters: {num_params}")

                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

                early_stopping = EarlyStopping(verbose=False, patience=60, activate=use_early_stopping)

                for epoch in range(num_epochs):
                    model.train()
                    optimizer.zero_grad()
                    out = model(train_data)
                    loss = F.nll_loss(out[train_mask], data.y[train_mask])
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_out = model(train_data)
                        val_loss = F.nll_loss(val_out[val_mask], data.y[val_mask])

                    early_stopping(val_loss.item(), model)

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    if record_att and round == 0 and epoch % record_att_interval == 0:
                        att_src, att_dst = model.get_att_vectors()
                        att_src = att_src.to('cpu')
                        att_dst = att_dst.to('cpu')
                        att_src_data[int(epoch / record_att_interval), :] = F.normalize(att_src)
                        att_dst_data[int(epoch / record_att_interval), :] = F.normalize(att_dst)

                    if epoch % 20 == 0:
                        _, pred = out.max(dim=1)
                        correct = float(pred[train_mask].eq(data.y[train_mask]).sum().item())
                        acc = correct / train_mask.sum().item()
                        print(f"Train Accuracy: {acc:.4f}, Train loss: {loss.item():.4f}")

                        _, val_pred = val_out.max(dim=1)
                        val_correct = float(val_pred[val_mask].eq(data.y[val_mask]).sum().item())
                        val_acc = val_correct / val_mask.sum().item()
                        print(f"Validation Accuracy: {val_acc:.4f}, Validation loss: {val_loss.item():.4f}")

                if early_stopping.activate:
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
                    accuracy_data[round, dataset_id, channel_id] = acc

                if record_att and round == 0:
                    path = './result/attval'
                    if restrict_W:
                        path += '_Restricted'
                    filename_src = dataset_name + '_src.csv'
                    filename_dst = dataset_name + '_dst.csv'
                    os.makedirs(path, exist_ok=True)
                    np.savetxt(os.path.join(path, filename_src), att_src_data.numpy(), delimiter=',')
                    np.savetxt(os.path.join(path, filename_dst), att_dst_data.numpy(), delimiter=',')

    if record_test_result:
        accuracy_data_mean = torch.mean(accuracy_data, dim=0)
        accuracy_data_std = torch.std(accuracy_data, dim=0)
        path = './result/Locator'
        if restrict_W:
            path += '_Restricted'
        os.makedirs(path, exist_ok=True)
        np.savetxt(os.path.join(path, "accuracy_data_mean.csv"), accuracy_data_mean.numpy(), delimiter=',')
        np.savetxt(os.path.join(path, "accuracy_data_std.csv"), accuracy_data_std.numpy(), delimiter=',')


def LocatorDEPRECATED(compression=True):
    num_epochs = 400
    num_rounds = 1
    channels_list = [20, 40, 60]
    datasets_list = ["recommendation", "citations", "fraud-detection", "fincen"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training on device {device}.')

    # accuracy_data_uncomp = torch.zeros(num_rounds, len(datasets_list), len(channels_list))
    # accuracy_data_comp = torch.zeros(num_rounds, len(datasets_list), len(channels_list))
    # dist_data = torch.zeros_like(accuracy_data)
    # param_data = torch.zeros(len(model_names), len(channels_list))
    # epoch_data = torch.zeros_like(param_data)

    # validation_losses = torch.zeros(len(channels_list), num_epochs)

    for round in range(num_rounds):
        for channel_id, channels in enumerate(channels_list):
            for dataset_id, dataset_name in enumerate(datasets_list):
                if dataset_name == "SBM":
                    num_blocks = 10
                    block_size = 1000
                    prob_out = 0.02
                    prob_degree = 0.05
                    block_sizes = [block_size] * num_blocks
                    assert prob_out < 1. / num_blocks
                    edge_probs = np.ones((num_blocks, num_blocks)) * prob_out * prob_degree + np.eye(num_blocks) * (
                            1 - prob_out * num_blocks) * prob_degree
                    inchannels = 100
                    dataset = StochasticBlockModelDataset(block_sizes=block_sizes, edge_probs=edge_probs,
                                                          is_undirected=False, num_channels=inchannels,
                                                          n_informative=inchannels, n_redundant=0, n_repeated=0,
                                                          flip_target=0.1)
                else:
                    dataset = LPGdataset(os.path.join('.', dataset_name), use_v_label=True,
                                         use_v_property=True,
                                         use_v_str_property=True, use_e_label=False,
                                         use_e_property=False,
                                         use_e_str_property=False)
                    v_mapping, e_mapping = dataset.get_mappings()
                    v_mapping = v_mapping / torch.norm(v_mapping, dim=-1).view(-1, 1)

                data = dataset[0].to(device)

                print(f"Dataset: {dataset_name}, Channels if compress: {channels}")
                train_mask, test_mask = data.train_mask, data.test_mask
                train_mask, val_mask = train_validation_split(train_mask, 0.7, device=device)

                if compression:
                    MLPmodel = MLP(dataset, hidden_dim=channels)
                    MLPmodel.to(device)
                    MLPmodel.train_model(data, train_mask=train_mask, val_mask=val_mask)

                    encoded_x = MLPmodel.encode(data.x)
                    v_encoded_mapping = MLPmodel.encode(v_mapping.to(device))
                    dataset = TempDataset(data, new_X=torch.mm(encoded_x, v_encoded_mapping.t()))
                    data = dataset[0].to(device)

                print(data)

                model = GAT(dataset, use_edge_attr=False).to(device)

                num_params = count_parameters(model)
                print(f"Number of parameters: {num_params}")

                # param_data[model_id, channel_id] = num_params

                optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

                early_stopping = EarlyStopping(verbose=False, patience=50, activate=True)

                for epoch in range(num_epochs):
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

                if early_stopping.activate:
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
                    # if compression:
                    #     accuracy_data_comp[round, dataset_id, channel_id] = acc
                    # else:
                    #     accuracy_data_uncomp[round, dataset_id, channel_id] = acc
                    # accuracy_data[round, model_id, channel_id] = acc
                    # dist_data[round, model_id, channel_id] = dist

    # accuracy_data_mean_uncomp = torch.mean(accuracy_data_uncomp, dim=0)
    # accuracy_data_mean_comp = torch.mean(accuracy_data_comp, dim=0)
    # accuracy_data_std_uncomp = torch.std(accuracy_data_uncomp, dim=0)
    # accuracy_data_std_comp = torch.std(accuracy_data_comp, dim=0)
    # path = './result/MLPtransfer'
    # os.makedirs(path, exist_ok=True)
    # np.savetxt(os.path.join(path, "accuracy_data_mean_uncomp.csv"), accuracy_data_mean_uncomp.numpy(), delimiter=',')
    # np.savetxt(os.path.join(path, "accuracy_data_mean_comp.csv"), accuracy_data_mean_comp.numpy(), delimiter=',')
    # np.savetxt(os.path.join(path, "accuracy_data_std_uncomp.csv"), accuracy_data_std_uncomp.numpy(), delimiter=',')
    # np.savetxt(os.path.join(path, "accuracy_data_std_comp.csv"), accuracy_data_std_comp.numpy(), delimiter=',')


if __name__ == '__main__':
    """
    for the first run, need to preprocess LPG data in the "raw_data" folder. 
    The processed files are stored in "datasets/DATASET_NAME/raw" (PYG convention)
    """
    # preprocess_LPG_data()

    """
    Next are tests performed in the project. Consult function definition for usage.
    """
    # original("MLP")
    # original("GAT")
    # SBMdata()
    # autoencoder()
    # MLPtransfer()
    # Locator(restrict_W=False)
    Locator(restrict_W=True, inductive=True)
