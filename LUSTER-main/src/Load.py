import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader

def BalanceNegSampler(inter, all_nodes, network_layer):
    inter = inter.drop_duplicates().sort_values(by='left', ignore_index=True)
    neg_samples = []
    group = inter.groupby('left')
    for node, pos_inters in group:
        pos_list = pos_inters['right'].tolist()
        choice_num = len(pos_list)
        pos_list.append(node)  # Node itself
        if choice_num <= len(all_nodes) - len(pos_list):
            neg_samples += random.sample(list(filter(lambda x: x not in pos_list, all_nodes)), choice_num)
        else:
            neg_samples += random.choices(list(filter(lambda x: x not in pos_list, all_nodes)), k=choice_num)
    inter['neg'] = neg_samples
    inter['layer_label'] = network_layer
    inter['yes'] = 1
    inter['no'] = 0
    pos_links = inter[['layer_label', 'left', 'right', 'yes']]
    neg_links = inter[['layer_label', 'left', 'neg', 'no']]
    pos_links.columns = ['layer', 'left', 'right', 'exist']
    neg_links.columns = ['layer', 'left', 'right', 'exist']
    result = pd.concat([pos_links, neg_links], axis=0).reset_index(drop=True)
    return result

def AllNegSampler(inter, all_nodes, network_layer):
    sample = []
    group = inter.groupby('left')
    for node, inter in group:
        pos_list = inter['right'].tolist()
        for temp_node in all_nodes:
            if temp_node == node:
                continue
            if temp_node in pos_list:
                sample.append([network_layer, node, temp_node, 1])
            else:
                sample.append([network_layer, node, temp_node, 0])
    sample = np.array(sample)
    sample = pd.DataFrame(sample)
    sample.columns = ['layer', 'left', 'right', 'exist']
    return sample

def get_loader(infor, batch_size):
    network = torch.LongTensor(infor[:, 0])
    leftnode = torch.LongTensor(infor[:, 1])
    rightnode = torch.LongTensor(infor[:, 2])
    link = torch.LongTensor(infor[:, 3])
    data_set = TensorDataset(network, leftnode, rightnode, link)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader

def gcndata_load(inters, all_nodes):
    pos_edge = np.array(inters).tolist()
    g = nx.Graph(pos_edge)
    g.add_nodes_from(all_nodes)
    adj = nx.to_scipy_sparse_matrix(g, nodelist=all_nodes, dtype=int, format='coo')
    edge_index = torch.LongTensor(np.vstack((adj.row, adj.col)))
    x = torch.unsqueeze(torch.FloatTensor(all_nodes), 1)
    gcn_data = Data(x=x, edge_index=edge_index)
    return gcn_data

def load_data(dataset, batch_size):
    datadir = 'data/' + dataset + '_data/'
    layerfiles = os.listdir(datadir)
    network_numbers = len(layerfiles)
    change = []
    whole_edges_num = 0
    for i in range(network_numbers):
        now_layer = datadir + dataset + str(i+1) + '.txt'
        now_inter = pd.read_csv(now_layer, sep=' ', header=None)
        now_nodes = list(set(np.array(now_inter).reshape(-1)))
        print('-----------------------------------')
        print('Nodes of layer ' + str(i + 1) + ": " + str(len(now_nodes)))
        print('Edges of layer ' + str(i + 1) + ": " + str(now_inter.shape[0]))
        whole_edges_num += now_inter.shape[0]
        change += now_nodes
    change = list(set(change))
    change_dict = {}
    for i in range(len(change)):
        change_dict[change[i]] = i
    whole_nodes = list(change_dict.values())
    print('-----------------------------------')
    print('Nodes of all layers: ', len(whole_nodes))
    print('Edges of all layers: ', whole_edges_num)
    print('-----------------------------------')
    data = pd.DataFrame()
    gcn_data = []

    for i in range(network_numbers):
        now_layer = datadir + dataset + str(i+1) + '.txt'
        now_inter = pd.read_csv(now_layer, sep=' ', header=None, names=['left', 'right'])
        now_inter['left'] = now_inter['left'].map(change_dict)
        now_inter['right'] = now_inter['right'].map(change_dict)
        # can choose BalanceNegSampler (get equal proportion of unconnected links)
        # or AllNegSampler (get all unconnected links)
        if dataset == 'TF' or dataset == 'Reddit':
            result = BalanceNegSampler(now_inter, whole_nodes, i)
        else:
            result = AllNegSampler(now_inter, whole_nodes, i)
        data = pd.concat([data, result], axis=0).reset_index(drop=True)
        gcn_data.append(gcndata_load(now_inter, whole_nodes))


    data = np.array(data)
    np.random.shuffle(data)
    # 80%train+10%valid+10%test
    train_infor, test_infor, train_label, test_label = train_test_split(data,  data[:, 3], test_size=0.2)
    valid_infor, test_infor, valid_label, test_label = train_test_split(test_infor, test_label, test_size=0.5)
    # whether training set oversampling
    if dataset == 'TF' or dataset == 'Reddit':  # the amount of training data is sufficient
        print("train counter: ", sorted(Counter(train_label).items()))
    else:
        print("train counter: ", sorted(Counter(train_label).items()))
        over = RandomOverSampler(sampling_strategy=1)
        train_infor, train_label = over.fit_resample(train_infor, train_label)
        print("train over sampling results: ", sorted(Counter(train_label).items()))
    print("valid counter: ", sorted(Counter(valid_label).items()))
    print("test counter: ", sorted(Counter(test_label).items()))
    print('-----------------------------------')
    train_loader = get_loader(train_infor, batch_size)
    valid_loader = get_loader(valid_infor, batch_size)
    test_loader = get_loader(test_infor, batch_size)

    return train_loader, valid_loader, test_loader, gcn_data, network_numbers