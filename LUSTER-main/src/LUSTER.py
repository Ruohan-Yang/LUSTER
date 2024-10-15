import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from sklearn import metrics
from src.Orthogonal import Ortho_algorithm

class GCN(nn.Module):
    def __init__(self, feature_dims, out_dims, hidden_dims):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_dims, hidden_dims)
        self.bn = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dims, out_dims)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

class LUSTER_Net(nn.Module):
    def __init__(self, embedding_dim, layer_number, gcn_data):
        super(LUSTER_Net, self).__init__()

        self.layer_number = layer_number
        self.node_dim = embedding_dim
        self.edge_dim = embedding_dim * 2
        self.gcn_data = gcn_data

        for i in range(self.layer_number):
            gcn = GCN(feature_dims=1, out_dims=self.node_dim, hidden_dims=64)
            setattr(self, 'gcn%i' % i, gcn)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.edge_dim, out_channels=self.edge_dim, kernel_size=7, padding=3),
            nn.ReLU())

        self.layer_classifier = nn.Linear(self.edge_dim, self.layer_number)
        self.link_classifier = nn.Linear(self.edge_dim, 2)



    def forward(self, now_layer, leftnode, rightnode):

        for i in range(self.layer_number):
            layer_embed = eval('self.gcn'+str(i))(self.gcn_data[i]).cuda()
            setattr(self, 'layer%i' % i, layer_embed)

        layer_names = ['self.layer'+str(i) for i in now_layer.cpu().numpy().tolist()]
        layer_specific = torch.Tensor().cuda()
        for (l, i, j) in zip(layer_names, leftnode, rightnode):
            temp = torch.cat((eval(l)[i], eval(l)[j]), dim=0).cuda()
            temp = torch.unsqueeze(temp, dim=0)
            layer_specific = torch.cat((layer_specific, temp), dim=0)

        shared = self.cnn(layer_specific.permute(1, 0)).permute(1, 0)
        discriminant_outs = self.layer_classifier(shared)

        Enhanced_representation = Ortho_algorithm(layer_specific, shared)
        prediction_outs = self.link_classifier(Enhanced_representation)

        return prediction_outs, discriminant_outs

    def metrics_eval(self, eval_data):
        scores = []
        labels = []
        preds = []
        for data in eval_data:
            network_labels, left_nodes, right_nodes, link_labels = data
            with torch.no_grad():
                network_labels = Variable(network_labels).cuda()
                left_nodes = Variable(left_nodes).cuda()
                right_nodes = Variable(right_nodes).cuda()
                link_labels = Variable(link_labels).cuda()
            output, _ = self.forward(network_labels, left_nodes, right_nodes)
            output = F.softmax(output, dim=1)
            _, argmax = torch.max(output, 1)
            scores += list(output[:, 1].cpu().detach().numpy())
            labels += list(link_labels.cpu().detach().numpy())
            preds += list(argmax.cpu().detach().numpy())

        acc = metrics.accuracy_score(labels, preds)
        pre = metrics.precision_score(labels, preds, average='weighted')
        f1 = metrics.f1_score(labels, preds, average='weighted')
        auc = metrics.roc_auc_score(labels, scores, average=None)

        return acc, pre, f1, auc
