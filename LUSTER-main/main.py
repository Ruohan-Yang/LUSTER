import os
import argparse
import datetime
import torch
from src.Load import load_data
from src.LUSTER import LUSTER_Net
from src.Train import run_model
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LUSTER', help='name of model')
    parser.add_argument('--dataset', type=str, default='Reddit', help='name of dataset')

    parser.add_argument('--learning_rate', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--ifDecay', action='store_false', help='If true, the learning rate decays (default true). If false, the learning rate is constant.')

    parser.add_argument('--epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--EarlyStop', action='store_true', help='Default is false. If true, enable early stopping strategy')
    parser.add_argument('--patience', type=int, default=50, help='early stop')

    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train data')
    parser.add_argument('--dim', type=int, default=16, help='dims of node embedding')

    parser.add_argument('--epsilon', type=int, default=7, help='epsilon of PGD')
    parser.add_argument('--alpha', type=int, default=3, help='alpha of PGD')
    parser.add_argument('--disturb_t', type=int, default=4, help='number of disturbances of PGD')

    parser.add_argument('--onlyTest', action='store_true', help='Default is false. If true, try to load an existing model.')
    parser.add_argument('--log', type=str, default='./log/', help='record file path')

    args = parser.parse_args()
    path = args.log
    if not os.path.exists(path):
        os.mkdir(path)

    print('***************************')
    print('The program starts running.')
    print('***************************')
    args.log = path + args.dataset + '-Result.txt'
    print(args)

    begin = datetime.datetime.now()
    print('Start time ', begin)
    time = str(begin.year) + '-' + str(begin.month) + '-' + str(begin.day) + '-' + str(begin.hour) + '-' + str(
        begin.minute) + '-' + str(begin.second)
    log = open(args.log, 'a', encoding='utf-8')
    write_infor = '\nStart time: ' + time + '\n'
    log.write(write_infor)
    write_infor = "lr:{}, IfDecay:{}, Epochs:{}, IfStop:{}, Patience:{}".format(args.learning_rate, args.ifDecay, args.epochs, args.EarlyStop, args.patience) + '\n'
    log.write(write_infor)
    write_infor = "Batch_size:{}, Dim:{}, Epsilon:{}, Alpha:{}, Disturb_t:{}".format(args.batch_size, args.dim, args.epsilon, args.alpha, args.disturb_t) + '\n'
    log.write(write_infor)
    log.close()

    # load data
    train_loader, valid_loader, test_loader, gcn_data, network_numbers = load_data(args.dataset, args.batch_size)
    # load model
    model = LUSTER_Net(embedding_dim=args.dim, layer_number=network_numbers, gcn_data=gcn_data)

    if torch.cuda.is_available():
        model = model.cuda()
        for i in range(network_numbers):
            gcn_data[i].x = gcn_data[i].x.cuda()
            gcn_data[i].edge_index = gcn_data[i].edge_index.cuda()
    run_model(train_loader, valid_loader, test_loader, model, args)

    # time
    end = datetime.datetime.now()
    print('End time ', end)
    print('Run time ', end - begin)


