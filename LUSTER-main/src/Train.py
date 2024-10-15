import os
from tqdm import trange
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from src.PGD import PGD

def train_phase(model, train_loader, valid_loader, args, log):
    model_path = 'save/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    best_valid_dir = model_path + 'best_model.pth'
    best_valid_metric = 0
    best_model_epoch = 0
    patience = args.patience
    current_patience = 0
    pgd = PGD(model, 'cnn', epsilon=args.epsilon, alpha=args.alpha)
    K = args.disturb_t
    criterion = nn.CrossEntropyLoss()

    print('Training...')
    log.write('Training...\n')

    for epoch in trange(args.epochs, desc="Training"):
        if args.ifDecay:
            p = epoch / (args.epochs - 1)
            learning_rate = args.learning_rate / pow((1 + 10 * p), 0.75)
        else:
            learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

        model.train()
        whole_loss_vec = []
        train_acc_vec = []
        for data in train_loader:
            network_labels, left_nodes, right_nodes, link_labels = data
            network_labels = Variable(network_labels).cuda()
            left_nodes = Variable(left_nodes).cuda()
            right_nodes = Variable(right_nodes).cuda()
            link_labels = Variable(link_labels).cuda()

            prediction_outs, discriminant_outs = model(network_labels, left_nodes, right_nodes)

            prediction_loss = criterion(prediction_outs, link_labels)
            discriminant_loss = criterion(discriminant_outs, network_labels)
            whole_loss = prediction_loss + discriminant_loss
            whole_loss_vec.append(whole_loss.cpu().detach().numpy())

            _, argmax = torch.max(prediction_outs, 1)
            batch_acc = (argmax == link_labels).float().mean()
            train_acc_vec.append(batch_acc.item())

            whole_loss.backward()

            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))

                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()

                _, adv_outs = model(network_labels, left_nodes, right_nodes)
                adv_loss = criterion(adv_outs, network_labels)
                adv_loss.backward()
            pgd.restore_origin_grad()
            pgd.restore()
            optimizer.step()
            optimizer.zero_grad()

        whole_loss = np.mean(whole_loss_vec)
        train_acc = np.mean(train_acc_vec)

        # write_infor = "Epoch:[{}/{}], lr:{:.4f}, Loss:{:.4f}, Train acc:{:.4f}".format(epoch+1, args.epochs, learning_rate, whole_loss, train_acc)
        # print(write_infor)
        # log.write('\n' + write_infor)

        model.eval()
        valid_acc, valid_pre, valid_f1, valid_auc = model.metrics_eval(valid_loader)
        # write_infor = "Valid acc:{:.4f}, Valid pre:{:.4f}, Valid f1:{:.4f}, Valid auc:{:.4f} ".format(valid_acc, valid_pre, valid_f1, valid_auc)
        # print(write_infor)
        # log.write('\n' + write_infor)

        # Update best_valid_auc and early stopping counter
        # Save the best model
        if valid_auc > best_valid_metric:
            best_valid_metric = valid_auc
            current_patience = 0
            torch.save(model.state_dict(), best_valid_dir)
            best_model_epoch = epoch
        else:
            current_patience += 1

        # Check if early stopping conditions are met
        if args.EarlyStop and current_patience >= patience:
            print("Early stopping!")
            break

    write_infor = 'Best Epoch:[{}/{}]'.format(best_model_epoch+1, args.epochs)
    print(write_infor)
    log.write(write_infor + '\n')
    return best_valid_dir

def test_phase(model, best_valid_dir, test_loader, log):
    # Load the best model for testing
    print('Load best model ' + best_valid_dir + ' ... ')
    model.load_state_dict(torch.load(best_valid_dir))
    model.eval()
    acc, pre, f1, auc = model.metrics_eval(test_loader)
    write_infor = "Test acc:{:.4f}, pre:{:.4f}, f1:{:.4f}, auc:{:.4f}".format(acc, pre, f1, auc)
    print(write_infor)
    log.write(write_infor + '\n')

def run_model(train_loader, valid_loader, test_loader, model, args):
    log = open(args.log, 'a', encoding='utf-8')
    best_valid_dir = 'save/' + args.dataset + '_best_model.pth'
    if args.onlyTest and os.path.exists(best_valid_dir):
        print('onlyTest')
        log.write('onlyTest\n')
    else:
        best_valid_dir = train_phase(model, train_loader, valid_loader, args, log)
    test_phase(model, best_valid_dir, test_loader, log)
    log.close()

