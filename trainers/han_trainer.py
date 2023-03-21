# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, RunningAverage
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

# 权重初始化，默认xavier
class Han_trainer(object):
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.device = args.device
        self.lr = args.lr
        self.l2 = args.l2
        self.num_epochs = args.epochs

    def init_network(self, model, method='xavier', exclude='embedding', seed=123):
        for name, w in model.named_parameters():
            if exclude not in name:
                if len(w.size()) < 2:
                    continue
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass


    def train(self, config, model, train_iter, dev_iter, test_iter):
        start_time = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2)
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        dev_best_f1 = float(0)
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        print('Present device:', self.device)
        interval_num = int(len(train_iter)*self.args.interval)
        scaler = GradScaler()

        for epoch in range(self.num_epochs):
            model.train()  # set model to training mode
            # summary for current training loop and a running average object for loss
            loss_avg = RunningAverage()
            loop = tqdm(enumerate(train_iter), total =len(train_iter))
            for i, (trains, labels) in loop:
                loop.set_description(f'Epoch [{epoch + 1}/{self.num_epochs}]')
                optimizer.zero_grad()
                with autocast():
                    outputs = model(trains)
                    loss = F.cross_entropy(outputs, labels)
                # labels_index = torch.max(labels, 1)[1]
                loss_avg.update(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if total_batch % interval_num == 0:
                    loop.set_description(f'Epoch [{epoch + 1}/{self.num_epochs}],Evaluating...')

                    # 每多少轮输出在训练集和验证集上的效果
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_f1 = metrics.f1_score(true, predic, average='micro')
                    dev_f1, dev_acc, dev_loss = self.evaluate(config, model, dev_iter)
                    if dev_f1 > dev_best_f1:
                        dev_best_f1 = dev_f1
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train f1: {2:>6.2%},  Val Loss: {3:>5.2},  Val f1: {4:>6.2%}, Val Acc: {5:>6.2%},  Time: {6} {7}'
                    loop.set_postfix_str(msg.format(total_batch, loss_avg(), train_f1, dev_loss, dev_f1, dev_acc, time_dif, improve))                    # model.train()
                total_batch += 1
                if total_batch - last_improve > config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        # self.test(config, model, test_iter, use_best=False)
        self.test(config, model, test_iter, start_time)


    def test(self, config, model, test_iter, start_time):

        # test
        model.load_state_dict(torch.load(config.save_path))

        model.eval()
        # start_time = time.time()
        test_f1, test_acc, test_loss, test_report, test_confusion = self.evaluate(config, model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test F1: {1:>6.2%}, Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_f1, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        with open (config.log_path, "a") as f_log:
            f_log.write('*'*50+'\n')
            f_log.write('model name: ' + config.model_name+'\n')
            f_log.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'\n')
            f_log.write('training time used: ' + str(time_dif)+'\n')
            f_log.write(msg.format(test_loss, test_f1, test_acc)+'\n')
            f_log.write(test_report+'\n')
            f_log.write(str(test_confusion)+'\n')
            
            


    def evaluate(self, config, model, data_iter, test=False):
        model.eval()
        loss_avg = RunningAverage()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in data_iter:
                with autocast():
                    outputs = model(texts)
                # 5分类和二分类不同，需要修改Loss和predict
                    loss = F.cross_entropy(outputs, labels)
                loss_avg.update(loss.item())
                # #             #将label进行Onehot编码
                # label_list = []
                # for i in labels:
                #     label = [0]*config.num_classes
                #     label[int(i)] = 1
                #     label_list.append(label)
                # labels = torch.tensor(label_list)
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        f1_score = metrics.f1_score(labels_all, predict_all, average= 'micro')
        if test:
            report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return f1_score, acc, loss_avg(), report, confusion
        model.train()
        return f1_score, acc, loss_avg()

    def run(self, mode, config, model, train_iter, dev_iter, test_iter):
        if mode == 'train':
            self.train(config, model, train_iter, dev_iter, test_iter)
        elif mode == 'test':
            start_time = time.time()
            self.test(config, model, test_iter, start_time)

        # else:
        #     self.predict()