# coding: UTF-8
import time
import torch
import numpy as np
from trainers.han_trainer import Han_trainer
from trainers.fishqa_trainer import Fishqa_trainer
from trainers.bert_trainer import Bert_trainer
from trainers.bert_hbqda_trainer import Bert_hbqda_trainer
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif, ReviewDataset, build_dataloader
import os

parser = argparse.ArgumentParser(description='Chinese document Classification')
parser.add_argument('--model_name', type=str, default='hbqda', help='choose a model: hans, fishqa, bert, bert_att, hbqa, fishqda, hbqda')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--process_data', type=bool, default=False, help='preprocess data or not')
parser.add_argument('--dataset', default='STKRCMD2021', help='choose dataset')
parser.add_argument('--seed', default=1, help='choose seed')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='choose device: cpu, cuda:0, cuda:1')
parser.add_argument('--mode', default='test', help='choose: train, test')
parser.add_argument('--lr', default=0.001, help='learning rate', type=float)
parser.add_argument('--l2', default=0.0005, help='weight decay', type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--interval', default=0.05, type=float, help='rate of train_iter to evaluate, less than 0.2')

               


if __name__ == '__main__':

    args = parser.parse_args()

    x = import_module('models.' + args.model_name)
    config = x.Config(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  

    start_time = time.time()
    print("Loading data...")


    if args.process_data:
        # 预处理数据
        print("process data...")
        train_data, dev_data, test_data, query_ids = build_dataset(config)
        torch.save(train_data, config.processed_train_path)
        torch.save(dev_data, config.processed_dev_path)
        torch.save(test_data, config.processed_test_path)
        torch.save(query_ids, config.processed_query_path)
    else:
        # 读取预处理数据
        print("load processed data...")
        train_data = torch.load(config.processed_train_path)
        dev_data = torch.load(config.processed_dev_path)
        test_data = torch.load(config.processed_test_path)
        query_ids = torch.load(config.processed_query_path)

    print('train_data_len = {}, dev_data_len = {}, test_data_len = {}'.format(len(train_data),len(dev_data),len(test_data)))

    train_loader = build_dataloader(train_data, args.device, args.batch_size, args.num_workers, config.is_bert)
    dev_loader = build_dataloader(dev_data, args.device, args.batch_size, args.num_workers, config.is_bert)
    test_loader = build_dataloader(test_data, args.device, args.batch_size, args.num_workers, config.is_bert)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(args.device)

    print('parameters_count:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print('trainer:', args.model_name)

    if args.model_name == 'hans':
        Han_trainer(args).run(args.mode, config, model, train_loader, dev_loader, test_loader)
    elif args.model_name == 'fishqa' or args.model_name == 'fishqda':
        Fishqa_trainer(args).run(args.mode, config, model, train_loader, dev_loader, test_loader, query_ids)
    elif args.model_name == 'hbqa' or args.model_name == 'hbqda':
        Bert_hbqda_trainer(args).run(args.mode, config, model, train_loader, dev_loader, test_loader, query_ids)
    elif args.model_name == 'bert' or args.model_name == 'bert_att':
        Bert_trainer(args).run(args.mode, config, model, train_loader, dev_loader, test_loader, query_ids)
