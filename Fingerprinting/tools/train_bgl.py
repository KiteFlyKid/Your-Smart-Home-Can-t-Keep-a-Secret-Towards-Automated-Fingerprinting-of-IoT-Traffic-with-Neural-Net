#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
sys.path.append('../../')
from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# from data.BGL_preprocessing import sliding_window_bgl
from nonFL.logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import sliding_window, session_window
from logdeep.tools.utils import (save_parameters, seed_everything,
                                 train_val_split)
def generate(name):
    window_size = 10
    hdfs = {}
    length = 0
    with open('/home/featurize/data/data/BGL/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Trainer():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.sample = options['sample']
        self.feature_num = options['feature_num']

        os.makedirs(self.save_dir, exist_ok=True)
        if self.sample == 'sliding_window':
            X=pd.read_pickle(self.data_dir+'BGL/bgl_swBGLTrain')
            train_logs=X.drop(columns=['label'],axis=1)
            train_labels=X.label

            # X = pd.read_pickle(self.data_dir + 'BGL/test_normal')
            # val_logs = X.drop(columns=['label'], axis=1)
            # val_labels = X.label
            # train_logs, train_labels = sliding_window(self.data_dir,
            #                                       datatype='train',
            #                                       window_size=self.window_size)
            # pd.DataFrame(train_logs).to_csv('sw_hdfs_train.csv',index=False)

            val_logs, val_labels = sliding_window(self.data_dir,
                                              datatype='val',
                                              window_size=self.window_size,
                                              sample_ratio=0.001)
        elif self.sample == 'session_window':
            train_logs, train_labels = session_window(self.data_dir,
                                                      datatype='train')
            val_logs, val_labels = session_window(self.data_dir,
                                                  datatype='val')
        else:
            raise NotImplementedError

        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)

        del train_logs
        # del val_logs
        gc.collect()

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_train_log = len(train_dataset)
        self.num_valid_log = len(valid_dataset)

        print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))
        print('Train batch size %d ,Validation batch size %d' %
              (options['batch_size'], options['batch_size']))

        self.model = model.to(self.device)

        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, self.save_dir + "parameters.txt")
        self.log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")
    def predict_unsupervised(self,models):
        models[0].eval()
        test_normal_loader, test_normal_length = generate('bgl_test_normal')
        test_abnormal_loader, test_abnormal_length = generate(
            'bgl_test_abnormal')
        TPs = [0]*len(models)
        FPs = [0]*len(models)
        FNs=[]
        Ps=[]
        Rs=[]
        F1s=[]
        # Test the model
        start_time = time.time()
        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                for idx, model in enumerate(models):
                    for i in range(len(line) - self.window_size):
                        seq0 = line[i:i + self.window_size]
                        label = line[i + self.window_size]
                        seq1 = [0] * 312
                        log_conuter = Counter(seq0)
                        for key in log_conuter:
                            seq1[key] = log_conuter[key]

                        seq0 = torch.tensor(seq0, dtype=torch.float).view(
                            -1, self.window_size, self.input_size).to(self.device)
                        seq1 = torch.tensor(seq1, dtype=torch.float).view(
                            -1, self.num_classes, self.input_size).to(self.device)
                        label = torch.tensor(label).view(-1).to(self.device)
                        output = model(features=[seq0, seq1], device=self.device)
                        predicted = torch.argsort(output,
                                              1)[0][-self.num_candidates:]
                        if label not in predicted:
                            FPs[idx] += test_normal_loader[line]
                            break
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                for idx, model in enumerate(models):
                    for i in range(len(line) - self.window_size):
                        seq0 = line[i:i + self.window_size]
                        label = line[i + self.window_size]
                        seq1 = [0] * 312
                        log_conuter = Counter(seq0)
                        for key in log_conuter:
                            seq1[key] = log_conuter[key]

                        seq0 = torch.tensor(seq0, dtype=torch.float).view(
                            -1, self.window_size, self.input_size).to(self.device)
                        seq1 = torch.tensor(seq1, dtype=torch.float).view(
                            -1, self.num_classes, self.input_size).to(self.device)
                        label = torch.tensor(label).view(-1).to(self.device)

                        output = model(features=[seq0, seq1], device=self.device)
                        predicted = torch.argsort(output,
                                                  1)[0][-self.num_candidates:]
                        if label not in predicted:
                            TPs[idx] += test_abnormal_loader[line]
                            break

        # Compute precision, recall and F1-measure
        for i in range(len(models)):
            FN = test_abnormal_length - TPs[i]
            P = 100 * TPs[i] / (TPs[i] + FPs[i])
            R = 100 * TPs[i] / (TPs[i] + FN)
            F1 = 2 * P * R / (P + R)
            print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FPs[i], FN, P, R, F1))
            print('Finished Predicting',self.model_path[i])
            F1s.append(F1)
            FNs.append(FN)
            Ps.append(P)
            Rs.append(R)

        path = '../result/'  +str(self.model_name)+'/'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + 'scores.txt', mode='w') as f:
            f.write('~'.join([str(i) for i in Ps]))
            f.write('\n')
            f.write('~'.join([str(i) for i in Rs]))
            f.write('\n')
            f.write('~'.join([str(i) for i in F1s]))
            f.write('\n')
            f.write('~'.join([str(i) for i in FNs]))
            f.write('\n')
            f.write('~'.join([str(i) for i in FPs]))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)
        start = time.strftime("%H:%M:%S")
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.train_loader, desc="\r")
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().detach().to(self.device))
            output = self.model(features=features, device=self.device)
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))

        self.log['train']['loss'].append(total_losses / num_batch)

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                output = self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
        print("Validation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        if total_losses / num_batch < self.best_loss:
            self.best_loss = total_losses / num_batch
            self.save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix="bestloss")

    def start_train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio
            self.train(epoch)
            if epoch >= self.max_epoch // 3 and epoch % 10 == 0:
                self.predict_unsupervised(models=[self.model])
                # self.valid(epoch)
                self.save_checkpoint(epoch,
                                     save_optimizer=True,
                                     suffix="epoch" + str(epoch))
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
            self.save_log() #what's this
