#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')

from Fingerprinting.models.LSTM import lstm
from Fingerprinting.tools.predict import Predicter
from Fingerprinting.tools.train import Trainer
from Fingerprinting.tools.utils import *

# Config Parameters

options = dict()

options['data_dir'] = '../dataset/'
options['window_size'] = 20
options['device'] = "cuda"




# Model
options['input_size'] = 11
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] =30

# Train
options['batch_size'] = 64
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 50
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "lstm"
options['save_dir'] = "../result/lstm/"

# Predict
# options['model_path'] =["../result/deeplog/deeplog_epoch{}.pth".format(i) for i in range(200, 370,10)]
options['model_path']='../result/lstm/lstm_epoch40.pth'

# options['dataset']='360drop_lan_eth0'

seed_everything(seed=1234)


def train():
    Model = lstm(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def predict():
    Model = lstm(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    predicter = Predicter(Model, options)
    results,metrics=predicter.predict_supervised()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'],default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
