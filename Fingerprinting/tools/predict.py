#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
from collections import Counter
sys.path.append('../../')
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
from Fingerprinting.dataset.log import smarthome_dataset
from sklearn import metrics
from Fingerprinting.dataset.sample import session_window


def generate(name):
    window_size = 10
    hdfs = {}
    length = 0
    with open('../data/hdfs/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs, length


class Predicter():
    def __init__(self, model, options):
        self.options=options
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model_name=options['model_name']
        self.model = model
        self.model_path = options['model_path']
        self.num_testmodel=len(self.model_path)
        self.window_size = options['window_size']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.batch_size = options['batch_size']

    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        target_names=['360drop_lan_eth0.csv', '360drop_lan_orivibo_lan_turnonoff_eth0.csv', '360drop_lan_orivibo_lan_turnonoff_wlan0.csv', '360drop_lan_wlan0.csv', '360drop_wan_orvibo_turnonoff_eth0.csv', '360drop_wan_orvibo_turnonoff_wlan0.csv', 'echo_dot_ask_questions_xiaomi_plug_turnon_off_eth0.csv', 'echo_dot_ask_questions_xiaomi_plug_turnon_off_wlan0.csv', 'google_home_ask_questions_tmall_assistant_sing_songs_broadlink_turnonoff_wlan_eth0.csv', 'google_home_ask_questions_tmall_assistant_sing_songs_broadlink_turnonoff_wlan_wlan0.csv', 'google_home_sing_songs_tmall_assistant_ask_questions_tplink_turn_onoff_eth0.csv', 'google_home_sing_songs_tmall_assistant_ask_questions_tplink_turn_onoff_wlan0.csv', 'google_home_sing_songs_tmall_assistant_sing_songs2_eth0.csv', 'google_home_sing_songs_tmall_assistant_sing_songs2_wlan0.csv', 'google_home_tmall_assistant_sing_songs_eth0.csv', 'google_home_tmall_assistant_sing_songs_wlan0.csv', 'mitu_story_eth0.csv', 'mitu_story_wlan0.csv', 'tmall_assistant_sing_songs_eth0.csv', 'tmall_assistant_sing_songs_wlan0.csv', 'xiaobai_camera_lan_eth0.csv', 'xiaobai_camera_lan_wlan0.csv', 'xiaobai_camera_wan2_eth0.csv', 'xiaobai_camera_wan2_wlan0.csv', 'xiaobai_camera_wan_eth0.csv', 'xiaobai_camera_wan_wlan0.csv', 'xiaomi_control_plug_tmall_assistant_sing_eth0.csv', 'xiaomi_control_plug_tmall_assistant_sing_wlan0.csv', 'xiaomi_control_plug_turnonoff_eth0.csv', 'xiaomi_control_plug_turnonoff_wlan0.csv']
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_dataset = smarthome_dataset(self.data_dir,'Dataset-Ind-test-{}.pkl'.format(self.window_size))
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        predicted_labels=[]
        target_labels=[]
        for i, (window_df, label) in enumerate(tbar):
            features=window_df.to(self.device)
            logits = self.model(features=features, device=self.device)
            predicted = torch.argmax(logits, dim=1).cpu().numpy()
            predicted_labels.extend(predicted)
            target_labels.extend(label.cpu().detach().numpy())


        cal_results=metrics.classification_report(target_labels, predicted_labels, target_names=target_names)
        print(cal_results)
        test_info=self.model_path.split('/')[-1].split('.')[0]
        with open('../result/{}-results.txt'.format(test_info), 'w') as f:
            f.write(cal_results)
        # accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        # precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
        # recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
        # f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')
        # return [accuracy, precision, recall, f1], ['accuracy', 'precision', 'recall', 'f1',' ']