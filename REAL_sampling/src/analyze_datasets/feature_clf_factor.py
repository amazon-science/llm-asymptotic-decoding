#the code is modified from https://github.com/microsoft/HaDes/blob/main/baselines/feature_clf.py

import random, os, sys
import codecs
import argparse
import spacy
import numpy as np
from collections import defaultdict
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, \
    f1_score, precision_score, recall_score
#import warnings
#warnings.filterwarnings("ignore")
from utils import binary_eval

import itertools

def subsets(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    ans = []
    def dfs(curpos, tmp):
        if tmp:
            ans.append(tmp[:])
        for i in range(curpos, len(nums)):
            tmp.append(nums[i])
            dfs(i+1, tmp)
            tmp.pop(-1)
    dfs(0, [])
    return ans

def plot_2_features(trainx, trainy, feature_keys):
    feature_idx = [0, 6]
    #feature_idx = [2, 4]
    #label_name = ['not hallucination', 'hallucination']
    label_name = ['not factual', 'factual']
    X = np.array(trainx)
    X1 = X[:, feature_idx[0]]
    X2 = X[:, feature_idx[1]]
    y = np.array(trainy)#.view(-1,1)
    plt.figure()
    plt.scatter(X1[y==1],X2[y==1], label=label_name[1], alpha=0.3, s=0.2)
    plt.scatter(X1[y==0], X2[y==0], label=label_name[0], alpha=0.3, s=0.2)
    plt.legend()
    #plt.ylim(0, 0.15)
    plt.xlim(0, 5)
    plt.xlabel(feature_keys[feature_idx[0]])
    plt.ylabel(feature_keys[feature_idx[1]])
    plt.savefig('outputs/factor/figs/per_ent_score1.png')
    
    plt.figure()
    #plt.scatter(X1[y==1],X2[y==1], label=label_name[1], alpha=0.3, s=0.2)
    plt.scatter(X1[y==0], X2[y==0], label=label_name[0], alpha=0.3, s=0.2)
    plt.legend()
    #plt.ylim(0, 0.15)
    plt.xlim(0, 5)
    plt.xlabel(feature_keys[feature_idx[0]])
    plt.ylabel(feature_keys[feature_idx[1]])
    plt.savefig('outputs/factor/figs/per_ent_score1_0.png')
    
    plt.figure()
    plt.scatter(X1[y==1],X2[y==1], label=label_name[1], alpha=0.3, s=0.2)
    plt.legend()
    #plt.ylim(0, 0.15)
    plt.xlim(0, 5)
    plt.xlabel(feature_keys[feature_idx[0]])
    plt.ylabel(feature_keys[feature_idx[1]])
    plt.savefig('outputs/factor/figs/per_ent_score1_1.png')


class ClfModel:
      def __init__(self, args):
            self.args = args
            self.select_idxs = [int(x) for x in args.feature_select_idxs.split(',')]
            self.device = args.device
            self.model = args.model
            if "svm" in self.model:
                self.clf = svm.LinearSVC()
            elif 'lr' in self.model:
                self.clf = make_pipeline(StandardScaler(),
                                         SGDClassifier(loss="log", max_iter=10000, tol=1e-5, shuffle=True))
            elif 'RF' in self.model:
                self.clf = RandomForestClassifier(max_depth=5, n_estimators=100)

      def filter_features(self, X):
          features_all = []
          for example_batch in X:
              for example in example_batch:
                  if type(example[0]) is list:
                      example = [x[0] for x in example]
                  entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2 = example
                  #features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, perplexity_tensor_large * ent_score1, ent_score1, per_score1, ent_score3, c_ent, c_per]
                  #features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3]
                  #features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3,  ent_score2, per_score2, curve_last_ent, curve_last_per]
                  features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3,  ent_score1, per_score1, c_ent, c_per]
                  #features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3,  ent_score1, c_ent]
                  #features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3,  per_score1, c_per]
                  #features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3,  ent_score2, per_score2, c_ent, c_per]
                              #perplexity_tensor_large, perplexity_tensor_large * ent_score1, ent_score1, per_score1, ent_score3, c_ent, c_per]
                              #entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2]
                  if len(self.select_idxs) > 0:
                      features = [features[idx] for idx in self.select_idxs]
                  features_all.append(features)
          return features_all

      def load_feature_file(self, f_in):
            X_pos, X_neg_1, X_neg_2, X_neg_3 = json.load(f_in)
            X_pos = self.filter_features(X_pos)
            X_neg_1 = self.filter_features(X_neg_1)
            X_neg_2 = self.filter_features(X_neg_2)
            X_neg_3 = self.filter_features(X_neg_3)
            X = np.array(X_pos+X_neg_1+X_neg_2+X_neg_3)
            y = np.array( [1]*len(X_pos)+[0]*len(X_neg_1)+[0]*len(X_neg_2)+[0]*len(X_neg_3) )
            print("pos avg features", np.mean(X_pos, axis=0))
            print("neg 1 avg features", np.mean(X_neg_1, axis=0))
            print("neg 2 avg features", np.mean(X_neg_2, axis=0))
            print("neg 3 avg features", np.mean(X_neg_3, axis=0))
            #return X, y, np.array(X_pos), np.array(X_neg)
            return X, y#, np.array(X_pos), np.array(X_neg)

      def train(self, epoch=10):

            #feature_keys = ["perplexity_tensor_large", 'ent_large', 'per_small', "perplexity_tensor_large*ent_score1", "ent_score1", "per_score1", 'ent_diff', "c_ent", "c_per"
            #feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff'
            #feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff', "ent_score2", "per_score2", "curve_last_ent", "curve_last_per"
            feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff', "ent_score1", "per_score1", "c_ent", "c_per"
            #feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff', "ent_score1", "c_ent"
            #feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff', "per_score1", "c_per"
            #feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff', "ent_score2", "per_score2", "c_ent", "c_per", 
                            # "perplexity_tensor_large", "perplexity_tensor_large*ent_score1", "ent_score1", "per_score1", 'ent_diff', "c_ent", "c_per"
                             #'ent_small', 'ent_large','ent_diff',
                             #'per_small','per_large','per_diff', 
                             #'c_ent','pred_last_ent', 'curve_last_ent', 'ent_score1', 'ent_score2',
                             #'c_per','pred_last_per', 'curve_last_per', 'per_score1', 'per_score2'
                            ]
            if len(self.select_idxs) > 0:
                feature_keys = [feature_keys[idx] for idx in self.select_idxs]
            feature_names = {}
            for i in range(len(feature_keys)):
                feature_names[feature_keys[i]] = i
            if 'pair' in self.args.output_score_name:
                combinations = list(itertools.combinations(feature_keys, 2) )
            else:
                combinations = subsets(feature_keys)
            
            print("Load Training Features ...")
            with open(args.input_train_feature_name) as f_in:
                #trainx, trainy, trainx_pos, trainx_neg = self.load_feature_file(f_in)
                trainx, trainy = self.load_feature_file(f_in)
                
            plot_scatter = False
            #plot_scatter = True
            if plot_scatter:
                plot_2_features(trainx, trainy, feature_keys)
                return

            with open(args.input_val_feature_name) as f_in:
                #testx, testy, testx_pos, testx_neg = self.load_feature_file(f_in)
                testx, testy = self.load_feature_file(f_in)

            fw = codecs.open(args.output_score_name, "w+", encoding="utf-8")
            infos = []
            for feats in combinations:
                real_trainx = []
                for fs in trainx:
                    new_fs = []
                    for featname in feats:
                        new_fs.append(fs[feature_names[featname]])
                    real_trainx.append(new_fs)
                real_testx = []
                for fs in testx:
                    new_fs = []
                    for featname in feats:
                        new_fs.append(fs[feature_names[featname]])
                    real_testx.append(new_fs)

                len_test = len(real_testx)
                testx_pos = real_testx[:int(len_test/4)]
                testx_neg_1 = real_testx[int(len_test/4):int(len_test/2)]
                testx_neg_2 = real_testx[int(len_test/2):int(len_test/4*3)]
                testx_neg_3 = real_testx[int(len_test/4*3):]

                self.clf.fit(real_trainx, trainy)
                predy = self.clf.predict(real_testx)
                predy_prob = self.clf.predict_proba(real_testx)[:,1]
                predy_prob_pos = self.clf.predict_proba(testx_pos)[:,1]
                predy_prob_neg_1 = self.clf.predict_proba(testx_neg_1)[:,1]
                predy_prob_neg_2 = self.clf.predict_proba(testx_neg_2)[:,1]
                predy_prob_neg_3 = self.clf.predict_proba(testx_neg_3)[:,1]

                print("Features: {}".format(" ".join(feats)), file=sys.stderr)
                print("="*20)
                print("Features: {}".format(" ".join(feats)))
                feat_str = "Features: {}".format(" ".join(feats))
                # fw.write("\n\nFeatures: {}\n".format(" ".join(feats)))
                acc, info = binary_eval(predy, testy, predy_prob, return_f1=False)
                pair_acc = np.sum( (predy_prob_pos > predy_prob_neg_1) & (predy_prob_pos > predy_prob_neg_2) & (predy_prob_pos > predy_prob_neg_3)  ) / len(testx_pos)
                pair_acc_str = "\n1 in 4 Acc : {}".format(pair_acc)
                print(pair_acc_str)
                info += pair_acc_str
                #infos.append([acc, feat_str, info])
                infos.append([pair_acc, feat_str, info])
            infos = sorted(infos, key=lambda x:-x[0])

            for item in infos:
                _, feat_str, info = item
                fw.write("\n\n"+feat_str+"\n")
                fw.write(info+"\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", default="svm", type=str) 
    #parser.add_argument("--model", default="lr", type=str) 
    parser.add_argument("--model", default="RF", type=str) 
    #parser.add_argument("--input_train_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/features/dialogue_data_train_wiki_smallOWT.json", type=str) 
    #parser.add_argument("--input_val_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/features/dialogue_data_val_wiki_smallOWT.json", type=str) 
    #parser.add_argument("--input_train_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/features/dialogue_data_train_OWT_perOWT.json", type=str) 
    #parser.add_argument("--input_val_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/features/dialogue_data_val_OWT_perOWT.json", type=str) 
    
    #parser.add_argument("--input_train_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/features/wiki_factor_train_wiki_smallOWT.json", type=str) 
    #parser.add_argument("--input_val_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/features/wiki_factor_val_wiki_smallOWT.json", type=str) 
    parser.add_argument("--input_train_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/features/wiki_factor_train_OWT_perOWT.json", type=str) 
    parser.add_argument("--input_val_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/features/wiki_factor_val_OWT_perOWT.json", type=str) 
    
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/scores/wiki_factor_wiki_smallOWT_lr.txt", type=str) 
    parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/scores/wiki_factor_OWT_perOWT_lr.txt", type=str) 

    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/scores/dialogue_data_RF5.txt", type=str) 
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/scores/dialogue_data_more_lr.txt", type=str) 
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/scores/dialogue_data_more2_lr.txt", type=str) 
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/scores/dialogue_data_more_smallOWT_lr.txt", type=str) 
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/scores/dialogue_data_more_wiki_lr.txt", type=str) 
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/scores/dialogue_data_more_OWT_lr.txt", type=str) 
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/scores/dialogue_data_wiki_smallOWT_lr.txt", type=str) 
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/scores/dialogue_data_no_pred_lr.txt", type=str) 
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    ent_model_names = ['OWT_perOWT', 'wiki_smallOWT']
    #feature_idx_set = [ ['large', [0,1]], ['small', [2,3]], ['large_small', [0,1,2,3,4,5]], ['large_small_pred1', [0,1,2,3,4,5,6,8]], ['large_small_pred2', [0,1,2,3,4,5,7,9]], ['large_small_pred12', [0,1,2,3,4,5,6,7,8,9]], ['large_small_pred12_pair', [0,1,2,3,4,5,6,7,8,9] ]  ]
    #feature_idx_set = [  ['large_small_pred34_pair', [0,1,2,3,4,5,6,7,8,9] ] , ['large_small_pred34', [0,1,2,3,4,5,6,7,8,9]] ]
    feature_idx_set = [  ['large_small_pred12_pair', [0,1,2,3,4,5,6,7,8,9] ] ]

    input_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/features/'
    output_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/scores/'
    log_folder = '/mnt/efs/Haw-Shiuan/true_entropy/log/factor/'

    #dataset = 'wiki_factor'
    dataset = 'news_factor'
    #dataset = 'expert_factor'

    for i, ent_model_name in enumerate(ent_model_names):
        args.input_train_feature_name = input_folder + dataset + '_train_' + ent_model_name + '.json'
        args.input_val_feature_name = input_folder + dataset + '_val_' + ent_model_name + '.json'
        if i == 0 or len(feature_idx_set) <= 3:
            idx_range = range(len(feature_idx_set))
        else:
            idx_range = range(3,len(feature_idx_set))
        for j in idx_range:
            feature_idx = feature_idx_set[j]
            feature_name = feature_idx[0]
            feature_use = feature_idx[1]
            args.feature_select_idxs = ','.join([str(x) for x in feature_use])
            if i == 0 and j < 3:
                args.output_score_name = output_folder + dataset + '_' + feature_name + '_' + args.model + '.txt'
                log_file_path = log_folder + dataset + '_' + feature_name + '_stdout_' + args.model
            else:
                args.output_score_name = output_folder + dataset + '_' + ent_model_name + '_' + feature_name + '_' + args.model + '.txt'
                log_file_path = log_folder + dataset + '_' + ent_model_name + '_' + feature_name + '_stdout_' + args.model
            with open(log_file_path, 'w') as sys.stdout:
                print(args)
                rep_op = ClfModel(args)
                rep_op.train()


