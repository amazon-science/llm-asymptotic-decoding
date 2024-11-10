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
from utils import binary_eval

import itertools

repo_folder = "/mnt/efs/Haw-Shiuan/true_entropy/"

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

def plot_2_features(trainx, trainy, feature_keys, fig_save_prefix):
    #feature_idx = [0, 4]
    feature_idx = [2, 6]
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
    #plt.xlim(1.5, 4.5)
    plt.xlabel(feature_keys[feature_idx[0]])
    plt.ylabel(feature_keys[feature_idx[1]])
    plt.savefig( fig_save_prefix + 'per_ent_score1.png')
    
    plt.figure()
    #plt.scatter(X1[y==1],X2[y==1], label=label_name[1], alpha=0.3, s=0.2)
    plt.scatter(X1[y==0], X2[y==0], label=label_name[0], alpha=0.3, s=0.2)
    plt.legend()
    #plt.ylim(0, 0.15)
    #plt.xlim(1.5, 4.5)
    plt.xlabel(feature_keys[feature_idx[0]])
    plt.ylabel(feature_keys[feature_idx[1]])
    plt.savefig( fig_save_prefix + 'per_ent_score1_label0.png')
    
    plt.figure()
    plt.scatter(X1[y==1],X2[y==1], label=label_name[1], alpha=0.3, s=0.2)
    plt.legend()
    #plt.ylim(0, 0.15)
    #plt.xlim(1.5, 4.5)
    plt.xlabel(feature_keys[feature_idx[0]])
    plt.ylabel(feature_keys[feature_idx[1]])
    plt.savefig( fig_save_prefix + 'per_ent_score1_label1.png')


class ClfModel:
      def __init__(self, args):
            self.args = args
            self.select_idxs = [int(x) for x in args.feature_select_idxs.split(',')]
            self.model = args.model
            if "svm" in self.model:
                self.clf = svm.LinearSVC()
            elif 'lr' in self.model:
                self.clf = make_pipeline(StandardScaler(),
                                         SGDClassifier(loss="log", max_iter=10000, tol=1e-5, shuffle=True))
            elif 'RF' in self.model:
                self.clf = RandomForestClassifier(max_depth=5, n_estimators=100)
      
      def simplify_features(sef, features):
          features_org = []
          for example_batch in features:
              if type(example_batch[0]) is list:
                  for example in example_batch:
                      if type(example[0]) is list:
                          example = [x[0] for x in example]
                      features_org.append(example)
              else:
                  features_org.append(example_batch)
          return features_org



      def filter_features(self, X_org, X_new):
          features_org = []
          for example in self.simplify_features(X_org):
              entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2 = example
              features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3]
              features_org.append(features)
          features_new = []
          for example in self.simplify_features(X_new):
              c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2 = example
              features = [c_ent, ent_score1]
              features_new.append(features)

          #for example_batch in X_org:
          #    for example in example_batch:
          #        if type(example[0]) is list:
          #            example = [x[0] for x in example]
          #        entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2 = example
          #        features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3]
          #        features_org.append(features)
          #features_new = []
          #for example_batch in X_new:
          #    for example in example_batch:
          #        if type(example[0]) is list:
          #            example = [x[0] for x in example]
          #        c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2 = example
          #        features = [c_ent, ent_score1]
          #        features_new.append(features)
          features_all = []
          for i in range(len(features_org)):
              features = features_org[i] + features_new[i]
              if len(self.select_idxs) > 0:
                  features = [features[idx] for idx in self.select_idxs]
              features_all.append(features)
          return features_all

      def filter_by_cat(self, X, y, output_cat, target_cat_set):
            valid_mask = [cat in target_cat_set for cat in output_cat]
            X = X[valid_mask, :]
            y = y[valid_mask]

            keep_idx = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
            print(keep_idx)
            X = X[keep_idx, :]
            y = y[keep_idx]
            return X, y

      def load_state_feature_file(self, f_in_org, f_in_new, dataset_cat):
            target_cat_list = ['animals_true_false','capitals_true_false','cities_true_false','companies_true_false',
            'conj_neg_companies_true_false','conj_neg_facts_true_false','elements_true_false',
            'facts_true_false','generated_true_false','inventions_true_false',
            'neg_companies_true_false','neg_facts_true_false','rJoke']
            target_cat_set = set(target_cat_list)

            if dataset_cat == 'state':
                all_features_org, all_labels, output_cat = json.load(f_in_org)
                all_features_new, all_labels, output_cat = json.load(f_in_new)
            elif dataset_cat == 'humor':
                all_features_org, all_labels, output_cat, labels_reg = json.load(f_in_org)
                all_features_new, all_labels, output_cat, labels_reg = json.load(f_in_new)
            all_labels = [x for label_b in all_labels for x in label_b]
            y = np.array(all_labels)
            all_features = self.filter_features(all_features_org, all_features_new)
            X = np.array(all_features)
            X, y = self.filter_by_cat(X, y, output_cat, target_cat_set)
            X_pos = X[y==1]
            X_neg = X[y==0]
            print("pos avg features", np.mean(X_pos, axis=0))
            print("neg avg features", np.mean(X_neg, axis=0))
            return X, y

      def load_HaDes_feature_file(self, f_in_org, f_in_new, text_file_path):
            print("Load Training Features ...")
            input_d2_features_train_org = json.load(f_in_org)
            input_d2_features_train_new = json.load(f_in_new)
            #trainx, trainy = [], []
            trainy = []
            trainx_org, trainx_new = [], []
            #example_list = []
            with codecs.open(text_file_path, "r", encoding="utf-8") as fr:
                cnt = 0
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    trainy.append(label)
                    trainx_org.append(input_d2_features_train_org[example["replaced"]])
                    trainx_new.append(input_d2_features_train_new[example["replaced"]])

                    cnt += 1
                    if cnt % 500 == 0:
                        print("Train {}".format(cnt))
            trainx = self.filter_features(trainx_org, trainx_new)
            X_pos = np.array(trainx)[ np.where( np.squeeze(np.array(trainy)==1))[0],:  ]
            X_neg = np.array(trainx)[ np.where( np.squeeze(np.array(trainy)==0))[0],:  ]
            print("hallu avg features", np.mean(X_pos, axis=0))
            print("not hallu avg features", np.mean(X_neg, axis=0))
            return trainx, trainy

      def load_factor_feature_file(self, f_in_org, f_in_new):
            X_pos_org, X_neg_1_org, X_neg_2_org, X_neg_3_org = json.load(f_in_org)
            X_pos_new, X_neg_1_new, X_neg_2_new, X_neg_3_new = json.load(f_in_new)
            X_pos = self.filter_features(X_pos_org, X_pos_new)
            X_neg_1 = self.filter_features(X_neg_1_org, X_neg_1_new)
            X_neg_2 = self.filter_features(X_neg_2_org, X_neg_2_new)
            X_neg_3 = self.filter_features(X_neg_3_org, X_neg_3_new)
            X = np.array(X_pos+X_neg_1+X_neg_2+X_neg_3)
            y = np.array( [1]*len(X_pos)+[0]*len(X_neg_1)+[0]*len(X_neg_2)+[0]*len(X_neg_3) )
            print("pos avg features", np.mean(X_pos, axis=0))
            print("neg 1 avg features", np.mean(X_neg_1, axis=0))
            print("neg 2 avg features", np.mean(X_neg_2, axis=0))
            print("neg 3 avg features", np.mean(X_neg_3, axis=0))
            #return X, y, np.array(X_pos), np.array(X_neg)
            return X, y#, np.array(X_pos), np.array(X_neg)

      def load_Halu_feature_file(self, f_in_org, f_in_new):
            X_pos_org, X_neg_org = json.load(f_in_org)
            X_pos_new, X_neg_new = json.load(f_in_new)
            X_pos = self.filter_features(X_pos_org, X_pos_new)
            X_neg = self.filter_features(X_neg_org, X_neg_new)
            X = np.array(X_pos+X_neg)
            y = np.array( [1]*len(X_pos)+[0]*len(X_neg) )
            print("pos avg features", np.mean(X_pos, axis=0))
            print("neg avg features", np.mean(X_neg, axis=0))
            return X, y

      def train(self, epoch=10):
            args = self.args
            #feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff', "ent_score1", "per_score1", "c_ent", "c_per"
            feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff', "ent_score1", "c_ent"]
            if len(self.select_idxs) > 0:
                feature_keys = [feature_keys[idx] for idx in self.select_idxs]
            feature_names = {}
            for i in range(len(feature_keys)):
                feature_names[feature_keys[i]] = i
            if 'pair' in args.output_score_name:
                combinations = list(itertools.combinations(feature_keys, 2) )
            else:
                combinations = subsets(feature_keys)
            
            print("Load Training Features ...")
            with open(args.input_train_feature_name) as f_in_train_new, open(args.input_val_feature_name) as f_in_test_new, open(args.input_train_org_name) as f_in_train_org, open(args.input_val_org_name) as f_in_test_org:
                #trainx, trainy, trainx_pos, trainx_neg = self.load_feature_file(f_in)
                if args.dataset_cat == 'HaDes':
                    text_file_path = repo_folder+"outputs/HaDes/all_train.txt"
                    trainx, trainy = self.load_HaDes_feature_file(f_in_train_org, f_in_train_new, text_file_path)
                    text_file_path = repo_folder+"outputs/HaDes/all_val.txt"
                    testx, testy = self.load_HaDes_feature_file(f_in_test_org, f_in_test_new, text_file_path)
                elif args.dataset_cat == 'Halu':
                    trainx, trainy = self.load_Halu_feature_file(f_in_train_org, f_in_train_new)
                    testx, testy = self.load_Halu_feature_file(f_in_test_org, f_in_test_new)
                elif args.dataset_cat == 'factor':
                    trainx, trainy = self.load_factor_feature_file(f_in_train_org, f_in_train_new)
                    testx, testy = self.load_factor_feature_file(f_in_test_org, f_in_test_new)
                elif args.dataset_cat == 'state' or args.dataset_cat == 'humor':
                    trainx, trainy = self.load_state_feature_file(f_in_train_org, f_in_train_new, args.dataset_cat)
                    testx, testy = self.load_state_feature_file(f_in_test_org, f_in_test_new, args.dataset_cat)

                
            plot_scatter = False
            #plot_scatter = True
            if plot_scatter:
                plot_2_features(trainx, trainy, feature_keys, args.fig_save_prefix)
                sys.exit(0)


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
                self.clf.fit(real_trainx, trainy)
                predy = self.clf.predict(real_testx)
                predy_prob = self.clf.predict_proba(real_testx)[:,1]

                print("Features: {}".format(" ".join(feats)), file=sys.stderr)
                print("="*20)
                print("Features: {}".format(" ".join(feats)))
                feat_str = "Features: {}".format(" ".join(feats))
                # fw.write("\n\nFeatures: {}\n".format(" ".join(feats)))
                acc, info = binary_eval(predy, testy, predy_prob, return_f1=False)
                if args.dataset_cat == 'Halu':
                    testx_pos = real_testx[:int(len_test/2)]
                    testx_neg = real_testx[int(len_test/2):]
                    predy_prob_pos = self.clf.predict_proba(testx_pos)[:,1]
                    predy_prob_neg = self.clf.predict_proba(testx_neg)[:,1]
                    pair_acc = np.sum(predy_prob_pos > predy_prob_neg) / len(testx_pos)
                    pair_acc_str = "\nPair Acc : {}".format(pair_acc)
                    print(pair_acc_str)
                    info += pair_acc_str
                    infos.append([pair_acc, feat_str, info])
                elif args.dataset_cat == 'factor':
                    testx_pos = real_testx[:int(len_test/4)]
                    testx_neg_1 = real_testx[int(len_test/4):int(len_test/2)]
                    testx_neg_2 = real_testx[int(len_test/2):int(len_test/4*3)]
                    testx_neg_3 = real_testx[int(len_test/4*3):]
                    predy_prob_pos = self.clf.predict_proba(testx_pos)[:,1]
                    predy_prob_neg_1 = self.clf.predict_proba(testx_neg_1)[:,1]
                    predy_prob_neg_2 = self.clf.predict_proba(testx_neg_2)[:,1]
                    predy_prob_neg_3 = self.clf.predict_proba(testx_neg_3)[:,1]
                    pair_acc = np.sum( (predy_prob_pos > predy_prob_neg_1) & (predy_prob_pos > predy_prob_neg_2) & (predy_prob_pos > predy_prob_neg_3)  ) / len(testx_pos)
                    pair_acc_str = "\n1 in 4 Acc : {}".format(pair_acc)
                    print(pair_acc_str)
                    info += pair_acc_str
                    infos.append([pair_acc, feat_str, info])
                else:
                    infos.append([acc, feat_str, info])
            infos = sorted(infos, key=lambda x:-x[0])

            for item in infos:
                _, feat_str, info = item
                fw.write("\n\n"+feat_str+"\n")
                fw.write(info+"\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", default="svm", type=str) 
    parser.add_argument("--model", default="lr", type=str) 
    #parser.add_argument("--model", default="RF", type=str) 

    parser.add_argument("--feature_select_idxs", default="", type=str)

    parser.add_argument("--dataset_cat", default="", type=str) 
    parser.add_argument("--input_train_org_name", default="", type=str) 
    parser.add_argument("--input_val_org_name", default="", type=str) 
    parser.add_argument("--input_train_feature_name", default="", type=str) 
    parser.add_argument("--input_val_feature_name", default="", type=str) 
    parser.add_argument("--output_score_name", default="", type=str) 
    
    args = parser.parse_args()
    
    dataset_cat_d2_names = {'factor': ['expert_factor', 'news_factor', 'wiki_factor'],
                     'HaDes': ['all'],
                     #'Halu': ['dialogue_data', 'qa_data', 'summarization_data_1024'],
                     'state': ['all'],
                     #'humor': ['all_128'],
                      }
    
    dataset_cat_d2_cr = {'factor': 'RF',
                     'HaDes': 'RF',
                     #'Halu': 'lr',
                     #'humor': 'RF',
                     'state': 'RF' }

    #datasets = ['dialogue_data', 'qa_data', 'dialogue_data_knowledge', 'summarization_data_1024']
    #datasets = ['summarization_data_1024']
    #datasets = ['dialogue_data', 'qa_data', 'dialogue_data_knowledge']

    org_model_name = 'wiki_smallOWT'
    #ent_model_names = ['wiki_70M', 'wiki_410M, OWT1024']
    #ent_model_names = ['wiki_70M']
    #ent_model_names = ['wiki_410M']
    #ent_model_names = ['OWT1024']
    #ent_model_names = ['OWT_b16']
    #ent_model_names = ['OWT_1b_b64']
    #ent_model_names = ['wiki_70M_b64']
    #ent_model_names = ['wiki_70M_a4_e3']
    #ent_model_names = ['OWT_70M_b64']
    #ent_model_names = ['OWT_70M_b64_e10']
    #ent_model_names = ['OWTwiki_1e7_70M_b32_e1']
    #ent_model_names = ['OWTwiki_1e7_70M_b8_a4_e1']
    #ent_model_names = ['per_OWT_1e6_70M_b128_a10_e3']
    ent_model_names = ['OWTwiki_1e7_70M_b128_a10_e3']
    #ent_model_names = ['OWTwiki_1e7_70M_b32_e3']
    #ent_model_names = ['OWTwiki_1e7_70M_b32_a4_e3']
    #ent_model_names = ['OWTwiki_1e7_70M_b32_a6_e3']
    #ent_model_names = ['OWTwiki_1e7_70M_b32_d08_e3']
    #ent_model_names = ['OWTwiki_1e7_70M_b32_e3_no_last']
    #ent_model_names = ['OWT_70M_b32_a4_e10']
    #ent_model_names = ['OWTwiki_1e7_70M_b8_e1']
    #ent_model_names = ['OWT_70M_b32_d08_e3']
    #ent_model_names = ['OWT_410M_b16_d08_e3']
    #ent_model_names = ['OWT_410M_b32_a4_e3']
    feature_idx_set = [ ['large_small_pred1', [0,1,2,3,4,5,6,7]], ['large_small_pred12_pair', [0,1,2,3,4,5,6,7] ]  ]
    #feature_idx_set = [ ['large_small_pred12_pair', [0,1,2,3,4,5,6,7,8,9] ] ]

    for ent_model_name in ent_model_names:
        for dataset_cat in dataset_cat_d2_names:
            input_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/'+dataset_cat+'/features/'
            output_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/'+dataset_cat+'/scores/'
            log_folder = '/mnt/efs/Haw-Shiuan/true_entropy/log/'+dataset_cat+'/'
            args.model = dataset_cat_d2_cr[dataset_cat]
            for dataset in dataset_cat_d2_names[dataset_cat]:
                args.input_train_org_name = input_folder + dataset + '_train_' + org_model_name + '.json'
                args.input_val_org_name = input_folder + dataset + '_val_' + org_model_name + '.json'
                args.input_train_feature_name = input_folder + dataset + '_train_' + ent_model_name + '.json'
                args.input_val_feature_name = input_folder + dataset + '_val_' + ent_model_name + '.json'
                idx_range = range(len(feature_idx_set))
                for j in idx_range:
                    feature_idx = feature_idx_set[j]
                    feature_name = feature_idx[0]
                    feature_use = feature_idx[1]
                    args.dataset_cat = dataset_cat
                    args.feature_select_idxs = ','.join([str(x) for x in feature_use])
                    args.output_score_name = output_folder + dataset + '_' + ent_model_name + '_' + feature_name + '_' + args.model + '.txt'
                    args.fig_save_prefix = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/'+dataset_cat+'/figs/' + ent_model_name
                    log_file_path = log_folder + dataset + '_' + ent_model_name + '_' + feature_name + '_stdout_' + args.model 
                    with open(log_file_path, 'w') as sys.stdout:
                        print(args)
                        rep_op = ClfModel(args)
                        rep_op.train()

