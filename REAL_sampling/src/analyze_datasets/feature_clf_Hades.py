from transformers import BertTokenizer, BertForMaskedLM, AutoModelWithLMHead, AutoTokenizer, BertConfig, AutoConfig
import torch
from nltk import sent_tokenize
from nltk.corpus import stopwords
import random, os, sys
import torch.nn as nn
import torch.nn.functional as F
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
from utils import binary_eval, remove_marked_sen
import itertools

repo_folder = '/mnt/efs/Haw-Shiuan/HaDes/'

def get_ppmi_matrix(voc, path=repo_folder+"data_collections/Wiki-Hades/train.txt"):
    def co_occurrence(sentences, window_size):
        d = defaultdict(int)
        vocab = set(voc)
        for text in sentences:
            # iterate over sentences
            # print(text)
            for i in range(len(text)):
                token = text[i]
                next_token = text[i+1 : i+1+window_size]
                for t in next_token:
                    if t in vocab and token in vocab:
                        key = tuple( sorted([t, token]) )
                        d[key] += 1
        # print(vocab)
        print(len(vocab))

        # formulate the dictionary into dataframe
        vocab = sorted(vocab) # sort vocab
        df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                          index=vocab,
                          columns=vocab)
        for key, value in d.items():
            df.at[key[0], key[1]] = value
            df.at[key[1], key[0]] = value
        return df

    def pmi(df, positive=True):
        col_totals = df.sum(axis=0)
        total = col_totals.sum()
        row_totals = df.sum(axis=1)
        expected = np.outer(row_totals, col_totals) / total
        df = df / expected
        return df

    corpus = []

    with codecs.open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            example = json.loads(line.strip())
            tgt, tgt_ids = example["replaced"], example["replaced_ids"]
            sen = remove_marked_sen(tgt, tgt_ids[0], tgt_ids[1])
            corpus.append(sen)

    df = co_occurrence(corpus, 1)
    ppmi = pmi(df, positive=True)
    print("finish")
    return ppmi


def get_idf_matrix(path=repo_folder+"data_collections/Wiki-Hades/train.txt"):
    corpus = []

    with codecs.open(path, "r", encoding="utf-8") as fr:
        for line in fr:
            example = json.loads(line.strip())
            tgt, tgt_ids = example["replaced"], example["replaced_ids"]
            sen = remove_marked_sen(tgt, tgt_ids[0], tgt_ids[1])
            corpus.append(" ".join(sen))

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    word = vectorizer.get_feature_names()
    num_doc, num_vocab = X.shape
    X = np.array(X>0, dtype=int)
    word_idf = np.log10(num_doc / (X.sum(0)+1))
    idf_dic = dict()
    for w, idf in zip(word, word_idf):
        idf_dic[w] = idf

    word_freq = X.sum(0)
    word_freq = word_freq / word_freq.sum()

    return idf_dic, word_freq


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


class ClfModel:
      def __init__(self, args):
            self.idf_dic, self.p_word = get_idf_matrix()
            #if not os.path.exists("Hades_ppmi.pkl"):
            #    print("reading ppmi ...")
            #    word_ppmi = get_ppmi_matrix(list(self.idf_dic.keys())[:])
            #    self.word_ppmi = word_ppmi
            #    word_ppmi.to_pickle("Hades_ppmi.pkl")
            #else:
            #    word_ppmi = pd.read_pickle("Hades_ppmi.pkl")
            #    self.word_ppmi = word_ppmi
            self.select_idxs = [int(x) for x in args.feature_select_idxs.split(',')]
            self.args = args
            self.device = args.device
            self.model = args.model
            #self.rep_model = AutoModelWithLMHead.from_pretrained("bert-base-uncased").to(self.device)
            #self.rep_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            if "svm" in self.model:
                self.clf = svm.LinearSVC()
            elif 'lr' in self.model:
                self.clf = make_pipeline(StandardScaler(),
                                         SGDClassifier(loss="log", max_iter=10000, tol=1e-5))
            elif 'RF' in self.model:
                self.clf = RandomForestClassifier(max_depth=5, n_estimators=100)

      def get_ppmi_features(self, rep_sen, rep_ids):
            rep_start_id, rep_end_id = rep_ids
            rep_tokens = remove_marked_sen(rep_sen, rep_start_id, rep_end_id)

            max_ppmi, mean_ppmi, min_ppmi = [], [], []
            for idx in range(rep_start_id, rep_end_id+1):
                ppmis = []
                for j in range(rep_start_id):
                    v = 0
                    if rep_tokens[idx] in self.word_ppmi.columns and \
                       rep_tokens[j] in self.word_ppmi.columns:
                        v = self.word_ppmi.at[rep_tokens[idx], rep_tokens[j]]
                        if v > 0:
                            v = max(0, np.log(v))
                        else:
                            v = 0 # not gona happen
                    ppmis.append(v)
                max_ppmi.append(max(ppmis))
                min_ppmi.append(min(ppmis))
                mean_ppmi.append(sum(ppmis)/len(ppmis))

            return sum(mean_ppmi) / len(mean_ppmi), sum(max_ppmi) / len(max_ppmi), sum(min_ppmi) / len(min_ppmi)

      def get_tfidf_features(self, rep_sen, rep_ids):
            rep_start_id, rep_end_id = rep_ids
            rep_tokens = remove_marked_sen(rep_sen, rep_start_id, rep_end_id)

            #  TF-IDF features
            tf_dic = dict()
            for token in rep_tokens:
                if token in self.idf_dic:
                    if token not in tf_dic:
                        tf_dic[token] = 1
                    else:
                        tf_dic[token] += 1

            tf_total = sum(tf_dic.values())
            tfidf_list = []
            for idx in range(rep_start_id, rep_end_id+1):
                if rep_tokens[idx] in self.idf_dic:
                    tfidf_list.append(tf_dic[rep_tokens[idx]] * self.idf_dic[rep_tokens[idx]] / tf_total)
            tfidf_max = max(tfidf_list) if tfidf_list else 0.
            tfidf_min = min(tfidf_list) if tfidf_list else 0.
            tfidf_mean = sum(tfidf_list)/len(tfidf_list) if tfidf_list else 0.
            return tfidf_mean, tfidf_max, tfidf_min

      def encode_bert(self, rep_sen, rep_ids):
            rep_start_id, rep_end_id = rep_ids
            rep_tokens = remove_marked_sen(rep_sen, rep_start_id, rep_end_id)
            #  Prob, Entropy features
            rep_subtokens = ["[CLS]"]
            tokenizer = self.rep_tokenizer
            model = self.rep_model
            rep_mask_start_id, rep_mask_end_id = 0, 0
            for id, rep_token in enumerate(rep_tokens):
                rep_subtoken = tokenizer.tokenize(rep_token)
                if id == rep_start_id:
                    rep_mask_start_id = len(rep_subtokens)
                if id == rep_end_id:
                    rep_mask_end_id = len(rep_subtokens) + len(rep_subtoken)
                if id >= rep_start_id and id <= rep_end_id:
                    rep_subtokens.extend(len(rep_subtoken) * ["[MASK]"])
                else:
                    rep_subtokens.extend(rep_subtoken)
            rep_subtokens.append("[SEP]")
            rep_input_ids = torch.LongTensor(tokenizer.convert_tokens_to_ids(rep_subtokens)).unsqueeze(0).to(self.device)
            prediction_scores = model(rep_input_ids)[0]
            prediction_scores = F.softmax(prediction_scores, dim=-1)

            scores = []
            for id in range(rep_mask_start_id, rep_mask_end_id):
                subtoken_score = prediction_scores[0, id, rep_input_ids[0][id]].item()
                scores.append(subtoken_score)

            entropies = []
            for id in range(rep_mask_start_id, rep_mask_end_id):
                vocab_scores = prediction_scores[0, id].detach().cpu().numpy()
                entropy = np.sum(np.log(vocab_scores+1e-11) * vocab_scores)
                entropies.append(-entropy)

            return sum(scores)/len(scores), max(scores), min(scores), \
                   sum(entropies)/len(entropies), max(entropies), min(entropies)

      def prepare_features(self, example, input_d2_features):
            #avgscore, maxscore, _, avgentro, maxentro, _ = self.encode_bert(example["replaced"], example["replaced_ids"])
            #avgtfidf, maxtfidf, _ = self.get_tfidf_features(example["replaced"], example["replaced_ids"])
            #avgppmi, maxppmi, _ = self.get_ppmi_features(example["replaced"], example["replaced_ids"])
            entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2 = input_d2_features[example["replaced"]]

            features = [perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, entropy_tensor_small, ent_score3, per_score3,  ent_score1, per_score1, c_ent, c_per]
            #features = [#avgscore, avgentro, avgtfidf, avgppmi,
                        #maxscore, maxentro, maxtfidf, maxppmi, 
                        #perplexity_tensor_large, entropy_tensor_large, perplexity_tensor_small, perplexity_tensor_large * ent_score1, ent_score1, per_score1, ent_score3, c_ent, c_per]
                        #perplexity_tensor_large, perplexity_tensor_large * ent_score1, ent_score1, per_score1, ent_score3, c_ent, c_per]
                        #entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2]

            if len(self.select_idxs) > 0:
                features = [features[idx] for idx in self.select_idxs]

            return features


      def train(self, trainpath=repo_folder+"data_collections/Wiki-Hades/train.txt",
              testpath=repo_folder+"data_collections/Wiki-Hades/valid.txt", epoch=10):
            
            #ent_train_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/train_features.json'
            #ent_valid_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/valid_features.json'
            #ent_train_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/train_features_wiki_small.json'
            #ent_valid_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/valid_features_wiki_small.json'
            ent_train_path = self.args.input_train_feature_name
            ent_valid_path = self.args.input_val_feature_name
            with open(ent_train_path) as f_in:
                input_d2_features_train = json.load(f_in)
            with open(ent_valid_path) as f_in:
                input_d2_features_valid = json.load(f_in)

            feature_keys = ["per_large", 'ent_large', 'per_small', "ent_small", 'ent_diff', 'per_diff', "ent_score1", "per_score1", "c_ent", "c_per"]
            #feature_keys = ["perplexity_tensor_large", 'ent_large', 'per_small', "perplexity_tensor_large*ent_score1", "ent_score1", "per_score1", 'ent_diff', "c_ent", "c_per"
                            #"avgscore", "avgentro", "avgtfidf", "avgppmi",
                            # "maxscore", "maxentro", "maxtfidf", "maxppmi",
                            # "perplexity_tensor_large", "perplexity_tensor_large*ent_score1", "ent_score1", "per_score1", 'ent_diff', "c_ent", "c_per"
                             #'ent_small', 'ent_large','ent_diff',
                             #'per_small','per_large','per_diff', 
                             #'c_ent','pred_last_ent', 'curve_last_ent', 'ent_score1', 'ent_score2',
                             #'c_per','pred_last_per', 'curve_last_per', 'per_score1', 'per_score2'
                            #]
            if len(self.select_idxs) > 0:
                feature_keys = [feature_keys[idx] for idx in self.select_idxs]
            feature_names = {}
            for i in range(len(feature_keys)):
                feature_names[feature_keys[i]] = i
            
            #feature_names = {"avgscore": 0, "avgentro": 1, "avgtfidf": 2, "avgppmi": 3,
            #                 "maxscore": 4, "maxentro": 5, "maxtfidf": 6, "maxppmi": 7,
            #                 'ent_small': 8, 'ent_large': 9,'ent_diff': 10,
            #                 'per_small': 11,'per_large': 12,'per_diff': 13,
            #                 'c_ent': 14,'pred_last_ent': 15, 'curve_last_ent': 16, 'ent_score1': 17, 'ent_score2': 18,
            #                 'c_per': 19,'pred_last_per': 20, 'curve_last_per': 21, 'per_score1': 22, 'per_score2': 23
            #                 }
            #feature_keys = list(feature_names.keys())[:]
            if 'pair' in self.args.output_score_name:
                combinations = list(itertools.combinations(feature_keys, 2) )
            else:
                combinations = subsets(feature_keys)

            #encode_func = self.encode_bert
            trainx, trainy = [], []
            print("Load Training Features ...")
            with codecs.open(trainpath, "r", encoding="utf-8") as fr:
                cnt = 0
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label == 2: continue
                    features = self.prepare_features(example, input_d2_features_train)

                    trainx.append(features)
                    trainy.append(label)
                    cnt += 1
                    if cnt % 500 == 0:
                        print("Train {}".format(cnt))
            #print(trainx[:30])
            #print(trainy)
            #print((np.array(trainy)))
            #print((np.array(trainy)==1))
            #print(np.squeeze(np.array(trainy)==1))
            #print(np.where( np.squeeze(np.array(trainy)==1)))
            X_pos = np.array(trainx)[ np.where( np.squeeze(np.array(trainy)==1))[0],:  ]
            X_neg = np.array(trainx)[ np.where( np.squeeze(np.array(trainy)==0))[0],:  ]
            print("hallu avg features", np.mean(X_pos, axis=0))
            print("not hallu avg features", np.mean(X_neg, axis=0))
            plot_scatter = False
            #plot_scatter = True
            if plot_scatter:
                feature_idx = [0, 6]
                label_name = ['not hallucination', 'hallucination']
                X = np.array(trainx)
                X1 = X[:, feature_idx[0]]
                X2 = X[:, feature_idx[1]]
                y = np.array(trainy)#.view(-1,1)
                plt.figure()
                #plt.scatter(X1[y==1],X2[y==1], label=label_name[1], alpha=0.3, s=0.2)
                plt.scatter(X1[y==0],X2[y==0], label=label_name[0], alpha=0.3, s=0.2)
                plt.legend()
                #plt.ylim(0, 0.15)
                plt.xlim(0, 14)
                plt.xlabel(feature_keys[feature_idx[0]])
                plt.ylabel(feature_keys[feature_idx[1]])
                #plt.savefig('per_ent_score1.png')
                plt.savefig('per_ent_score1_0.png')
                
                plt.figure()
                plt.scatter(X1[y==1],X2[y==1], label=label_name[1], alpha=0.3, s=0.2)
                plt.legend()
                #plt.ylim(0, 0.15)
                #plt.xlim(0, 6)
                plt.xlim(0, 14)
                plt.xlabel(feature_keys[feature_idx[0]])
                plt.ylabel(feature_keys[feature_idx[1]])
                plt.savefig('per_ent_score1_1.png')
                sys.exit()
                #return

            testx, testy = [], []
            with codecs.open(testpath, "r", encoding="utf-8") as fr:
                cnt = 0
                for line in fr:
                    example = json.loads(line.strip())
                    label = example["hallucination"]
                    if label == 2: continue
                    features = self.prepare_features(example, input_d2_features_valid)

                    testx.append(features)
                    testy.append(label)
                    cnt += 1
                    if cnt % 500 == 0:
                        print("Test {}".format(cnt))

            #fw = codecs.open("feature_combination_us_lr.txt", "w+", encoding="utf-8")
            fw = codecs.open(self.args.output_score_name, "w+", encoding="utf-8")
            #fw = codecs.open("feature_combination_RF5.txt", "w+", encoding="utf-8")
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

                self.clf.fit(real_trainx, trainy)
                predy = self.clf.predict(real_testx)
                predy_prob = self.clf.predict_proba(real_testx)[:,1]

                print("Features: {}".format(" ".join(feats)), file=sys.stderr)
                print("="*20)
                print("Features: {}".format(" ".join(feats)))
                feat_str = "Features: {}".format(" ".join(feats))
                # fw.write("\n\nFeatures: {}\n".format(" ".join(feats)))
                acc, info = binary_eval(predy, testy, predy_prob, return_f1=False)
                infos.append([acc, feat_str, info])
            infos = sorted(infos, key=lambda x:-x[0])

            for item in infos:
                _, feat_str, info = item
                fw.write("\n\n"+feat_str+"\n")
                fw.write(info+"\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model", default="svm", type=str) # svm or lr
    #parser.add_argument("--model", default="lr", type=str) # svm or lr
    parser.add_argument("--model", default="RF", type=str) # svm or lr
    #parser.add_argument("--input_train_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/train_features.json", type=str)
    #parser.add_argument("--input_val_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/valid_features.json", type=str)
    #parser.add_argument("--input_train_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/train_features_wiki_small.json", type=str)
    #parser.add_argument("--input_val_feature_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/valid_features_wiki_small.json", type=str)

    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/scores/all_OWT_perOWT_lr", type=str)
    parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/scores/all_OWT_perOWT_RF", type=str)
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/scores/all_wiki_smallOWT_lr", type=str)
    #parser.add_argument("--output_score_name", default="/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/scores/all_wiki_smallOWT_RF", type=str)

    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    ent_model_names = ['OWT_perOWT', 'wiki_smallOWT']
    feature_idx_set = [ ['large', [0,1]], ['small', [2,3]], ['large_small', [0,1,2,3,4,5]], ['large_small_pred1', [0,1,2,3,4,5,6,8]], ['large_small_pred2', [0,1,2,3,4,5,7,9]], ['large_small_pred12', [0,1,2,3,4,5,6,7,8,9]], ['large_small_pred12_pair', [0,1,2,3,4,5,6,7,8,9] ]  ]
    #feature_idx_set = [ ['large_small_pred12_pair', [0,1,2,3,4,5,6,7,8,9] ] ]
    #feature_idx_set = [ ['large_small_pred12_plot', [0,1,2,3,4,5,6,7,8,9]]  ]

    input_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/features/'
    output_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/scores/'
    log_folder = '/mnt/efs/Haw-Shiuan/true_entropy/log/HaDes/'

    dataset = 'all_span'

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

