'''
    This code is adapted from https://github.com/ari-holtzman/degen/blob/master/metrics/distinct_n.py by Ali Holtzman.
'''
import argparse
import json
import os
import logging
from collections import Counter

from os import listdir
from os.path import isfile, join
from tqdm import tqdm

from nltk.tokenize import word_tokenize, sent_tokenize

import multiprocessing
import itertools
import numpy as np

logger = logging.getLogger(__name__)

'''
    Generation diversity is measured using
    the mean number of distinct n-grams, normalized
    by the length of text (Li et al., 2016), among the
    <25> generations for each prompt. We report Dist-1,
    Dist-2, and Dist-3 scores for distinct uni-, bi-, and
    trigrams, respectively
'''

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str)
    parser.add_argument("--file_template", type=str)
    parser.add_argument("--number_of_seeds", type=int, default=10)
    # parser.add_argument("N", type=int, help="N in distinct-N metric")
    parser.add_argument("--numbers-only", action="store_true")
    parser.add_argument("--num_eval_sent", type=int, default=1)
    return parser.parse_args()


def distinct_n(num_eval_sent, n, factual_examples):
    distinct_set = set()
    n_total = 0

    for example in tqdm(factual_examples, total=len(factual_examples)):
        if example['text'].strip() != "":
            #use_first_sent_only=True
            #use_first_sent_only=False
            gen = ' '.join(sent_tokenize(example['text'])[:num_eval_sent])
            #if use_first_sent_only:
            #    gen = sent_tokenize(example['text'])[0] 
            #else:
            #    gen = example['text']


            #if "WikiNamePrefix" in f_name_template:
            #    wikiPrefix = example['prompt'].split(". ")[-1].strip()
            #    gen = gen.replace(wikiPrefix, " ")

            gen_tokens = word_tokenize(gen)
            for token in zip(*(gen_tokens[i:] for i in range(n))):
                distinct_set.add(token)
                n_total += 1

    return len(distinct_set), n_total, n

def distinct_n_wrapper(_args):
    return distinct_n(*_args)

def distinct_n_local(num_eval_sent, n, factual_zip_examples):
    dist_n_list = []

    for examples in tqdm(factual_zip_examples, total=len(factual_zip_examples)):
        distinct_set = set()
        n_total = 0
        for example in examples:
            if example['text'].strip() != "":
                gen = ' '.join(sent_tokenize(example['text'])[:num_eval_sent])

                #if "WikiNamePrefix" in f_name_template:
                #    wikiPrefix = example['prompt'].split(". ")[-1].strip()
                #    gen = gen.replace(wikiPrefix, " ")

                gen_tokens = word_tokenize(gen)
                for token in zip(*(gen_tokens[i:] for i in range(n))):
                    distinct_set.add(token)
                    n_total += 1
        if n_total >0:
            dist_n = float( len(distinct_set) / n_total )
            dist_n_list.append(dist_n)

    return sum(dist_n_list), len(dist_n_list), n

def distinct_n_local_wrapper(_args):
    return distinct_n_local(*_args)


def computing_all_dist(factual_target_files, args, dir, workers, file_prefix):
    factual_examples = []
    factual_zip_examples = []

    for target_file in factual_target_files:
        with open("{}/{}".format(dir, target_file), "r") as fin:
            examples = [json.loads(l.strip()) for l in fin]
            factual_examples.extend( examples )
            factual_zip_examples.append( examples )
    
    factual_zip_examples = list(zip(*factual_zip_examples))

    # factual prompts
    factual_res_dict = {}
    #for (_n_distinct, _n_total, _n) in workers.imap_unordered(distinct_n_wrapper, zip([args.num_eval_sent]*3, [2,3,4], itertools.repeat(factual_examples), itertools.repeat(f_template))):
    for (_n_distinct, _n_total, _n) in workers.imap_unordered(distinct_n_wrapper, zip([args.num_eval_sent]*3, [2,3,4], itertools.repeat(factual_examples) )):
        #if 'greedy' in f_template:
        #    _n_total = _n_total * args.number_of_seeds # greedy will always generate same. So test on just one generation file, and multiply by # of seed used

        # print(_n, _n_distinct, _n_total)
        factual_res_dict[_n] = float(_n_distinct/_n_total)

    factual_local_res_dict = {}
    for (_n_distinct, _n_total, _n) in workers.imap_unordered(distinct_n_local_wrapper, zip([args.num_eval_sent]*4, [1,2,3,4], itertools.repeat(factual_zip_examples) )):
        factual_local_res_dict[_n] = float(_n_distinct/_n_total)

    #if args.use_first_sent_only:
    #    score_folder_name = 'scores'
    #else:
    #    score_folder_name = 'scores_all'
    if args.num_eval_sent == 1:
        score_folder_name = 'scores'
    else:
        score_folder_name = 'scores_s'+str(args.num_eval_sent)

    f_gen_path = "{}/{}/{}_{}".format(dir, score_folder_name, file_prefix, args.file_template)
    output_folder = dir+'/'+score_folder_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f_res_path = f_gen_path.replace(".jsonl", "_results.jsonl")
    with open(f_res_path, 'a') as outfile:
        res_obj = {}
        for n in [4,3,2]:
            key_ = "factual-distinct-{}".format(n)
            res_obj[key_] = factual_res_dict[n]
        for n in [4,3,2,1]:
            key_ = "factual-local-distinct-{}".format(n)
            res_obj[key_] = factual_local_res_dict[n]

        json.dump(res_obj, outfile)
        outfile.write("\n")

def main():
    args = parse_args()

    dir = args.gen_dir
    f_template = args.file_template

    print(f_template)

    factual_target_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f_template in f and 'nonfactual' not in f]
    nonfactual_target_files = [f for f in listdir(dir) if isfile(join(dir, f)) and f_template in f and 'nonfactual' in f]

    if 'greedy' not in f_template:
        print(len(factual_target_files))
        assert len(factual_target_files) == args.number_of_seeds
        if len(nonfactual_target_files) > 0:
            assert len(nonfactual_target_files) == args.number_of_seeds
    
    workers = multiprocessing.Pool(4)

    computing_all_dist(factual_target_files, args, dir, workers, 'factual')
    if len(nonfactual_target_files) > 0:
        computing_all_dist(nonfactual_target_files, args, dir, workers, 'nonfactual')

    #nonfactual_examples = []
    #for target_file in nonfactual_target_files:
    #    with open("{}/{}".format(dir, target_file), "r") as fin:
    #        nonfactual_examples.extend([json.loads(l.strip()) for l in fin])

    # nonfactual prompts
    #nonfactual_res_dict = {}
    #for (_n_distinct, _n_total, _n) in workers.imap_unordered(distinct_n_wrapper, zip([args.num_eval_sent]*3, [2,3,4], itertools.repeat(nonfactual_examples), itertools.repeat(f_template))):
    #    
    #    if 'greedy' in f_template:
    #        _n_total = _n_total * args.number_of_seeds # greedy will always generate same. So test on just one generation file, and multiply by # of seed used

    #    # print(_n, _n_distinct, _n_total)
    #    nonfactual_res_dict[_n] = float(_n_distinct/_n_total)

    #nf_gen_path = "{}/{}/nonfactual_{}".format(dir, score_folder_name, f_template)
    #nf_res_path = nf_gen_path.replace(".jsonl", "_results.jsonl")
    #with open(nf_res_path, 'a') as outfile:
    #    res_obj = {}
    #    for n in [4,3,2]:
    #        key_ = "nonfactual-distinct-{}".format(n)
    #        res_obj[key_] = nonfactual_res_dict[n]
    #    json.dump(res_obj, outfile)
    #    outfile.write("\n")
        

    
if __name__ == '__main__':
    main()

