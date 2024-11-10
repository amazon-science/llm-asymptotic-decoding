import pandas as pd

folder_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/state/'

#file_suffix = '_val'
file_suffix = '_train'

output_file_name = 'all'

input_cat_list = ['animals_true_false',
    'capitals_true_false',
    'cities_true_false',
    'companies_true_false',
    'conj_neg_companies_true_false',
    'conj_neg_facts_true_false',
    'elements_true_false',
    'facts_true_false',
    'generated_true_false',
    'inventions_true_false',
    'neg_companies_true_false',
    'neg_facts_true_false']

df_all = None

for cat in input_cat_list:
    df_cat = pd.read_csv(folder_path + cat + file_suffix+'.csv')
    df_cat['category'] = cat
    df_all = pd.concat([df_all,df_cat])

df_all.to_csv(folder_path+output_file_name+file_suffix+'.csv')
