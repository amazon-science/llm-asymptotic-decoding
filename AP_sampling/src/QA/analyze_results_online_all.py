import scipy 
import pandas as pd

#input_file_name = 'outputs/commonqa/final_Pythia_commonqa_neg_train_topk20.csv'
#input_file_name = 'outputs/qasc/final_Pythia_qasc_neg_fact_train_topk20.csv'
#input_file_name = 'outputs/qasc/final_Pythia_qasc_neg_train_topk20.csv'
#input_file_name = 'outputs/arc/final_Pythia_arc_neg_all_train_topk20.csv'
#input_file_name = 'outputs/socialiqa/final_Pythia_socialiqa_neg_train_topk20.csv'

#input_file_name = 'outputs/lambda/final_Pythia_logsoftmax_lambda_topk20_1e-2.csv'

have_acc = 1
have_acc = 0

include_apo = True
#include_apo = False

#input_file_arr = ['outputs/commonqa/online_all_Pythia_commonqa_neg_train_topk20.csv']
#input_file_arr = ['outputs/lambda/online_all_Pythia_logsoftmax_lambda_topk20_1e-2.csv']

input_file_arr = ['outputs/commonqa/online_fixed_all_Pythia_commonqa_neg_train_topk20.csv', 'outputs/qasc/online_fixed_all_Pythia_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/online_fixed_all_Pythia_qasc_neg_train_topk20.csv', 'outputs/arc/online_fixed_all_Pythia_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/online_fixed_all_Pythia_socialiqa_neg_train_topk20.csv', 'outputs/commonqa/online_fixed_all_Qwen_w1_commonqa_neg_train_topk20.csv', 'outputs/qasc/online_fixed_all_Qwen_w1_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/online_fixed_all_Qwen_w1_qasc_neg_train_topk20.csv', 'outputs/arc/online_fixed_all_Qwen_w1_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/online_fixed_all_Qwen_w1_socialiqa_neg_train_topk20.csv' ]

input_file_arr = ['outputs/squad/online_fixed_all_Pythia_logsoftmax_squad_val_topk20.csv', 'outputs/squad/online_fixed_all_Pythia_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/online_fixed_all_Pythia_logsoftmax_multirc_train_topk20.csv', 'outputs/lambda/online_fixed_all_Pythia_logsoftmax_lambda_topk20_1e-2.csv', 'outputs/squad/online_fixed_all_Qwen_w1_logsoftmax_squad_val_topk20.csv', 'outputs/squad/online_fixed_all_Qwen_w1_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/online_fixed_all_Qwen_w1_logsoftmax_multirc_train_topk20.csv', 'outputs/lambda/online_fixed_all_Qwen_w1_logsoftmax_lambda_topk20_1e-2.csv']

#input_file_arr = ['outputs/commonqa/online_all_Pythia_commonqa_neg_train_topk20.csv', 'outputs/qasc/online_all_Pythia_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/online_all_Pythia_qasc_neg_train_topk20.csv', 'outputs/arc/online_all_Pythia_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/online_all_Pythia_socialiqa_neg_train_topk20.csv', 'outputs/commonqa/online_all_Qwen_w1_commonqa_neg_train_topk20.csv', 'outputs/qasc/online_all_Qwen_w1_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/online_all_Qwen_w1_qasc_neg_train_topk20.csv', 'outputs/arc/online_all_Qwen_w1_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/online_all_Qwen_w1_socialiqa_neg_train_topk20.csv' ]

#input_file_arr = ['outputs/squad/online_all_Pythia_logsoftmax_squad_val_topk20.csv', 'outputs/squad/online_all_Pythia_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/online_all_Pythia_logsoftmax_multirc_train_topk20.csv', 'outputs/lambda/online_all_Pythia_logsoftmax_lambda_topk20_1e-2.csv', 'outputs/squad/online_all_Qwen_w1_logsoftmax_squad_val_topk20.csv', 'outputs/squad/online_all_Qwen_w1_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/online_all_Qwen_w1_logsoftmax_multirc_train_topk20.csv', 'outputs/lambda/online_all_Qwen_w1_logsoftmax_lambda_topk20_1e-2.csv']

#input_file_arr = ['outputs/commonqa/online_Pythia_commonqa_neg_train_topk20.csv', 'outputs/qasc/online_Pythia_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/online_Pythia_qasc_neg_train_topk20.csv', 'outputs/arc/online_Pythia_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/online_Pythia_socialiqa_neg_train_topk20.csv', 'outputs/commonqa/online_OPT_commonqa_neg_train_topk20.csv', 'outputs/qasc/online_OPT_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/online_OPT_qasc_neg_train_topk20.csv', 'outputs/arc/online_OPT_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/online_OPT_socialiqa_neg_train_topk20.csv', 'outputs/commonqa/online_Qwen_w1_commonqa_neg_train_topk20.csv', 'outputs/qasc/online_Qwen_w1_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/online_Qwen_w1_qasc_neg_train_topk20.csv', 'outputs/arc/online_Qwen_w1_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/online_Qwen_w1_socialiqa_neg_train_topk20.csv' ]

#input_file_arr = ['outputs/lambda/online_OPT_logsoftmax_lambda_topk20_1e-2.csv', 'outputs/squad/online_OPT_logsoftmax_squad_val_topk20.csv', 'outputs/squad/online_OPT_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/online_OPT_logsoftmax_multirc_train_topk20.csv']

#input_file_arr = ['outputs/squad/online_Pythia_logsoftmax_squad_val_topk20.csv', 'outputs/squad/online_Pythia_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/online_Pythia_logsoftmax_multirc_train_topk20.csv', 'outputs/lambda/online_Pythia_logsoftmax_lambda_topk20_1e-2.csv', 'outputs/lambda/online_OPT_logsoftmax_lambda_topk20_1e-2.csv', 'outputs/squad/online_OPT_logsoftmax_squad_val_topk20.csv', 'outputs/squad/online_OPT_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/lambda/online_OPT_logsoftmax_lambda_topk20_1e-2.csv', 'outputs/multirc/online_OPT_logsoftmax_multirc_train_topk20.csv', 'outputs/squad/online_Qwen_w1_logsoftmax_squad_val_topk20.csv', 'outputs/squad/online_Qwen_w1_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/online_Qwen_w1_logsoftmax_multirc_train_topk20.csv', 'outputs/lambda/online_Qwen_w1_logsoftmax_lambda_topk20_1e-2.csv']

#input_file_arr = ['outputs/commonqa/final_Pythia_commonqa_neg_train_topk20.csv', 'outputs/qasc/final_Pythia_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/final_Pythia_qasc_neg_train_topk20.csv', 'outputs/arc/final_Pythia_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/final_Pythia_socialiqa_neg_train_topk20.csv', 'outputs/commonqa/final_OPT_commonqa_neg_train_topk20.csv', 'outputs/qasc/final_OPT_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/final_OPT_qasc_neg_train_topk20.csv', 'outputs/arc/final_OPT_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/final_OPT_socialiqa_neg_train_topk20.csv', 'outputs/commonqa/final_Qwen_w1_commonqa_neg_train_topk20.csv', 'outputs/qasc/final_Qwen_w1_qasc_neg_fact_train_topk20.csv', 'outputs/qasc/final_Qwen_w1_qasc_neg_train_topk20.csv', 'outputs/arc/final_Qwen_w1_arc_neg_all_train_topk20.csv', 'outputs/socialiqa/final_Qwen_w1_socialiqa_neg_train_topk20.csv' ]

#input_file_arr = ['outputs/squad/final_Pythia_logsoftmax_squad_val_topk20.csv', 'outputs/squad/final_OPT_logsoftmax_squad_val_topk20.csv', 'outputs/squad/final_Pythia_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/squad/final_OPT_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/final_Pythia_logsoftmax_multirc_train_topk20.csv', 'outputs/multirc/final_OPT_logsoftmax_multirc_train_topk20.csv', 'outputs/lambda/final_Pythia_logsoftmax_lambda_topk20_1e-2.csv', 'outputs/lambda/final_OPT_logsoftmax_lambda_topk20_1e-2.csv']

#input_file_arr = ['outputs/squad/final_Qwen_w1_logsoftmax_squad_val_topk20.csv', 'outputs/squad/final_Qwen_w1_logsoftmax_squad_val_no_pass_topk20.csv', 'outputs/multirc/final_Qwen_w1_logsoftmax_multirc_train_topk20.csv', 'outputs/lambda/final_Qwen_w1_logsoftmax_lambda_topk20_1e-2.csv']

for input_file_name in input_file_arr:

    df = pd.read_csv(input_file_name)
    df = df.set_index('Unnamed: 0')

    ppl_mrr_row_name = 'ppl_llm_mean,ppl_llm_std,mrr_llm_mean,mrr_llm_std, ppl_hlm_mean,ppl_hlm_std,mrr_hlm_mean,mrr_hlm_std,num,ratio '
    diff_acc_row_name = 'ppl_diff_llm_mean,ppl_diff_llm_std,acc_llm_mean,acc_llm_std, ppl_diff_hlm_mean,ppl_diff_hlm_std,acc_hlm_mean,acc_hlm_std,num,ratio '

    def get_best_results(df, method_name, get_max):
        if get_max:
            min_idx = df.loc[method_name + '_mean'].argmax()
        else:
            min_idx = df.loc[method_name + '_mean'].argmin()
        best_mean = df.loc[method_name + '_mean', str(min_idx)]
        best_temp = df.loc[method_name + '_temp', str(min_idx)]
        best_std = df.loc[method_name + '_std', str(min_idx)]
        return best_mean, best_std, best_temp

    ppl_llm_mean, ppl_llm_std, mrr_llm_mean, mrr_llm_std, ppl_hlm_mean, ppl_hlm_std, mrr_hlm_mean, mrr_hlm_std, num_ppl  = df.loc[ ppl_mrr_row_name, [str(x) for x in range(9)] ].tolist()

    ap_ppl_best_mean, ap_ppl_best_std, ap_ppl_best_temp = get_best_results(df, 'ap_ppl', get_max = 0)
    cd_ppl_best_mean, cd_ppl_best_std, cd_ppl_best_temp = get_best_results(df, 'cd_ppl', get_max = 0)
    ap_mrr_best_mean, ap_mrr_best_std, ap_mrr_best_temp = get_best_results(df, 'ap_mrr', get_max = 1)
    cd_mrr_best_mean, cd_mrr_best_std, cd_mrr_best_temp = get_best_results(df, 'cd_mrr', get_max = 1)
    if include_apo:
        apo_ppl_best_mean, apo_ppl_best_std, apo_ppl_best_temp = get_best_results(df, 'apo_ppl', get_max = 0)
        apo_mrr_best_mean, apo_mrr_best_std, apo_mrr_best_temp = get_best_results(df, 'apo_mrr', get_max = 1)
    else:
        apo_ppl_best_mean = 0
        apo_mrr_best_mean = 0
        apo_ppl_best_temp = 0
        apo_mrr_best_temp = 0

    print(input_file_name)
    print('ppl ', ppl_llm_mean, cd_ppl_best_mean, ap_ppl_best_mean, apo_ppl_best_mean,  ppl_hlm_mean, 100 * (cd_ppl_best_mean - ap_ppl_best_mean) / (ppl_llm_mean - ppl_hlm_mean), 100 * (ppl_llm_mean - ap_ppl_best_mean ) / (ppl_llm_mean - ppl_hlm_mean), apo_ppl_best_temp  )
    print('mrr ', mrr_llm_mean, cd_mrr_best_mean, ap_mrr_best_mean, apo_mrr_best_mean, mrr_hlm_mean, 100 * (ap_mrr_best_mean - cd_mrr_best_mean) / (mrr_hlm_mean - mrr_llm_mean), 100 * (ap_mrr_best_mean - mrr_llm_mean  ) / (mrr_hlm_mean - mrr_llm_mean), apo_mrr_best_temp  )
    #print(scipy.stats.ttest_ind_from_stats(ap_ppl_best_mean, ap_ppl_best_std, num_ppl, cd_ppl_best_mean, cd_ppl_best_std, num_ppl, 'less' ))
    #print(scipy.stats.ttest_ind_from_stats(ap_mrr_best_mean, ap_mrr_best_std, num_ppl, cd_mrr_best_mean, cd_mrr_best_std, num_ppl, 'greater' ))
    #print(scipy.stats.ttest_ind_from_stats(ap_acc_best_mean, ap_acc_best_std, num_acc, cd_acc_best_mean, cd_acc_best_std, num_acc, 'greater' ))

    print('ppl_p ', scipy.stats.ttest_ind_from_stats(ap_ppl_best_mean, ap_ppl_best_std, num_ppl, cd_ppl_best_mean, cd_ppl_best_std, num_ppl ).pvalue )
    print('mrr_p ', scipy.stats.ttest_ind_from_stats(ap_mrr_best_mean, ap_mrr_best_std, num_ppl, cd_mrr_best_mean, cd_mrr_best_std, num_ppl ).pvalue )
    print(num_ppl)

    if have_acc:
        ppl_diff_llm_mean, ppl_diff_llm_std, acc_llm_mean, acc_llm_std, ppl_diff_hlm_mean, ppl_diff_hlm_std, acc_hlm_mean, acc_hlm_std, num_acc  = df.loc[ diff_acc_row_name, [str(x) for x in range(9)] ].tolist()
        ap_acc_best_mean, ap_acc_best_std, ap_acc_best_temp = get_best_results(df, 'ap_acc', get_max = 1)
        cd_acc_best_mean, cd_acc_best_std, cd_acc_best_temp = get_best_results(df, 'cd_acc', get_max = 1)
        if include_apo:
            apo_acc_best_mean, apo_acc_best_std, apo_acc_best_temp = get_best_results(df, 'apo_acc', get_max = 1)
        else:
            apo_acc_best_mean = 0

        print('acc ', acc_llm_mean, cd_acc_best_mean, ap_acc_best_mean, apo_acc_best_mean, acc_hlm_mean, 100 * (ap_acc_best_mean - cd_acc_best_mean) / (acc_hlm_mean - acc_llm_mean), 100 * (ap_acc_best_mean - acc_llm_mean) / (acc_hlm_mean - acc_llm_mean), apo_acc_best_temp)
        print('acc_p ',scipy.stats.ttest_ind_from_stats(ap_acc_best_mean, ap_acc_best_std, num_acc, cd_acc_best_mean, cd_acc_best_std, num_acc ).pvalue)
        print(num_acc)
