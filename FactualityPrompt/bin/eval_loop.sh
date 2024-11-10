#!/bin/bash

FOLDER_PATH="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factual_gen/"
PROMPT_TYPE="factual_test7k"

GEN_MODEL="6.9b"
#GEN_MODEL="OpenLLaMA2-7b"
#GEN_MODEL="OPT-6.7b"

#METHOD_ARR=( 'CD' )
#MODEL_ARR=( 'opt-125m')
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.05;0.1;0.3;0.6;0.8' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_CD_topp' 'CD' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'opt-125m')
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.05;0.1;0.3;0.6;0.8' )
#DT_ARR=( '0.7;1.0;2.0;4.0;8.0' '1.0' )

#METHOD_ARR=( 'fe_topp' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '1.0' '1.0;0.8;0.7;0.6;0.5;0.3;0.0' )
#DT_ARR=( '8.0;4.0;3.0;2.0;1.0;0.7;0.5' '1.0' )

#METHOD_ARR=( 'AP_topp')
#MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e3_only_top_last_w_10_l1_reg_w_5_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.6' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'AP_topp' 'CD_topp' )
#MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_10_lr-4' 'opt-125m' )
#SUBMETHOD_ARR=( 'a' 'a' )
#P_ARR=( '0.2' '0.2;0.4' )
#DT_ARR=( '1.0' '0.5' )

#METHOD_ARR=( 'AP_topp' 'CD_topp' )
#MODEL_ARR=( 'prob_wiki_ext_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_10_lr-4' 'pythia-70m-deduped' )
#SUBMETHOD_ARR=( 'a' 'a' )
#P_ARR=( '0.2' '0.2' )
#DT_ARR=( '1.0' '0.5' )


#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_opt_wiki_ext_new_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_03_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.6' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_opt_wiki_ext_new_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_04_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.1' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'CD' )
#MODEL_ARR=( 'opt-125m' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.2;0.6' )
#DT_ARR=( '0.25' )

#METHOD_ARR=( 'fe_EAD_no_ELI' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '-0.5' )
#DT_ARR=( '8.0' )

#METHOD_ARR=( 'fe_EAD_no_ELI' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '-1.0' )
##DT_ARR=( '8.0' )
#DT_ARR=( '16.0' )

METHOD_ARR=( 'fe_EAD_no_ELI' )
MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
SUBMETHOD_ARR=( 'a' )
P_ARR=( '0.0' )
DT_ARR=( '3.0' )
#DT_ARR=( '1.2' )
#DT_ARR=( '1.5' )

#METHOD_ARR=( 'EAD_no_ELI' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.2' )
#DT_ARR=( '0.0' )

#METHOD_ARR=( 'microstat' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '1.0' )
#DT_ARR=( '2.0' )



#METHOD_ARR=( 'EDT' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '3.0' )
#DT_ARR=( '1.5;0.4;0.6' )

#METHOD_ARR=( 'adaptive' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.05;0.1' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'DoLa' )
#MODEL_ARR=( 'even' )
#SUBMETHOD_ARR=( '0,2,4,6,8,10,12,14,32' )
#P_ARR=( '0.6' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'DoLa' )
#MODEL_ARR=( 'last' )
#SUBMETHOD_ARR=( '16,18,20,22,24,26,28,30,32' )
#P_ARR=( '0.6' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_1b_bsz_32_exp_pred_last_a10_e3' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '2.0;1.0' )
#DT_ARR=( '0.5;4.0' )
#DT_ARR=( '1.5' )
#DT_ARR=( '8.0' )
#DT_ARR=( '0.5;1.0;1.5;2.0;4.0;8.0' )

#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
##P_ARR=( '0.25' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.5;1.0;1.5;2.0;4.0' )

#METHOD_ARR=( 'CD' )
#MODEL_ARR=( 'pythia-70m-deduped')
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.1' )
#DT_ARR=( '0.5' )

#METHOD_ARR=( 'CD' )
#MODEL_ARR=( 'opt-125m' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.1' )
#DT_ARR=( '0.25' )


#METHOD_ARR=( 'CD_topp' )
#MODEL_ARR=( 'opt-125m' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.3' )
#DT_ARR=( '0.25' )


#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OPT_OWT_wiki_1e7_70M_bsz_64_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '0.25' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_AP_topp'  )
#MODEL_ARR=( 'OPT_OWT_wiki_1e7_70M_bsz_64_exp_pred_last_a10_e3_prob_opt_wiki_ext_new_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_04_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OPT_OWT_wiki_1e7_70M_bsz_64_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.5' )

#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_160M_bsz_32_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.2' )
#DT_ARR=( '1.0' )


#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_1_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.6' )
#DT_ARR=( '1.0' )


#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_ld_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_scaled_a3_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.2' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_a3_lr-4' )
#MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_10_lr-4' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '4.0' )

#METHOD_ARR=( 'CD_topp' )
#MODEL_ARR=( 'pythia-70m-deduped' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.15' )
#P_ARR=( '0.2' )
#P_ARR=( '0.6' )
#DT_ARR=( '0.5' )

#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_a3_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_04_logit_a3_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_1_logit_logistic_decay_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_ld_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_04_logit_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_02_logit_lr-4' )
##MODEL_ARR=( 'prob_wiki_ext_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_10_temp05_lr-4' )
##MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_75_lr-4' )
###MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_10_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_1_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.6' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '0.5' )
#DT_ARR=( '1.0' )



#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '0.5' )
#DT_ARR=( '0.7' )

#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_opt_wiki_ext_new_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_04_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.6' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'CD_topp' )
#MODEL_ARR=( 'opt-125m' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.2;0.4;0.6' )
#DT_ARR=( '0.25' )

#METHOD_ARR=( 'CD_topp' )
#MODEL_ARR=( 'pythia-70m-deduped' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.4' )
#DT_ARR=( '0.25;0.75' )

#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' )
#METHOD_ARR=( 'a' )
#P_ARR=( '0.15' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_AP_topp' )
#MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_04_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'AP_topp' 'CD_topp' 'fe_AP_topp' 'fe_CD_topp' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' 'pythia-70m-deduped' 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'a' 'a' 'exp_1_win' 'exp_1_win' )
#P_ARR=( '0.1;0.2;0.3;0.4;0.6;0.8' '0.1;0.15' '1.0' '0.5' )
#DT_ARR=( '1.0' '1.0' '0.7;1.0;2.0;4.0' '1.0;2.0;4.0' )

#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '0.5' )
#DT_ARR=( '2.0' )

#METHOD_ARR=( 'fe_AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' )
##MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_a3_lr-4' )
##MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_10_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_1_logit_logistic_decay_lr-4' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_1_logit_exp_decay_lr-4' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.7' )



#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' )
##MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_04_logit_a3_lr-4' )
##MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_a3_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.4;0.8' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_1b_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.5;4.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a1_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0;1.2;1.5;2.0;3.0;4.0;8.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a6_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0;1.2;1.5;2.0;3.0;4.0;8.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a4_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0;1.2;1.5;2.0;3.0;4.0;8.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a2_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0;1.2;1.5;2.0;3.0;4.0;8.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a3_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0;1.2;1.5;2.0;3.0;4.0;8.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_scaled_a5_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '2.0' )

#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.2;0.85' )

#METHOD_ARR=( 'fe_topp' 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_logistic_decay_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_scaled_a5_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' )
#P_ARR=( '1.0' '1.0' )
#DT_ARR=( '0.1' '2.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_exp_decay_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.05;0.1;0.15;0.2;0.3;0.5;1.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_logistic_decay_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.05;0.1;0.15;0.2;0.3;0.5;1.0' )

#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_logistic_decay_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.05;0.1;0.2;0.5;1.0' )

#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_exp_decay_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.05;0.1;0.2;0.5;1.0' )

#METHOD_ARR=( 'fe_topp' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_exp_decay_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a6_e3' 'OWT_wiki_1e7_1b_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'exp_1_win'  )
#P_ARR=( '1.0' '1.0' '1.0' )
#DT_ARR=( '0.2;0.5' '1.2;1.5' '0.5;4.0' )

#METHOD_ARR=( 'fe_topp' 'fe_topp' 'fe_topp' 'fe_topp' 'fe_topp')
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a4_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a3_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a2_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a1_e3' 'OWT_wiki_1e7_1b_bsz_32_exp_pred_last_a10_e3')
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'exp_1_win' 'exp_1_win' 'exp_1_win' )
#P_ARR=( '1.0' '1.0' '1.0' '1.0' '1.0' )
#DT_ARR=( '1.2;1.5;4.0' '1.2;1.5;4.0' '1.2;1.5;4.0' '1.2;1.5;4.0' '1.0;1.5;3.0;8.0' )

#METHOD_ARR=( 'AP_topp' 'AP_topp' 'CD_topp' )
#MODEL_ARR=( 'prob_opt_wiki_ext_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_75_lr-4' 'prob_opt_wiki_ext_1e6_70M_bsz_32_e3_only_top_last_w_10_l1_reg_w_10_lr-4' 'opt-125m' )
#SUBMETHOD_ARR=( 'a' 'a' 'a' )
#P_ARR=( '0.6' '0.6' '0.6' )
#DT_ARR=( '1.0' '1.0' '1.0;0.5' )

#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_20_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.6')
#DT_ARR=( '0.5' )

#METHOD_ARR=( 'AP_topp' )
#MODEL_ARR=( 'prob_wiki_ext_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_10_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.4;0.6' )
#DT_ARR=( '1.0' )


#METHOD_ARR=( 'CD_topp' )
#MODEL_ARR=( 'pythia-70m-deduped' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.8' )
#DT_ARR=( '0.5' )

#METHOD_ARR=( 'CD_topp' 'AP_topp' 'AP_topp' )
#MODEL_ARR=( 'pythia-70m-deduped' 'prob_wiki_ext_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_5_lr-4' 'prob_wiki_ext_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_10_lr-4' )
#SUBMETHOD_ARR=( 'a' 'a' 'a' )
#P_ARR=( '0.6' '0.6' '0.4' )
#DT_ARR=( '1.0' '1.0' '1.0' )


#METHOD_ARR=( 'AP_topa' 'AP_topa' )
#MODEL_ARR=( 'prob_wiki_ext_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_20_lr-4' 'prob_wiki_ext_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_10_lr-4' )
#SUBMETHOD_ARR=( 'a' 'a' )
#P_ARR=( '0.1' '0.6;0.3')
#DT_ARR=( '1.0' '1.0' )

#METHOD_ARR=( 'CD_topk' 'AP_topk' )
#MODEL_ARR=( 'pythia-70m-deduped' 'prob_wiki_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_20_lr-4' )
#SUBMETHOD_ARR=( 'a' 'a' )
#P_ARR=( '10.0;5.0' '5.0' )
#DT_ARR=( '1.0' '1.0' )

#METHOD_ARR=( 'AP_topk' )
#MODEL_ARR=( 'prob_wiki_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_20_lr-4' )
#MODEL_ARR=( 'prob_wiki_1e6_70M_bsz_64_e3_only_top_last_w_10_l1_reg_w_5' )
#MODEL_ARR=( 'prob_OWT_1e6_70M_bsz_128_a10_e3_lr-4' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '10.0' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'CS' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '5.0' )
#DT_ARR=( '0.6' )
#P_ARR=( '5.0;10.0' )
#DT_ARR=( '0.4;0.6' )

#METHOD_ARR=( 'topp' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.5' )

#METHOD_ARR=( 'fe_topk' 'fe_topp_period' 'fe_CD_topp' 'CD' 'fe_topp' 'topp' 'eta' 'typical' 'decay_period' 'topk' 'fe_topp' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'pythia-70m-deduped' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' '' '' '' '' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'exp_1_win' 'a' 'exp_1_win' 'a' 'a' 'a' 'a' 'a' 'exp_real_e_only_win' 'exp_e_only_win' 'exp_1_win' )
#P_ARR=( '' '0.9' '1.0' '0.25' '1.0' '1.0;0.8;0.7;0.6;0.5;0.4;0.3;0.0' '' '0.1' '0.9' '' '1.0' '1.0' '1.0' )
#DT_ARR=( '1.0' '5.0' '' '1.0' '' '1.0' '1' '1' '' '1' '' '3.0;6.0' '' )


#METHOD_ARR=( 'fe_topk' 'fe_topp_period' 'fe_CD_topp' 'CD' 'fe_topp' 'topp' 'eta' 'typical' 'decay_period' 'topk' 'fe_topp' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'pythia-70m-deduped' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' '' '' '' '' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'exp_1_win' 'a' 'exp_1_win' 'a' 'a' 'a' 'a' 'a' 'exp_real_e_only_win' 'exp_e_only_win' 'exp_1_win' )
#P_ARR=( '3.0;5.0;10.0;20.0;40.0' '0.9' '1.0' '0.8;0.6;0.4;0.3;0.2;0.1;0.05' '1.0' '1.0;0.8;0.7;0.6;0.5;0.4;0.3;0.0' '0.1;0.3;0.8' '0.9;0.5;0.3' '0.9' '10.0;5.0;3.0;2.0;1.0' '1.0' '1.0' '1.0' )
#DT_ARR=( '1.0' '2.0;3.0;4.0;6.0;8.0' '0.7;1.0;1.5;2.0;4.0;8.0' '1.0' '0.5;0.7;1.0;1.2;1.5;2.0;3.0;4.0;8.0' '1.0' '1' '1' '0.95;0.9;0.7;0.5;0.3;0.1' '1' '2.0;3.0;4.0;6.0;8.0' '2.0;4.0;8.0' '0.5;1.0;1.5;2.0;3.0;4.0;6.0' )

#METHOD_ARR=( 'fe_topk' 'fe_topk' 'fe_topp_period' 'typical' 'fe_topp')
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'exp_1_win' 'a' 'exp_1_win' )
#P_ARR=( '10.0' '3.0;5.0;10.0;20.0;40.0' '0.9;0.95' '0.9' '1.0' )
#DT_ARR=( '0.5;0.7;1.0;2.0' '1.0' '2.0;3.0;4.0' '1' '0.5;1.5' )

#METHOD_ARR=( 'fe_topp' 'topp' 'eta' 'typical' 'decay_period' 'topk' 'fe_topp' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' '' '' '' '' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' 'a' 'a' 'a' 'a' 'exp_real_e_only_win' 'exp_e_only_win' 'exp_1_win' )
#P_ARR=( '1.0' '0.8;0.7;0.6;0.5;0.4;0.3' '0.1;0.3;0.8' '2.0;0.9;0.5;0.3' '0.9' '10.0;5.0;3.0;2.0;1.0' '1.0' '1.0' '1.0' )
#DT_ARR=( '0.7;1.0;1.2;1.5;2.0;3.0;4.0' '1' '1' '1' '0.95;0.9;0.7;0.5;0.3;0.1' '1' '2.0;4.0;8.0' '2.0;4.0;8.0' '0.5;1.0;1.5;2.0;4.0' )

#METHOD_ARR=( 'CD' 'topp' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( 'pythia-70m-deduped' '' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'a' 'a' 'exp_real_e_only_win' 'exp_e_only_win' )
#P_ARR=( '0.6;0.4;0.2;0.1;0.05' '1.0' '1.0' '1.0' )
#DT_ARR=( '1.0' '1.0' '3.0;6.0' '3.0;6.0' )

#METHOD_ARR=( 'fe_CD' 'CD' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'pythia-70m-deduped')
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '0.4;0.6' '0.6;0.4' )
#DT_ARR=( '1.0' '1.0' )

#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3')
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0;4.0;8.0' )

#METHOD_ARR=( 'fe_topp' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.0;0.3;0.5;0.6;0.7;1.0' )
#DT_ARR=( '0.5;0.7;1.0;2.0;3.0;4.0' '1.0' )

#METHOD_ARR=( 'fe_topp' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.8' )
#DT_ARR=( '8.0' '1.0' )

#METHOD_ARR=( 'fe_CD_topp' 'CD' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'opt-125m' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.05;0.8' )
#DT_ARR=( '0.7;8.0' '1.0' )

#METHOD_ARR=( 'fe_CD_topp' 'CD' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'opt-125m' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.6;0.3;0.1' )
#DT_ARR=( '1.0;2.0;4.0' '1.0' )

#METHOD_ARR=( 'fe_CD_topp' 'fe_topp_period' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '')
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.9' '1.0' )
#DT_ARR=( '4.0;0.7;1.5' '5.0' '0.1;0.9' )

#METHOD_ARR=( 'topp' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.3;0.7' )

#METHOD_ARR=( 'topp' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '1.0' )
#DT_ARR=( '0.5' )

#METHOD_ARR=( 'CD' )
#MODEL_ARR=( 'pythia-70m-deduped')
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.3' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'topp' 'CD' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( '' 'pythia-70m-deduped' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3')
#SUBMETHOD_ARR=( 'a' 'a' 'exp_1_win' 'exp_e_only_win' )
#P_ARR=( '1.0' '0.25' '1.0' '1.0' )
#DT_ARR=( '0.5' '1.0' '8.0' '6.0' )

#METHOD_ARR=( 'fe_topp' 'topp' 'eta' 'typical' 'decay_period' 'topk'  )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' '' '' '' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' 'a' 'a' 'a' 'a' )
#P_ARR=( '1.0' '0.8' '0.8' '2.0' '0.9' '5.0'  )
#DT_ARR=( '1.2;3.0' '1' '1' '1' '0.95;0.9;0.7;0.5' '1' )

#METHOD_ARR=( 'fe_topk' 'fe_topp_period' 'typical' 'decay_period' 'topk' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' '' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'a' 'a' 'a' )
#P_ARR=( '10.0' '0.95' '0.1' '0.9' '1.0;3.0' )
#DT_ARR=( '4.0' '4.0' '1' '0.3;0.1' '1' )

#METHOD_ARR=( 'typical' 'fe_topp' )
#MODEL_ARR=( '' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'a' 'exp_1_win' )
#P_ARR=( '0.9' '1.0' )
#DT_ARR=( '1' '0.5;1.5' )

#METHOD_ARR=( 'fe_topp_period' 'typical' 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' 'exp_1_win' )
#P_ARR=( '0.9' '0.9' '1' )
#DT_ARR=( '2.0;3.0;4.0;6.0;8.0' '1' '0.5;1.5' )

#SEED_NUM=3
SEED_NUM=4

#TEMPERATURE=1.0

#NUM_EVAL_SENT=3
NUM_EVAL_SENT=1

#sample_method='topp'

#top_p='1.0'

#SEED_ARR=( 2 )
CUDE_IDX_F=( 0 1 2 3 )
CUDE_IDX_NF=( 4 5 6 7 )

for j in "${!METHOD_ARR[@]}"; do
        MODEL=${MODEL_ARR[$j]}
        sample_method=${METHOD_ARR[$j]}
	if [[ $sample_method == "DoLa" ]]; then
		SEED_ARR=( 0 1 2 3 )
		NONSEED_ARR=( 4 5 6 7 )
	else
		SEED_ARR=( 1 2 3 4 )
		NONSEED_ARR=( 1 2 3 4 )
	fi
        sample_sub_method=${SUBMETHOD_ARR[$j]}
        top_p_all=${P_ARR[$j]}
        decay_temperature_all=${DT_ARR[$j]}

        IFS=";" read -r -a top_p_list <<< "${top_p_all}"
        IFS=";" read -r -a decay_temperature_list <<< "${decay_temperature_all}"
        for top_p in "${top_p_list[@]}"; do
                for decay_temperature in "${decay_temperature_list[@]}"; do
			if [[ $sample_method == "fe_topp" ]]; then
				#GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_p${top_p}_fixed_${MODEL}"
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_p${top_p}_${MODEL}"
			elif [[ $sample_method == "fe_topp_period" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_our_period_r${top_p}_${MODEL}"
			elif [[ $sample_method == "fe_CD" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_alpha${top_p}_fixed_${MODEL}"
			elif [[ $sample_method == "fe_CD_topp" ]]; then
				#GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_p${top_p}_fixed_${MODEL}"
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_p${top_p}_${MODEL}"
			elif [[ $sample_method == "fe_AP_topp" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_ait${top_p}_fixed_${MODEL}"
			elif [[ $sample_method == "fe_topk" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_k${top_p}_fixed_${MODEL}"
			elif [[ $sample_method == "eta" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
			elif [[ $sample_method == "typical" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
			elif [[ $sample_method == "topk" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_k${top_p}"
			elif [[ $sample_method == "CS" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_k${top_p}_alpha_${decay_temperature}"
			elif [[ $sample_method == "AP_topk" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_k${top_p}_${MODEL}"
			elif [[ $sample_method == "topp" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_temp_${decay_temperature}"
			elif [[ $sample_method == "decay_period" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_dt_${decay_temperature}"
			elif [[ $sample_method == "CD_topk" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_k${top_p}_${MODEL}"
			elif [[ $sample_method == "CD" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_p${top_p}_${MODEL}"
			elif [[ $sample_method == "CD_topp" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_p${top_p}_${MODEL}"
			elif [[ $sample_method == "AP_topa" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_alpha${top_p}_${MODEL}"
			elif [[ $sample_method == "AP_topp" ]]; then
				#GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_${MODEL}"
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_dt_${decay_temperature}_${MODEL}"
			elif [[ $sample_method == "DoLa" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_alpha${top_p}_${MODEL}"
			elif [[ $sample_method == "adaptive" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_a${top_p}"
			elif [[ $sample_method == "EDT" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_temp${decay_temperature}_theta${top_p}"
			elif [[ $sample_method == "microstat" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_t${decay_temperature}"
			elif [[ $sample_method == "EAD_no_ELI" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_alpha${decay_temperature}"
			elif [[ $sample_method == "fe_EAD_no_ELI" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_alpha${top_p}_dt_${decay_temperature}_${MODEL}"
			fi
			FILE_HYPER="${PROMPT_TYPE}_${GEN_MODEL}_${sample_method}_p${top_p}"
			if [[ $sample_method == "DoLa" ]]; then
				FILE_HYPER="${PROMPT_TYPE}_${GEN_MODEL}_${sample_method}_p1_dolalayers${sample_sub_method}"
			fi
			GEN_CONFIG="${PROMPT_TYPE}_${GEN_HYPER}"
			echo "python src/distinct_n_each_gen.py --gen_dir ${FOLDER_PATH}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT"
			python src/distinct_n_each_gen.py --gen_dir ${FOLDER_PATH}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT
			pids=()
			#echo $GEN_CONFIG
			#for SEED_IDX in "${SEED_ARR[@]}";
			#do
			for i in "${!SEED_ARR[@]}"; do
				#echo $SEED_IDX
				SEED_IDX=${SEED_ARR[$i]}
				echo "PYTHONPATH=. python src/repetition.py ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --final --output --num_eval_sent $NUM_EVAL_SENT"
				PYTHONPATH=. python src/repetition.py ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --final --output --num_eval_sent $NUM_EVAL_SENT
			done
			for i in "${!NONSEED_ARR[@]}"; do
				SEED_IDX=${NONSEED_ARR[$i]}
				PYTHONPATH=. python src/repetition.py ${GEN_CONFIG}/non${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --final --output --num_eval_sent $NUM_EVAL_SENT
			done
			for i in "${!SEED_ARR[@]}"; do
				SEED_IDX=${SEED_ARR[$i]}
				if [[ $sample_method == "DoLa" ]]; then
					SEED_IDX=${CUDE_IDX_F[$i]}
				fi
				echo "PYTHONPATH=. python src/evaluate_v3_final_all.py --prompt_type $PROMPT_TYPE --gen_path ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --num_eval_sent $NUM_EVAL_SENT &> ./temp &"
				export CUDA_VISIBLE_DEVICES=${CUDE_IDX_F[$i]}
				PYTHONPATH=. python src/evaluate_v3_final_all.py --prompt_type $PROMPT_TYPE --gen_path ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --num_eval_sent $NUM_EVAL_SENT &> ./temp &
				pids+=($!)
				if [[ $sample_method == "DoLa" ]]; then
					SEED_IDX=${CUDE_IDX_NF[$i]}
				fi
				export CUDA_VISIBLE_DEVICES=${CUDE_IDX_NF[$i]}
				PYTHONPATH=. python src/evaluate_v3_final_all.py --prompt_type non$PROMPT_TYPE --gen_path ${GEN_CONFIG}/non${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --num_eval_sent $NUM_EVAL_SENT &> ./temp &
				pids+=($!)
			done
			echo "${pids[@]}"
                        wait "${pids[@]}"
		done
	done
done

