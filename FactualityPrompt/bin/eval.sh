#!/bin/bash

FOLDER_PATH="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factual_gen/"
PROMPT_TYPE="factual_1000"

#GEN_HYPER="70M_p0.7"
#SEED_NUM=2

GEN_MODEL="6.9b"
#GEN_MODEL="OpenLLaMA2-7b"
#GEN_MODEL="OPT-6.7b"

#GEN_HYPER="${GEN_MODEL}_p0.7"
#GEN_HYPER="${GEN_MODEL}_p0.3"
#GEN_HYPER="${GEN_MODEL}_decay_p0.9"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p1.0"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p1.0_wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3"
#GEN_HYPER="${GEN_MODEL}_fe_topp_exp_p1.0_wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3"
#GEN_HYPER="${GEN_MODEL}_fe_topp_exp_half_p1.0_wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p0.4_wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p1.0_wiki_1e6_410M_bsz_8_exp_pred_last"
#GEN_HYPER="${GEN_MODEL}_fe_topp_exp_p1.0_wiki_1e6_410M_bsz_8_exp_pred_last"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p1.0_OWT_wiki_1e7_70M_bsz_32_exp_pred_last_e3"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p1.0_OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a6_e3"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p1.0_OWT_wiki_1e7_70M_bsz_8_exp_pred_last_a4_e1"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p1.0_OWT_wiki_1e7_70M_bsz_32_exp_pred_last_d08_e3"
#GEN_HYPER="${GEN_MODEL}_fe_topp_p0.5"
#FILE_HYPER="${PROMPT_TYPE}_${GEN_MODEL}_fe_topp_p1.0"
#FILE_HYPER="${PROMPT_TYPE}_${GEN_MODEL}_fe_topp_p0.4"

SEED_NUM=3

NUM_EVAL_SENT=3
#NUM_EVAL_SENT=1

#sample_method='topk'
#sample_method='topp'
#sample_method='decay'
#sample_method='decay_period'
sample_method='fe_topp'
#sample_method='eta'
#sample_method='typical'
#sample_method='fecut_topp'

#top_p='0.001'
#top_p='0.004'
#top_p='0.02'
#top_p='0.3'
#top_p='10.0'
#top_p='2.0'
#top_p='5.0'
top_p='1.0'
#top_p='0.9'
#top_p='0.5'
#top_p='0.6'
#top_p='0.3'
#top_p='0.7'

#MODEL_ARR=( 'wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3' 'wiki_1e6_410M_bsz_8_exp_pred_last' 'wiki_1e6_70M_bsz_8_exp_pred_last' 'OWT_wiki_1e7_70M_bsz_8_exp_pred_last_a4_e1' 'OWT_wiki_1e7_70M_bsz_32_exp_pred_last_e3' 'OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a4_e3' 'OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a6_e3' 'OWT_wiki_1e7_70M_bsz_32_exp_pred_last_d08_e3' 'OWT_1e6_70M_bsz_32_exp_pred_last_d08_e3' 'OWT_1e6_410m_bsz_32_exp_pred_last' 'OWT_1024_1e6_410M_bsz_8_exp_pred_last' )
#MODEL_ARR=( 'wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3' 'wiki_1e6_410M_bsz_8_exp_pred_last' 'wiki_1e6_70M_bsz_8_exp_pred_last' 'OWT_wiki_1e7_70M_bsz_8_exp_pred_last_a4_e1' 'OWT_1e6_70M_bsz_32_exp_pred_last_d08_e3'  'OWT_1024_1e6_410M_bsz_8_exp_pred_last' )
#MODEL_ARR=( 'per_OWT_1e6_70M_bsz_128_exp_pred_last_a10_e3' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
MODEL_ARR=( 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a6_e3' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_32_exp_pred_last_d08_e3' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a4_e3' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_32_exp_pred_last_e3' )
#MODEL_ARR=( 'wiki_1e6_410M_bsz_8_exp_pred_last' )
#MODEL_ARR=( 'wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3' )
#MODEL_ARR=( 'OWT_1e6_70M_bsz_32_exp_pred_last_d08_e3' )
#MODEL_ARR=( 'OWT_1e6_410m_bsz_32_exp_pred_last' )
#MODEL_ARR=( 'OWT_1e6_1024_70M_bsz_64_exp_pred_last' 'OWT_1e6_1024_70M_bsz_64_exp_pred_last_e10' 'wiki_1e6_70M_bsz_32_exp_pred_last_a10_e3' 'OWT_wiki_1e7_410M_bsz_16_exp_pred_last_a6_e1')
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_32_exp_e3' 'OWT_1e6_1024_70M_bsz_64_exp_pred_last' 'OWT_1e6_1024_70M_bsz_64_exp_pred_last_e10' 'wiki_1e6_70M_bsz_32_exp_pred_last_a10_e3' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_32_exp_e3' )
#MODEL_ARR=( 'wiki_1e6_70M_bsz_64_exp_pred_last_a10_e10' )
#MODEL_ARR=( 'OWT_wiki_1e7_410M_bsz_16_exp_pred_last_a6_e1' )
#MODEL_ARR=( 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a6_e1' )
#MODEL_ARR=( 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_e1' )
#MODEL_ARR=( 'dummy' )

#SEED_IDX=1
SEED_ARR=( 1 2 3 )
#SEED_ARR=( 1 )
#SEED_ARR=( 2 3 )
#SEED_ARR=( 3 )
#SEED_ARR=( )


for MODEL in "${MODEL_ARR[@]}";
do
	if [[ $sample_method == "fe_topp" ]]; then
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_half_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_dt_2.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_2_dt_1.5_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_dt_0.5_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_dt_1.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_win_20_dt_1.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_win_40_dt_4.0_p${top_p}_${MODEL}"
		GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_win_40_dt_2.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_win_40_dt_1.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_raw_e_only_win_40_dt_4.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_real_e_only_win_40_dt_4.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_e_only_win_40_dt_8.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_e_only_win_40_dt_4.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_e_only_win_40_dt_2.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_large_win_40_dt_2.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_large_win_40_dt_1.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_large_win_40_dt_0.5_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_large_win_40_dt_0.25_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_win_40_dt_0.5_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_dt_2.0_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_dt_1.5_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_norm_dt_0.5_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_norm_dt_0.25_p${top_p}_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_norm_dt_0.35_p${top_p}_${MODEL}"
	elif [[ $sample_method == "fecut_topp" ]]; then
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_0.0_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_-0.5_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_-0.3_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_-0.7_${MODEL}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_-0.8_${MODEL}"
		GEN_HYPER="${GEN_MODEL}_${sample_method}_exp_1_dt_-0.7_${MODEL}"
	elif [[ $sample_method == "eta" ]]; then
		GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
	elif [[ $sample_method == "typical" ]]; then
		GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
	elif [[ $sample_method == "topk" ]]; then
		GEN_HYPER="${GEN_MODEL}_${sample_method}_k${top_p}"
	elif [[ $sample_method == "topp" ]]; then
		GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_temp_0.5"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
		#GEN_HYPER="${GEN_MODEL}_p${top_p}"
	elif [[ $sample_method == "decay" ]]; then
		GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
	elif [[ $sample_method == "decay_period" ]]; then
		GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_dt_0.5"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
		#GEN_HYPER="${GEN_MODEL}_${sample_method}"
	fi
	FILE_HYPER="${PROMPT_TYPE}_${GEN_MODEL}_${sample_method}_p${top_p}"
	#FILE_HYPER="${PROMPT_TYPE}_${GEN_MODEL}_p${top_p}"
	GEN_CONFIG="${PROMPT_TYPE}_${GEN_HYPER}"
	#python src/distinct_n.py --gen_dir ${FOLDER_PATH}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT
	#echo "python src/distinct_n_each_gen.py --gen_dir ${FOLDER_PATH}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT"
	python src/distinct_n_each_gen.py --gen_dir ${FOLDER_PATH}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT
	for SEED_IDX in "${SEED_ARR[@]}";
	do
		echo $SEED_IDX
		#echo "PYTHONPATH=. python src/evaluate_v3_final.py --prompt_type $PROMPT_TYPE --gen_path ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl"
		PYTHONPATH=. python src/evaluate_v3_final_all.py --prompt_type $PROMPT_TYPE --gen_path ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --num_eval_sent $NUM_EVAL_SENT
		PYTHONPATH=. python src/evaluate_v3_final_all.py --prompt_type non$PROMPT_TYPE --gen_path ${GEN_CONFIG}/non${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --num_eval_sent $NUM_EVAL_SENT
		PYTHONPATH=. python src/repetition.py ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --final --output --num_eval_sent $NUM_EVAL_SENT
		PYTHONPATH=. python src/repetition.py ${GEN_CONFIG}/non${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --final --output --num_eval_sent $NUM_EVAL_SENT
	done
done
