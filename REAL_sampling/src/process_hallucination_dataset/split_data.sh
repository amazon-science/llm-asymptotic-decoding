input_folder="/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/"

#input_file="${input_folder}summarization_data_2048.json"
#mid_file="${input_folder}summarization_data_2048_rnd.json"
#output_prefix="${input_folder}summarization_data_2048_"
input_file="${input_folder}summarization_data_1024.json"
mid_file="${input_folder}summarization_data_1024_rnd.json"
output_prefix="${input_folder}summarization_data_1024_"

#input_file="${input_folder}qa_data.json"
#mid_file="${input_folder}qa_data_rnd.json"
#output_prefix="${input_folder}qa_data_"
#input_file="${input_folder}qa_data_knowledge.json"
#mid_file="${input_folder}qa_data_knowledge_rnd.json"
#output_prefix="${input_folder}qa_data_knowledge_"

#input_file="${input_folder}dialogue_data.json"
#mid_file="${input_folder}dialogue_data_rnd.json"
#output_prefix="${input_folder}dialogue_data_"
#input_file="${input_folder}dialogue_data_knowledge.json"
#mid_file="${input_folder}dialogue_data_knowledge_rnd.json"
#output_prefix="${input_folder}dialogue_data_knowledge_"

sort -R $input_file > $mid_file

num_files=10
#num_files=5
total_lines=$(wc -l <${mid_file})
((lines_per_file = (total_lines + num_files - 1) / num_files))
split -l ${lines_per_file} ${mid_file}

cat xaa xab xac xad xae xaf xag xah > ${output_prefix}train.json
mv xai ${output_prefix}val.json
mv xaj ${output_prefix}test.json
rm xa*

#cat xac xad xae > ${output_prefix}train.json
#mv xaa ${output_prefix}val.json
#mv xab ${output_prefix}test.json
#rm xa*
