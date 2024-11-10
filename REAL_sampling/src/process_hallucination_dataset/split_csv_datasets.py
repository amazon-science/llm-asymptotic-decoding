import pandas as pd
import os

input_folder = "/mnt/efs/Haw-Shiuan/factor/data/"
output_folder = "/mnt/efs/Haw-Shiuan/true_entropy/outputs/factor/"

#input_folder = "/mnt/efs/Haw-Shiuan/Probes/datasets/"
#output_folder = "/mnt/efs/Haw-Shiuan/true_entropy/outputs/state/"

training_ratio = 0.5
val_ratio = 0.5
#test_ratio = 0.1

assert training_ratio + val_ratio == 1
#assert training_ratio + val_ratio + test_ratio == 1

for input_file in os.listdir(input_folder):
    #print(input_file)
    input_path = input_folder + input_file
    if not os.path.isfile(input_path):
        continue
    
    #input_name = os.path.basename(input_file)
    output_path = output_folder + input_file.replace('.csv', '_{}.csv')

    df = pd.read_csv(input_path)

    training_size = int( len(df) * training_ratio )
    val_size = int( len(df) * val_ratio )

    df_part = df.sample(n = training_size)
    df_part.to_csv(output_path.format('train'), index = False)

    df = df.drop(df_part.index)
    df_part = df.sample(n = val_size)
    df_part.to_csv(output_path.format('val'), index = False)

    #df_part = df.drop(df_part.index)
    #df_part.to_csv(output_path.format('test'), index = False)
