import pandas as pd
import os
import math

save_as = 'packetCounts'

#Folder with predictions:
folderpath_pred = 'C:\Githubs\kandidat\High_freq_files\predictions'

main_dic = {}
for filename in os.listdir(folderpath_pred):
    filepath_pred = os.path.join(folderpath_pred, filename)
    # checking if it is a file
    if os.path.isfile(filepath_pred):
        df = pd.read_csv(filepath_pred)

        for file in df.columns:
            count = sum(1 for x in df[file] if not math.isnan(x))
            print(count)
            data_dict = {
            "packetCount": count
            }
            main_dic[file] = data_dict
            print(str(file) + ' complete')
    else:
        print("Couldn't find file at {filepath_pred}.")

new_df = pd.DataFrame.from_dict(main_dic, orient='index')
new_df = new_df.transpose()
new_df.to_pickle(save_as)
  
       
       