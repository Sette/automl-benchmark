import pandas as pd
import os
PATH_DIR = "datasets/plant-seedlings/"
PATH_DIR_TRAIN = "datasets/plant-seedlings/train/"
PATH_DIR_FILES_TRAIN = PATH_DIR+"train_files/"
targets = os.listdir(PATH_DIR_TRAIN)

try:
	os.mkdir(PATH_DIR_FILES_TRAIN)
	print("Directory TRAIN_FILES created")
except:
	print("Diretório já criado")

all_dict = {"id":[],"target":[]}
for target in targets:
	all_dict["target"]+=[target for i in range(len(os.listdir(PATH_DIR_TRAIN+target)))]
	all_dict["id"]+=os.listdir(PATH_DIR_TRAIN+target)
	#[print('cp '+ PATH_DIR_TRAIN+target+target+ '/' + image + ' ' + PATH_DIR_FILES_TRAIN + image) for image in os.listdir(PATH_DIR_TRAIN+target)]
	#dict = {"target": [target for i in range(len(os.listdir(PATH_DIR_TRAIN+target)))],
	#	"id": os.listdir(PATH_DIR_TRAIN+target)
	#}
	#all_dict.update(dict)

print(all_dict)
df_out = pd.DataFrame(all_dict)
df_out.to_csv(PATH_DIR+"train.csv",index=False)
