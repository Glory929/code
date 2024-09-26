import os
from sklearn.model_selection import train_test_split

data_path = r"/home/ww/wlr/datasets/BraTSout"
train_and_val_ids = os.listdir(data_path)

train_ids, val_ids = train_test_split(train_and_val_ids, test_size=0.2,random_state=42)
val_ids, test_ids = train_test_split(val_ids, test_size=0.5,random_state=21)
print("Using {} images for training, {} images for validation, {} images for test.".format(len(train_ids),len(val_ids),len(test_ids)))

with open(r'/home/ww/wlr/datasets/BraTSout_txt/train.txt', 'w') as f:
    f.write('\n'.join(train_ids))

with open(r'/home/ww/wlr/datasets/BraTSout_txt/valid.txt', 'w') as f:
    f.write('\n'.join(val_ids))

with open(r'/home/ww/wlr/datasets/BraTSout_txt/test.txt', 'w') as f:
    f.write('\n'.join(test_ids))