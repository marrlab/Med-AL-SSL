import pandas as pd
import numpy as np
import matplotlib.style as style
from PIL import Image
from sklearn.model_selection import train_test_split
import os

style.use('fivethirtyeight')
# mpl.font_manager._rebuild()

# Use the newly integrated Roboto font family for all text.
# plt.rc('font', family='Open Sans')

root = '/home/qasima/datasets/thesis/stratified/isic/'
curr_root = '/home/qasima/datasets/thesis/stratified/isic_data/'

df = pd.read_csv(os.path.join(root, 'annotations.csv'))
cols = df.columns.tolist()[1:-1]
num_classes = []
targets = np.zeros(df.shape[0], dtype=np.int)

for i, col in enumerate(cols):
    num_classes.append(df[df[col] == 1].shape[0])
    targets[df[col] == 1] = i
    # os.mkdir(os.path.join(root, 'train', col))
    # os.mkdir(os.path.join(root, 'test', col))

train_indices, test_indices = train_test_split(np.arange(df.shape[0]), test_size=0.2, shuffle=True, stratify=targets)

for i, idx in enumerate(train_indices):
    img = df['image'].iloc[idx]
    img_data = Image.open(os.path.join(curr_root, img + '.jpg'))
    width, height = img_data.size
    img_data_resized = img_data.resize((144, 144), Image.LANCZOS)
    img_data_resized.save(os.path.join(root, 'train', cols[targets[idx]], img + '.jpg'))
    print(f'Train {img} Image: {i}')

print('Done for train')

for i, idx in enumerate(test_indices):
    img = df['image'].iloc[idx]
    img_data = Image.open(os.path.join(curr_root, img + '.jpg'))
    img_data_resized = img_data.resize((144, 144), Image.LANCZOS)
    img_data_resized.save(os.path.join(root, 'test', cols[targets[idx]], img + '.jpg'))
    print(f'Test {img} Image: {i}')
