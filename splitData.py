import pandas as pd

labels = pd.read_csv('csvfiles\\ISIC_2019_Training_GroundTruth.csv')
labels = labels.sample(frac=1).reset_index(drop=True)
train_set = labels.iloc[3334:, :]
eval_set = labels.iloc[:3334, :]

df_class = pd.DataFrame(data=eval_set.iloc[:, 0], columns=['image'])
df_class['class'] = eval_set.iloc[:, 1:].idxmax(axis=1)
df_class = df_class.reset_index()
df_class.to_csv('csvfiles\\validation_class.csv', index=False)
eval_set.to_csv('csvfiles\\validation.csv', index=False)
