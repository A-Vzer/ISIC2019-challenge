import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import pickle
from efficientnet_pytorch import EfficientNet
from isicDataset import ISICDataset
from EfficientEnsemble import EfficientEnsemble
import os
torch.manual_seed(0)

# csv files
test_meta = 'csvfiles\\ISIC_2019_Test_Metadata.csv'
train_path = 'csvfiles\\train.csv'
eval_path = 'csvfiles\\validation.csv'
class_path = 'csvfiles\\validation_class.csv'


# Put location to train and test images
train_img_path = 'A:\\Datasets\\ISIC\\ISIC_2019\\ISIC_2019\\ISIC_2019_Training_Input\\ISIC_2019_Training_Input'
test_img_path = 'A:\\Datasets\\ISIC\\ISIC_2019\\ISIC_2019\\ISIC_2019_Test_Input\\ISIC_2019_Test_Input'

classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
images = pd.read_csv(test_meta)
df_class = pd.read_csv(class_path)

# Pipeline with image augmentations
data_transforms = {
    'train': transforms.Compose([transforms.Resize([299, 299]),
                                transforms.RandomRotation(degrees=180),
                                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                                transforms.ToTensor(),
                                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0,
                                                         inplace=False),
                                transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))]),
    'val': transforms.Compose([transforms.Resize([299, 299]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))]),
}

# Create dataset and dataloader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_set_train = ISICDataset(
    train_img_path, transform=data_transforms['train'],
    csv_path=train_path)

data_set_val = ISICDataset(
    img_path=train_img_path, transform=data_transforms['val'],
    csv_path=eval_path)

dataloader_val = DataLoader(
    data_set_val, batch_size=64, shuffle=False, pin_memory=True)

data_set_test = ISICDataset(
    test_img_path, transform=data_transforms['val'], csv_path=test_meta,
    test=True)

dataloader_test = DataLoader(
    data_set_test, batch_size=16, shuffle=False, pin_memory=True)

losses = []
softmax = nn.Softmax(dim=-1)


# Train ensemble model or individual EfficientNet
def train_model(model, batchSize, lr, eps, dec, mom, version, ensemble=False, pretr=True):
    if not ensemble:
        modelFileName = f'efficientnet/efficientnetb{version}_lr_{lr}_bs_{batchSize}_dec_{dec}_mom_{mom}_ep_{eps}' \
                        f'_pretr_{pretr}.pth'
        lossesFileName = f'efficientnet/efficientnetb{version}_losses_lr_{lr}_bs_{batchSize}_dec_{dec}_mom_{mom}_ep_' \
                         f'{eps}_pretr_{pretr}'
    else:
        modelFileName = f'efficientnet/Ensemble_lr_{lr}_bs_{batchSize}_dec_{dec}_mom_{mom}_ep_{eps}.pth'
        lossesFileName = f'efficientnet/Ensemble_losses_lr_{lr}_bs_{batchSize}_dec_{dec}_mom_{mom}_ep_{eps}_pretr_{pretr}'

    print(device)
    print(f'Batch size: {batchSize}, Learning rate: {lr}, Decay: {dec}, Momentum {mom}, Epochs: {eps}')

    torch.cuda.empty_cache()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=dec, momentum=mom)
    dataloader = DataLoader(data_set_train, batch_size=batchSize, shuffle=True, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(eps):
        model.train()
        for batch, sample in enumerate(dataloader):
            running_loss = 0.0
            optimizer.zero_grad()
            output = softmax(model(sample['image'].to(device)))
            loss = criterion(output, torch.max(sample['label'], 2)[1].squeeze(-1).to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch % 100 == 0:
                print(f'{epoch + 1}, {batch}, {running_loss / batchSize}')
            losses.append(running_loss)
            del running_loss
            break

    with open(lossesFileName, 'wb+') as myfile:
        pickle.dump(losses, myfile)

    validate(model)
    torch.save(model.state_dict(), modelFileName)
    time.sleep(3)
    return model, modelFileName


# Run through validation set
def validate(model):
    df = pd.DataFrame(columns=classes)
    model.eval()
    with torch.no_grad():
        print('validating...')
        for idx, sample in enumerate(dataloader_val):
            outputs = model(sample['image'].to(device))
            outputs = softmax(outputs)
            outputs = outputs.cpu().numpy()
            df = df.append(pd.DataFrame(data=outputs, columns=classes))

        df['truth'] = df.idxmax(axis=1)
        df = df.reset_index()
        df['accuracy'] = (df['truth'] == df_class['class'])
        accuracy = df['accuracy'].values.sum() / len(df['accuracy'])
        print(accuracy)


# Combine two EfficientNet models
def create_ensemble(model1, model2):
    for param in model1.parameters():
        param.requires_grad_(False)

    for param  in model2.parameters():
    param.requires_grad_(False)

    return EfficientEnsemble(model1, model2)


# Test model
def test_model(model, modelPath):
    if not os.path.exists("results"):
        os.makedirs("results")
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    model = model.to(device)
    df = pd.DataFrame(columns=classes)
    with torch.no_grad():
        for idx, sample in enumerate(dataloader_test):
            outputs = model(sample['image'].to(device))
            outputs = softmax(outputs)
            outputs = outputs.cpu().numpy()
            df = df.append(
                pd.DataFrame(data=outputs, columns=classes))
        df = df.reset_index()
        del df['index']
        df.insert(0, 'image', images['image'])
    df.to_csv(f'results/test_results.csv', index=False)


if __name__ == "__main__":
    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=9).cuda()
    model, fileName = train_model(model, 16, 0.01, 1, 0, 0, '3')
    validate(model)
    test_model(model, fileName)

