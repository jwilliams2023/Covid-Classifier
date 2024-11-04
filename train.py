import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.CovidClassifier import CovidClassifier
from data.CovidDataset import CovidDataset
from torch.utils.data import DataLoader

# Hyperparameters ----------------------------------------------------------------
EPOCHS = 20
LEARNING_RATE = .0003
BATCH_SIZE = 256

# Static Parameters ---------------------------------------------------------------
IN_DIM = 13
OUT_DIM = 1
SEED = 42

# Torch Setup ---------------------------------------------------------------------
torch.manual_seed(SEED)

model_save_path = f'./saved_models/model/best_gen.pth'
dict_save_path = f'./saved_models/state_dict/best_gen.pth'

# CUDA ----------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)
print(f'Using Device: {device}\n')

# Data Loading --------------------------------------------------------------------
dataset = CovidDataset('./data/covid_data_cleaned.csv')
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True)

# Model Setup --------------------------------------------------------------------
model = CovidClassifier(IN_DIM, OUT_DIM).to(device)
optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        batch_loss = np.array([])
        for batch_idx, (data, target) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            batch_loss = np.append(batch_loss, loss.item())

        print(f'\nAVG Loss {np.round(batch_loss.mean().item(), 5)}\n')
