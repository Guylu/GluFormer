import pandas as pd

import numpy as np
from torch.utils.data import Dataset, DataLoader
import wandb
import random
import torch

hyperparameter_defaults = dict(
    epochs=110,
    lr=3e-5,
    weight_decay=0,
    step_size=10,
    gamma=0.99,
    dropout=0.1,

    n_embd=1024,
    n_heads=16,
    n_layers=16,
    dim_feedforward=2048,
    max_seq_length=25000,

    chunk_size=1200,

    seed=42,
    batch_per_gpu=20,

    name="10K_data2",
)
print(hyperparameter_defaults)
wandb.init(config=hyperparameter_defaults, project="CGM_Foundation_embd_",
           allow_val_change=True)  # , mode="disabled")
config = wandb.config

# fixing seeds
seed = config.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

chunks = 460
chunk_size = config.chunk_size  # Fixed size for each chunk
PAD_TOKEN = chunks  # Assuming -1 is not a valid glucose measurement
name = wandb.config.name
path = f"cgm_diet_filtered_processed_aligned_tokenized_tensors_train.pt"
discretized_data = torch.load(path)['tokens']

class GlucoseDataset2(Dataset):
    def __init__(self, data, split=3):
        self.data = data
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = self.data.index[idx]
        glucose = self.data.loc[idx].values
        glucose = glucose[glucose != PAD_TOKEN]
        return torch.tensor(glucose, dtype=torch.long), idx


# Create a DataLoader for batching
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = GlucoseDataset2(discretized_data)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

vocab_size = chunks
n_embd = config.n_embd
dim_feedforward = config.dim_feedforward
# Assuming the model and dataloader are already defined
model = TransformerModel(vocab_size, n_embd, n_heads=config.n_heads, n_layers=config.n_layers,
                          max_seq_length=25000,
                          dropout=config.dropout, dim_feedforward=dim_feedforward).to(device)

# Model_peach-plasma-78.pt
model.load_state_dict(torch.load(
    "Model_best.pt",
    map_location=device))

model = model.eval()
model = model.to(device)
embds = []
ids = []
for i, (glucoses, idx) in enumerate(dataloader):
    print(f"sample {i}/{len(dataloader)}")
    idx = idx[0]
    glucoses = glucoses.squeeze().to(device)
    embd = model(glucoses.unsqueeze(0), mask_token_id=PAD_TOKEN, ret_embds=True)[1]
    embd = embd.squeeze().max(dim=0)[0]
    embds.append(embd.squeeze().detach().cpu().numpy())
    ids.append(idx)
embds = np.stack(embds)
embds_df = pd.DataFrame(embds.squeeze(), index=ids)
embds_df.to_csv(
    "embds/embds_df.csv")
print("saved")
