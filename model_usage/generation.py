import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import wandb
import random
import torch
import gc
import os

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
    new=4,
    x=0,
    # pre will be 10 days (each maesaurement is 15 mins) so PRE should be 10*24*4
    GEN=10,
    DAYS=1,
    k=10,  # Control the diversity
    CONTINUATIONS=3,  # Number of continuations per sample
    plus=1,
)
print(hyperparameter_defaults)
wandb.init(config=hyperparameter_defaults, project="CGM_Foundation_10k_gen",
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
path = f"cgm_diet_filtered_processed_aligned_tokenized_tensors_test.pt"
discretized_data = torch.load(path)['tokens']
times = torch.load(path)['time_expanded']

# make sure to remove duped indecies from both, and to align them by index
# remove duped indecies
discretized_data = discretized_data[~discretized_data.index.duplicated(keep='first')]
times = times[~times.index.duplicated(keep='first')]
# align by index
discretized_data = discretized_data.loc[times.index]


class GlucoseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].values, self.data.index[idx]


# Create a DataLoader for batching
batch_size_per_gpu = config.batch_per_gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = GlucoseDataset(discretized_data)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

vocab_size = chunks
n_embd = config.n_embd
dim_feedforward = config.dim_feedforward
# Assuming the model and dataloader are already defined
model = TransformerModel(vocab_size, n_embd, n_heads=config.n_heads, n_layers=config.n_layers, max_seq_length=25000,
                          dropout=config.dropout, dim_feedforward=dim_feedforward).to(device)
print(f"num of parameters: {sum(p.numel() for p in model.parameters())}")
wandb.log({"num of parameters": sum(p.numel() for p in model.parameters())})
# move to all GPUs in dataparralele
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
num_epochs = config.epochs
# clear gpu memory
gc.collect()
torch.cuda.empty_cache()
num_epochs = config.epochs

# Model_peach-plasma-78.pt
model.load_state_dict(torch.load(
    "Model_best.pt",
    map_location=device))


def top_k_logits(logits, k):
    """Filter logits to keep only the top k."""
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).expand_as(logits)
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


def sample_from_logits(logits, k):
    """Sample an index from top k logits."""
    filtered_logits = top_k_logits(logits, k=k)
    probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
    next_point = torch.multinomial(probabilities, 1)
    return next_point


DAYS = wandb.config.DAYS
PRE = DAYS * 24 * 4
GEN = wandb.config.GEN
k = wandb.config.k  # Control the diversity
NUM = wandb.config.NUM  # Number of samples to generate continuations for

# select from val_dataloader all samples that have more than 1000 measurements (i.e 1000 values that are not 128)
test_samples = []
ids = []
for i, (sample, id) in tqdm(enumerate(dataloader)):
    for s, id_ in zip(sample, id):
        test_samples.append(s.cpu().numpy())
        ids.append(id_)

test_samples = pd.DataFrame(test_samples, index=ids)
print(f'\t\t\t\t\tshape of samples: {test_samples.shape}')

reals = []
generateds = []
ids = []
times_ = []

x = wandb.config.x

NUM = (x, x + wandb.config.plus)
print(f'Generating continuations for samples {NUM[0]} to {NUM[1]}')
for B in range(*NUM):
    print(f'Sample {B + 1}')
    first_sample = test_samples.iloc[B]
    first_sample = torch.tensor(first_sample.values, dtype=torch.long).unsqueeze(0).to(device)
    real_sequence = first_sample[:, :PRE + GEN].squeeze().cpu().numpy()

    distances = []
    generated_sequences = []
    # Generate and plot 3 number of sequences
    for c in range(3):
        start_point = first_sample[:, :PRE].to(device)
        generated_sequence = start_point.clone()

        # Generate sequence
        for _ in tqdm(range(GEN)):
            with torch.no_grad():
                logits = model(generated_sequence)[:, -1, :]
                next_point = sample_from_logits(logits, k)
                generated_sequence = torch.cat([generated_sequence, next_point], dim=1)

        # Prepare for plotting
        generated_sequence = generated_sequence.squeeze().cpu().numpy()
        generated_sequences.append(generated_sequence)

    reals.append(real_sequence)
    # stack them
    generateds.append(np.stack(generated_sequences))
    ids.append(test_samples.index[B])
    times_.append(times.loc[test_samples.index[B]].values[:PRE + GEN])

# turn reals and generateds to dataframes with ids as index
reals = pd.DataFrame(reals, index=ids)
generateds = pd.DataFrame(np.vstack(generateds), index=np.repeat(ids, 3))
times_ = pd.DataFrame(times_, index=ids)

data = []

# Add real data
for i in ids:
    date_series = times_.loc[i].squeeze()  # Assuming times_ is a DataFrame with a column for dates
    value_series = reals.loc[i].squeeze()  # Assuming reals is a DataFrame
    for date, value in zip(date_series, value_series):
        data.append({'date': date, 'value': value + 40, 'id': i})

# Add generated data
for i in ids:
    date_series = times_.loc[i].squeeze()  # Assuming times_ is a DataFrame with a column for dates
    for j in range(3):
        value_series = generateds.loc[i].iloc[j].squeeze()  # Assuming generateds has T rows per id
        for date, value in zip(date_series, value_series):
            new_id = f"{i}_{j + 1}"
            data.append({'date': date, 'value': value + 40, 'id': new_id})

# Create the new DataFrame
new_df2 = pd.DataFrame(data)
new_df2.set_index('id', inplace=True)

path = f"generated_files/10_K"

# under 10_K create a new folder (if not existed) named after DAYS
if not os.path.exists(f"{path}/{DAYS}"):
    os.makedirs(f"{path}/{DAYS}")

new_df2.to_csv(f"{path}/{DAYS}/generateds.csv")
