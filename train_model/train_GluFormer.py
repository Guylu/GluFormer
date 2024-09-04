import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
import torch
import os

hyperparameter_defaults = dict(
    epochs=76,
    lr=5e-5,
    weight_decay=0,
    step_size=100,
    gamma=0.99,
    dropout=0.1,

    n_embd=1024,
    n_heads=16,
    n_layers=16,
    dim_feedforward=2048,
    max_seq_length=25000,

    chunk_size=1200,

    seed=42,
    batch_per_gpu=32,

)
print(hyperparameter_defaults)
wandb.init(config=hyperparameter_defaults, project="CGM_Foundation", allow_val_change=True)
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

chunks = 128
chunk_size = config.chunk_size  # Fixed size for each chunk
PAD_TOKEN = chunks  # Assuming -1 is not a valid glucose measurement


class GlucoseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns a single item at index `idx` from the data
        return self.data[idx]


class TransformerModel(nn.Module):
    # Should be like BERT
    def __init__(self, vocab_size, n_embd, mask_prob=0.15, n_heads=8, n_layers=4, max_seq_length=672, dropout=0.1,
                 dim_feedforward=512):
        super(TransformerModel, self).__init__()
        self.n_embd = n_embd
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embedding = nn.Embedding(vocab_size + 1, n_embd)
        self.pos_embedding = self.create_pos_embedding(max_seq_length, n_embd)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=n_layers)
        self.linear = nn.Linear(n_embd, vocab_size)

    def create_pos_embedding(self, max_seq_length, n_embd):
        # Initialize the position matrix
        position = torch.arange(max_seq_length).unsqueeze(1)
        # Compute the div term
        div_term = torch.exp(torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd))
        # Apply sine to even indices in the array; 2i
        pos_embedding = torch.zeros(max_seq_length, n_embd)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the array; 2i+1
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding

    def forward(self, tokens, mask=None):
        # Compute the embeddings
        token_embeddings = self.embedding(tokens)  # [Batch, Seq, Emb]
        position_embeddings = self.pos_embedding[:tokens.size(1), :].unsqueeze(0).to(tokens.device)
        embeddings = token_embeddings + position_embeddings

        # Create the causal mask
        seq_length = tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).to(tokens.device)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))

        # for this version set src_key_padding_mask to None
        src_key_padding_mask = mask

        # Pass embeddings, causal mask, and padding mask to the transformer
        transformer_output = self.transformer(embeddings.permute(1, 0, 2), mask=causal_mask,
                                              src_key_padding_mask=src_key_padding_mask)
        logits = self.linear(transformer_output.permute(1, 0, 2))

        return logits


# Create a DataLoader for batching
batch_size_per_gpu = config.batch_per_gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_path = f"cgm_diet_filtered_processed_aligned_tokenized_tensors_train.pt"
train_dataset = GlucoseDataset(torch.load(my_path)['tokens'])
my_path = f"cgm_diet_filtered_processed_aligned_tokenized_tensors_val.pt"
val_dataset = GlucoseDataset(torch.load(my_path)['tokens'])

# print sizes
print(f"train size: {len(train_dataset)}")
print(f"val size: {len(val_dataset)}")

if torch.cuda.is_available():
    batch_size = batch_size_per_gpu * torch.cuda.device_count()
else:
    batch_size = batch_size_per_gpu

print(f'Batch size: {batch_size}')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

vocab_size = chunks
n_embd = config.n_embd
dim_feedforward = config.dim_feedforward
# Assuming the model and dataloader are already defined
model = TransformerModel(vocab_size, n_embd, n_heads=config.n_heads, n_layers=config.n_layers, max_seq_length=25000,
                         dropout=config.dropout, dim_feedforward=dim_feedforward).to(device)
print(f"num of parameters: {sum(p.numel() for p in model.parameters())}")
wandb.log({"num of parameters": sum(p.numel() for p in model.parameters())})
# move to all GPUs in dataparralele
model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
num_epochs = config.epochs

for epoch in range(num_epochs):
    print(f'\n\n\nEpoch {epoch}\n\n\n')
    wandb.log({"epoch": epoch})
    model.train()
    for i, batch in enumerate(train_dataloader):
        inputs = batch.to(device)
        mask = (inputs == PAD_TOKEN)
        # Shift the inputs to the right for the target sequence
        inputs, targets = inputs[:, :-1], inputs[:, 1:]

        model.zero_grad()
        logits = model(inputs, mask=mask)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=PAD_TOKEN)

        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        print(f'Epoch {epoch}, batch {i}, Train accuracy: {accuracy.item()}')
        wandb.log({"Train accuracy": accuracy.item()})

        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f' {loss.item()}')
        wandb.log({"train loss": loss.item()})
        print(f' {scheduler.get_last_lr()[0]}')
        wandb.log({"learning rate": scheduler.get_last_lr()[0]})

        # half the size of len(train_dataloader) for testing
        if (i + 1) % ((len(train_dataloader) // 2) - 3) == 0:
            model.eval()
            with torch.no_grad():
                loss_avg = 0
                acc_logits = []
                acc_targets = []
                for i, batch in enumerate(val_dataloader):
                    print(f"batch {i}")
                    inputs = batch.to(device)
                    mask = (inputs == PAD_TOKEN)
                    # Shift the inputs to the right for the target sequence
                    inputs, targets = inputs[:, :-1], inputs[:, 1:]

                    logits = model(inputs, mask=mask)
                    loss_ = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=PAD_TOKEN)
                    loss_avg = loss_avg + loss_.item()
                    acc_logits.append(logits)
                    acc_targets.append(targets)

                loss_avg = loss_avg / len(val_dataloader)
                print(f' \t\t\t\t\t\t\t\t\t\t\t\t {loss_avg}')
                wandb.log({"val loss": loss_avg})

                logits = torch.cat(acc_logits, dim=0)
                targets = torch.cat(acc_targets, dim=0)
                accuracy = (logits.argmax(dim=-1) == targets).float().mean()
                print(f'\t\t\t\t\t\t\t\t\t\t\t\t Epoch {epoch}, batch {i}, VAl accuracy: {accuracy.item()}')
                wandb.log({"Val accuracy": accuracy.item()})
            model.train()

    # save checkpoint of the model with time and date and epoch/epochs
    path = "Models/" + str(wandb.run.name)
    # if not created, create a folder with the name of the wand.run.name
    if not os.path.exists(path):
        os.makedirs(path)
    path += f"/Model_{str(wandb.run.name)}.pt"
    # save model dict
    torch.save(model.module.state_dict(), path)
