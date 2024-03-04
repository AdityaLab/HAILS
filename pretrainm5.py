import torch
from hails.hails import HAILS_Univ
from hails.seq_layers import DLinear, NLinear
from torch.optim import AdamW
from torch.utils.data import DataLoader
from ts_utils.datasets import HierarchicalTimeSeriesDataset
from ts_utils.m5_dataset import get_dataset, get_datasets
from ts_utils.utils import prob_poisson, prob_poisson_dispersion, set_seed

SEED = 42
PRED_LEN = 28
SEQ_LEN = 112  # past 4 months
NUM_WORKERS = 1
USE_DISPERSION = False
MODEL_TYPE = "DLinear"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
BATCH_SIZE = 128
PRE_TRAIN_EPOCHS = 100

set_seed(SEED)
train_dataset, text_dataset, _, _ = get_datasets()
train_dataset, train_hmatrix = get_dataset(train_dataset)
text_dataset, text_hmatrix = get_dataset(text_dataset)
print(train_dataset.shape, train_hmatrix.shape)
print(text_dataset.shape, text_hmatrix.shape)

dist_mask = (
    prob_poisson(train_dataset).to(DEVICE)
    if not USE_DISPERSION
    else prob_poisson_dispersion(train_dataset).to(DEVICE)
)
print(
    f"Percentage of Poisson distributed nodes: {dist_mask.sum().item()/dist_mask.size(0)*100:.2f}%"
)


train_dataset_obj = HierarchicalTimeSeriesDataset(
    train_dataset, PRED_LEN, SEQ_LEN, None, train_hmatrix
)
test_dataset_obj = HierarchicalTimeSeriesDataset(
    text_dataset, PRED_LEN, SEQ_LEN, None, text_hmatrix
)

print(f"{len(train_dataset_obj.time_series_dataset)= }")

train_loader = DataLoader(
    train_dataset_obj.time_series_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset_obj.time_series_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


hails = HAILS_Univ(
    num_nodes=train_dataset.shape[-1],
    seq_len=SEQ_LEN,
    pred_len=PRED_LEN,
    pred_model=DLinear if MODEL_TYPE == "DLinear" else NLinear,
    corem_c=5,
).to(DEVICE)

print(hails)

# Pre-train
optimizer = AdamW(hails.parameters(), lr=LR)


def pre_train_step():
    hails.train()
    losses = []
    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        mu, logstd = hails._forward_base(x)
        loss = hails.get_ll_loss(mu, logstd, y, dist_mask).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


print("Pre-training...")
for ep in range(PRE_TRAIN_EPOCHS):
    loss = pre_train_step()
    print(f"Epoch {ep+1}/{PRE_TRAIN_EPOCHS}, Loss: {loss:.4f}")
print("Pre-training done!")
