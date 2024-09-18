import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from hails.hails import HAILS_Univ
from hails.seq_layers import DLinear, NLinear
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
PRETRAIN_LR = 1e-3
BATCH_SIZE = 16
PRE_TRAIN_EPOCHS = 1
TRAIN_LR = 1e-3
LAMBDA = 0.5
TRAIN_EPOCHS = 100
SCALE_PREC = False

set_seed(SEED)
train_dataset, text_dataset, _, _ = get_datasets()
train_dataset, train_hmatrix = get_dataset(train_dataset)
text_dataset, text_hmatrix = get_dataset(text_dataset)
print(f"{train_dataset.shape=}, {train_hmatrix.shape=}")
print(f"{text_dataset.shape=}, {text_hmatrix.shape=}")
train_hmatrix = train_hmatrix.to(DEVICE)
# text_hmatrix = text_hmatrix.to(DEVICE)

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
optimizer = AdamW(hails.parameters(), lr=PRETRAIN_LR)
scaler = GradScaler(device=DEVICE)


# Load pre-trained model
hails.load_state_dict(torch.load("pretrained_m5.pth"))
print("Pre-trained model loaded!")

# Training


def train_step():
    hails.train()
    losses = [[], [], []]
    for x, y in tqdm(train_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        if SCALE_PREC:
            with autocast(device_type=DEVICE):
                mu, logstd = hails(x)
                ll_loss = hails.get_ll_loss(mu, logstd, y, dist_mask).mean()
                dch_loss = hails.get_jsd_loss(
                    mu, logstd, train_hmatrix, dist_mask
                ).mean()
                loss = (1 - LAMBDA) * ll_loss + LAMBDA * dch_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            mu, logstd = hails(x)
            ll_loss = hails.get_ll_loss(mu, logstd, y, dist_mask).mean()
            dch_loss = hails.get_jsd_loss(mu, logstd, train_hmatrix, dist_mask).mean()
            loss = (1 - LAMBDA) * ll_loss + LAMBDA * dch_loss
            loss.backward()
            optimizer.step()
        losses[0].append(loss.item())
        losses[1].append(ll_loss.item())
        losses[2].append(dch_loss.item())
    return (
        sum(losses[0]) / len(losses[0]),
        sum(losses[1]) / len(losses[1]),
        sum(losses[2]) / len(losses[2]),
    )


print("Training...")
for ep in range(TRAIN_EPOCHS):
    loss, ll_loss, dch_loss = train_step()
    print(
        f"Epoch {ep+1}/{TRAIN_EPOCHS}, Loss: {loss:.4f}, LL Loss: {ll_loss:.4f}, DCH Loss: {dch_loss:.4f}"
    )
print("Training done!")

# Save trained model
torch.save(hails.state_dict(), "trained_m5.pth")
print("Trained model saved!")
