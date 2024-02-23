import numpy as np
import pandas as pd
import torch as th
from hails.layers import Corem, GRUEncoder
from hails.utils import device
from torch.optim import Adam
from torch.utils.data import Dataset
from tqdm import tqdm


def float_tensor(x):
    return th.tensor(x, dtype=th.float32, device=device)

def long_tensor(x):
    return th.tensor(x, dtype=th.int64, device=device)

SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed(SEED)
BATCH_SIZE = 32
TRAIN_EPOCHS = 100
LAMBDA = 0.1

sell_prices = pd.read_csv("dataset/M5/Dataset/sell_prices.csv")
calendar = pd.read_csv("dataset/M5/Dataset/calendar.csv")
sales_train_validation = pd.read_csv("dataset/M5/Dataset/sales_train_validation.csv")
sales_test_validation = pd.read_csv("dataset/M5/Dataset/sales_test_validation.csv")
sales_train_evaluation = pd.read_csv("dataset/M5/Dataset/sales_train_evaluation.csv")
sales_test_evaluation = pd.read_csv("dataset/M5/Dataset/sales_test_evaluation.csv")

train_data = pd.concat([sales_train_validation, sales_train_evaluation])
test_data = pd.concat([sales_test_validation, sales_test_evaluation])
train_data.head()

def jsd_norm(mu1, mu2, var1, var2):
    mu_diff = mu1 - mu2
    t1 = 0.5 * (mu_diff ** 2 + (var1) ** 2) / (2 * (var2) ** 2)
    t2 = 0.5 * (mu_diff ** 2 + (var2) ** 2) / (2 * (var1) ** 2)
    return t1 + t2 - 1.0

def jsd_loss(mu, logstd, hmatrix, train_means, train_std):
    lhs_mu = (((mu * train_std + train_means) * hmatrix).sum(1) - train_means) / (
        train_std
    )
    lhs_var = (((th.exp(2.0 * logstd) * (train_std ** 2)) * hmatrix).sum(1)) / (
        train_std ** 2
    )
    ans = th.nan_to_num(jsd_norm(mu, lhs_mu, (2.0 * logstd).exp(), lhs_var))
    return ans.mean()

def generate_hmatrix():
    ans = np.zeros((len(data_obj.idx_dict), len(data_obj.idx_dict)))
    for i, n in enumerate(data_obj.nodes):
        if len(n.children) == 0:
            ans[n.idx, n.idx] = 1
        c_idx = [x.idx for x in n.children]
        ans[n.idx, c_idx] = 1.0
    return float_tensor(ans)

cat_encode = {"HOBBIES": 1, "HOUSEHOLD": 2, "FOODS": 3}
state_encode = {"CA": 1, "TX": 2, "WI": 3}
train_data["cat_id"] = train_data["cat_id"].map(cat_encode)
train_data["state_id"] = train_data["state_id"].map(state_encode)

num_nodes = train_data.shape[1]

class SeqDataset(Dataset):
    def __init__(self, dataset):
        self.X, self.Y = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

rnn_predict = [GRUEncoder(1, 1, True).to(device) for _ in range(num_nodes)]
corem = Corem(num_nodes).to(device)

opt = Adam(list(rnn_predict.parameters()) + list(corem.parameters()), lr=1e-3)

def train_epoch():
    encoder = [encoder.train() for encoder in rnn_predict]
    corem.train()
    losses = []
    means, stds, gts = [], [], []
    opt.zero_grad()
    ref_x = float_tensor(train_data[:, :, None])
    hmatrix = generate_hmatrix()
    th_means = float_tensor(train_means)
    th_std = float_tensor(train_std)
    meta_x = long_tensor(np.arange(ref_x.shape[0]))
    for i in tqdm(train_idx):
        x = ref_x[:, : i - 1, :]
        y = ref_x[:, i, :]
        ref_out_x = encoder(ref_x, meta_x)
        out_x = encoder(x, meta_x)
        mean_sample1, logstd_sample1, log_py1, log_pqz, py1 = decoder(
            ref_out_x, out_x, y
        )
        mean_sample, logstd_sample, log_py, py = corem(
            mean_sample1.squeeze(), logstd_sample1.squeeze(), y
        )
        loss1 = -(log_py + log_pqz) / x.shape[0]
        loss2 = (
            jsd_loss(
                mean_sample.squeeze(),
                logstd_sample.squeeze(),
                hmatrix,
                th_means,
                th_std,
            )
            / x.shape[0]
        )
        loss = loss1 + LAMBDA * loss2
        if th.isnan(loss):
            import pdb

            pdb.set_trace()
        loss.backward()
        losses.append(loss.detach().cpu().item())
        print(f"Loss1: {loss1.detach().cpu().item()}")
        print(f"Loss2: {loss2.detach().cpu().item()}")
        means.append(mean_sample.detach().cpu().numpy())
        stds.append(logstd_sample.detach().cpu().numpy())
        gts.append(y.detach().cpu().numpy())
        if (i + 1) % BATCH_SIZE == 0:
            opt.step()
            opt.zero_grad()
    if i % BATCH_SIZE != 0:
        opt.step()
    return np.mean(losses), np.array(means), np.array(stds)

print("Training....")
for ep in tqdm(range(TRAIN_EPOCHS)):
    loss, means, stds = train_epoch()
    print(f"Epoch {ep} loss: {loss}")