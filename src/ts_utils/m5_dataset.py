import pandas as pd
import torch

ROOT_FOLDER = "dataset/M5/Dataset"
HIERARCHY = ["country_id", "state_id", "store_id", "cat_id", "dept_id", "item_id"]


def get_datasets(root_folder: str = ROOT_FOLDER):
    train_sales = pd.read_csv(f"{root_folder}/sales_train_validation.csv")
    test_sales = pd.read_csv(f"{root_folder}/sales_train_evaluation.csv")
    calendar = pd.read_csv(f"{root_folder}/calendar.csv")
    prices = pd.read_csv(f"{root_folder}/sell_prices.csv")
    return train_sales, test_sales, calendar, prices


def get_dataset(sales_df: pd.DataFrame):
    sales_df["country_id"] = "US"
    time_series_np = sales_df.drop(
        columns=["country_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ).values.T
    time_series = torch.tensor(time_series_np, dtype=torch.float32)
    hmatrix = torch.eye(len(sales_df), len(sales_df))
    for col in reversed(HIERARCHY[1:]):
        ts, matrix, sales_df = get_aggregate(sales_df, col)
        time_series = torch.cat([ts, time_series], dim=-1)  # type: ignore
        hmatrix_new = torch.zeros(
            hmatrix.shape[0] + matrix.shape[0], hmatrix.shape[0] + matrix.shape[0]
        )
        hmatrix_new[matrix.shape[0] :, matrix.shape[0] :] = hmatrix
        hmatrix_new[
            : matrix.shape[0], matrix.shape[0] : matrix.shape[0] + matrix.shape[1]
        ] = matrix
        hmatrix = hmatrix_new
    return time_series, hmatrix


def get_aggregate(df: pd.DataFrame, agg_over: str):
    # Groupby over all the columns except the agg_over column
    df_group = df.groupby(
        [col for col in HIERARCHY if col in df.columns and col != agg_over]
    ).sum()
    df_group = df_group.drop(columns=[agg_over])
    df["initialIndex"] = df.index
    df_group2 = df.groupby(
        [col for col in HIERARCHY if col in df.columns and col != agg_over]
    )["initialIndex"].apply(list)
    df_group["initialIndex"] = df_group2
    matrix = torch.zeros(len(df_group), len(df), dtype=torch.float32)
    for i, (index, row) in enumerate(df_group.iterrows()):
        matrix[i, row["initialIndex"]] = 1.0
    time_series = df_group.drop(columns=["initialIndex"]).values.T
    time_series = torch.tensor(time_series, dtype=torch.float32)
    df.drop(columns=["initialIndex"], inplace=True)
    return time_series, matrix, df_group.drop(columns=["initialIndex"]).reset_index()
