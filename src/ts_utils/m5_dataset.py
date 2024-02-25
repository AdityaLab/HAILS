import pandas as pd

ROOT_FOLDER = "dataset/M5/Dataset"


def get_datasets(root_folder: str = ROOT_FOLDER):
    train_sales = pd.read_csv(f"{root_folder}/sales_train_validation.csv")
    test_sales = pd.read_csv(f"{root_folder}/sales_train_evaluation.csv")
    calendar = pd.read_csv(f"{root_folder}/calendar.csv")
    prices = pd.read_csv(f"{root_folder}/sell_prices.csv")
    return train_sales, test_sales, calendar, prices
