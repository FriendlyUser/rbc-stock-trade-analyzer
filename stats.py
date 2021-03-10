import argparse
import pandas as pd 
import os
def main(args):
    file_path = args.file
    df = pd.read_csv(file_path)
    get_stat(calc_input, df)
    get_stat(calc_profits, df)
    get_stat(count_commissions, df)

# func with file path
def get_stat(func, file_path):
    print(f"Calling {str(func.__name__)}")
    stat = func(file_path)
    print(f"\t {stat} \n")
    return stat

def calc_profits(df: pd.DataFrame):
    df = df[df.Activity != "Deposits & Contributions"]
    df = df[df.Activity != "Dividends"]
    df = df[df.Activity != "Withdrawals & De-registrations"]
    df = df[df.Symbol != "RBF558"]
    df = df[df.Symbol != "RBF556"]
    return int(df["Value"].round().sum())

# counts commisions based on number of transactions
def count_commissions(df: pd.DataFrame, tnx_cost = 9.95):
    activity_df = df[df.Activity.isin(["Buy", "Sell"])]

    total_cost = len(activity_df) * tnx_cost

    return int(total_cost)

def calc_input(df: pd.DataFrame):
    deposit_df = df[df.Activity == "Deposits & Contributions"]
    divid_df = df[df.Activity == "Dividends"]
    with_df = df[df.Activity == "Withdrawals & De-registrations"]
    # Dividends at 229.7 + 208.44 + 7
    total = int(deposit_df["Value"].sum() + with_df["Value"].sum() - 438.14)
    return total

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stats for given csv file')
    trades_csv_path = os.path.join('..', '..', '2021', 'activities', 'data_66544658.csv')
    parser.add_argument(
        '--file',
        default=trades_csv_path,
        help='File for the %(prog)s program to analyze'
    )
    args = parser.parse_args()
    if os.path.exists(args.file) == False:
        raise Exception('Enter a valid file path')
    main(args)
