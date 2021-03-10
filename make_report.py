# Makes a report from given set of trades
__author__ = 'David Li (FriendlyUser)'
__license__ = 'GNU Lesser General Public License v2.1'
__copyright__ = 'Copyright 2021 by David Li'

import pandas as pd
import os
import numpy as np
import yaml
from datetime import datetime
import argparse
from jinja2 import Template
from stats import calc_input, count_commissions
import datapane as dp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from datetime import datetime

pd.set_option('display.max_rows', 500)
def price_colors(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'green'
    return 'color: %s' % color

class FinReport:
    def __init__(self, csvPath, config):
        self.config = config
        self.starting_year = config["settings"].get("starting_year", 2019)
        self.tnx_cost = config["settings"].get("tnx_cost", 9.95)

        df = self.read_csv(csvPath)
        self.original_df = df
        self.inputAmount = calc_input(df)
        cad_df = self.clean_data(df)
        df = self.filter_tickers(df)
        self.commissions = count_commissions(df, self.tnx_cost)
        self.df = cad_df
        usd_df = self.clean_data(df, "USD")
        self.usd_df = usd_df

    def gen_report_dp(self):
        """
            Creates html datapane report.

            It is possible to add web reports with datapane publishing. 

            I do not want to send my trading data to an external website.
        """
        df = self.df
        ticker_data = self.parse_csv(df, tnx_cost=self.tnx_cost)
        total_profit = self.calc_profit(ticker_data)
        pages = []
        cad_options = {
            "ticker_data": ticker_data,
            "version": "Version 1",
            "commissions": self.commissions,
            "input": self.inputAmount,
            "total_profit": int(total_profit)
        }
        if self.config["settings"]["currencies"] in ["ALL", "CAD"]:
            cad_stocks = self.map_ticker_data_to_blocks(ticker_data)
            cad_statistics = self.stats_for_report_data(cad_options)
            curr_date = datetime.today().strftime('%Y-%m-%d')
            header = dp.Group(
                f'# Stock Reports {curr_date}',
                f"Profit Calculation only counts when there is a zero sum value for the TotalQty.",
                cad_statistics,
            )
            pages.append(   
                dp.Page(
                    label="Canadian Tickers",
                    blocks=[
                        header,
                        dp.Group(*cad_stocks)
                    ]
                )
            )
        # Get usd stats
        if self.config["settings"]["currencies"] in ["ALL", "USD"]:
            usd_ticker_data = self.parse_csv(self.usd_df, "USD", tnx_cost=self.tnx_cost)
            total_profit = self.calc_profit(usd_ticker_data)
            usd_options = {
                "ticker_data": usd_ticker_data,
                "version": "Version 1",
                "commissions": self.commissions,
                "input": self.inputAmount,
                "total_profit": int(total_profit)
            }
            usd_stocks = self.map_ticker_data_to_blocks(usd_ticker_data)
            usd_statistics = self.stats_for_report_data(usd_options)

            pages.append(
                dp.Page(
                    label="Usd Tickers",
                    blocks=[
                        usd_statistics,
                        dp.Group(*usd_stocks)
                    ]
                )
            )
        if len(pages) == 0:
            return
        r = dp.Report(
            *pages
        )
        output_path = self.config["output"]["path"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        r.save(path=output_path)


    def gen_report_jinja(self):
        df = self.df
        ticker_data = self.parse_csv(df, tnx_cost=self.tnx_cost)
        # get date and render report
        total_profit = self.calc_profit(ticker_data)
        options = {
            "ticker_data": ticker_data,
            "version": "Version 1",
            "commissions": self.commissions,
            "input": self.inputAmount,
            "total_profit": int(total_profit)
        }
        self.render_report(**options)

    @staticmethod 
    def stats_for_report_data(options):
        num_transactions = len(options["ticker_data"])
        commissions = options["commissions"]
        total_profit = options["total_profit"]
        statistics = dp.Group(
                dp.BigNumber(
                    heading="Profit", 
                    value=total_profit,
                    is_upward_change=True
                ),
                dp.BigNumber(
                    heading="Commissions", 
                    value=commissions,
                    change=num_transactions,
                    is_upward_change=False
                ),
                columns=2
            )
        return statistics

    @staticmethod 
    def map_ticker_data_to_blocks(ticker_data: list):
        block_list = []

        if len(ticker_data) == 0:
            return []
        for ticker in ticker_data:
            df = ticker.get('data')
            plt.figure(figsize=(8, 10))
            fig_df = df
            fig_df['Date'] = fig_df['Date'].dt.date

            ticker_profit = ticker.get('profit')
            fig_caption = f"{df.iloc[0]['Symbol']} ({df.iloc[0]['Currency']}) {df.iloc[0]['Account']} - Profit({ticker_profit})"
            fig = fig_df[["Date", "TotalQty", "Value"]].plot.bar(x='Date', rot=15, title=fig_caption )
            fig.xaxis.set_major_locator(mdates.AutoDateLocator())

            table = df.drop(['Account', 'Currency', 'Settlement Date'], axis=1)
            table = table.style.\
                applymap(price_colors, subset=['Value', 'Quantity']).\
                hide_index().\
                format(
                    {
                        'Date': "{:%b %d, %Y}", 'Settlement Date': "{:%b %d, %Y}",
                        'Value': "{:.0f}", 'Quantity': "{:.0f}",
                        "Price": "{:.2f}", 'TotalQty': "{:.0f}",
                        'average_price': "{:.2f}"
                    }
                )

            ticker_block = dp.Group(
                blocks=[
                    dp.Plot(fig),
                    table
                ]
            )
            plt.close('all')

            block_list.append(
                ticker_block
            )

        return block_list

    @staticmethod
    def calc_profit(ticker_data: list)-> float:
        total_profit = 0
        for ticker in ticker_data:
            ticker_profit = ticker.get("profit")
            if ticker_profit != "N/A":
                total_profit += ticker_profit
        return total_profit
            
    def filter_tickers(self, df: pd.DataFrame)-> None:
        # filters tickers based on config
        df = FinReport.calc_totalqty(df)
        if self.config["settings"]["show_only_current_holdings"] == True:
            for ticker in df.Symbol.unique():
                ticker_df = df[df['Symbol'] == ticker]
                ticker_df = ticker_df.reset_index(drop=True)
                currShares = ticker_df.iloc[-1]['TotalQty']
                if currShares == 0:
                    df = df[df['Symbol'] != ticker]
        return df
    
    @staticmethod
    def read_csv(csv_path: str)-> pd.DataFrame:
        dateparse = lambda x: datetime.strptime(x, "%B %d, %Y")
        df = pd.read_csv(csv_path, parse_dates=['Date'], date_parser=dateparse)
        return df

    @staticmethod
    def clean_data(df: pd.DataFrame, currency: str = "CAD")->pd.DataFrame:

        df = df[df.Activity != "Deposits & Contributions"]
        df = df[df.Activity != "Dividends"]
        df = df[df.Activity != "Withdrawals & De-registrations"]
        # add option to remove transfers in config
        # df = df[df.Activity != "Transfers"]
        df = df[df.Symbol != "RBF558"]
        df = df[df.Symbol != "RBF556"]
        df = df[df['Currency'] == currency]
        df = df[df['Activity'] != 'Reorganization']
        return df

    @staticmethod
    def calc_totalqty(df: pd.DataFrame, save_csv: bool = False)-> pd.DataFrame:
        df = df.sort_values(by=['Date', 'Activity'], ascending=[True, True])
        # Groups by ticker and then applies summation to culumative summation to quantity column
        # saves to TotalQty.
        # Given number of owned shares at any given time
        df['TotalQty'] = df.groupby(['Symbol'])['Quantity'].apply(lambda x: x.cumsum())
        # Make saving the csv an option
        if save_csv == True:
            df.to_csv('totalqty.csv', index=False)
        return df

    @staticmethod
    def calc_acb(sorted_df: pd.DataFrame)-> pd.DataFrame:
        """calculate acb for a sorted dataframe only containing
        trades with a single ticker
        
        see https://stackoverflow.com/questions/45448532/dataframe-calculating-average-purchase-price
        this method does not account for stock purchasing fees.
        """
        df = sorted_df
        df['prev_cum_qty'] = df['TotalQty'].shift(1, fill_value=0)
        df['average_price'] = np.nan
        for i, row in df.iterrows():
            # transfers dont work very well
            # if you transfer a cad stock to usd and
            # thats the only time you it was bought
            # it works
            if row['Activity'] == 'Transfers':
                # grab last value
                if i == 0:
                    df.at[i, 'average_price'] = 0
                else:
                    df.at[i, 'average_price'] = df.at[i-1, 'average_price']
                continue
            # Quantity positive for buying
            if row['Quantity'] > 0:
                if i == 0:
                    df.at[i, 'average_price'] = abs(df.at[i, 'Price'])
                else:
                    # check for transfer
                    # get share value by grabbing last average_price and current cumulative shares
                    share_value = df['average_price'].shift(1, fill_value=df.at[i, 'Price'])[i] * row['prev_cum_qty']
                    # Buying is treated as a loss of money
                    # Selling treated as a gain of money in Value
                    acb = (-row['Value'] + share_value) / row['TotalQty']
                    df.at[i, 'average_price'] = abs(acb)
            # quantity negative for selling
            else:
                # use last value, selling transaction does not impact average_price
                # sales do not impact acb as selling does not lower average cost basis
                if i == 0:
                    df.at[i, 'average_price'] = abs(df.at[i, 'Price'])
                else:
                    df.at[i, 'average_price' ] = df['average_price'][i-1]
        # TODO add function to control rounding
        # or let pandas only show 2 decimal places
        df = df.round(decimals=3)
        return df

    @staticmethod
    def calc_profit_for_years(ticker_df: pd.DataFrame, txn_cost=9.95):
        """ Calculates the profit for all years in the given df for a particular ticker
        Returns an array of object, year: profit
        """
        profits = []
        ticker_df['tnx_year'] = ticker_df['Date'].dt.strftime('%Y')
        for tnx_year in ticker_df["tnx_year"].unique():
            year_df = ticker_df[ticker_df.tnx_year == tnx_year]
            sell_df = year_df[year_df.Activity == 'Sell'] 
            raw_profit = ((sell_df['Price'] - sell_df['average_price']) * abs(sell_df['Quantity'])).sum().round(2)
            num_txns = len(year_df[year_df.Activity.isin(['Sell', 'Buy'])])
            yearly_profit = raw_profit - num_txns * txn_cost
            profits.append({
                "year": tnx_year,
                "profit": yearly_profit,
                "raw_profit": raw_profit,
                "num_txns": num_txns
            })
        return profits

    # parses a csv from rbc
    @staticmethod
    def parse_csv(df: pd.DataFrame, currency: str = 'CAD', txn_cost = 9.95)-> list:
        df = FinReport.clean_data(df, currency)
        df = FinReport.calc_totalqty(df)
        report_data = []
        if len(df) == 0:
            print("dataframe should not be empty")
            return []

        if not os.path.exists("output"):
            os.makedirs("output")
        # perform calculations for each ticker
        for ticker in df.Symbol.unique():
            ticker_df = df[df['Symbol'] == ticker]
            ticker_df = ticker_df.reset_index(drop=True)
            ticker_df = FinReport.calc_acb(ticker_df)
            output_path = os.path.join("output", f"{ticker}.csv")
            description = ticker_df.iloc[0]['Description']
            start_date = ticker_df.iloc[0]['Date']
            end_date = ticker_df.iloc[-1]['Date']
            # remove uneeded columns for analysis
            del ticker_df['Symbol Description']
            del ticker_df['Description']
            del ticker_df['prev_cum_qty']

            ticker_df.to_csv(output_path, index=False)

            sell_df = ticker_df[ticker_df.Activity == 'Sell'] 
            raw_profit = ((sell_df['Price'] - sell_df['average_price']) * abs(sell_df['Quantity'])).sum().round(2)

            profit = raw_profit - len(ticker_df[ticker_df.Activity.isin(['Sell', 'Buy'])]) * txn_cost
            profit_by_year = FinReport.calc_profit_for_years(ticker_df, txn_cost)

            ticker_data = {
                "ticker": ticker,
                "data": ticker_df,
                "description": description,
                "start_date": start_date,
                "end_date": end_date,
                "raw_profit": raw_profit,
                "profit": profit,
                "profit_by_year": profit_by_year
            }
            report_data.append(ticker_data)
        return report_data

    @staticmethod
    def render_report(report_path='index.html', **options):
        with open("report.template") as file_:
            template = Template(file_.read())
        index_html = template.render(**options)
        with open(report_path, "w") as file_:
            file_.write(index_html)

    def gen_profit_stats(self):
        currency = self.config["settings"]["currencies"]
        cad_df = self.df
        cad_data = []
        if currency in ["ALL", "CAD"]:
            cad_data = self.parse_csv(cad_df, "CAD", self.tnx_cost)
        # calc profit from cad_stock
        usd_df = self.clean_data(self.original_df, "USD")
        usd_data = []
        if currency in ["ALL", "USD"]:
            usd_data = self.parse_csv(usd_df, "USD", self.tnx_cost)
        full_data = [*cad_data, *usd_data]

        # Do logging to file instead of all these print statements
        if self.config["settings"]["verbose"] == True:
            for ticker in full_data:
                name = ticker.get('ticker')
                profit_by_year = ticker.get("profit_by_year")
                print(f"Parsing {name}")
                for yearly_profit in profit_by_year:
                    print(f"Year: {yearly_profit.get('year')} \t Profit: {yearly_profit.get('profit')}")
        
        # increment next year
        # TODO figure out how to parameterize the number of years
        # or scan the csv at run time
        yearly_profits = [0]*4
        for ticker in full_data:
            name = ticker.get('ticker')
            profit_by_year = ticker.get("profit_by_year")
            for yearly_profit in profit_by_year:
                profit_num = yearly_profit.get('profit')
                tnx_year = yearly_profit.get('year')
                index_num = int(tnx_year) - self.starting_year
                yearly_profits[index_num] += profit_num

        report_name = f"{self.config['name']} ({currency})"
        profit_series = pd.Series(
            yearly_profits,
            index=[self.starting_year + x for x in range (0,len(yearly_profits))],
            name=report_name
        ) 
        print(profit_series)
        print()
        pass

    def gen_profit_stock_stats(self):
        """
            Generate profit per stock per year
            rough estimation (should be fairly close to real values)
        """
        cad_df = self.df
        cad_data = []
        currency = self.config["settings"]["currencies"]
        if currency in ["ALL", "CAD"]:
            cad_data = self.parse_csv(cad_df, tnx_cost=self.tnx_cost)

        usd_df = self.clean_data(self.original_df, "USD")
        usd_data = []
        if currency in ["ALL", "USD"]:
            usd_data = self.parse_csv(usd_df, "USD", tnx_cost=self.tnx_cost)
        full_data = [*cad_data, *usd_data]

        # TODO logging to file instead of all these print statements
        if self.config["settings"]["verbose"] == True:
            # print values
            for ticker in full_data:
                name = ticker.get('ticker')
                profit_by_year = ticker.get("profit_by_year")
                print(f"Parsing {name}")
                for yearly_profit in profit_by_year:
                    print(f"Year: {yearly_profit.get('year')} \t Profit: {yearly_profit.get('profit')}")
        

        valid_years = [self.starting_year + x for x in range(0,4)]
        tickers = [ ticker.get('ticker') for ticker in full_data]
        report_idx = pd.MultiIndex.from_product([tickers, valid_years],
                           names=['Ticker', 'Year'])

        df_columns = ['profit']

        report_name = f"{self.config['name']} ({currency})"
        stock_profit_df = pd.DataFrame(index=report_idx, columns=df_columns)
        for ticker in full_data:
            name = ticker.get('ticker')
            profit_by_year = ticker.get("profit_by_year")
            for yearly_profit in profit_by_year:
                profit_num = yearly_profit.get('profit')
                tnx_year = yearly_profit.get('year')
                stock_profit_df.at[(name, int(tnx_year)), 'profit'] = profit_num
        stock_profit_df = stock_profit_df.dropna()
        print(report_name)
        print(stock_profit_df)
        print()
        pass

def main(cfg):

    for report_name in cfg["reports"]:
        report_cfg = cfg["reports"][report_name]
        csv_path = report_cfg["file"]
        finreport = FinReport(csv_path, report_cfg)
        report_type = report_cfg["type"]
        if report_type == 'dp':
            finreport.gen_report_dp()
        elif report_type == 'stats':
            finreport.gen_profit_stats()
        elif report_type == 'stock_stats':
            finreport.gen_profit_stock_stats()
        else:
            finreport.gen_profit_stats()

if __name__ == '__main__':
    # scan document
    parser = argparse.ArgumentParser(description='Generate Report for given rbc trade file')
    conf_file = "config.yml"
    parser.add_argument(
        '--cfg_file',
        default=conf_file,
        help='Config File for the %(prog)s program to analyze'
    )
    args = parser.parse_args()
    with open(args.cfg_file, 'r') as stream:
        cfg = yaml.safe_load(stream)
    main(cfg)
