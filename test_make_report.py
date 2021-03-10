import unittest
import os
import pandas as pd
from make_report import FinReport, main as mk_report
from datetime import datetime
import yaml

class TestMakeReport(unittest.TestCase):
    def setUp(self):
        trades_csv_path = os.path.join('test.csv')
        dateparse = lambda x: datetime.strptime(x, "%B %d, %Y")
        df = pd.read_csv(trades_csv_path, parse_dates=['Date'], date_parser=dateparse)
        df = FinReport.clean_data(df)
        self.df = df

    def test_totalqty(self):
        # create total qty without crashing
        t_df = FinReport.calc_totalqty(self.df)

    def test_calc_acb(self):
        # create total qty without crashing
        df = FinReport.calc_totalqty(self.df)
        df = df[df['Symbol'] == 'BB']
        df = df.reset_index(drop=True)
        fin_df = FinReport.calc_acb(df)

    def test_mk_report(self):
        with open('config.yml', 'r') as stream:
            cfg = yaml.safe_load(stream)
        mk_report(cfg)