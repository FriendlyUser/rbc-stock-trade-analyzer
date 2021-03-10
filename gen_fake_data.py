import pandas as pd 
import random
import time

def str_time_prop(start, end, format, prop, output_format='%B %d, %Y'):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(output_format, time.localtime(ptime))


def random_date(start, end, prop):
    return str_time_prop(start, end, '%m/%d/%Y', prop)

column_names = [
    "Date",
    "Activity",
    "Symbol",
    "Quantity",
    "Price",
    "Value",
    "Currency",
    "Description",
    "Symbol Description"
]

# fields that matter for my script
# date, activity, symbol, quantity, price, value, currency
df = pd.DataFrame(columns=column_names)
activities = ["Buy", "Sell"]
stocks = ["BB", "AMC", "GME"]
starting_year = 2019
def get_currency(stock: str)-> str:
    if stock == "BB":
        return "CAD"
    return "USD"


for x in range(0, 15):
    symbol = random.choice(stocks)
    currency = get_currency(symbol)
    if x == 0:
        quantity = random.randint(30, 100)
    else:
        quantity = random.randint(0, 10)

    month = 1 + x % 12
    year = int(starting_year + x / 12)
    activity = random.choice(activities)
    ran_date = random_date(f"1/{month}/{year}", f"1/{month}/{year+1}", random.random())
    price = random.randint(1,15)
    if activity == 'Buy':
        value = -price*quantity
    else:
        value = price*quantity
    df = df.append({
        "Date": ran_date,
        "Symbol": symbol,
        "Quantity": quantity,
        "Price": price,
        "Activity": activity,
        "Value": value,
        "Currency": currency,
        "Description": "lame",
        "Symbol Description": "test",
    }, ignore_index=True)

df.to_csv("test.csv", index=False)
    # make random symbol from meme stocks
    # random quantity
    # random price
    # claculate value