from datetime import date
from pathlib import Path

import mlflow
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta

if __name__ == "__main__":
    print(f"{yf.__version__=}")
    # Workaround to handle issue https://github.com/pydata/pandas-datareader/issues/868
    USER_AGENT = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    sesh = requests.Session()
    sesh.headers.update(USER_AGENT)

    with mlflow.start_run(run_name="load_raw_data") as run:
        mlflow.set_tag("mlflow.runName", "load_raw_data")
        end = date.today()
        start = end + relativedelta(months=-3)

        df = yf.download("BTC-USD", start=start, end=end)
        # df = df.drop(df.index[[0]])
        df = df.reset_index()
        print(df.head())
        out_df_path = Path("./data/raw/data.csv")
        out_df_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out_df_path, index=False)
