import os
import numpy as np
import pandas as pd

PATH_5MIN  = r'/Users/jan/Documents/working papers/project 1/data/btc_5m.csv'
PATH_15MIN = r'/Users/jan/Documents/working papers/project 1/data/btc_15m.csv'
PATH_1H    = r'/Users/jan/Documents/working papers/project 1/data/btc_1h.csv'

OUT_DIR = r'/Users/jan/Documents/working papers/project 1/data'

def add_log_return(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path)
    # columns: timestamp,open,high,low,close,volume
    df["log_return"] = np.log(df["close"]).diff().astype("float32") #have to discuss the float size
    df = df.dropna(subset=["log_return"])
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    out_5min  = os.path.join(OUT_DIR, "btc_5min_logreturns.csv")
    out_15min = os.path.join(OUT_DIR, "btc_15min_logreturns.csv")
    out_1h    = os.path.join(OUT_DIR, "btc_1h_logreturns.csv")

    add_log_return(PATH_5MIN,  out_5min)
    add_log_return(PATH_15MIN, out_15min)
    add_log_return(PATH_1H,    out_1h)

    print("Saved:")
    print(out_5min)
    print(out_15min)
    print(out_1h)\

