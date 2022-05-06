import pandas as pd


def print_progress(i, total, extra_data=""):
    print(f"{i} of {total} ({extra_data})" + " " * 20, end="\r" if i != total else "\n", flush=True)


def memory_usage(df: pd.DataFrame):
    bytes_used = df.memory_usage(index=True).sum()
    return bytes_used / 1024 / 1024
