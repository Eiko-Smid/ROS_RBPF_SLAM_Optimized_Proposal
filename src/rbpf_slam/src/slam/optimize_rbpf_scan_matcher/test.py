#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def analyze_rolling_mean():
    trans_err = np.ones(10)
    trans_err[5] = 15
    window = 3

    print(trans_err)

    # Compute rolling mean
    trans_err_ser = pd.Series(trans_err**2)
    roll_mean_trans_err = trans_err_ser.rolling(window=window).mean()
    sqrt_roll_mean_trans_err = np.sqrt(roll_mean_trans_err)
    
    print(f"roll_mean_trans_err: {roll_mean_trans_err}")
    

    # Plot values
    fig = plt.figure(figsize=(10, 6))
    # plt.plot(trans_err, label="trans_err")
    plt.plot(roll_mean_trans_err, label=f"Rolling mean (window={window})")
    plt.plot(sqrt_roll_mean_trans_err, label=f"Sqrt of rolling mean (window={window})")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Rolling Mean and its Square Root")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    analyze_rolling_mean()


if __name__ == "__main__":
    main()