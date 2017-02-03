#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from subprocess import check_call

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main(name):
    result_fn = name + ".csv"
    with open(result_fn, "w") as f:
        check_call(["cargo", "run", "--release", "--bin", name], stdout=f)
    data = pd.read_csv(result_fn).set_index("time")[1:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs=data["x"], ys=data["y"], zs=data["z"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(name + ".png")


if __name__ == '__main__':
    main("lorenz63")
    main("roessler")
