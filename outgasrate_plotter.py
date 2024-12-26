#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter
import datetime
import glob
import re
from outgasrate import fit_linear, read_file

DATE_FORMAT = "%Y%m%d%H%M%S"


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", help="Start time of the range.")
    parser.add_argument("end", nargs="?", help="End time of the range. If not specified or invalid format, end is set to time to run this software.")
    parser.add_argument("--start-pressure", "-p", type=float, help="Start pressure of the range.", dest="start_pressure")
    parser.add_argument("--prefix", "-p", default="outgas_")
    parser.add_argument("--outputfile_namebase", "-f", default="outgas_plot", help="filename header of png file")
    parser.add_argument("--show", "-s", action="store_true", help="If true, figure screen will be shown.")
    parser.add_argument("--no-png", action="store_true", dest="no_png", help="If specified, png file will not be saved.")
    namespace = parser.parse_args()
    return namespace


def select_file(filelist, prefix, start_time, end_time):
    file_to_read = []
    re_pattern = prefix + "(\\d*)\\.csv"
    reg = re.compile(re_pattern)
    for file in filelist:
        res = re.fullmatch(reg, file)
        if (res is not None):
            time = datetime.datetime.strptime(res[1], DATE_FORMAT)
            if (time < end_time) and (time > start_time):
                file_to_read.append((time, file))
    file_to_read.sort(key=lambda x: x[1])
    return file_to_read


def main():
    namespace = parse_argument()
    try:
        start_time = datetime.datetime.strptime(namespace.start, DATE_FORMAT)
    except ValueError:
        start_time = datetime.datetime.now()
        print(f"Parcing end time failed. Use default value: {start_time.strftime(DATE_FORMAT)}")
    try:
        if namespace.end is None:
            end_time = datetime.datetime.now()
        else:
            end_time = datetime.datetime.strptime(namespace.end, DATE_FORMAT)
    except ValueError:
        end_time = datetime.datetime.now()
        print(f"Parcing end time failed. Use default value: {end_time.strftime(DATE_FORMAT)}")
    if namespace.start_pressure is not None:
        print(f"Start pressure: {namespace.start_pressure:0.3e} Pa")
        if (namespace.start_pressure < 0):
            raise ValueError("Start pressure must be positive.")
    path = namespace.prefix + "*.csv"
    filelist = glob.glob(path)
    file_to_read = select_file(filelist, namespace.prefix, start_time, end_time)
    x = []
    y = []
    err = []
    stoh = 3600
    for file in file_to_read:
        times, datas, errors = read_file(file[1], True, start_pressure=namespace.start_pressure)
        times_arr = np.array(times)
        datas_arr = np.array(datas)
        errors_arr = np.array(errors)
        try:
            a, a_error, b, b_error = fit_linear(times_arr, datas_arr, errors_arr)
        except Exception as e:
            print(f"Failed to fit data in {file[1]}")
            print(e)
            continue
        x.append(file[0])
        y.append(a * stoh)
        err.append(a_error * stoh)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=err, fmt="o", capsize=10, markersize=6, ecolor='black', markeredgecolor="black", color='w')
    ax.set_xlabel("time")
    ax.set_ylabel("Out Gas Rate [Pa/hour]")
    ax.set_yscale("log")
    ax.grid()
    fig.autofmt_xdate()
    save_path = namespace.outputfile_namebase + "_" + start_time.strftime(DATE_FORMAT) + "_" + end_time.strftime(DATE_FORMAT) + ".png"
    if (not namespace.no_png):
        plt.savefig(save_path)
        print(f"Saved image to {save_path}")
    if (namespace.show):
        plt.show()


if __name__ == "__main__":
    main()
