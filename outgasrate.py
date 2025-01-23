#! /usr/bin/env python3
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from GL840.MongoDBHandler import MongoDBPuller
import datetime
from typing import Any, Callable
from time import sleep
import argparse
import os
from argparse import RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter


def calcchi(params, model_func, xvalues, yvalues, yerrors):
    model = model_func(xvalues, yvalues, params)
    chi = (yvalues - model) / yerrors
    return (chi)


def solve_leastsq(xvalues, yvalues, yerrors, param_init, model_func):
    param_result, covar_output, _, _, _ = leastsq(
        calcchi,
        param_init,
        args=(model_func, xvalues, yvalues, yerrors),
        full_output=True)
    error_result = np.sqrt(np.diag(covar_output))
    dof = len(xvalues) - 1 - len(param_init)
    chi2 = np.sum(np.power(calcchi(param_result, model_func, xvalues, yvalues, yerrors), 2.0))
    return ([param_result, error_result, chi2, dof])


def mymodel(xvalues, yvalues, params):
    assert type(xvalues) is np.ndarray, "xvalues must be numpy.ndarray"
    p0, p1 = params
    model_y = p0 * xvalues + p1
    return model_y


def conversionMPT200(data: Any) -> Any:
    return 10**(1.667 * data - 9.333)


def fetchData(data_name: str, func: Callable = lambda x: x, ip: str = "192.168.160.22", port=None) -> None | tuple[float | int, float]:
    puller = MongoDBPuller(ip, port)
    data = puller.pull_one("GL840", "GL840")
    if data is None:
        return None
    unixtime = data.unixtime
    ch_data = func(data.sections["GL840"].contents[data_name])
    return unixtime, ch_data


class TimeKeeper():
    def __init__(self, duration: datetime.timedelta) -> None:
        self.starttime: datetime.datetime = datetime.datetime.now()
        self.duration = duration

    def start(self) -> None:
        self.starttime = datetime.datetime.now()

    def print_starttime(self) -> None:
        print(f'Start Time: {self.starttime.strftime("%Y/%m/%d %H:%M:%S")}')

    def check(self) -> bool:
        if (datetime.datetime.now() - self.starttime) > self.duration:
            print(f"End Time: {self.starttime.strftime("%Y/%m/%d %H:%M:%S")}")
            return True
        return False


def MPT200Error(value: float) -> float:
    '''
    MPT200Error
    -

    Error of MPT200 (in case of N2)

    Parameters
    --

    Value: float
        Measurement value of MPT200


    Returns
    --

    Error: float
        Error of MPT200 (Half width)

    '''
    if (value > 1000.0 and value < 100000.0):
        return value * 0.3
    elif (value < 1000.0 and value > 2e-3):
        return value * 0.1
    elif (value < 2e-3 and value > 1e-8):
        return value * 0.25
    else:
        raise ValueError("Pressure value is incorrect, so cant estimate error.")


class MyHelper(RawTextHelpFormatter, RawDescriptionHelpFormatter, ArgumentDefaultsHelpFormatter):
    pass


def read_file(filename, use_auto_error, error_fix=0, start_pressure=None):
    with open(filename, "r") as f:
        print(f"CSV file read: {filename}")
        lines = f.readlines()
        times = []
        datas = []
        errors = []
        for line in lines[1:]:
            time, data = line.strip().split(",")
            if (start_pressure is not None and float(data) < start_pressure):
                continue
            times.append(float(time))
            datas.append(float(data))
            err = error_fix
            if (use_auto_error):
                err = MPT200Error(float(data))
            errors.append(err)
    return times, datas, errors


def fit_linear(times_arr, datas_arr, errors_arr, init_params=np.array([0, 100])):
    try:
        res = solve_leastsq(times_arr, datas_arr, errors_arr, init_params, mymodel)
    except Exception as e:
        print(f"Fit Error: {e}")
        raise e
    result, error, chi2, dof = res
    a = result[0]
    a_error = error[0]
    b = result[1]
    b_error = error[1]
    return a, a_error, b, b_error


if __name__ == "__main__":
    use_auto_error = False
    parser = argparse.ArgumentParser(formatter_class=MyHelper)
    parser.add_argument("--duration", "-d", default=180, type=int, help="duration of measurement [s]")
    parser.add_argument("--error-type", "-t", default="auto", choices=['auto', "constant", "const"], help="Error type(auto: use specification of MPT200, constant or const: use constant error, see --error)", dest="error_type")
    parser.add_argument("--error", "-e", default=0.05, type=float, help="constant error of measurement [Pa]")
    parser.add_argument("--filename", "-f", default=f"data/outgas_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}", help="filename header of csv file")
    parser.add_argument("--start-pressure", "-p", type=float, help="Start pressure of measurement [Pa]", dest="start_pressure")
    parser.add_argument("--show", "-s", action="store_true", help="After the measurement, figure screen will be shown if true.")
    parser.add_argument("--read", "-r", help="read mode", type=str)
    parser.add_argument("--display-number", help="How many data points are plotted in output figure. If 0, all data points are plotted.", default=10, type=int, dest="display_number")
    parser.add_argument("--dataname", default="Ch2", help="Name of data")
    parser.add_argument("--no-png", action="store_true", dest="no_png", help="If specified, png file will not be saved.")
    namespace = parser.parse_args()
    if (namespace.duration <= 0):
        raise ValueError("duration must be positive")
    if (namespace.error <= 0):
        raise ValueError("error must be positive")
    if (namespace.error_type == "auto"):
        use_auto_error = True
    elif (namespace.error_type == "constant" or namespace.error_type == "const"):
        use_auto_error = False
    else:
        raise ValueError(f"Unknown error type: {namespace.error_type}")
    if (namespace.display_number < 0):
        raise ValueError("display period must be non-negative.")
    if (namespace.filename == ""):
        raise ValueError("filename must be set")
    if (namespace.no_png):
        print("Png file will NOT be saved.")
    if (namespace.show is True):
        print("plt.show() will be called.")
    if (namespace.start_pressure is not None):
        print(f"Start pressure: {namespace.start_pressure} Pa")
        if (namespace.start_pressure < 0):
            raise ValueError("Start pressure must be positive.")
    if (namespace.read is not None):
        print("Read mode")
        read_file(namespace.read, use_auto_error, namespace.error, namespace.start_pressure)
    else:
        timekeeper = TimeKeeper(datetime.timedelta(0, namespace.duration))
        timekeeper.start()
        times = []
        errors = []
        datas = []
        while (not timekeeper.check()):
            try:
                tup = fetchData(namespace.dataname, conversionMPT200, ip="192.168.1.30")
                if tup is None:
                    print("Data is None. Retry after 2.0s")
                    sleep(2.0)
                    continue
                if (namespace.start_pressure is not None and tup[1] < namespace.start_pressure):
                    print(f"Pressure ({tup[1]:0.3e} Pa) is lower than start pressure.")
                    sleep(2.0)
                    continue
                if len(times) == 0:
                    print("---------------Out Gas Rate Measurement Start!!---------------")
                    timekeeper.print_starttime()
                    with open(namespace.filename + ".csv", "w") as f:
                        print(f"Time,{namespace.dataname}", file=f)
                        print(f"CSV file created: {namespace.filename}.csv")
                time = tup[0]
                data = tup[1]
                print(f"Time:{datetime.datetime.fromtimestamp(time).strftime("%Y/%m/%d %H:%M:%S")} data:{data:0.3e}")
                with open(namespace.filename + ".csv", "a") as f:
                    print(f"{time},{data}", file=f)
                times.append(time)
                datas.append(data)
                err = namespace.error
                if (use_auto_error):
                    err = MPT200Error(float(data))
                errors.append(err)
                sleep(2.0)
            except KeyboardInterrupt:
                length_min = min(len(times), len(datas), len(errors))
                if (len(datas) != length_min):
                    print("Some data were removed")
                    sz = len(datas)
                    for i in range(sz - length_min):
                        datas.pop()
                if (len(times) != length_min):
                    print("Some time data were removed")
                    sz = len(times)
                    for i in range(sz - length_min):
                        times.pop()
                if (len(errors) != length_min):
                    print("Some error data were removed")
                    sz = len(errors)
                    for i in range(sz - length_min):
                        errors.pop()
                break

        print("----------------------Stop Measurement!!----------------------")
    times_arr = np.array(times)
    datas_arr = np.array(datas)
    errors_arr = np.array(errors)
    init_params = np.array([0, 100])
    fit_success = True
    try:
        a, a_error, b, b_error = fit_linear(times_arr, datas_arr, errors_arr, init_params)
    except Exception as e:
        print(f"Fit Error: {e}")
        fit_success = False
    stoh = 3600
    time_datetime = [datetime.datetime.fromtimestamp(x) for x in times_arr]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if fit_success:
        ax.plot(time_datetime, a * times_arr + b, "-", color="r", label="slope : {:.4} ± {:.2} Pa/h".format(a * stoh, a_error * stoh), )
        ax.legend()
    skip_num = 1
    sz_time = len(times)
    if (namespace.display_number == 0):
        skip_num = 1
    elif (sz_time > namespace.display_number):
        skip_num = len(times_arr) // namespace.display_number
    ax.errorbar(time_datetime[::skip_num], datas_arr[::skip_num], yerr=errors_arr[::skip_num], fmt="o", capsize=10, markersize=6, ecolor='black', markeredgecolor="black", color='w')
    ax.set_xlabel("time [s]")
    ax.set_ylabel("inner pressure [Pa]")
    fig.autofmt_xdate()
    save_path = namespace.filename + ".png"
    if (not namespace.no_png):
        plt.savefig(save_path)
        print(f"Saved image to {save_path}")
    if fit_success:
        print(f"\nResult: Outgas Rate = {a * stoh:.4} ± {a_error * stoh:.2} Pa/h")
    os.system('afplay /System/Library/Sounds/Ping.aiff')
    if (namespace.show):
        plt.show()
