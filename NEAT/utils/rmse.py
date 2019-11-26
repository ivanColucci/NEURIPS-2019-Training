import matplotlib.pyplot as plt
import numpy as np


def linear(start, value, interval, neg=False):
    a = []
    uptime = round(interval*2/3)
    step = value/uptime
    for i in range(interval):
        a.append(start)
        if i < uptime:
            if neg:
                start -= step
            else:
                start += step
    return a

def get_foot_time_function(steps, interval, value, start_left, start_right):
    left = False
    left_value = start_left
    right_value = start_right
    a = []
    b = []
    first = True
    for i in range(round(steps/interval)):
        left = not left
        if left:
            a.extend(linear(left_value, value, interval))
            for j in range(interval):
                b.append(right_value)
            left_value += value
        else:
            for j in range(interval):
                a.append(left_value)
            b.extend(linear(right_value, value, interval))
            right_value += value
        if first:
            value = value*2
            first = False

    return a[:steps], b[:steps], rmse(np.zeros(len(a[:steps])), a[:steps])+rmse(np.zeros(len(b[:steps])), b[:steps])

def get_foot_time_function2(steps, interval, value):
    start = 0
    a = []
    neg = True
    # for i in range(10):
    #     a.append(start)
    for i in range(round(steps/interval)):
        a.extend(linear(start, value, interval, neg))
        if neg:
            start -= value
        else:
            start += value
        for j in range(interval):
            a.append(start)
        neg = not neg
        a.extend(linear(start, value, interval, neg))
        if neg:
            start -= value
        else:
            start += value

    return a[:steps], rmse(np.zeros(len(a[:steps])), a[:steps])


def rmse(real, expected):
    if len(expected) < len(real):
        real = np.array(real)[-len(expected):]
    else:
        real = np.append(real, np.zeros(len(expected)-len(real)))
        expected = np.array(expected)
    diff = expected-real
    if len(diff) > 0:
        return np.sqrt(((diff) ** 2).mean())
    else:
        return -1


def array_diff(x, y):
    a = []
    for i in range(len(x)):
        a.append(x[i]-y[i])
    return a


def calculate_sin(x):
    peaks_value = []
    peaks_time = []
    possible_peak = dict()
    possible_peak['value'] = 0
    possible_peak['time'] = 0
    signs = np.sign(x)
    first_sign = 1
    for i in range(len(x)-1):
        #aggiungi condizione per lo startup
        if signs[i] != signs[i+1]:
            # print("zero!")
            peaks_value.append(abs(possible_peak['value']))
            if len(peaks_time) > 0:
                peaks_time.append(possible_peak['time'] - sum(peaks_time))
            else:
                peaks_time.append(possible_peak['time'])
                first_sign = signs[possible_peak['time']]
            # print("Picco trovato: ({}, {})".format(possible_peak, i))
            possible_peak['value'] = 0
            possible_peak['time'] = 0
        if signs[i] == 1 and possible_peak['value'] < x[i] or signs[i] == -1 and possible_peak['value'] > x[i]:
            possible_peak['value'] = x[i]
            possible_peak['time'] = i

    if len(peaks_time) > 0:
        peaks_value.append(abs(possible_peak['value']))
        peaks_time.append(possible_peak['time'] - sum(peaks_time))
        #version 0
        # useless_values = x[:peaks_time[0]]
        #version 1
        first_peak_time = peaks_time[0]
        interval = np.mean(peaks_time)
        if round(interval) == 0:
            interval = 1
        value = np.mean(peaks_value)
        max_value = max(peaks_value)
    else:
        interval = 1
        value = 0
        first_peak_time = 0
        max_value = 1
    return get_sin_function(len(x), first_peak_time, interval, value, first_sign, max_value)


def get_sin_function(steps, first_peak_time, interval, value, sign, max_value):
    if value == 0:
        return [], 0
    if value < 0.5:
        value = 0.5
    a = []
    temp = first_peak_time/(2*interval)
    offset = (temp-np.floor(temp))*2*interval
    for i in range(steps):
        a.append(sign*value*np.sin(np.pi/interval*i+np.pi/2-np.pi/interval*offset))
    b = [max_value for i in range(len(a[:steps]))]
    return a[:steps], rmse(b, a[:steps])


# a, error = get_sin_function(1000, 200, 200, 0.5, -1, 0.5)
# plt.title("Error: "+str(error))
# plt.plot(a, label="expected")
# b = [0.5 for i in range(len(a))]
# plt.plot(b, label="real")
# plt.ylabel('distance')
# plt.xlabel('#steps')
# plt.legend()
# plt.grid()
# plt.show()
