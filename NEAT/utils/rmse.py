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


def calculate_sin(x, params_only=False):
    peaks_value = []
    peaks_time = []
    info = dict()
    possible_peak = dict()
    possible_peak['value'] = 0
    possible_peak['time'] = 0
    signs = np.sign(x)
    first_sign = 1
    for i in range(len(x)-1):
        if signs[i] != signs[i+1]:
            peaks_value.append(abs(possible_peak['value']))
            if len(peaks_time) > 0:
                peaks_time.append(possible_peak['time'] - sum(peaks_time))
            else:
                peaks_time.append(possible_peak['time'])
                first_sign = signs[possible_peak['time']]
            possible_peak['value'] = 0
            possible_peak['time'] = 0
        if signs[i] == 1 and possible_peak['value'] < x[i] or signs[i] == -1 and possible_peak['value'] > x[i]:
            possible_peak['value'] = x[i]
            possible_peak['time'] = i

    if len(peaks_time) > 0:
        peaks_value.append(abs(possible_peak['value']))
        peaks_time.append(possible_peak['time'] - sum(peaks_time))
        first_peak_time = peaks_time[0]
        info['interval'] = np.mean(peaks_time)
        if round(info['interval']) == 0:
            info['interval'] = 1
        info['value'] = np.mean(peaks_value)
        info['max_value'] = max(peaks_value)
        info.update(mean_peak_variance(peaks_value))
        temp = first_peak_time / (2 * info['interval'])
        info['phase_shift'] = -2 * np.pi * (temp - np.floor(temp))
        if first_sign == -1:
            info['phase_shift'] += np.pi
    else:
        info['interval'] = 1
        info['value'] = 0
        info['max_value'] = 1
        info['phase_shift'] = 0
    if params_only:
        return info
    else:
        return get_sin_function(len(x), info)


def get_sin_function(steps, param):
    if param['value'] == 0:
        return [], 0
    a = []
    for i in range(steps):
        a.append(param['value']*np.sin(np.pi/param['interval']*i+np.pi/2+param['phase_shift']))
    b = [param['max_value'] for _ in range(len(a[:steps]))]
    return a[:steps], rmse(b, a[:steps])

def mean_peak_variance(peaks_value):
    top = []
    down = []
    info = dict()
    for i in range(len(peaks_value)):
        if i % 2 == 0:
            top.append(peaks_value[i])
        else:
            down.append(peaks_value[i])
    info['mean+'] = np.mean(np.array(top))
    info['mean-'] = np.mean(np.array(down))
    info['var+'] = np.std(np.array(top))
    info['var-'] = np.std(np.array(down))

    return info

if __name__ == '__main__':
    temp = 600 / (2 * 200)
    a, error = get_sin_function(1000, {'max_value': 0.5, 'interval': 200,
                                       'value': 0.5, 'phase_shift': -2*np.pi*(temp - np.floor(temp))})
    plt.title("Error: "+str(error))
    plt.plot(a, label="expected")
    b = [0.5 for i in range(len(a))]
    plt.plot(b, label="real")
    plt.ylabel('distance')
    plt.xlabel('#steps')
    plt.legend()
    plt.grid()
    plt.show()
