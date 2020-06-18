#  Author: Kay Hartmann <kg.hartma@gmail.com>

import matplotlib.pyplot as plt
import numpy as np

EEG_10_20_POSITIONS = [
    ['', '', 'Fp1', '', 'Fp2', '', ''],
    ['', 'F7', 'F3', 'Fz', 'F4', 'F8', ''],
    ['M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2'],
    ['', 'P7', 'P3', 'Pz', 'P4', 'P8', ''],
    ['', '', 'O1', '', 'O2', '', '']]


def get_sensor_pos(sensor_name, sensor_map=EEG_10_20_POSITIONS):
    sensor_pos = np.where(
        np.char.lower(np.char.array(sensor_map)) == sensor_name.lower())
    # unpack them: they are 1-dimensional arrays before
    assert len(sensor_pos[0]) == 1, (
        "there should be a position for the sensor "
        "{:s}".format(sensor_name))
    return sensor_pos[0][0], sensor_pos[1][0]


def plot_head_signals_tight(signals, signals2=None, sensor_names=None, figsize=(12, 7),
                            plot_args=None, hspace=0.35,
                            sensor_map=EEG_10_20_POSITIONS,
                            tsplot=False, sharex=True, sharey=True):
    assert sensor_names is None or len(signals) == len(sensor_names), ("need "
                                                                       "sensor names for all sensor matrices")
    assert sensor_names is not None
    if plot_args is None:
        plot_args = dict()
    figure = plt.figure(figsize=figsize)
    sensor_positions = [get_sensor_pos(name, sensor_map) for name in
                        sensor_names]
    sensor_positions = np.array(sensor_positions)  # sensors x 2(row and col)
    maxima = np.max(sensor_positions, axis=0)
    minima = np.min(sensor_positions, axis=0)
    max_row = maxima[0]
    max_col = maxima[1]
    min_row = minima[0]
    min_col = minima[1]
    rows = max_row - min_row + 1
    cols = max_col - min_col + 1
    first_ax = None
    for i in range(0, len(signals)):
        sensor_name = sensor_names[i]
        sensor_pos = sensor_positions[i]
        assert np.all(sensor_pos == get_sensor_pos(sensor_name, sensor_map))
        # Transform to flat sensor pos
        row = sensor_pos[0]
        col = sensor_pos[1]
        subplot_ind = (
                              row - min_row) * cols + col - min_col + 1  # +1 as matlab uses based indexing
        if first_ax is None:
            ax = figure.add_subplot(rows, cols, subplot_ind)
            first_ax = ax
        elif sharex is True and sharey is True:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax,
                                    sharex=first_ax)
        elif sharex is True and sharey is False:
            ax = figure.add_subplot(rows, cols, subplot_ind,
                                    sharex=first_ax)
        elif sharex is False and sharey is True:
            ax = figure.add_subplot(rows, cols, subplot_ind, sharey=first_ax)
        else:
            ax = figure.add_subplot(rows, cols, subplot_ind)

        signal = signals[i]
        if signals2 is not None:
            signal2 = signals2[i]
        if tsplot is False:
            ax.plot(signal, **plot_args)
            if signal2 is not None:
                ax.plot(signal2, **plot_args)
        else:
            x = np.arange(signal.shape[1])
            colors = []
            y_tmp = signal.mean(axis=0)
            tube_tmp = signal.std(axis=0)
            p = ax.fill_between(x, y_tmp + tube_tmp, y_tmp - tube_tmp, alpha=0.5)
            colors.append(p._original_facecolor)
            if signals2 is not None:
                y_tmp2 = signal2.mean(axis=0)
                tube_tmp2 = signal2.std(axis=0)
                p = ax.fill_between(x, y_tmp2 + tube_tmp2, y_tmp2 - tube_tmp2, alpha=0.5)
                colors.append(p._original_facecolor)

            ax.plot(x, y_tmp, lw=2, color=colors[0])
            if signals2 is not None:
                ax.plot(x, y_tmp2, lw=2, color=colors[1])

        ax.set_title(sensor_name)
        ax.set_yticks([])
        if len(signal) == 600:
            ax.set_xticks([150, 300, 450])
            ax.set_xticklabels([])
        else:
            ax.set_xticks([])

        ax.xaxis.grid(True)
        # make line at zero
        ax.axhline(y=0, ls=':', color="grey")
        figure.subplots_adjust(hspace=hspace)
    return figure
