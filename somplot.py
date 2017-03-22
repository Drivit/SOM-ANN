import matplotlib.pyplot as plt
import numpy as np

COLOR_MAP = 'jet'

_som = None
_training_set = None

def somplot(som, training_set):
    '''Plots the SOM topology reflecting neighbor distances.

    Parameters:
    -----------
    som - A trained SOM.
    '''
    global _som, _training_set

    _som = som
    _training_set = training_set

    _plot_colors()
    _plot_mapped()

    plt.show()

def _plot_colors():
    rows = _som._cols
    cols = _som._rows

    # get grid dimensions
    Y, X = np.mgrid[:rows + 1, :cols + 1]

    # this is bad, will work only with 1-D inputs
    colors = [i[0] for i in _som._maps_weights]
    colors = np.array(colors).reshape((rows, cols))

    plt.pcolormesh(X, Y, colors, cmap=COLOR_MAP)
    plt.colorbar()

def _plot_mapped():
    results = _calculate_maps()

    for n, neuron_map in enumerate(results):
        y, x = _som._vector_to_matrix_point(n)

        for output in neuron_map:
            s = '{}: {}'.format(output, neuron_map[output])
            plt.text(x, y, s, fontsize=12)
            y += 0.1

def _calculate_maps():
    maps = [{} for n in _som._maps_weights]

    for inputs, output in _training_set:
        neuron = _som.map(inputs, type='v_index')
        neuron_map = maps[neuron]

        if output in neuron_map:
            neuron_map[output] += 1
        else:
            neuron_map[output] = 1

    return maps
