import matplotlib.pyplot as plt
import numpy as np

COLOR_MAP = 'jet'

def somplot(som):
    '''Plots the SOM topology reflecting neighbor distances.

    Parameters:
    -----------
    som - A trained SOM.
    '''

    # get grid corners
    x, y = np.mgrid[:som._rows + 1, :som._cols + 1]


    ref = som._maps_weights[0] # use first neuron as reference

    colors = []

    for neuron, weights in enumerate(som._maps_weights):
        # calculate distance between the reference neuron
        distance = som._euclidean_distance(ref, weights)

        # use distance as color value, should we normalize color values?
        colors.append(distance)

    colors = np.array(colors).reshape((som._rows, som._cols))

    plt.pcolormesh(x, y, colors, cmap=COLOR_MAP)
    plt.colorbar()

    plt.show()
