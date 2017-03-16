import math
import random
import numpy as np
import matplotlib.pyplot as plt

class SOM:
	'''Self-organizing Maps Implementation (SOM)'''

	def __init__(self, args):
		'''Creates a SOM.

        Parameters:
        -----------
        args - an iterable describing the network architecture.
        e.g. (2, 4, 8) is a network that works in a 2D space with
        a map's weights of 4x8.
        '''

		# Catch the args for the SOM
		self._dimension = args[0]
		self._cols = args[1]
		self._rows = args[2]

		#Initialize random map's weights
		self._maps_weights = []
		self._class_list = []
		
		for counter in range(self._cols*self._rows):
			self._class_list.append('class_' + str(counter))
			weight = np.random.rand(self._dimension)
			self._maps_weights.append(weight)


	def _euclidean_distance(self, vector_1, vector_2):
		result = 0.0
		for x in range(len(vector_1)):
			result += (vector_1[x] - vector_2[x])**2

		return math.sqrt(result)

	def _calculate_sinaptic_potential(self, entry):
		'''
		Function to calculate the sinaptic potential from an entry
		to all the neurons in the network.
		'''
		temp_distance = []

		for neuron in self._maps_weights:
			distance = self._euclidean_distance(entry, neuron)
			temp_distance.append(distance)

		return temp_distance

	def _get_best_nueron(self, distances_vector):
		min_value = min(distances_vector)
		min_index = distances_vector.index(min_value)
		return min_index

	def _calculate_neuron_position(self, neuron):
		'''
		Function to get the position of a neuron in the map's weights.
		'''
		neuron_position = -1

		for i in range(len(self._maps_weights)):
			if np.array_equal(self._maps_weights[i], neuron):
				neuron_position = i
				break

		return neuron_position

	def _vector_to_matrix_point(self, neuron_position):
		'''
		Function to transform index position to matrix point.
		'''
		neuron_col = neuron_position / self._cols
		neuron_row = neuron_position % self._rows

		return neuron_col, neuron_row

	def _matrix_to_vector_position(self, col, row):
		'''
		Function to transform matrix point to vector position.
		'''
		return col*self._cols + row

	def _calculate_neighbours(self, neuron, type='vector', radius=1):
		'''
		This function will return a vector with the position of the neuron's
		neighbours.

		Parameters
		----------
		neuron: neuron from the map's weights
		radius: radius of the neighborhood
		type: type of position that will be return
			
			Values
			------
			vector: the positions are vector index
			matrix: the positions are in matrix coordinates
			both: return both kind of positions

		'''
		neuron_col, neuron_row = self._vector_to_matrix_point(neuron)

		neighbours_matrix = []
		neighbours_index = []
		
		#Calculate neighbours in the radius
		for i in xrange(1, radius+1):
			#Up
			if (neuron_row) - i >= 0:
				neighbours_matrix.append((neuron_col, neuron_row-i))
			#Down
			if (neuron_row) + i < self._rows:
				neighbours_matrix.append((neuron_col, neuron_row+i))
			#Left
			if (neuron_col) - i >= 0:
				neighbours_matrix.append((neuron_col-i, neuron_row))
			#Right
			if (neuron_col) + i < self._cols:
				neighbours_matrix.append((neuron_col+i, neuron_row))

		#Calculate the vector index for each neighbour
		for neighbour in neighbours_matrix:
			vector_index = self._matrix_to_vector_position(neighbour[0], neighbour[1])
			neighbours_index.append(vector_index)

		if type == 'vector':
			return neighbours_index
		elif type == 'matrix':
			return neighbours_matrix
		elif type == 'both':
			return neighbours_index, neighbours_matrix


	def _weights_adjustment(self, 
							entry,
							best_neuron, 
							learning_rate, 
							function='zero',
							radius=1):
		'''
		The weights adjustment applied to the network.

		Parameters
		----------
		entry: Wv to calculate the adjustment
		best_neuron: closer neuron to the entry
		learning_rate: learning rate in iteration 'k' of the program
		function: the weight adjustment will be applied based on the function
				  of the neighborhood.

				  Values
				  ------
				  zero: only the winner neuron's weigths will be updated
				  quadratic: the winner and the next 4 neighbours_matrix neuron's 
				  			 weights will be updated
				  hexagon: the winner and the next 6 neighbours_matrix neuron's 
				  		   weights will be updated
		'''

		#Calculate new weights for winner neuron
		old_weights = self._maps_weights[best_neuron]
		new_weights = old_weights+learning_rate*(np.subtract(entry, old_weights))
		self._maps_weights[best_neuron] = new_weights

		
		#Calculate weights for winner neuron's neighbours
		if function == 'quadratic':
			neighbours = self._calculate_neighbours(best_neuron)

			for neighbour in neighbours:
				old_weights = self._maps_weights[neighbour]
				new_weights = old_weights+learning_rate*(np.subtract(entry, old_weights))
				self._maps_weights[neighbour] = new_weights
		elif function == 'hexagon':
			pass

		#TODO: Add rest of the adjustment functions

	def _determine_learning_rate(self, 
								 max_epochs, 
								 iteration, 
								 function='default'):
		'''
		Calculate the learning rate in function of the iterations in the program.

		Parameters
		----------
		max_epochs: max epochs in the program
		iteration: actual iteration in the program
		function: funtion that will be used to calculate the learning rate
			Values
			------
			'default' : 1-(k/T)
			'euclidean': e^-(k/T)
		'''
		if function == 'default':
			return (1-(float(iteration)/float(max_epochs))) #Default learning rate
		elif function == 'euclidean':
			return math.exp(-(float(iteration)/float(max_epochs))) #Exponential learning rate

	def _determine_neighborhood_radius(self, 
									   max_epochs,
									   iteration):
		'''
		Function to calculate a neighborhood radius based on
		the number of iterations executed. 

		NOTE: Max radius is 5
		'''
		return (5-(iteration*6/max_epochs))

	def train(self,
			  training_set,
			  max_epochs,
			  min_learning_rate=-1,
			  neighborhood_function='zero'):
		'''
		Non-supervised training in the SOM

		Parameters
		----------
		training_set: vector with N entries of M-Dimension
		max_epochs: max epochs to be simulated
		min_learning_rate: min learning rate to stop the learning phase, if the
						   value is -1, the training phase will stop when the
						   'max_epochs' are reached
		'''
		self._training_set = training_set
		epochs_converged = max_epochs

		if min_learning_rate == -1:
			converged = True
		else:
			converged = False

		for iteration in range(max_epochs):
			for entry in training_set:
				distance_vector = self._calculate_sinaptic_potential(entry)
				best = self._get_best_nueron(distance_vector)
				learning_rate = self._determine_learning_rate(max_epochs, iteration)
				neighborhood_radius = self._determine_neighborhood_radius(max_epochs, 
																		  iteration)
				self._weights_adjustment(entry,
										 best,
										 learning_rate,
										 neighborhood_function)

				if min_learning_rate != -1:
					if learning_rate <= min_learning_rate:
						converged = True
						epochs_converged = iteration+1
						return (converged, epochs_converged)
		
		return (converged, epochs_converged)

	def map(self, entry):
		'''
		Function to mapping an entry in the SOM.

		Parameters
		----------
		entry: vector of M-Dimension to be mapped
		'''
		distance = self._calculate_sinaptic_potential(entry)
		best = self._get_best_nueron(distance)
		class_name = self._class_list[best]

		return class_name

	def get_mapped_classes(self):
		'''
		Function to get the mapped objects of the 'entry_set' into the SOM.

		This function will return a dictionary with the structure:
		{'class_X': n_elements}
		'''

		objects_counter = {}
		for object_class in self._class_list:
			objects_counter.update({object_class: 0})

		for entry in self._training_set:
			class_name = self.map(entry)
			objects_counter[class_name] = objects_counter[class_name] + 1

		return objects_counter