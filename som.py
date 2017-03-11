import math
import random
import numpy as np

class SOM:
	'''Self-organizing Maps Implementation (SOM)'''

	def __init__(self, args):
		'''Creates a SOM.

        Parameters:
        -----------
        args - an iterable describing the network architecture.
        e.g. (2, 4, 8) is a network that works in a 2D space with
        a map's weights of 4x8.

        NOTE: the algorithm works with an update weights function of
        ZERO, this means that only the weights of the winner neuron
        will be updated.
        '''

		# Catch the args for the SOM
		self._neurons = args[0]
		self._cols = args[1]
		self._rows = args[2]
		self._neighborhood_function = 'zero'
		
		#Catch the function name to use
		if len(args) == 4:
			self._neighborhood_function = args[3]

		#Initialize random map's weights
		self._maps_weights = []
		self._class_list = []
		
		for counter in range(self._cols*self._rows):
			self._class_list.append('class_' + str(counter))
			weight = np.random.rand(self._neurons)
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

	def _weights_adjustment(self, 
							entry,
							best_neuron, 
							learning_rate, 
							function='zero'):
		'''
		The weights adjustment applied to the network.

		Parameters
		----------
		entry: Wv to calculate the adjustment
		best_neuron: closer neuron to the entrys
		learning_rate: learning rate in iteration 'k' of the program
		function: the weight adjustment will be applied based on the function
				  of the neighborhood.

				  Values
				  ------
				  zero: only the winner neuron's weigths will be updated
				  cuadratic: the winner and the next 4 neighbors neuron's 
				  			 weights will be updated
				  hexagon: the winner and the next 6 neighbors neuron's 
				  		   weights will be updated
		'''

		if function == 'zero':
			old_weights = self._maps_weights[best_neuron]
			new_weights = old_weights+learning_rate*(np.subtract(entry, old_weights))
			self._maps_weights[best_neuron] = new_weights

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

	def train(self,
			  training_set,
			  max_epochs,
			  min_learning_rate=-1):
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
				self._weights_adjustment(entry,
										 best,
										 learning_rate,
										 self._neighborhood_function)

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
	
	"""
	def get_mapped_classes(self, entry_set):
		'''
		Function to get the mapped objects of the 'entry_set' into the SOM.

		This function will return a dictionary with the structure:
		{'class_X': n_elements}
		'''

		objects_counter = {}
		for object_class in self._class_list:
			objects_counter.update({object_class: 0})

		for entry in entry_set:
			class_name = self.map(entry)
			objects_counter[class_name] = objects_counter[class_name] + 1

		return objects_counter
	"""

if __name__ == '__main__':

	#Create and train a SOM with the XOR problem
	som_ann = SOM((2, 2, 2))

	training_set = [
		np.array([1, 0]),
		np.array([0, 1]),
		np.array([1, 1]),
		np.array([0, 0]),
	]

	converged, epochs = som_ann.train(training_set, 200)
	if not converged:
		print 'El SOM no convergio con el error minimo establecido\n'
	else:
		print 'El SOM convergio en {0} iteraciones\n'.format(epochs)

	#Mapping the values
	for entry in training_set:
		class_name = som_ann.map(entry)
		print 'El objeto {0} pertenece a la clase {1}'.format(entry, class_name)

	#Get the full mapped classes
	print ''
	elements = som_ann.get_mapped_classes()
	print elements