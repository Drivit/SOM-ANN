import numpy as np
from som import SOM

def main():
	'''
	Create and train a SOM with the XOR problem

	SOM settings
	------------
	The SOM used in this test works in a 2D space with a 
	map's weights of 2x2

	Train settings
	---------------
	The SOM will be trained with the XOR problem, this training
	will run in 2000 iterations. In the weights adjustment function, a 
	zero neighborhood radius, that mean that only the winner neuron's 
	weights will be updated.
	'''
	som_ann = SOM((2, 2, 2))

	training_set = [
		np.array([1, 0]),
		np.array([0, 1]),
		np.array([1, 1]),
		np.array([0, 0]),
	]

	#Train network
	converged, epochs = som_ann.train(training_set, 2000, -1, 'zero')
	if not converged:
		print 'SOM didn\'t converge with the minimum learning rate\n'
	else:
		print 'SOM converged in {0} epochs\n'.format(epochs)

	#Mapping values
	for entry in training_set:
		class_name = som_ann.map(entry)
		print 'The object {0} belongs to class: {1}'.format(entry, class_name)

	#Get the full mapped classes
	print ''
	elements = som_ann.get_mapped_classes()
	print elements


if __name__ == '__main__':
	main()	