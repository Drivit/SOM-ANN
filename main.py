from som import SOM
from somplot import somplot
import numpy as np

def main():
	som_ann = SOM((2, 5, 9))

	training_set = [np.random.rand(2) for i in range(100)]

	som_ann.train(training_set, 200)

	somplot(som_ann)


if __name__ == '__main__':
	main()
