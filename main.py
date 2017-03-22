from som import SOM
from somplot import somplot
import numpy as np
from openpyxl import load_workbook

def main():
	#Load .xls .xlsx file
	file_name = raw_input('File name: ')
	work_book = load_workbook(file_name)

	#Load sheet
	print (work_book.get_sheet_names())
	sheet_name = raw_input('\nSheet name: ')
	work_sheet = work_book.get_sheet_by_name(sheet_name)

	#Select workspace in sheet
	begin_position = raw_input('Begin position: ')
	end_position = raw_input('End position: ')
	workspace = work_sheet[begin_position : end_position] #Workspace(begin, end)

	#Load data
	training_set = []
	for row in workspace:
		class_name = row[0].value # first column is class name

		for item in row[1:]: # skip first column
			inputs = np.array([item.value])
			training_set.append((inputs, class_name))

	#Create SOM ANN
	som_ann = SOM((1, 4, 4))

	#Train network
	print 'Wait, I\'m traning...'
	som_ann.train([i[0] for i in training_set], 200, -1, 'quadratic')

	somplot(som_ann, training_set)


if __name__ == '__main__':
	main()
