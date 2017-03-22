from som import SOM
from somplot import somplot
import numpy as np
from openpyxl import load_workbook

def main():
	som_ann = SOM((28, 2, 2))

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
	training_set = np.array([[i.value for i in j] for j in workspace])

	#Train network
	print 'Wait, I\'m traning...'
	som_ann.train(training_set, 200)

	somplot(som_ann)


if __name__ == '__main__':
	main()
