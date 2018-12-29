"""
Copyright 2018 LIN Lu (ncble)
"""
# _*_ coding: utf-8 _*_
import os, glob
import numpy as np 
from time import time
from numpy import linalg as LA
from operator import itemgetter #, attrgetter # To sort a list
from tqdm import tqdm # Progress bar
# import itertools # Construct "S^d" using k-combinaisons of {1,...,d+1}
from collections import defaultdict, Counter # Construct dictionary with default value
import matplotlib.pyplot as plt


class Simplex(object):
	"""docstring for Simplex"""
	def __init__(self, val, dim, vert):
		self.val = val
		self.dim = dim
		self.vert = vert

class sparse_matrix(object):
	"""docstring for sparse_matrix"""
	def __init__(self):
		self.dict_col2row_set = defaultdict(lambda:set({}))
	def toy_example(self):
		# For debug purpose
		self.row = [0,0,1,2,2]
		self.col = [0,2,2,0,1]
		self.val = [1,1,1,1,1]
		self.nb_element = len(self.row)
	def sample_example(self):
		# For debug purpose
		self.row = [0,0,0,1,1,2,2,2,5,5,7,8,9]
		self.col = [3,7,8,3,4,4,6,8,6,7,9,9,9]
		self.val = [1,1,1,1,1,1,1,1,1,1,1,1,1]
		self.nb_element = len(self.row)

	def build_matrix_for_sanity_check(self, load_example_path = ""):

		A = np.loadtxt(open(load_example_path, "rb"), delimiter=" ")
		n, m = A.shape
		assert n == m
		for j in range(n):
			for i in range(n):
				if A[i, j] != 0:
					self.dict_col2row_set[j].add(i)
		return n

	def to_array_from_dict_set(self, N):
		""" For sanity check purpose """
		A = np.zeros((N,N))

		for key in self.dict_col2row_set:
			for row_index in self.dict_col2row_set[key]:
				A[row_index, key] = 1
		return A

	def to_array(self, N):
		A = np.zeros((N,N))
		for i in range(self.nb_element):
			A[self.row[i], self.col[i]] = self.val[i]

		return A

	def to_array_from_dict(self, N):
		A = np.zeros((N,N))
		for key in iter(self.dict_col2row):
			if -1 in self.dict_col2row[key]:
				pass
			else:
				A[self.dict_col2row[key], key] = 1

		# for i in range(self.nb_element):
		# 	A[self.row[i], self.col[i]] = self.val[i]

		return A
	

	def add_col_i_by_j_opt2_4(self, i, j, N, tempoI, new = False): # 90 times better ! 
		assert i != j
		
		
		# if new:
		# 	tempoI = np.zeros(N, dtype=bool)
		if new:
			tempoI[self.dict_col2row[i]] = 1
		tempoI[self.dict_col2row[j]] = (tempoI[self.dict_col2row[j]]+int(1)) %2
		
		
		if np.any(tempoI):
			self.dict_col2row[i] = np.arange(N)[tempoI]
			max_non_null_row_index_of_col_i = np.max(self.dict_col2row[i])
			
		else:
			# self.dict_col2row[i] = np.array([]) No change to self.dict_col2row[i]
			max_non_null_row_index_of_col_i = -1
		return max_non_null_row_index_of_col_i

	
	def add_col_i_by_j_opt5(self, i, j):
		assert i != j
		# import ipdb; ipdb.set_trace()
		self.dict_col2row_set[i].symmetric_difference_update(self.dict_col2row_set[j])
		# tempoI[self.dict_col2row[i]] = 1
		# tempoI[self.dict_col2row[j]] = (tempoI[self.dict_col2row[j]]+int(1)) %2
		# print(self.dict_col2row_set[i])
		if self.dict_col2row_set[i]:
			max_non_null_row_index_of_col_i = max(self.dict_col2row_set[i])
		# if np.any(tempoI):
		# 	self.dict_col2row[i] = np.arange(N)[tempoI]
		# 	max_non_null_row_index_of_col_i = np.max(self.dict_col2row[i])
			
		else:
			# self.dict_col2row[i] = np.array([]) No change to self.dict_col2row[i]
			max_non_null_row_index_of_col_i = -1
		return max_non_null_row_index_of_col_i




class Persistent_homology(object):
	"""docstring for parse"""
	def __init__(self):
		
		self.data = []
		self.N = 0 # self.N = len(self.data)
		self.bmatrix_sparse = sparse_matrix()#None # = 
		self.max_non_null_index = None
		self.pivot = defaultdict(lambda:-1) # {row_index : column_index}
		
		self.dict_hash_vertice = {}

	def load_data(self, path):
		print("Loading/parsing data...")

		with open(path, "rb") as file:
			for line in tqdm(file):
				line = line.split()
				self.data.append(Simplex(float(line[0]), int(line[1]), map(int,line[2:])))
				
		self.N = len(self.data)
		print("Done.")

	def build_Rips_filtration(self, filepath, truncate= np.arange(0,100)):
		A = np.loadtxt(open(filepath, "rb"), delimiter= " ")
		A = A[truncate, :] # first 'truncate' points
		
		row = len(A)
		distance_matrix = np.zeros((row,row))
		
		count = 0
		print("Calculation distance matrix...")
		for i in xrange(row):
			for j in xrange(i+1, row):
				distance_matrix[i, j] = LA.norm(A[i,:]-A[j,:])
				distance_matrix[j, i] = distance_matrix[i, j]
				# all_dist.append(distance_matrix[i, j])
				# get_ij_from_index[count] = (i,j)
				count += 1
		print("Done.")

		filename = filepath.split("/")[-1].split(".")[0]
		print("Building Rips filtration...")
		for i in xrange(row):
			# print distance_matrix[i,i], 0, i
			self.data.append(Simplex(distance_matrix[i,i], 0, [i]))
			for j in xrange(i+1, row):
				# print distance_matrix[i,j], 1, i, j
				self.data.append(Simplex(distance_matrix[i,j], 1, [i, j]))
				for k in xrange(j+1, row):
					# print np.max((distance_matrix[i,j], distance_matrix[i,k], distance_matrix[j,k])), 2, i, j, k
					dist_max = np.max((distance_matrix[i,j], distance_matrix[i,k], distance_matrix[j,k]))
					self.data.append(Simplex(dist_max, 2, [i, j, k]))
		self.N = len(self.data)
		print("Done.")

		return filename


	def set_data(self, new_data):
		self.data = new_data
		self.N = len(self.data)

	def print_data(self):
		for sim in self.data:
			
			print("{val="+str(sim.val)+"; dim="+str(sim.dim)+"; "+str(sim.vert)+"}\n")
	def sort_by_lexicographic(self):
		print("Sorting data by lexicographic (filtration value, dim)")
		# Way 1
		self.data.sort(key=lambda x: (x.val, x.dim)) # , reverse=True


		# Way 2
		# newlist = sorted(self.data, key=attrgetter('val'))
		# self.data = newlist
		print("Done.")
	def initialize_dict_for_sparse_matrix(self):
		print("Initializing dictionary dim-vertice/hash-dict for sparse matrix use...")
		self.dict_hash_vertice[hash(frozenset([]))] = -1
		for i in xrange(self.N):
			simplex = self.data[i]
			self.dict_hash_vertice[hash(frozenset(simplex.vert))] = i # NEW LU TODO
		print("Done.")

	def initialize_for_sanity_check(self, path):
		# A = np.loadtxt(open(path, "rb"), delimiter= " ")
		# n = len(A)
		
		# self.bmatrix_sparse = sparse_matrix()
		n = self.bmatrix_sparse.build_matrix_for_sanity_check(load_example_path = path)
		A = self.bmatrix_sparse.to_array_from_dict_set(n)
		self.N = n
		self.max_non_null_index = np.zeros(n)-1
		for j in range(n):
			self.max_non_null_index[j] = A[:,j].cumsum().argmax()
		return A

	def calculate_boundary_matrix_sparse(self):
		print("Calculating boundary matrix (sparse)...")
		# Maximum non null row index of each column
		self.max_non_null_index = np.zeros(self.N)-1

		# self.bmatrix_sparse = sparse_matrix()

		for i in xrange(self.N):
			current_simplex = self.data[i]
			current_dim = current_simplex.dim
			vertice_list = current_simplex.vert
			max_non_null_row_index_of_col_i = -1
			boundary_dim = current_dim-1

			# Methode 0

			for j in xrange(len(vertice_list)):
				boundary_of_simplex = vertice_list[:j] + vertice_list[j+1:]
				
				row_index = self.dict_hash_vertice[hash(frozenset(boundary_of_simplex))]
				
				self.bmatrix_sparse.dict_col2row_set[i].add(row_index) # NEW TODO 25/11/2017
				max_non_null_row_index_of_col_i = max(row_index, max_non_null_row_index_of_col_i)
			
			self.max_non_null_index[i] = max_non_null_row_index_of_col_i

		print("Done.")

	def reduce_bmatrix_using_gauss_sparse(self):
		print("Processing gaussian elimination algorithm...")
	
		for j in tqdm(xrange(self.N)):
			element =self.max_non_null_index[j]
			
			while (self.pivot[element] != -1):

				# update (Gaussian elimination)
				column = self.pivot[element]
				# element = self.bmatrix_sparse.add_col_i_by_j_opt2_4(j, column, self.N, tempoI, new = first_time_execute) # TODO NEW Lu
				element = self.bmatrix_sparse.add_col_i_by_j_opt5(j, column) 

				

			if (element >= 0) and (self.pivot[element] == -1):
				# element not repeated
				self.pivot[element] = j


		print("Done.")
	def save_barcode(self, path, approxi_eps = 1e-10, mode_COCO = False):
		
		has_been_written = defaultdict(lambda:-1)
		print("Saving barcode to file : "+path)
		with open(path, "ab") as output_file:

			for i in xrange(self.N):
				if self.pivot[i] != -1:
					if (self.data[self.pivot[i]].val-self.data[i].val) < approxi_eps: # Neglect 
						has_been_written[i] = 0
						has_been_written[self.pivot[i]] = 0
					else:
						output_file.write(str(self.data[i].dim)+" "+str(self.data[i].val)+" "+str(self.data[self.pivot[i]].val)+"\n")
						has_been_written[i] = 0
						has_been_written[self.pivot[i]] = 0
				elif (self.pivot[i]==-1) and has_been_written[i] == -1:
					if mode_COCO and self.data[i].dim >= 2:
						pass #continue# 'pass' will still execute lines below
					else:
						output_file.write(str(self.data[i].dim)+" "+str(self.data[i].val)+" "+"inf\n")





def debug_test():
	obj = Persistent_homology()
	obj.load_data("./sample.txt")
	print np.array(obj.data)
	obj.print_data()
	obj.sort_by_lexicographic()
	print("=========")
	obj.print_data()
	print("=========")
	print('Boundary matrix')
	print("=========")
	obj.calculate_boundary_matrix()
	print(obj.bmatrix)
	obj.reduce_bmatrix_using_gauss()
	print("=========")
	print('Boundary matrix after gaussian elimination')
	print("=========")
	print(obj.bmatrix)
	obj.save_barcode("./output.txt")


def main_sparse(filtration_file_path, save2dir = "./", letter = "", debug = False, mode_COCO=False):
	
	filename = filtration_file_path.split("/")[-1].split(".")[0]

	obj = Persistent_homology()
	if debug:
		filtration_file_path = "./sample.txt"
	obj.load_data(filtration_file_path)
	obj.sort_by_lexicographic()

	# New for sparse version
	obj.initialize_dict_for_sparse_matrix()
	# import ipdb; ipdb.set_trace()
	obj.calculate_boundary_matrix_sparse()
	if debug:
		print(obj.bmatrix_sparse.to_array(10)) # for ./sample.txt
	obj.reduce_bmatrix_using_gauss_sparse()

	if not os.path.exists(save2dir):
		os.mkdir(save2dir)
	obj.save_barcode(save2dir+filename+"_barcode"+letter+".txt", mode_COCO = mode_COCO)


def draw_barcode(filepath, H_max_dim = 1, savefig = False, upper_bound_multiplied_by = 2):
	plt.clf() # clears the entire current figure 
	plt.close()


	filename = filepath.split("/")[-1].split(".")[0]
	
	# filename = "_".join(filename.split("_")[1:])

	
	print("Loading/parsing data from file '{}' ...".format(filepath))
	H_dict = defaultdict(lambda:[])
	interval_upper_bound = 0
	
	with open(filepath, "rb") as file:
		for line in file:
			line = line.split()
			if line[2] == "inf":
				continue
			else:
				interval_upper_bound = max(interval_upper_bound, float(line[2]))
			
	
	with open(filepath, "rb") as file:
		for line in file:
			line = line.split()
			
			if line[2] == "inf":
				
				H_dict[int(line[0])].append((float(line[1]), interval_upper_bound*upper_bound_multiplied_by))
			else:
				H_dict[int(line[0])].append((float(line[1]), float(line[2])))

	
	f, ax_tuple = plt.subplots(H_max_dim+1, sharex=True, sharey=True)

	f.subplots_adjust(hspace=0)
	plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

	for h_dim in range(H_max_dim+1):
		k = 0
		
		for interval in H_dict[h_dim]:
			k += 1
			ecart = 0.5 / len(H_dict[h_dim])
			if k == 1:
				ax_tuple[h_dim].plot(interval, (1-k*ecart,1-k*ecart), color = "b", label = "H"+str(h_dim))
			else:
				ax_tuple[h_dim].plot(interval, (1-k*ecart,1-k*ecart), color = "b")
		

		ax_tuple[h_dim].legend(loc = "best")

	if savefig:
		coco_fun_name = "_".join(filename.split("_")[1:-2])
		plt.savefig("./Figures/Barcodes/"+coco_fun_name+"_barcode.png")
	else:
		plt.show()
	print("Done.")
	
def draw_barcode_diagram(filepath, upper_bound_multiplied_by = 2, H_max_dim = 1, savefig = False):
	plt.clf() # clears the entire current figure 
	plt.close()
	
	filename = filepath.split("/")[-1].split(".")[0]
	
	# filename = "_".join(filename.split("_")[1:])
	Dim = []
	X = [] # Creat time (born)
	Y = [] # Vanish time (dead)
	interval_upper_bound = 0
	interval_lower_bound = np.inf

	print("Calculating Diagram's lower/upper bound...")
	with open(filepath, "rb") as file:
		for line in file:
			line = line.split()
			# i += 1
			if line[2] == "inf":
				interval_lower_bound = min(interval_lower_bound, float(line[1]))
				continue
			else:
				
				interval_lower_bound = min(interval_lower_bound, float(line[1]))
				interval_upper_bound = max(interval_upper_bound, float(line[2]))
			
	print("Done.")


	with open(filepath, "rb") as file:
		for line in file:
			line = line.split()
			if int(line[0])> H_max_dim:
				pass
			elif line[2] == "inf":
				Dim.append(int(line[0]))
				X.append(float(line[1]))
				Y.append(2*interval_upper_bound)
			else:
				Dim.append(int(line[0]))
				X.append(float(line[1]))
				Y.append(float(line[2]))

	interval_upper_bound *= upper_bound_multiplied_by
	interval_lower_bound = interval_lower_bound-0.1*np.abs(interval_lower_bound)
	axis_x = np.arange(interval_lower_bound, interval_upper_bound, 0.01)
	plt.figure(0)
	plt.title("Diagram of barcode")
	plt.fill_between(axis_x, interval_lower_bound, axis_x)
	plt.plot([interval_lower_bound, interval_upper_bound], [interval_lower_bound, interval_upper_bound], color = "b")
	plt.scatter(X, Y, s = 10, c = Dim)
	
	if savefig:
		coco_fun_name = "_".join(filename.split("_")[1:-2])
		plt.savefig("./Figures/Diagrams/"+coco_fun_name+"_diag.png")
	else:
		plt.show()
	print("Done.")


def new_pipeline(cloud_path, save2dir = "./", letter = "", debug = False, mode_COCO=True):
	obj = Persistent_homology()
	if debug:
		filtration_file_path = "./sample.txt"
	filename = obj.build_Rips_filtration(cloud_path)
	obj.sort_by_lexicographic()

	# New for sparse version
	obj.initialize_dict_for_sparse_matrix()
	# import ipdb; ipdb.set_trace()
	obj.calculate_boundary_matrix_sparse()
	if debug:
		print(obj.bmatrix_sparse.to_array(10)) # for ./sample.txt
	obj.reduce_bmatrix_using_gauss_sparse()

	if not os.path.exists(save2dir):
		os.mkdir(save2dir)
	obj.save_barcode(save2dir+filename+"_barcode"+letter+".txt", mode_COCO = mode_COCO)	

def sanity_check(path = "./sanity_test/test.txt"):
	# obj = sparse_matrix()
	# obj.build_matrix_for_sanity_check(load_example_path = path)
	obj = Persistent_homology()
	A = obj.initialize_for_sanity_check(path)
	n = len(A)
	print("Original matrix is: ")
	print(A)
	print("="*40)
	obj.reduce_bmatrix_using_gauss_sparse()
	print("After Gaussian elimination process: ")
	print(obj.bmatrix_sparse.to_array_from_dict_set(n))


if __name__ == "__main__":

	st = time()
	# draw_barcode_diagram("./3.txt")
	# new_pipeline("./coco_data/bbob_f001_i02_d02.txt", save2dir="./new_test/")



	################ Sanity Chekck #################
	# sanity_check(path = "./sanity_test/test.txt")#
	################################################

	

	# for filepath in glob.glob("../TD7/filtrations_height/*.txt"):
	# 	main_sparse(filepath, save2dir = "../TD7/barcodes2/", debug = False)
	
	# for dir_number in range(79,100):
		############  Calculate COCO funtions' barcode (Persistent Homology) #############
		# counter = 0
		# Files_list = glob.glob("./All_coco_filtrations/coco_Rips_filtrations{}/*.txt".format(dir_number))
		# total = len(Files_list)
		# for filepath in Files_list:
		# 	counter += 1
		# 	print("Proceeding {}/{} of batch {}...".format(counter, total, dir_number))
		# 	main_sparse(filepath, save2dir ="./All_coco_barcodes/coco_barcodes{}/".format(dir_number), mode_COCO=True)
		# 	print("="*20+"Done."+"="*20)
		# print("All done.")
		##################################################################################

	############  Draw COCO functions' barcode and diagram. ############
	# for filepath in tqdm(glob.glob("./coco_barcodes/*.txt")):
	# 	draw_barcode(filepath, H_max_dim = 1, savefig = True, upper_bound_multiplied_by = 2)
	# 	draw_barcode_diagram(filepath, H_max_dim = 1, savefig = True, upper_bound_multiplied_by = 2)
	####################################################################






	et = time()
	print("Total elapsed time: "+str(et-st))

	

	
