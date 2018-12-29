"""
Copyright 2018 LIN Lu (ncble)
"""
# _*_ coding: utf-8 _*_
import sys
sys.path.append("/home/lulin/Desktop/Desktop/Python_projets/my_packages")
# from draw import draw_surface_level, draw_clouds
from utils import generator, split_file_name, makedirs_advanced
import os, glob
from time import time
import numpy as np 
from operator import itemgetter #, attrgetter # To sort a list
from tqdm import tqdm # Progress bar
import itertools # Construct "S^d" using k-combinaisons of {1,...,d+1}
from collections import defaultdict, Counter, OrderedDict # Construct dictionary with default value



class Get_filtration_from_off(object):
	"""docstring for Get_filtration_from_off"""
	def __init__(self):
		# super(Get_filtration_from_off, self).__init__()
		# self.arg = arg
		self.data_triangles = []#{}#defaultdict(lambda:[])
		self.data_index = []#{} #defaultdict(lambda:[])
		self.nv = 0
		self.nf = 0
		self.filtration = {}
		self.ordered_dict= {}
		
	def load_off(self, filepath):

		with open(filepath, "rb") as file:
			count = 0
			for line in tqdm(file):
				count += 1
				# print(type(line.split(" ")))
				if count == 1:
					continue
				elif count == 2:
					self.nv, self.nf, _ = line.split(" ")

				# elif count > 10:
				# 	break
				else:
					tempo = (line.strip().split(" "))

					if len(tempo) == 3:
						# print tempo
						x, z, y = tempo
						self.data_triangles.append((count-3, float(x), float(y), float(z)))
					else:
						assert len(tempo) == 4
						# print("YEs")
						_, ind1, ind2, ind3 = tempo
						z1 = self.data_triangles[int(ind1)][-1]
						z2 = self.data_triangles[int(ind2)][-1]
						z3 = self.data_triangles[int(ind3)][-1]
						index_argsort = np.argsort((z1, z2, z3))
						index_sorted = np.array((ind1,ind2, ind3)).astype(int)[index_argsort]
						val_sorted = np.array((z1, z2, z3))[index_argsort]
						# val = max(z1, z2, z3)
						# triangle = [(ind1, z1), (ind2, z2), (ind3, z3)]
						# triangle.sort(key=lambda x: x[-1])
						# triangle_str = ""
						# triangle_str += str(triangle[0][-1])+" 0 "+str(triangle[0][0])+"\n"
						# triangle_str += str(triangle[1][-1])+" 0 "+str(triangle[1][0])+"\n"
						# triangle_str += str(triangle[1][-1])+" 1 "+str(triangle[0][0])+" "+str(triangle[1][0])+"\n"
						# triangle_str += str(triangle[2][-1])+" 0 "+str(triangle[2][0])+"\n"
						# triangle_str += str(triangle[2][-1])+" 1 "+str(triangle[0][0])+" "+str(triangle[2][0])+"\n"
						# triangle_str += str(triangle[2][-1])+" 1 "+str(triangle[1][0])+" "+str(triangle[2][0])+"\n"
						# triangle_str += str(triangle[2][-1])+" 2 "+str(triangle[0][0])+" "+str(triangle[1][0])+" "+str(triangle[2][0])+"\n"
						# self.filtration.append((val, triangle_str))
						
						# self.filtration.append((val, simplex))
						self.filtration[hash((val_sorted[0], index_sorted[0]))] = (val_sorted[0], 0, index_sorted[0])
						self.filtration[hash((val_sorted[1], index_sorted[1]))] = (val_sorted[1], 0, index_sorted[1])
						self.filtration[hash((val_sorted[2], index_sorted[2]))] = (val_sorted[2], 0, index_sorted[2])

						self.filtration[hash((val_sorted[1], index_sorted[0], index_sorted[1]))] = (val_sorted[1], 1, index_sorted[0], index_sorted[1])

						self.filtration[hash((val_sorted[2], index_sorted[0], index_sorted[2]))] = (val_sorted[2], 1, index_sorted[0], index_sorted[2])
						self.filtration[hash((val_sorted[2], index_sorted[1], index_sorted[2]))] = (val_sorted[2], 1, index_sorted[1], index_sorted[2])
						self.filtration[hash((val_sorted[2], index_sorted[0], index_sorted[1], index_sorted[2]))] = (val_sorted[2], 2, index_sorted[0], index_sorted[1], index_sorted[2])


						# self.data_index.append((val, ind1, ind2, ind3))
					
	
				# print count
	def sort_data(self):
		# self.data_triangles.sort(key=lambda x: x[-1])
		# self.data_index.sort(key=lambda x: x[0])
		self.ordered_dict = OrderedDict(sorted(self.filtration.items(), key=lambda x: x[1][0]))
	def save_filtration(self, filepath):
		print len(self.filtration)
		with open(filepath, "ab") as file:
			for key in self.ordered_dict:
				
				file.write(" ".join(map(lambda x: str(x), self.ordered_dict[key]))+"\n")

	

class Barcode_embeding(object):
	"""docstring for Barcode_embeding"""
	def __init__(self):
		# super(Barcode_embeding, self).__init__()
		# self.arg = arg
		self.data = defaultdict(lambda:[]) # {} #np.array([])  Attention!!! TODO (self.data[key] will be change to np.ndarray later...it's not good)
		self.distance_matrix_dict = {}
		self.max_dim_of_homology = -1
		self.data_array = defaultdict(lambda:np.array([])) # or 'None' ?  # Duplicated, TOO WASTE !


	# def load_barcode_from_file(self, filepath, filter_by_H_max_dim = 2, upper_bound_multiplied_by = 100):


	# 	print("Getting data max bound...")
	# 	intervals_max_bound = 0
	# 	with open(filepath, "rb") as file:
	# 		for line in file:
	# 			line = line.split()
	# 			# if line[2]=="inf", it's ok!
	# 			# self.data[int(line[0])].append(np.array((float(line[1]), float(line[2]), float(line[2])-float(line[1])))) 
	# 			if line[2] =="inf":
	# 				continue
	# 			intervals_max_bound = max(intervals_max_bound, float(line[2]))
	# 	print("Done.")

	# 	intervals_max_bound = upper_bound_multiplied_by*intervals_max_bound


	# 	print("Loading data...")
	# 	with open(filepath, "rb") as file:
	# 		for line in file:
	# 			line = line.split()

	# 			# if int(line[0]) > filter_by_H_max_dim: # TODO (Not necessary ?)
	# 			# 	pass

	# 			if line[2]=="inf": #, it's ok!
	# 				self.data[int(line[0])].append(np.array((float(line[1]), intervals_max_bound, intervals_max_bound-float(line[1])))) 
	# 			else:
	# 				self.data[int(line[0])].append(np.array((float(line[1]), float(line[2]), float(line[2])-float(line[1])))) 
				
	# 	print("Done.")
	# 	print("Calculating distance matrix...")
	# 	for key in self.data:  # key = dim of Homology
	# 		self.max_dim_of_homology += 1
	# 		self.data_array[key] = np.array(self.data[key]) # TOO WASTE

	# 		# sort data according to length of intervals (increase order)
	# 		index_argsort = np.argsort(self.data_array[key][:, -1], axis = 0)

	# 		self.data_array[key] = self.data_array[key][index_argsort] # TOO WASTE

	# 		A = self.data_array[key][:, 0][:, np.newaxis] # TOO WASTE # A is x-axis values of points
	# 		B = self.data_array[key][:, 1][:, np.newaxis] # TOO WASTE # B is y-axis values of points
	# 		C = (self.data_array[key][:, 2])**2 / 2 # Distance of each points to the diagonal
			
	# 		D_matrix = (A-A.T)**2 + (B-B.T)**2  # Distance matrix of each pair-points
	# 		np.fill_diagonal(D_matrix, np.inf)
	# 		D_matrix = np.minimum(D_matrix, C) # It's necessary ! 
	# 		D_matrix = np.minimum(D_matrix, C[:,np.newaxis])    # C[:,np.newaxis] == C.reshape
	# 		# import ipdb; ipdb.set_trace()
			
			

	# 		n_points, _ = self.data_array[key].shape
	# 		entries = [D_matrix[i, j] for i in range(n_points) for j in range(i+1)] # inferior triangle
	# 		entries.sort()

	# 		self.distance_matrix_dict[key] = entries

	# 	print("Done.")



	def load_barcode_from_file_advanced(self, filepath, upper_bound = 100):


		print("Getting data max bound...")
		intervals_max_bound = 0
		with open(filepath, "rb") as file:
			for line in file:
				line = line.split()
				# if line[2]=="inf", it's ok!
				# self.data[int(line[0])].append(np.array((float(line[1]), float(line[2]), float(line[2])-float(line[1])))) 
				if line[2] =="inf":
					continue
				intervals_max_bound = max(intervals_max_bound, float(line[2]))
		print("Done.")

		# intervals_max_bound = upper_bound_multiplied_by*intervals_max_bound
		


		print("Loading data...")
		with open(filepath, "rb") as file:
			for line in file:
				line = line.split()

				# if int(line[0]) > filter_by_H_max_dim: # TODO (Not necessary ?)
				# 	pass

				if line[2]=="inf": #, it's ok!
					scaled_birth_of_barcode = float(line[1])/intervals_max_bound * upper_bound
					scaled_death_of_barcode = upper_bound
					life_length = upper_bound - scaled_birth_of_barcode
				else:
					scaled_birth_of_barcode = float(line[1])/intervals_max_bound * upper_bound
					scaled_death_of_barcode = float(line[2])/intervals_max_bound * upper_bound
					life_length = scaled_death_of_barcode - scaled_birth_of_barcode

				self.data[int(line[0])].append(np.array((scaled_birth_of_barcode, scaled_death_of_barcode, life_length))) 
				
		print("Done.")
		print("Calculating distance matrix...")
		for key in self.data:  # key = dim of Homology
			self.max_dim_of_homology += 1
			self.data_array[key] = np.array(self.data[key]) # TOO WASTE

			# sort data according to length of intervals (increase order)
			index_argsort = np.argsort(self.data_array[key][:, -1], axis = 0)

			self.data_array[key] = self.data_array[key][index_argsort] # TOO WASTE

			A = self.data_array[key][:, 0][:, np.newaxis] # TOO WASTE # A is x-axis values of points
			B = self.data_array[key][:, 1][:, np.newaxis] # TOO WASTE # B is y-axis values of points
			C = (self.data_array[key][:, 2])**2 / 2 # Distance of each points to the diagonal
			
			D_matrix = (A-A.T)**2 + (B-B.T)**2  # Distance matrix of each pair-points
			np.fill_diagonal(D_matrix, np.inf)
			D_matrix = np.minimum(D_matrix, C) # It's necessary ! 
			D_matrix = np.minimum(D_matrix, C[:,np.newaxis])    # C[:,np.newaxis] == C.reshape
			# import ipdb; ipdb.set_trace()
			
			

			n_points, _ = self.data_array[key].shape
			entries = [D_matrix[i, j] for i in range(n_points) for j in range(i+1)] # inferior triangle
			entries.sort()

			self.distance_matrix_dict[key] = entries


		print("Done.")
	# def feature_vector(self, dim_homology, n_longest, save_path = None, verbose=False):
	# 	"""

	# 	A mapping procedure that turns a barcode to a vector of dim = (dim_homology+1)*n_longest*(n_longest+1)/2
		
	# 	dim_homology, which is the maximal desired homological dimension.
	# 	n_longest, which is the maximal desired number of barcode intervals.

	# 	return: 
	# 	"""
	# 	if self.max_dim_of_homology<dim_homology:
	# 		print("Maximum dimension of homology is: {}".format(self.max_dim_of_homology))
	# 		print("Padding the rest by 0...")
	# 		# dim_homology = self.max_dim_of_homology
			
	# 	# max_dim_non_null_homology = min(self.max_dim_of_homology, dim_homology)

	# 	feature_vec = np.array([])
	# 	for k in xrange(self.max_dim_of_homology+1): # Python2
	# 		# First, compute
	# 		# n_points, _ = self.data[k].shape
	# 		n_points = len(self.data[k])
	# 		if n_points<n_longest:
	# 			first = np.zeros(n_longest)
	# 			first[:n_points] = self.data_array[k][:, -1]
	# 		else:

	# 			first = self.data_array[k][:,-1][-n_longest:]

	# 		tempo = self.distance_matrix_dict[k][-n_longest*(n_longest-1)/2 :]
			
	# 		if len(tempo) < n_longest*(n_longest-1)/2:
	# 			second = np.zeros(n_longest*(n_longest-1)/2)
	# 			second[:len(tempo)] = np.array(tempo)
	# 		else:
	# 			second = np.array(tempo)
	# 		# Two step = more efficient ??
	# 		# import ipdb; ipdb.set_trace()
	# 		feature_vec = np.hstack((feature_vec, first.ravel()))
	# 		feature_vec = np.hstack((feature_vec, second))
	# 		# print(feature_vec.shape)
		
	# 	### Pad the rest of dim by 0
	# 	padding = np.zeros((dim_homology- self.max_dim_of_homology)*n_longest*(n_longest+1)/2)
	# 	feature_vec = np.hstack((feature_vec, padding))

	# 	if save_path is not None:
	# 		np.savetxt(open(save_path, "ab"), feature_vec[np.newaxis, :], delimiter = " ")
	# 	if verbose:
	# 		print("Diagram's feature vector is of shape {}".format(feature_vec.shape))
	# 	return feature_vec #[np.newaxis, :]

	def feature_vector_modified(self, dim_homology, n_longest, save_path = None, verbose=False):
		"""

		A mapping procedure that turns a barcode to a vector of dim = (dim_homology+1)*n_longest*(n_longest+1)/2
		
		dim_homology, which is the maximal desired homological dimension.
		n_longest, which is the maximal desired number of barcode intervals.

		return: 
		"""
		if self.max_dim_of_homology<dim_homology:
			print("Maximum dimension of homology is: {}".format(self.max_dim_of_homology))
			print("Padding the rest by 0...")
			# dim_homology = self.max_dim_of_homology
			
		# max_dim_non_null_homology = min(self.max_dim_of_homology, dim_homology)

		feature_vec = np.array([])
		for k in xrange(self.max_dim_of_homology+1): # Python2
			if k == 0:
				continue
			# First, compute
			# n_points, _ = self.data[k].shape
			n_points = len(self.data[k])
			if n_points<n_longest:
				first = np.zeros(n_longest)
				first[:n_points] = self.data_array[k][:, -1]
			else:

				first = self.data_array[k][:,-1][-n_longest:]

			tempo = self.distance_matrix_dict[k][-n_longest*(n_longest-1)/2 :]
			
			if len(tempo) < n_longest*(n_longest-1)/2:
				second = np.zeros(n_longest*(n_longest-1)/2)
				second[:len(tempo)] = np.array(tempo)
			else:
				second = np.array(tempo)
			# Two step = more efficient ??
			# import ipdb; ipdb.set_trace()
			feature_vec = np.hstack((feature_vec, first.ravel()))
			feature_vec = np.hstack((feature_vec, second))
			# print(feature_vec.shape)
		
		### Pad the rest of dim by 0
		padding = np.zeros((dim_homology- self.max_dim_of_homology)*n_longest*(n_longest+1)/2)
		feature_vec = np.hstack((feature_vec, padding))

		if save_path is not None:
			np.savetxt(open(save_path, "ab"), feature_vec[np.newaxis, :], delimiter = " ")
		if verbose:
			print("Diagram's feature vector is of shape {}".format(feature_vec.shape))
		return feature_vec #[np.newaxis, :]





def verify(dir1, dir2):

	All_close = True
	for file in glob.glob(dir1+"/*.txt"):
		A = np.loadtxt(open(file,'rb'), delimiter = " ")
		file2 = dir2+"/"+file.split("/")[-1]
		B = np.loadtxt(open(file2,'rb'), delimiter = " ")
		if not np.allclose(A,B):
			All_close = False
			return 
	print("All close : ", All_close)

def Diag2Vec(filepath, max_H_dim = 1, n_longest = 30, save2dir ="./coco_diag_vec", upper_bound_multiplied_by = 100):

	#######################################################
	# shape = dim_homology * n_longest * (n_longest+1) /2 #
	#######################################################
	if not os.path.exists(filepath):
		print("File {} not exists or 'Wrong file path'. Please check again.".format(filepath))
		return
	if not os.path.exists(save2dir):
		os.mkdir(save2dir)
	# Attetion: Adapt special COCO format of file name ! Don't change that.
	# Ex: filtration_bbob_f001_i01_d02_barcode.txt
	filename = filepath.split("/")[-1].split(".")[0]
	coco_fun_name = "_".join(filename.split("_")[1:-1])
	filename = "Diag_vec_"+coco_fun_name+".txt"
	
	obj = Barcode_embeding()
	obj.load_barcode_from_file(filepath, upper_bound_multiplied_by = upper_bound_multiplied_by)
	obj.feature_vector(max_H_dim, n_longest, save_path = os.path.join(save2dir,filename))


def Diagram2Vec(filepath, max_H_dim = 1, n_longest = 15, save2file =None, upper_bound_multiplied_by = 100, verbose=False):

	#######################################################
	# shape = dim_homology * n_longest * (n_longest+1) /2 #
	#######################################################
	# if not os.path.exists(filepath):
	# 	print("File {} not exists or 'Wrong file path'. Please check again.".format(filepath))
	# 	return
	# if not os.path.exists(save2file):
	# 	os.mkdir(save2file)

	# Attetion: Adapt special COCO format of file name ! Don't change that.
	# Ex: filtration_bbob_f001_i01_d02_barcode.txt
	# filename = filepath.split("/")[-1].split(".")[0]
	# coco_fun_name = "_".join(filename.split("_")[1:-1])
	# filename = "Diag_vec_"+coco_fun_name+".txt"
	
	obj = Barcode_embeding()
	# obj.load_barcode_from_file(filepath, upper_bound_multiplied_by = upper_bound_multiplied_by)
	obj.load_barcode_from_file_advanced(filepath, upper_bound = upper_bound_multiplied_by)
	# feature_vec = obj.feature_vector(max_H_dim, n_longest, save_path = save2file, verbose=verbose)
	feature_vec = obj.feature_vector_modified(max_H_dim, n_longest, save_path = save2file, verbose=verbose)
	
	return feature_vec

def load_barcode_save_feature_vec(root_dir="./embeddings/dionysus", output_path="./embeddings/All_dgm_vec", file_type='txt', save2filename="train", one_hot_form=False):
	dirpath = makedirs_advanced(output_path, set_count=1)

	def file_label_fun_360(filename):
		name = filename.split(".")[0]
		label = "_".join(name.split("_")[:-1])
		return label
	def file_label_fun_24(filename):
		name = filename.split(".")[0]
		# label = "_".join(name.split("_")[:2])
		label = name.split("_")[1]

		return label
	count = 0
	X = []
	Y = []
	all_true = True
	gen = generator(root_dir=root_dir, file_type=file_type, file_label_fun=file_label_fun_24, stop_after = None, verbose=0)
	with open(os.path.join(dirpath, "badkids.txt"), "ab") as bad_bug:

		for filepath, classe in gen:
			count += 1
			features_vec = Diagram2Vec(filepath, verbose=0, upper_bound_multiplied_by = 100)
			all_true = all_true* (len(features_vec)==120)
			if len(features_vec)==120:
				X.append(features_vec)
				Y.append(int(classe[1:]))
			else:
				print("Warning there is a bad kid in my data !!!!!")
				bad_bug.write(filepath+"     ,  "+classe+"\n")
				continue

		# print(filepath,classe)
	print("All true ? {}".format(all_true))
	print(count)

	X = np.stack(X)
	Y = np.stack(Y)

	np.save(os.path.join(dirpath, "{}_X.npy".format(save2filename)), X)
	np.save(os.path.join(dirpath, "{}_Y.npy".format(save2filename)), Y)





if __name__ == "__main__":
	# verify("./barcodes", "./barcodes2")
	# verify("./barcodes2", "./barcodes(copy)")

	#################### TD7-8 ####################
	# obj = Get_filtration_from_off()
	# obj.load_off("./shapes/tr_reg_000.off")
	# obj.sort_data()
	# obj.save_filtration("./tr_reg_000_filtr.txt")
	###############################################

	
	# A = np.loadtxt(open("/home/lu/Desktop/INF556/TD7/matrix_d2_n10_height.txt", "rb"), delimiter = " ")
	# print A.shape


	### One file test ###
	# obj = Barcode_embeding()
	# obj.load_barcode_from_file("./barcodes/tr_reg_000_barcode.txt")
	# obj.feature_vector(2, 10, save_path = "./feature_vec_000_copy.txt")

	############ Example ############
	# Diag2Vec("../TD6/old_coco_data/coco_barcodes3/filtration_bbob_f008_i01_d02_barcode.txt", save2dir ="./coco_diag_vec/", upper_bound_multiplied_by = 100, max_H_dim = 1, n_longest = 30)
	#################################


	# for dir_number in range(0,100):
	# 	############  Calculate COCO funtions' feature vector  #############
	# 	counter = 0
	# 	Files_list = glob.glob("../TD6/All_coco_barcodes/coco_barcodes{}/*.txt".format(dir_number))
	# 	total = len(Files_list)
	# 	for filepath in Files_list:
	# 		counter += 1
	# 		print("Proceeding {}/{} of batch {}...".format(counter, total, dir_number))
	# 		Diag2Vec(filepath, save2dir ="./All_coco_vec_10/coco_diag_vec{}/".format(dir_number), upper_bound_multiplied_by = 100, max_H_dim = 1, n_longest = 10)
	# 		print("="*20+"Done."+"="*20)
	# 	print("All done.")
	# 	##################################################################################





	# A = Diagram2Vec("./embeddings/dionysus/All_barcodes2/batch_0/bbob_f002_i03_d2_100.txt", n_longest=15, upper_bound_multiplied_by = 100)
	# print(A)
	# print(A.shape)

	# load_barcode_save_feature_vec(root_dir="./embeddings/dionysus",save2filename="train")
	# load_barcode_save_feature_vec(root_dir="./embeddings/test_set",save2filename="test")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_0", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_1", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_2", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_3", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_4", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_5", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_6", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_7", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_8", output_path="./output/Exp",save2filename="examin")
	# load_barcode_save_feature_vec(root_dir="./output/dionysus/mimima_clouds_test/batch_9", output_path="./output/Exp",save2filename="examin")


	
	# X = np.load("./embeddings/All_dgm_vec_2/testX.npy")
	# Y = np.load("./embeddings/All_dgm_vec_2/testY.npy")

	# Diagram2Vec()
	# print(X)
	# print(Y)
	# print(X.shape)
	# print(Y.shape)
	# import ipdb; ipdb.set_trace()


	############  Calculate COCO funtions' feature vector  #############
	# dir_number = 3
	# counter = 0
	# Files_list = glob.glob("../TD6/old_coco_data/coco_barcodes{}/*.txt".format(dir_number))
	# total = len(Files_list)
	# for filepath in Files_list:
	# 	counter += 1
	# 	print("Proceeding {}/{} of batch {}...".format(counter, total, dir_number))
	# 	Diag2Vec(filepath, save2dir ="./coco_diag_vec/", upper_bound_multiplied_by = 100, max_H_dim = 1, n_longest = 30)
	# 	print("="*20+"Done."+"="*20)
	# print("All done.")
	##################################################################################
