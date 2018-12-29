"""
Copyright 2018 LIN Lu (ncble)
"""
from __future__ import print_function
# source activate TDA
import sys
sys.path.append("/home/lulin/Desktop/Desktop/Python_projets/my_packages")
from utils import makedirs_advanced
import os
import numpy as np
import dionysus as phd
from time import time
import itertools # itertools.combinations(np.arange(5), 2) : choose 2 within 0~4
import matplotlib.pyplot as plt
# filepath = "./coco_data/bbob_f001_i02_d02.txt"
# truncate= np.arange(0,100)

# np.random.choice(20, 5, replace=False)

def coco_funs_num(batch = np.arange(1, 25)):
	func_list = []
	for i in batch:
		func_list.append(int2str(i, length=3))
	# print(func_list)
	return func_list

def coco_funs_instances_num(batch = np.arange(1, 16)):
	func_list = []
	for i in batch:
		func_list.append(int2str(i, length=2))
	# print(func_list)
	return func_list

def get_diagram(filepath, truncate= np.arange(0,100)):
	A = np.loadtxt(open(filepath, "rb"), delimiter= " ")
	A = A[truncate, :]
	f1 = phd.fill_rips(A, 2, 1e15)
	m1 = phd.homology_persistence(f1)
	dgm1 = phd.init_diagrams(m1, f1)
	return dgm1

def get_diagram_Zp(filepath, truncate= np.arange(0,100), field = 2):
	A = np.loadtxt(open(filepath, "rb"), delimiter= " ")
	A = A[truncate, :]
	# f1 = phd.Filtration(A, 2, 1e15)
	f1 = phd.fill_rips(A, 2, 1e15)
	ofp = phd.omnifield_homology_persistence(f1)
	dgm1 = phd.init_diagrams(ofp, f1, field)
	return dgm1


def bottleneck_distance(dgm1, dgm2, dim=1):
	return phd.bottleneck_distance(dgm1[dim], dgm2[dim])
def wasserstein_distance(dgm1, dgm2, dim=1):
	return phd.wasserstein_distance(dgm1[dim], dgm2[dim], q=2)

def int2str(a,length=2):
	s = str(a)
	if len(s)< length:
		s = "0"*(length-len(s))+str(a)
	return s


def paire_wise_btw_instances(fun = "001", instances = np.arange(1,4)):
	
	filepath = "./Bottleneck_distances/bbob_f{}_ixxx_d02.txt".format(fun)
	if os.path.exists(filepath):
		print("Warning file exists. Existing...")
		return 
	dgms_list = [(get_diagram("./coco_data/bbob_f{}_i{}_d02.txt".format(fun, int2str(i))), i) for i in instances]
	A = itertools.combinations(dgms_list, 2)

	with open(filepath, "ab") as file:

		for paire in A:
			dgm1, dgm2 = paire
			file.write("{} {} {}\n".format(dgm1[1],dgm2[1],bottleneck_distance(dgm1[0], dgm2[0])))
			# print("Distance between instances ({}, {}) is : {}".format(dgm1[1],dgm2[1],bottleneck_distance(dgm1[0], dgm2[0])))

def paire_wise_btw_sobol_seed(num_seed = 7):
	
	for fun in coco_funs_num():
		print("="*50)
		print("Paire wise distances for function index {} instances xxx".format(fun))
		for instance in coco_funs_instances_num():
			filepath = "./Bottleneck_distances_seed/bbob_f{}_i{}_d02.txt".format(fun, instance)
			if os.path.exists(filepath):
				print("Warning file exists. Existing...")
				return 
			dgms_list = [(get_diagram("./coco_data/bbob_f{}_i{}_d02.txt".format(fun, instance), truncate = np.arange(100*i,100*(i+1))), i) for i in range(num_seed)]
			A = itertools.combinations(dgms_list, 2)

			with open(filepath, "ab") as file:

				for paire in A:
					dgm1, dgm2 = paire
					file.write("{} {} {}\n".format(dgm1[1],dgm2[1],bottleneck_distance(dgm1[0], dgm2[0])))
					# print("Distance between instances ({}, {}) is : {}".format(dgm1[1],dgm2[1],bottleneck_distance(dgm1[0], dgm2[0])))

def get_fun_name(fun):
	"""
	It's useless
	"""
	return str(fun).split()[1]


def paire_wise_from_list(dgms_list):
	A = itertools.combinations(enumerate(dgms_list), 2)
	for paire in A:
		dgm1, dgm2 = paire
		# name1 = get_fun_name(dgms_list[dgm1[0]])
		# name2 = get_fun_name(dgms_list[dgm2[0]])
		print("Distance between instances ({}, {}) is : {}".format(dgm1[0],dgm2[0],bottleneck_distance(dgm1[1], dgm2[1])))
		# print("Distance between instances ({}, {}) is : {}".format(name1, name2,bottleneck_distance(dgm1[1], dgm2[1])))


def save_barcodes(diagrams, save_to= "./test.txt"):
	"""
	Decode dionysus diagram object. Save it to a barcode file (classical format).

	"""
	dir_name = "/".join(save_to.split("/")[:-1])
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	with open(save_to, "ab") as file:
		# for dim in range(len(diagram)):
		# 	for barcodes in diagram[dim]:
		for dim, dgm in enumerate(diagrams):
			if dim >1:
				return
			for pt in dgm:
 				file.write("{} {} {}\n".format(dim, pt.birth, pt.death))


def load_my_barcodes_to_dionysus(filepath):
	"""
	Willing list (seems to be impossible)

	"""
	return
def plot_barcode(diagram, type="dgm", save_to="barcode.png"):
	if os.path.exists(save_to):
		return
	if len(diagram[1]) == 0:
		return

	dir_name = "/".join(save_to.split("/")[:-1])
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

	plt.figure()
	if type=="dgm":
		phd.plot.plot_diagram(diagram[1], show = False)
	elif type=="bars":
		phd.plot.plot_bars(diagram[1], show = False)
	else:
		raise ValueError("type should be dgm or bars.")
	plt.savefig(save_to)
	plt.close()
	# plt.show()

if __name__ == "__main__":
	print("Start")
	from tqdm import tqdm
	st = time()
	# fun_name = "003"
	# instance = "02"
	#### Main code to produce barcodes !! 25/2/2018 ########
	# output_folder_path = "./embeddings/dionysus/All_barcodes5/"
	# for i in tqdm(range(75,100)):
	# 	for fun_name in coco_funs_num():
	# 		for instance in coco_funs_instances_num():
	# 			if os.path.exists(output_folder_path+"batch_{}/bbob_f{}_i{}_d2_100.txt".format(i, fun_name, instance)):
	# 				continue
	# 			dg1 = get_diagram("./data/coco_data5/bbob_f{}_i{}_d02.txt".format(fun_name, instance), truncate = np.arange(100*i, 100*(i+1)))
	# 			# plot_barcode(dg1, save_to=output_folder_path++"batch_{}/bbob_f{}_i{}_d2_100.png".format(i, fun_name, instance))
	# 			save_barcodes(dg1, save_to=output_folder_path+"batch_{}/bbob_f{}_i{}_d2_100.txt".format(i, fun_name, instance))
	#########################################################
	# output_folder_path = "./output/dionysus/mimima_clouds_test/"
	# for i in tqdm(range(10)):
	# 	for fun_name in coco_funs_num():
	# 		for instance in coco_funs_instances_num():
	# 			if os.path.exists(output_folder_path+"batch_{}/bbob_f{}_i{}_d2_{}.txt".format(i, fun_name, instance, 10*(i+1))):
	# 				continue

	# 			dg1 = get_diagram("./data/coco_data5/bbob_f{}_i{}_d02.txt".format(fun_name, instance), truncate = np.arange(0, 10*(i+1)))
	# 			# import ipdb; ipdb.set_trace()
	# 			plot_barcode(dg1, save_to=output_folder_path+"batch_{}/bbob_f{}_i{}_d2_{}.png".format(i, fun_name, instance, 10*(i+1)))
	# 			save_barcodes(dg1, save_to=output_folder_path+"batch_{}/bbob_f{}_i{}_d2_{}.txt".format(i, fun_name, instance, 10*(i+1)))


	# fun_name = "003"
	# dg1 = get_diagram("./coco_data/bbob_f{}_i01_d02.txt".format(fun_name), truncate = np.arange(100))
	# plot_barcode(dg1, save_to="{}.png".format(fun_name))

	# Z_prime = 41
	# dg1 = get_diagram_Zp("./coco_data/bbob_f001_i01_d02.txt", truncate = np.arange(100, 200), field=Z_prime)
	# dg2 = get_diagram_Zp("./coco_data/bbob_f001_i01_d02.txt", truncate = np.arange(1100, 1200), field=Z_prime)
	# dg3 = get_diagram_Zp("./coco_data/bbob_f001_i01_d02.txt", truncate = np.arange(500, 600), field=Z_prime)
	# dg4 = get_diagram_Zp("./coco_data/bbob_f001_i01_d02.txt", truncate = np.arange(700, 800), field=Z_prime)
	
	# plot_barcode(dg1, save_to="./dionysus_test/test/1_with_Z_{}.png".format(Z_prime))
	# plot_barcode(dg2, save_to="./dionysus_test/test/2_with_Z_{}.png".format(Z_prime))
	# plot_barcode(dg3, save_to="./dionysus_test/test/3_with_Z_{}.png".format(Z_prime))
	# plot_barcode(dg4, save_to="./dionysus_test/test/4_with_Z_{}.png".format(Z_prime))
	# paire_wise_from_list([dg1, dg2, dg3, dg4])


	# Z_prime = 41
	# dg1 = get_diagram_Zp("./coco_data/bbob_f001_i01_d02.txt", truncate = np.arange(100, 200), field=2)
	# dg2 = get_diagram_Zp("./coco_data/bbob_f001_i01_d02.txt", truncate = np.arange(100, 200), field=37)
	# paire_wise_from_list([dg1, dg2])



	# save_barcodes(dg1, save_to="1.txt")
	# save_barcodes(dg2, save_to="2.txt")
	# save_barcodes(dg3, save_to="3.txt")
	# save_barcodes(dg4, save_to="4.txt")

	# phd.plot.plot_diagram(dg1[1], show = True)
	# phd.plot.plot_diagram(dg2[1], show = True)
	# phd.plot.plot_diagram(dg3[1], show = True)

	# plot_barcode(dg1, save_to = "100.png")
	# plot_barcode(dg2, save_to = "200.png")
	# dg2 = get_diagram("./coco_data/bbob_f005_i01_d02.txt")
	# dg3 = get_diagram("./coco_data/bbob_f005_i02_d02.txt")
	# dg4 = get_diagram("./coco_data/bbob_f005_i03_d02.txt")
	# save_barcodes(dg1)
	


	
	# print(bottleneck_distance(dg1, dg2))
	# print(wasserstein_distance(dg1, dg2))

	# print(bottleneck_distance(dg1, dg3))
	# print(bottleneck_distance(dg2, dg3))

	# print(bottleneck_distance(dg4, dg1))
	# print(bottleneck_distance(dg4, dg2))
	# print(bottleneck_distance(dg4, dg3))

	
	# func_list = coco_funs_num()
	# for fun_num in func_list:
	# 	print("="*50)
	# 	print("Paire wise distances for function index {}".format(fun_num))
	# 	paire_wise_btw_instances(fun = fun_num, instances = np.arange(1,16))


	# paire_wise_btw_sobol_seed(num_seed = 7)

	# plot_barcode(dg1)
	# import ipdb; ipdb.set_trace()
	
	et = time()
	print("Total elapsed time: "+str(et-st))
