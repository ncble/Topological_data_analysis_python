"""
Copyright 2018 LIN Lu (ncble)
"""
# _*_ coding: utf-8 _*_

import numpy as np 
import pandas as pd
from time import time
from numpy import linalg as LA
from tqdm import tqdm
import os, glob
from time import time


def centrer(X):
	moy = np.mean(X, axis = 0)
	# print("Mean : ", moy)
	return X-moy, moy

def PCA_lu(X):
	row = len(X) # == 72
	val_propre, vec_propre = np.linalg.eig((1./row) * X.T.dot(X))
	D_eig = np.diag(val_propre)

	print()
	return

def MDS_lu(X):
	val_propre, vec_propre = np.linalg.eig(X.dot(X.T))
	D_eig_sqrt = np.diag(np.sqrt(val_propre))
	vec_propre.dot()
	return

def load_cloud_to_matrix(filepath):
	# filepath = "./clouds/cloud1.txt"
	object_list = []
	with open(filepath, "rb") as file:
		for line in file:
			object_list.append(map(float, line.split()))
	return np.vstack(object_list)

# np.loadtxt()
# A = np.loadtxt(open("./clouds/cloud1.txt", "rb"), delimiter=" ")

# data = pd.read_csv("./clouds/cloud1.txt", header = None, delimiter=" ")
# print(data)



def Rips_filtration(filepath, truncate= np.arange(0,100), save2dir = "./"):
	# if not os.path.exists(save2dir):
	# 	os.mkdir(save2dir)

	A = load_cloud_to_matrix(filepath)
	A = A[truncate, :] # first 'truncate' points
	# print(A.shape)
	row = len(A)
	distance_matrix = np.zeros((row,row))
	# all_dist = []
	# get_ij_from_index = {}
	count = 0
	print("Calculation distance matrix...")
	for i in tqdm(xrange(row)):
		for j in xrange(i+1, row):
			distance_matrix[i, j] = LA.norm(A[i,:]-A[j,:])
			distance_matrix[j, i] = distance_matrix[i, j]
			# all_dist.append(distance_matrix[i, j])
			# get_ij_from_index[count] = (i,j)
			count += 1
	print("Done.")
	# all_dist = np.array(all_dist)
	# ind = all_dist.argsort()
	# all_dist= all_dist[ind]
	

	# for k, item in enumerate(all_dist):
	# 	i, j = get_ij_from_index[ind[k]] # i
	# 	print item, 1, i, j
	
	filename = filepath.split("/")[-1].split(".")[0]
	print("Writing down Rips filtration to file..")
	with open(save2dir+"filtration_"+filename+".txt", "ab") as file:

		for i in tqdm(xrange(row)):
			# print distance_matrix[i,i], 0, i
			file.write(str(distance_matrix[i,i])+" "+str(0)+" "+str(i)+"\n")
			for j in xrange(i+1, row):
				# print distance_matrix[i,j], 1, i, j
				file.write(str(distance_matrix[i,j])+" "+str(1)+" "+str(i)+" "+str(j)+"\n")
				for k in xrange(j+1, row):
					# print np.max((distance_matrix[i,j], distance_matrix[i,k], distance_matrix[j,k])), 2, i, j, k
					file.write(str(np.max((distance_matrix[i,j], distance_matrix[i,k], distance_matrix[j,k])))\
						+" "+str(2)+" "+str(i)+" "+str(j)+" "+str(k)+"\n")



	print("Done.")
	# return distance_matrix


def write_config_file(dir_path = "", number_of_points = 100, dir_number = 0):
	with open(dir_path+"/ReadMe", "wb") as file:
		file.write("{} points of Sobol sequence of range ({}, {}).\n".format(number_of_points, dir_number*100, (dir_number+1)*100))

def COCO_filtrations(truncate = np.arange(0,100), save2dir = "./coco_Rips_filtrations/", read_clouds_from_dir = "./coco_data/"):
	counter = 0
	Dir_number = truncate[0]/len(truncate)
	save2dir = save2dir[:-1] + str(Dir_number)+"/"
	if not os.path.exists(save2dir):
		os.mkdir(save2dir)

	total = len(glob.glob(read_clouds_from_dir+"*.txt")) # COCO clouds of points (1e4 points sample from Sobol sequence.) 
	
	write_config_file(dir_path = save2dir, number_of_points = len(truncate), dir_number =Dir_number)

	for filepath in glob.glob(read_clouds_from_dir+"*.txt"):
		counter += 1
		print("Proceeding {}/{} of batch {}...".format(counter, total,Dir_number))
		Rips_filtration(filepath, truncate = truncate, save2dir = save2dir)
	


if __name__ =="__main__":
	print(17)
	# Rips_filtratsion("./clouds/cloud1.txt")
	# Rips_filtration("./coco_data/bbob_f001_i01_d02.txt", save2dir="./coco_Rips_filtrations/")
	# Rips_filtration("./coco_data/bbob_f021_i03_d02.txt", save2dir="./coco_Rips_filtrations/")
	
	st = time()
	# for i in range(99,100):
	# 	COCO_filtrations(truncate = np.arange(100*i,100*(i+1)), read_clouds_from_dir = "./coco_data/")

	et = time()
	print("Total elapsed time: "+str(et-st))



