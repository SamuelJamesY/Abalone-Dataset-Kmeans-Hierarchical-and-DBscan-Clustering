'''
Abalone dataset Kmeans, Hierachial Clustering and Dbscan clustering
Elbow plot, Performance comparison and Dendogram
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from scipy.cluster import hierarchy

def load_data():
	'''
	Load and scale the data
	'''
	df = pd.read_csv('abalone.txt',header=None)
	df.columns = ['sex','length','diameter','height','wholeweight','shuckedweight','visceraweight','shellweight','rings']
	df['sex'] = df['sex'].replace({'M':0,'F':1,'I':-1})
	scaler = MinMaxScaler()
	df.iloc[:,:-1] = scaler.fit_transform(df.iloc[:,:-1])
	print(df.head())
	return df 

def extract_features(df):
	'''
	Get the predictor variables and convert to numpy array
	'''
	X = df.iloc[:,:-1].to_numpy()
	return X

def kmeans_clustering(features,k):
	'''
	Perform k means clustering for different amount of clusters k
	# fit and predict using the kmeans clustering and return inertia and siloutte scores
	'''
	kmeans = KMeans(n_clusters=k,random_state=1)
	pred_class = kmeans.fit_predict(features)
	sil_score = silhouette_score(features,pred_class)
	return kmeans.inertia_, sil_score

def hierachial_clustering(features,k):
	'''
	Perform hierachial (agglomerative clustering) algorithm and report silhoutte score
	'''
	# affinity means find point with closses eucildean distance in ndim space
	# ward means minimize the variance of clusters being merged
	hier = AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage='ward')
	pred_class = hier.fit_predict(features)
	sil_score = silhouette_score(features,pred_class)
	return sil_score

def dbscan_clustering(features,eps):
	'''
	Perform DBScan algorithm for different epsilon sizes
	'''
	db = DBSCAN(eps=eps,min_samples=5)
	pred_class = db.fit_predict(features)
	sil_score = silhouette_score(features,pred_class)
	return sil_score

def inertia_plot(cluster_lst,inertia_lst):
	'''
	Construct an inertia (elbow) plot for the kmeans algorithm
	'''
	plt.plot(cluster_lst,inertia_lst)
	plt.scatter(cluster_lst,inertia_lst)
	plt.xlabel('Number of clusters')
	plt.ylabel('Inertia Score')
	plt.title('Inertia Graph')
	plt.savefig('Inertia_Graph_P1.png')
	plt.clf()

def siloutte_plot(cluster_lst,sil_scores_lst):
	'''
	Construct a siloutte plot for the kmeans algorithm
	'''
	plt.plot(cluster_lst,sil_scores_lst)
	plt.scatter(cluster_lst,sil_scores_lst)
	plt.xlabel('Number of clusters')
	plt.ylabel('Siloutte Score')
	plt.title('Siloutte Graph')
	plt.savefig('Siloutte_Graph_P1.png')
	plt.clf()

def main():
	df = load_data()
	features = extract_features(df)
	cluster_lst = list(range(2,10))
	inertia_lst_kmeans = []
	sil_scores_lst_kmeans = []
	print('The problem is K means clustering:\n')
	for k in cluster_lst:
		inertia, sil_score = kmeans_clustering(features,k)
		inertia_lst_kmeans.append(inertia)
		sil_scores_lst_kmeans.append(sil_score)
	print(cluster_lst,'List of cluster sizes')
	print(inertia_lst_kmeans,'List of inertia values')
	print(sil_scores_lst_kmeans,'List of siloutte scores for Kmeans')
	inertia_plot(cluster_lst,inertia_lst_kmeans) # this is the elbow plot --> here it seems 2 clusters is best
	siloutte_plot(cluster_lst,sil_scores_lst_kmeans) # this is the siloutte plot --> here it seems 2 clusters is best

	print('The problem is Hierachial Clustering clustering:\n')
	
	sil_scores_lst_hier = []
	for k in cluster_lst:
		sil_score = hierachial_clustering(features,k)
		sil_scores_lst_hier.append(sil_score)
	print(cluster_lst,'List of cluster sizes')
	print(sil_scores_lst_hier,'List of siloutte scores for Hierachial')

	print('The problem is DBSCAN clustering:\n')

	# siloutte score is mean of each siloutte coefficient 
	# and indicates whether data is well within its only cluster if closer to 1 

	eps_lst = np.arange(0.05,0.15,0.01) # make an epsilon list
	sil_scores_lst_dbscan = []
	for eps in eps_lst:
		sil_score = dbscan_clustering(features,eps)
		sil_scores_lst_dbscan.append(sil_score)
	print(eps_lst,'List of cluster sizes')
	print(sil_scores_lst_dbscan,'List of siloutte scores for DBscan')

if __name__ == '__main__':
	main()