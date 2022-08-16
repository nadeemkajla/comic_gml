import os
import argparse
import gensim
import numpy as np

__author__ = "Nadeem Iqbal Kajla"
__email__ = "nadeem.iqbal@mnsuam.edu.pk"

import get_embeddings_uniform
import get_embeddings_similar_neighbours
import get_embeddings_similar_any

def main(input_file, output_folder,directed=False,walks_per_node=10,steps=80,size=300,metric='jaccard',window=10, workers=1, verbose=False):  
    """
    Getting Uniform embedding
    """
    folder_path = os.path.join(output_folder, 'Uniform')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.isfile(os.path.join(folder_path, 'embedd_' + str(size/3) + '.csv')):
        get_embeddings_uniform.get_embeddings(input_file, folder_path, directed=directed, walks_per_node=walks_per_node, steps=steps, size=size,window=window, workers=workers,verbose=verbose)
        
    """
    Getting Similar neighbour embedding
    """
    folder_path = os.path.join(output_folder, 'Similar_neighbours')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.isfile(os.path.join(folder_path, 'embedd_' + str(size/3) + '.csv')):
        get_embeddings_similar_neighbours.get_embeddings(input_file, folder_path, directed=directed, walks_per_node=walks_per_node, steps=steps, size=size, window=window, workers=workers,verbose=verbose)
        
        
    """
    Getting Similar any embedding
    """
    folder_path = os.path.join(output_folder, 'Similar_any')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.isfile(os.path.join(folder_path, 'embedd_' + str(size/3) + '.csv')):
        get_embeddings_similar_any.get_embeddings(input_file, folder_path, directed=directed, walks_per_node=walks_per_node, steps=steps, size=size, window=window, workers=workers,verbose=verbose)
        
    folder_path = os.path.join(output_folder, 'all')
    print('.')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print('.')
    if not os.path.isfile(os.path.join(folder_path, 'embedd_' + str(int(size/3)) + '.csv')):
        embeddingsI = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(output_folder, 'Uniform', 'embedd_' + str(int(size/3)) + '.csv'))
        print('.')
        embeddingsII = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(output_folder, 'Similar_neighbours', 'embedd_' + str(int(size/3)) + '.csv'))
        print('.')
        embeddingsIII = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(os.path.join(output_folder, 'Similar_any', 'embedd_' + str(int(size/3)) + '.csv'))
        embeddings = gensim.models.keyedvectors.KeyedVectors(size)
        for node in embeddingsI.index2word:
            values = np.concatenate((embeddingsI[node],embeddingsII[node],embeddingsIII[node]))
            embeddings.add(node, values)
        embeddings.save_word2vec_format(os.path.join(folder_path, 'embedd_' + str(int(size)) + '.csv'))
