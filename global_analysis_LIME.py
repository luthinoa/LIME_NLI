#!/usr/bin/env python
# coding: utf-8

# In[34]:


import json 
import time 
from wordcloud import WordCloud
import numpy as np 
from sklearn.cluster import KMeans 
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt 
import matplotlib
from tqdm import tqdm 
import sys
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd 
import collections
import re
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA

timestr = time.strftime("%Y%m%d-%H%M%S") 
FILE_NAME = 'snli_explanations'
FILE_NAME_JSON = FILE_NAME+'.jsonl'
pdf = PdfPages("SNLITEST"+timestr+".pdf")

plt.rc('text', usetex=False)
explained_data = []

with open(FILE_NAME_JSON) as json_file: 
    explained_data = json.load(json_file)

# explained_data = explained_data[:10000]

def get_word_set(data_instances):
    word_set = []

    for instance in data_instances: 
        expl = instance['explanation']
        for word,weigth in expl:
            if(word not in (word_set)):
                word_set.append(word)
    return word_set


# In[5]:


#filter words in explanation by thershold (i.e if words appear less than x times in all of explanations - remove from dict)
def filter_words_in_explanations(data_instances):
    word_count = {}
    for instance in data_instances: 
        expl = instance['explanation']
        for word,weight in expl:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    
    
    word_count = {key:val for key, val in word_count.items() if val>100}
    
    
    return word_count.keys()


# In[6]:


def get_clusters_by_vectors(clusters,vectors):
    #create dictionary with all clusters. 
    clusterd_vectors = {}
    #iterate through instances' cluster label
    for i in range(len(clusters)):
        cluster = clusters[i]
        value = vectors[i].tolist()
        clusterd_vectors.setdefault(str(clusters[i]), []).append(value)
    
    return clusterd_vectors        


# In[7]:


#get all the words contained in a kmenas cluster
def get_words_in_each_cluster(word_set,clusters_dict):
    
    clusterd_words = {}
    
    for cluster_num in clusters_dict.keys():
        cluster_vectors = clusters_dict[cluster_num]
        for vector in cluster_vectors:
            for i in range(len(vector)):
                weight = vector[i]
                word = word_set[i]
                if(weight!=0):                        
                    if(cluster_num in clusterd_words.keys() and word not in clusterd_words[cluster_num]):
                        clusterd_words[cluster_num].append(word)
                    elif(cluster_num not in clusterd_words.keys()):
                        clusterd_words.setdefault(cluster_num,[]).append(word_set[i])
    


# In[8]:


#method for determining number of clusters
def elbow(X_emb):
    number_clusters = range(1, 16)

    kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters]


    score = [kmeans[i].fit(X_emb).score(X_emb) for i in range(len(kmeans))]

    plt.plot(number_clusters, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Method')
    plt.show()


# In[9]:


#get all the explanations contained in a certain kmeans cluster
def get_explanations_in_cluster(explanations,clusters):
    
    clusterd_explanations = {}
    for i in range(len(clusters)):
        cluster = clusters[i]
        value = explanations[i]
        clusterd_explanations.setdefault(str(clusters[i]), []).append(value)
    return clusterd_explanations


# In[10]:


#count words in all of explanations
def count_words(explanation, d, num_of_top_features):
    top_words = get_top_features(explanation,num_of_top_features)
    for word_feature in top_words:
        word = word_feature[0].lower()
        if word in d:
            d[word]+=1
        else:
            d[word] = 1
    d = sorted(d.items(), key=operator.itemgetter(1), reverse = True)
    return d

def remove_from_dict(dict_data,key):
    r = dict(dict_data)
    del r[key]
    return r

#this functions sorts the explanation according to absolute weights and returns a list of the top three weighted words
def get_top_features(explanation, num_of_top_features):

    if len(explanation) <= num_of_top_features:
        return explanation

    explanation.sort(key=lambda row: abs(row[1]), reverse=True)
    return explanation[:num_of_top_features]

#filter dictionary according to a word count threshold.
def filter_word_count(dict_data,threshold):

    labels = dict_data.keys() #neutral,cont,ent
    for label in labels:
        sub_labels = dict_data[label].keys()  #neutral,cont,ent
        for inner_label in sub_labels:
            words = dict_data[label][inner_label].keys() #all words in word count
            for word in words:
                count = dict_data[label][inner_label][word]
                if(count<=threshold):
                    inner_dict = dict_data[label][inner_label]
                    dict_data[label][inner_label] = remove_from_dict(inner_dict,word)
    return dict_data


def get_all_explanations(data_instances):
    expl = []
    for instance in data_instances:
        expl.append(instance["explanation"])
    return expl


# In[11]:


#get top occuring unigrams from explanations.
def top_unigrams_per_cluster_and_label(clusters):
    clusters_word_count = {}
    for cluster in clusters.keys():
        clust = clusters[cluster]
        res = {
            "neutral": {
                "neutral": {},
                "contradiction": {},
                "entailment": {}
            },
            "contradiction": {
                "neutral": {},
                "contradiction": {},
                "entailment": {}
            },
            "entailment": {
                "neutral": {},
                "contradiction": {},
                "entailment": {}
            }
        }
        for instance in clust:
            gold_label = instance['gold_label']
            predicted_label = instance['label']
            explanation = instance['explanation']
            if(gold_label=='-'):
                continue
            count_words(explanation, res[gold_label][predicted_label], 20) 
        filter_word_count(res,15)
        clusters_word_count[cluster] = res 
    return clusters_word_count


# In[12]:


#plot top unigrams by labels.
def plot_unigrams(clusters_word_count):
    key_set = clusters_word_count.keys()
    for cluster in clusters_word_count.keys():
        cluster_count = clusters_word_count[cluster]

        for key in cluster_count.keys():

            df = pd.DataFrame(cluster_count[key])
            df = df.sort_values('entailment',ascending=False)
            df = df.sort_values('neutral',ascending=False)
            df = df.sort_values('contradiction',ascending=False)
            f = df.plot(kind="bar",figsize=(18,10),fontsize=14).get_figure()
            f.suptitle("cluster: "+cluster+", "+key)
            plt.xlabel('Word', fontsize=11)
            plt.ylabel('Occurence', fontsize=11)
            pdf.savefig(f)


# In[13]:


#plot unigrams by kmeans clusters and labels.
def plot_all_unigrams(clusters,threshold):  
    cluster_count = {}
    for key in clusters.keys():
        cluster = clusters[key]
        counts = {}
        for instance in cluster:
            explanation = instance['explanation']
            
            for word,weight in explanation:
                word = word.lower()
                if word not in counts:
                    counts[word] = 0 
                counts[word] += 1
        
        for word in counts.keys():
            count = counts[word]
            if(count<=threshold):
                counts = delete_from_dict(counts,word)
        cluster_count[key]=counts
    
    for key in cluster_count.keys():
        
        wordcount = cluster_count[key]
        mc = sorted(wordcount.items(), key=lambda k_v: k_v[1], reverse=True)
        mc = dict(mc)
        names = list(mc.keys())
        values = list(mc.values())
        
        fig = plt.figure(figsize=(12,10))
        plt.bar(range(len(mc)),values,tick_label=names)
        plt.title('SNLI - K-means Cluster '+key, fontsize=11)
        plt.xticks(rotation='vertical')
        plt.xlabel('Words', fontsize=14)
        plt.ylabel('Occurence', fontsize=14)
        pdf.savefig(fig)
    
def delete_from_dict(word_freq,key):
    d = dict(word_freq)
    del d[key]
    return d


# In[14]:


#plot NMF and LDA topics and top explanations in each topic
def plot_by_topic(list_by_topic):
    i=1
    for topic in list_by_topic: 
        print(i)
        for expl in topic:
            features = expl['explanation']
            sen1 = expl['sentence1']
            sen2 = expl['sentence2']
            label = expl['label']
            gold_label = expl['gold_label']
            
            words =[]
            weights=[]
            for f in features:
                words.append(f[0])
                weights.append(f[1])
            

            plt.bar(words, weights)
            plt.title("Premise: "+sen1+"\n Hypothesis: "+sen2+"\n Label: "+label+"\nGold Label: "+gold_label)
            plt.xlabel("Word")
            plt.ylabel("Weight")
            plt.xticks(rotation=90)
            plt.show()
        i=i+1


# In[15]:


def display_topics_2(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


# In[26]:


#print generated topic modeling (both for NMF and LDA)
def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    explanations_by_topic = []
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:no_top_documents]
        similar_explanation = []  
        
        for doc_index in top_doc_indices:
            similar_explanation.append(explained_data[doc_index])
            print(str(documents[doc_index]))
        explanations_by_topic.append(similar_explanation)
    plot_by_topic(explanations_by_topic)


# In[33]:


def topic_modeling():
    no_features = 1000
    documents = []
    for instance in tqdm(explained_data):
        explanation = instance["explanation"]
        words = ""
        for word, weight in explanation: 
            words = words+" "+word
        documents.append(words)

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 5

    # Run NMF
    nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tf)
    nmf_W = nmf_model.transform(tf)
    nmf_H = nmf_model.components_
    
     # Run LDA
    lda_model = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    lda_W = lda_model.transform(tf)
    lda_H = lda_model.components_

    no_top_words = 6
    no_top_documents = 5
    display_topics(nmf_H, nmf_W, tf_feature_names, documents, no_top_words, no_top_documents)
    #display_topics(lda_H, lda_W, tf_feature_names, documents, no_top_words, no_top_documents)


# In[19]:


def kmeans_analysis(): 
    
    #code for kmeans, pca, t-sne analysis 
        
    word_set = sorted(filter_words_in_explanations(explained_data))
    
    #populate word_to_position vector which holds weights to all explanations words in the word's position in vocabulary. 
    
    word_to_position = {word: position for position, word in enumerate(word_set)}
    X = np.zeros(shape=(len(explained_data),len(word_to_position)))
    X_lst = []

    for index, instance in tqdm(enumerate(explained_data)):
        explanation = instance['explanation']
        explanation_vector = np.zeros(len(word_to_position))
        for word, weight in explanation:
            if word in word_set:
                position = word_to_position[word]
                explanation_vector[position] = weight
        X[index] = explanation_vector
        if np.any(np.abs(explanation_vector) > 1e-4):
            X_lst += [explanation_vector]
    
    #if vector is 0 remove. (i.e if no words are a part of the vocabulary)
    X = np.concatenate([x.reshape(1, -1) for x in X_lst], axis=0)
    X = X[~np.all(X == 0, axis=1)]    
    
    #pca - dimension reduction to 61
    pca_50 = PCA(n_components=61)
    pca_result_50 = pca_50.fit_transform(X)
    print(format(np.sum(pca_50.explained_variance_ratio_)))
    
    #t-SNE dimension reduction to 2
    X_emb = TSNE(n_components=2,perplexity=30).fit_transform(pca_result_50)
    
    #elbow method for determining number of clusters:
    #elbow(X_emb)
    
    #kmeans with 6 clusters
    kmeans = KMeans(n_clusters=6)
    kmeans.fit_predict(X_emb)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    clustered_vectors = get_clusters_by_vectors(labels,X)
    words_by_cluster = get_words_in_each_cluster(word_set,clustered_vectors)
    explanation_by_cluster = get_explanations_in_cluster(explained_data,labels)
    
    fig1 = plt.figure(figsize=(8,5))
    plt.scatter(X_emb[:,0], X_emb[:,1], c=labels, cmap='rainbow')

    plt.scatter(centers[:,0],centers[:,1],c='black',s=50)   
    plt.title('FEVER Explanations - K-means Clustering', fontsize=11)
    pdf.savefig(fig1)
    
    top_unigrams = top_unigrams_per_cluster_and_label(explanation_by_cluster)
    plot_unigrams(top_unigrams)
    plot_all_unigrams(explanation_by_cluster,50)

    pdf.close()

