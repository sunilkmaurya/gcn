import pickle as p
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import networkx as nx
from prettytable import PrettyTable
graph_path = './gcn/data/'
data_loc = './../data/'

def simple_classify_f1(dataset_name,write_to_file=None):
    ''' To take input planetoid classified data and perform statistical analysis'''
    results_str = []
    with open(dataset_name+"_gcn.pickle","rb") as f:
        _,test_results,test_ind = p.load(f)
    with open(dataset_name+"_proper.labels","rb") as f1:
        y_test = p.load(f1)

    g_file = open(graph_path+"ind."+dataset_name+".graph",'rb')
    graph = nx.from_dict_of_lists(p.load(g_file))

    #convert probabilities to one-hot encoding
    new_mat = np.zeros(test_results.shape)
    for i in range(test_results.shape[0]):
        index = np.argmax(test_results[i,:])
        new_mat[i,index]=1

    test_results = new_mat

    averages = ["micro", "macro", "samples", "weighted"]
    str_output = "Train ratio: Default with planetoid\n"
    for avg in averages:
        str_output += avg + ': ' + str(f1_score(test_results[test_ind], y_test[test_ind],average=avg)) + '\n'

    str_output += "Accuracy: " + str(accuracy_score(test_results[test_ind], y_test[test_ind])) + '\n'
    results_str.append(str_output)
    print(nx.info(graph))
    print("Simple classification results")
    if write_to_file:
        with open(write_to_file, 'w') as f:
            f.write(info)
            f.writelines(results_str)
    print(''.join(results_str))
    return write_to_file


def modified_classify_f1(dataset_name,write_to_file=None):
    '''this function spreads the label of neighbor to test node'''
    results_str = []
    with open(dataset_name+"_gcn.pickle","rb") as f:
        _,test_results,test_ind = p.load(f)

    with open(dataset_name+"_proper.labels","rb") as f1:
        y_test = p.load(f1)

    g_file = open(graph_path+"ind."+dataset_name+".graph",'rb')
    graph = nx.from_dict_of_lists(p.load(g_file))

    #convert probabilities to one-hot encoding
    new_mat = np.zeros(test_results.shape)
    for i in range(test_results.shape[0]):
        index = np.argmax(test_results[i,:])
        new_mat[i,index]=1

    test_results = new_mat

    #with open(data_loc+dataset_name+".labels","rb") as f:
    #        labels = p.load(f)
    #modifying labels with respect to neighbors, degree can be changed where

    print("Propagating neighbor labels for degree 1 nodes.")
    #f1 = open(graph_path+"ind."+dataset_name+".ally")
    #temp_x = p.load(f1)
    #index = temp_x.shape[0]

    for c in range(len(test_ind)):
        if test_ind[c]==True:
            if graph.degree(c)==1:
                neighbor = list(graph.neighbors(c))
                if test_ind[neighbor[0]]==False:
                    test_results[c,:] = y_test[neighbor[0],:]

    averages = ["micro", "macro", "samples", "weighted"]
    str_output = "Train ratio: Default with planetoid\n"
    for avg in averages:
        str_output += avg + ': ' + str(f1_score(test_results[test_ind], y_test[test_ind],average=avg)) + '\n'

    str_output += "Accuracy: " + str(accuracy_score(test_results[test_ind], y_test[test_ind])) + '\n'
    results_str.append(str_output)
    print(nx.info(graph))
    print("Simple classification results")
    if write_to_file:
        with open(write_to_file, 'w') as f:
            f.write(info)
            f.writelines(results_str)
    print(''.join(results_str))
    return write_to_file


def classify_analysis(dataset_name):
    with open(dataset_name+"_gcn.pickle","rb") as f:
        _,test_results,test_ind = p.load(f)

    with open(dataset_name+"_proper.labels","rb") as f1:
        y_test = p.load(f1)

    g_file = open(graph_path+"ind."+dataset_name+".graph",'rb')
    graph = nx.from_dict_of_lists(p.load(g_file))

    #import full labels
    #with open(data_loc+dataset_name+".labels","rb") as f:
    #    labels = p.load(f)

    #convert probabilities to one-hot encoding
    new_mat = np.zeros(test_results.shape)
    for i in range(test_results.shape[0]):
        index = np.argmax(test_results[i,:])
        new_mat[i,index]=1

    test_results = new_mat

    #read training set file and load the starting index of test file.
    #f1 = open(graph_path+"ind."+dataset_name+".ally")
    #temp_x = p.load(f1)
    #index = temp_x.shape[0]
    table = PrettyTable(['Serial No.','Node no.','True Label','Predicted Label','Degree','Neighbor label','Neighbor Match'])
    counter = 0
    match_counter = 0

    for i in range(len(test_ind)):
        if test_ind[i]==True:
            if (np.array_equal(y_test[i,:],test_results[i,:]))==False:
                if graph.degree(i)==1:
                    counter = counter + 1
                    neighbor = list(graph.neighbors(i))

                    if np.array_equal(y_test[i,:],y_test[neighbor[0],:]):
                        match = True
                        match_counter += 1
                    else:
                        match = False
                    table.add_row([counter,str(i)+' '+str(neighbor[0]),np.where(y_test[i,:]!=0),np.where(test_results[i,:]!=0),graph.degree(i),np.where(y_test[neighbor[0],:]!=0),match])

    print(table)
    print("Total cases of neighbor label matches are: "+str(match_counter))
    #to count number of wrongly classified nodes
    print("For test set label classification prediction results")
    counts = dict()

    #this counts the number of wrong predictions of nodes with degrees
    for i in range(len(test_ind)):
        if test_ind[i]==True:
            if (np.array_equal(test_results[i,:],y_test[i,:]))== False:
                degree = graph.degree(i)
                counts[degree] = counts.get(degree,0)+1

    #for total number of nodes
    counts_total = dict()
    for i in range(len(test_ind)):
        if test_ind[i]==True:
            degree = graph.degree(i)
            counts_total[degree] = counts_total.get(degree,0)+1
    total_test_nodes=0
    for j in counts_total.keys():
        total_test_nodes = total_test_nodes + counts_total[j]

    print("Statistics for dataset: "+dataset_name)
    print("---------------------------------")

    for i in sorted(counts.keys()):
        print("For degree "+str(i)+" total: "+str(counts_total[i])+" ("+str(int((counts_total[i]/total_test_nodes)*100))+"%) and wrong prediction: "+ str(counts[i])+"("+str(int(counts[i]/counts_total[i]*100))+"%)")
