import tensorflow as tf
import numpy as np
from itertools import islice
import random
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from RelationCompositionModel import RelComp
import math
#------------End of Import Libraries----------
def readEmbeddings(name):
    embeddings=np.load("RelWalk_Embeddings/Embeddings_d=50.npy").item()
    entity = embeddings['ent_embeddings']
    d = entity.shape[1]
    r1 = embeddings['R1'].reshape(-1, d, d)
    r2 = embeddings['R2'].reshape(-1, d, d)
    return entity, r1, r2

def read_relations():
    relations = {}
    file = open("FB15K474/relation2id.txt")
    for relation in islice(file, 1, None):
        rel = relation.split()[0]
        id = int(relation.split()[1])
        relations[rel] = id
    R = set()
    relcompstats = []
    file2 = open("Composition_Constraints/relcompstats_filter.txt")
    for line in file2:
        rA_id = relations[line.split()[0]]
        rB_id = relations[line.split()[1]]
        rC_id = relations[line.split()[2]]
        relcompstats.append((rA_id, rB_id, rC_id))
        R.update([rA_id, rB_id, rC_id])

    return R, relcompstats

def add_layer(inputs, in_size, out_size, keep_prob, activation_function=None):
        Weights = tf.Variable(tf.random_normal([in_size, out_size], stddev=1, seed=1))
        mul = tf.matmul(inputs, Weights) 
        mul = tf.nn.dropout(mul, keep_prob)
        if activation_function is None:
            outputs = mul
        else:
            outputs = activation_function(mul)
        return Weights, outputs
    
def construct_data(dim):
    entity, r1, r2 = readEmbeddings("FB474-Embeddings_d=" + str(dim))
    
    X = np.zeros(shape=(154,4* dim* dim))
    Y = np.zeros(shape=(154,2* dim* dim))
    i = 0
  
    for relation_tuples in relation_compose:
        rA_id = relation_tuples[0]
        rB_id = relation_tuples[1]
        rC_id = relation_tuples[2]

        rA_linear = np.append(r1[rA_id].reshape(-1),r2[rA_id].reshape(-1))
        rB_linear = np.append(r1[rB_id].reshape(-1),r2[rB_id].reshape(-1))
        input_feature = np.append(rA_linear, rB_linear)
        target_output = np.append(r1[rC_id].reshape(-1),r2[rC_id].reshape(-1))

        X[i] = input_feature
        Y[i] = target_output
        i = i+1
         
    relations = np.zeros(shape=(474,2*dim*dim))
    for i in range(474):
        relations[i] = np.append(r1[i].reshape(-1), r2[i].reshape(-1))
    return X, Y, relations    

# --------------------------------- Unsupervised Methods -------------------------------
def multiply(rA_1, rA_2, rB_1, rB_2):
    rC_1 = np.dot(rA_1, rB_1)
    rC_2 = np.dot(rA_2, rB_2)
    return rC_1, rC_2

def elemMultiply(rA_1, rA_2, rB_1, rB_2):
    rC_1 = np.multiply(rA_1, rB_1)
    rC_2 = np.multiply(rA_2, rB_2)
    return rC_1, rC_2

def add(rA_1, rA_2, rB_1, rB_2):
    rC_1 = np.add(rA_1, rB_1)
    rC_2 = np.add(rA_2, rB_2)
    return rC_1, rC_2

def multiply_trans(rA_1, rA_2, rB_1, rB_2):
    rC_1 = np.dot(rA_1, rB_1.T)
    rC_2 = np.dot(rA_2.T, rB_2)
    return rC_1, rC_2        
        
def compose(method, dim, X_test, Y_test, eval_set):
    entity, r1_embeddings, r2_embeddings = readEmbeddings("FB474-Embeddings_d=" + str(dim))
    
    # for 154 composed relations
    rank_sum = 0
    inverse_rank_sum = 0
    top10_sum = 0
    rank_num = 0
    for i in range(len(X_test)):
        rA_1 = X_test[i][:dim*dim].reshape(dim,dim)
        rA_2 = X_test[i][dim*dim:2*dim*dim].reshape(dim,dim)
        rB_1 = X_test[i][2*dim*dim:3*dim*dim].reshape(dim,dim)
        rB_2 = X_test[i][3*dim*dim:].reshape(dim,dim)
        
        target_1 = Y_test[i][:dim*dim].reshape(dim,dim)
        target_2 = Y_test[i][dim*dim:2*dim*dim].reshape(dim,dim)

        # composed embeddings
        if(method == "multiply"):
            rC_1_com, rC_2_com = multiply(rA_1, rA_2, rB_1, rB_2)
        elif(method == "elemMultiply"):
            rC_1_com, rC_2_com = elemMultiply(rA_1, rA_2, rB_1, rB_2)
        elif(method == "add"):
            rC_1_com, rC_2_com = add(rA_1, rA_2, rB_1, rB_2)
        elif (method == "multiply_trans"):
            rC_1_com, rC_2_com = multiply_trans(rA_1, rA_2, rB_1, rB_2)
        
        correct_score = np.linalg.norm(target_1 - rC_1_com, ord='fro') + np.linalg.norm(target_2 - rC_2_com, ord='fro')
        score_rA = np.linalg.norm(rA_1 - rC_1_com, ord='fro') + np.linalg.norm(rA_2 - rC_2_com, ord='fro')
        score_rB = np.linalg.norm(rB_1 - rC_1_com, ord='fro') + np.linalg.norm(rB_2 - rC_2_com, ord='fro')
        scores = []
        correct = ()
   
        # for each 154 relation, compute the score with data set(474)
        for rel_id in eval_set:
            rC_1 = r1_embeddings[rel_id]
            rC_2 = r2_embeddings[rel_id]
            score = np.linalg.norm(rC_1 - rC_1_com, ord='fro') + np.linalg.norm(rC_2 - rC_2_com, ord='fro')
            scores.append(score)
        scores.sort()   
             
        rank = scores.index(correct_score) + 1
        rank_rA = scores.index(score_rA)+1 
        rank_rB = scores.index(score_rB)+1 
        if rank_rA < rank or rank_rB < rank:
            rank_num += 1 
        rank_sum += rank 
        inverse_rank_sum += 1/float(rank)
        if(rank <= 10):
            top10_sum = top10_sum + 1
    mean_rank = rank_sum / len(X_test)
    mean_inverse_rank = inverse_rank_sum / len(X_test)
    H10 = float(top10_sum) / len(X_test)
    return mean_rank, mean_inverse_rank, H10, rank_num
# -----------------------------------------------------------------------------------------------------------------------     
def evaluation(model, eval_set):
    results = []
   
    print("rank within " + str(eval_set) +  " relations")
    mean_rank, mean_inverse_rank, H10, rank_num = model.evaluation(eval_set, "test")
    results.append(mean_rank)
    results.append(mean_inverse_rank)
    results.append(H10)
    results.append(rank_num)
    print(" MR: %g, MRR: %g, H10: %g, rA/rB rank: %g (Supervised Method)"  %(mean_rank, mean_inverse_rank, H10, rank_num))
    
    mean_rank, mean_inverse_rank, H10, rank_num = compose("multiply", model.dim, model.X_test, model.Y_test, eval_set)
    results.append(mean_rank)
    results.append(mean_inverse_rank)
    results.append(H10)
    print(" MR: %g, MRR: %g, H10: %g, rA/rB rank: %g (Multiply)"  %(mean_rank, mean_inverse_rank, H10, rank_num))
    
    mean_rank, mean_inverse_rank, H10, rank_num = compose("elemMultiply", model.dim, model.X_test, model.Y_test, eval_set)
    results.append(mean_rank)
    results.append(mean_inverse_rank)
    results.append(H10)
    print(" MR: %g, MRR: %g, H10: %g, rA/rB rank: %g (Element_wise Multiply)"  %(mean_rank, mean_inverse_rank, H10, rank_num))
    
    mean_rank, mean_inverse_rank, H10, rank_num = compose("add", model.dim, model.X_test, model.Y_test, eval_set)
    results.append(mean_rank)
    results.append(mean_inverse_rank)
    results.append(H10)
    print(" MR: %g, MRR: %g, H10: %g, rA/rB rank: %g (Add)"  %(mean_rank, mean_inverse_rank, H10, rank_num))  
    
    return results
# -----------------------------------------------------------------------------------------------------------------------             
if __name__=="__main__":
    # read relation embeddings
    R, relation_compose = read_relations()
    print("E=50")
    model_E50 = RelComp(d=50, layer_num=2, m={1:100,2:600}, training_epoch = 25000, dropout=0.5, l2_reg=True)
    #---------------------------Evaluate the trained model----------------------------
    model_E50.construct_predictions()
    model_E50.train_model(plot=False)
    print("rank within 474 relations")
    results = evaluation(model_E50,range(474))
    model_E50.output_predict()

