import tensorflow as tf
import numpy as np
from sklearn.cross_validation import KFold
import math
import random

class RelComp():
    def __init__(self,d, layer_num, m, training_epoch, dropout, l2_reg):
        self.dim = d
        self.embedding_dim = d*d
        self.hidden_layer_num = layer_num
        self.hidden_layer_tuples = m
        self.learning_rate = 5e-4
        self.training_epoch = training_epoch
        self.dropout = dropout
        self.reg = l2_reg
        self.sess = tf.Session()       
        self.current_folder = 0          
# ----------------------------------------------------------------------------------------            
    def construct_predictions(self):
        # placeholders for inputs,outputs
        self.x = tf.placeholder(tf.float32,shape=(None,4*self.embedding_dim),name='features')  
        self.y_ = tf.placeholder(tf.float32,shape=(None,2*self.embedding_dim),name='labels')
        self.data_num = tf.placeholder(tf.int32)  # the number of input data 
        self.length = tf.identity(self.data_num)
        self.keep_prob = tf.placeholder(tf.float32)
        
        # the outputs of each layer, start with 0 layer which is the inputs x
        output = {}
        output[0] = self.x
        # m is the tuples numbers of each layer outputs, start with 0 layer, is the tuples in the input x
        self.hidden_layer_tuples[0] = 4*self.embedding_dim 
        # w is the weight between layers
        w = {} 
        # ---------- add L2 regularisation -----------
        for i in range(self.hidden_layer_num):
            w[i], output[i+1] = add_layer(inputs=output.get(i), in_size=self.hidden_layer_tuples.get(i), out_size=self.hidden_layer_tuples.get(i+1), keep_prob=self.keep_prob, activation_function=tf.nn.tanh)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, w[i]) 
        u, self.y = add_layer(inputs=output.get(self.hidden_layer_num), in_size=self.hidden_layer_tuples.get(self.hidden_layer_num), out_size=2*self.embedding_dim, keep_prob=self.keep_prob,activation_function=None)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, u)
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-10)
        reg_term = tf.contrib.layers.apply_regularization(regularizer)
        
        # define cost
        # rel_1/2 are target relation embeddings, rel1/2 are produced relation embeddings
        rel_1 = tf.slice(self.y_, [0, 0], [self.length, self.embedding_dim])
        rel_2 = tf.slice(self.y_, [0, self.embedding_dim ], [self.length, self.embedding_dim])
        rel1 = tf.slice(self.y, [0, 0], [self.length, self.embedding_dim])
        rel2 = tf.slice(self.y, [0, self.embedding_dim], [self.length, self.embedding_dim])

        self.score = tf.sqrt( tf.reduce_sum( tf.square(rel_1 - rel1), axis = 1) ) + tf.sqrt( tf.reduce_sum( tf.square(rel_2 - rel2), axis = 1) ) 
        if self.reg:
            self.cost = tf.reduce_mean( self.score ) + reg_term
        else:
            self.cost = tf.reduce_mean( self.score ) 
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
# ----------------------------------------------------------------------------------------       
    def output_predict(self,fold):
        predict = self.sess.run(self.y, feed_dict={self.x:self.X, self.keep_prob:1})
        d = int(math.sqrt(self.X.shape[1] /4)) 
        predict_1 = predict[ : , 0:d*d]
        predict_2 = predict[ : , d*d:2*d*d]
        f_name = "predict_E=" + str(d) + "_fold%d.npz"%fold
        np.savez(f_name, R1=predict_1, R2 = predict_2)

        embeddings = np.load(f_name)
        predict_1 = embeddings["R1"].reshape(-1,d,d)
        predict_2 = embeddings["R2"].reshape(-1,d,d)
# ----------------------------------------------------------------------------------------   
    def evaluation(self,eval_set, data_set):
        if data_set=="train": 
            feed_x = self.X_train
            feed_y = self.Y_train
        if data_set=="test":
            feed_x = self.X_test
            feed_y = self.Y_test
        predict = self.sess.run(self.y, feed_dict={self.x:feed_x, self.keep_prob:1})
        test_num = self.sess.run(self.length, feed_dict={self.data_num:len(feed_x)})
        
        rank_sum = 0
        inverse_rank_sum = 0
        top10_sum = 0
        accurate_num = 0  

        rank_num = 0 # this the number of composed relations which is more related with rA and rB, not target relation

        for i in range(test_num):
            rA_1 = feed_x[i][:self.embedding_dim]
            rA_2 = feed_x[i][self.embedding_dim:2*self.embedding_dim]
            rB_1 = feed_x[i][2*self.embedding_dim:3*self.embedding_dim]
            rB_2 = feed_x[i][3*self.embedding_dim:4*self.embedding_dim]
            composed_1 = predict[i][:self.embedding_dim]
            composed_2 = predict[i][self.embedding_dim:2*self.embedding_dim]
            score_rA = np.sqrt(np.sum(np.square(composed_1 - rA_1))) + np.sqrt(np.sum(np.square(composed_2 - rA_2)))
            score_rB = np.sqrt(np.sum(np.square(composed_1 - rB_1))) + np.sqrt(np.sum(np.square(composed_2 - rB_2)))              
            target_1 = feed_y[i][:self.embedding_dim]
            target_2 = feed_y[i][self.embedding_dim:2*self.embedding_dim]
            target_score = np.sqrt(np.sum(np.square(composed_1 - target_1))) + np.sqrt(np.sum(np.square(composed_2 - target_2)))

            norms = []
            for j in eval_set:
                RelWalk_1 = self.relations[j][:self.embedding_dim]
                RelWalk_2 = self.relations[j][self.embedding_dim:2*self.embedding_dim]
                norm = np.sqrt(np.sum(np.square(composed_1 - RelWalk_1))) + np.sqrt(np.sum(np.square(composed_2 - RelWalk_2)))
                norms.append(norm)
            norms.sort()                    
            rank = norms.index(target_score)+1   
            # compare rank with rA rank and rB rank
            rank_rA = norms.index(score_rA)+1 
            rank_rB = norms.index(score_rB)+1 
            if rank_rA < rank or rank_rB < rank:
                rank_num += 1                    
            rank_sum += rank 
            inverse_rank_sum += 1/rank
            if(rank <= 10):
                top10_sum = top10_sum + 1
            if rank ==1 : accurate_num += 1
        mean_rank = rank_sum / test_num
        mean_inverse_rank = inverse_rank_sum / test_num
        H10 = top10_sum / test_num  
        
        ratio = float(accurate_num/test_num)    
        
        return mean_rank, mean_inverse_rank, H10, rank_num
# ----------------------------------------------------------------------------------------        
    def train_model(self, plot):
        #self.set_inputs_outputs()
        init = tf.global_variables_initializer()  
        self.sess.run(init)
        
        x_axis = []
        test_MR = []
        test_MRR = []
        test_H10 = []
        test_losses = []
        hist_loss = []
        
        # apply stochastic gradient descent
        batch_size = 25
        stop = False
        epoch_num = 0        
        patience_cnt = 0

        self.X, self.Y, self.relations = construct_data(self.dim)
        self.relation_compose=np.zeros((154,3))
        for i,(r1,r2,r3) in enumerate(relation_compose):
        	self.relation_compose[i,0]=r1
        	self.relation_compose[i,1]=r2
        	self.relation_compose[i,2]=r3

        randomize = np.arange(154)
        np.random.shuffle(randomize)
        self.X = self.X[randomize]
        self.Y=self.Y[randomize]
        self.relation_compose=self.relation_compose[randomize]

        kf = KFold(154, n_folds=5)

        Iter=0

        for train, test in kf:
			print ("Test_Fold",Iter)
			print (self.relation_compose[test])
			save_test_ids(self.relation_compose[test],Iter)
			self.X_train,self.Y_train=self.X[train],self.Y[train]
			self.X_test,self.Y_test=self.X[test],self.Y[test]

			for epoch_num in range(self.training_epoch):
				slice = random.sample(range(len(self.X_train)), batch_size) 
				x_batch = self.X_train[slice]
				y_batch = self.Y_train[slice]

				# add dropout
				self.sess.run(self.train_op,feed_dict={self.x:x_batch , self.y_:y_batch, self.data_num:batch_size, self.keep_prob:self.dropout}) 

			test_loss = self.sess.run(self.cost, feed_dict={self.x:self.X_test , self.y_:self.Y_test, self.data_num:len(self.X_test), self.keep_prob:1}) 
			#------------------------- evaluation matrices on testing data ---------------------------------------------
			mean_rank, mean_inverse_rank, H10, rank_num = self.evaluation(range(474),"test")
			test_MR.append(mean_rank)
			test_MRR.append(mean_inverse_rank)
			test_H10.append(H10)

			self.output_predict(Iter)
			Iter+=1
# ------------------------- End of Relation Composition Class -------------------------