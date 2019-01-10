import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm_notebook

# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, x, y_):

        in_dim = int(x.get_shape()[1]) 
        out_dim = int(y_.get_shape()[1])

        self.x = x 

        
        W1 = weight_variable([in_dim,50])
        b1 = bias_variable([50])

        W2 = weight_variable([50,out_dim])
        b2 = bias_variable([out_dim])

        h1 = tf.nn.relu(tf.matmul(x,W1) + b1) 
        self.y = tf.matmul(h1,W2) + b2 

        self.var_list = [W1, b1, W2, b2]

        
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
        self.set_vanilla_loss()

        
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def compute_fisher(self, imgset, sess, num_samples=200):
        
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        for i in tqdm_notebook(range(num_samples)):
            
            im_ind = np.random.randint(imgset.shape[0])
            
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
            
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
                
       
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples

    def star(self):
        
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))
    
    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

    def update_ewc_loss(self, lam):
        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = self.cross_entropy

        for v in range(len(self.var_list)):
            self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)



