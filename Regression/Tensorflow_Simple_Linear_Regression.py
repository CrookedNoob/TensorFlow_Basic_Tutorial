
# coding: utf-8

# In[1]:


#Simple Linear Regression with TensorFlow


# In[2]:


#Predict Life Expectancy from Birth Rate


# In[3]:


#Dataset- https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/data/birth_life_2010.txt
#Data Dictionary:
#File Name: birth_rate.txt
#Predictor: Birth rate
#Dependent: Life Expectancy

#Assumption:
#The relationship between "birth rate" and "Life Expectancy" is linear
#Y = wX + b, where Y= Life Expectancy, X= Birth Rate, w= Weight(scalar), b= Bias(scalar)
#w, b will be calculated using One Layer Neural Network with BackPropagation Technique
#Loss to be calculated using MSE(Mean Squared Error)
#MSE will be calculated after each epoch


# In[4]:


import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt


# In[5]:


#Import the data
File= "C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\Tensorflow_Basics\\birth_rate.txt"

def read_birth_data(filename):
    text= open(filename, 'r').readlines()[1:] #Open the file 
    data= [line[:-1].split('\t') for line in text] #Ignore the first row as it contains row names
    births= [float(line[1]) for line in data] #store birth rate
    lifes= [float(line[2]) for line in data] #store life expectancy
    data= list(zip(births, lifes)) #Bind birth rate and life expectancy in a list
    n_samples= len(data) #fin the total number of observations 
    data= np.asarray(data, dtype=np.float32) #Converting the list into an array of data type float32
    return data, n_samples

data, n_samples = read_birth_data(File)


# In[6]:


#Create Dataset and Iterator
dataset= tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
iterator= dataset.make_initializable_iterator()
X,Y = iterator.get_next()


# In[7]:


#Create Weight(w) and Bias(b) and initialize them to zero
w= tf.get_variable("weight", initializer=tf.constant(0.0))
b= tf.get_variable("bias", initializer=tf.constant(0.0))


# In[8]:


#Build a model to predict Life Expectancy(Y) from Birth Rate(X)
Y_predicted= w*X + b


# In[9]:


#MSE to calculate loss
loss= tf.square(Y- Y_predicted, name="loss")


# In[10]:


#Using Gradient Descent with learning rate of 0.01 to minimise loss
optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


# In[11]:


start= time.time()
with tf.Session() as sess:
    #Initialize the variable w and b
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('C:\\Users\\soumyama\\Documents\\Python Scripts\\personal tutorial\\Tensorflow_Basics\\graphs\\linreg', sess.graph)
    #train the model
    for i in range(100): #epochs
        sess.run(iterator.initializer) #Initialize the iterator
        total_loss= 0
        try:
            while True:
                _, l= sess.run([optimizer, loss])
                total_loss += 1
        except tf.errors.OutOfRangeError:
            pass
        
        
    writer.close()

 # Step 9: output the values of w and b
    w_out, b_out = sess.run([w, b]) 
    print('w: %f, b: %f' %(w_out, b_out))
print('Took: %f seconds' %(time.time() - start))

    # plot the results
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data with squared error')

plt.legend()
plt.show()


# In[12]:


#This Image shows negative correlation between birth rate and life expectancy

