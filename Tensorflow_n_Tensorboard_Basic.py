
# coding: utf-8

# In[1]:


#Writing a simple TensorFlow code and visualize it on TensorBoard


# In[9]:


import tensorflow as tf

a = tf.constant(10)
b = tf.constant(30)
x = tf.add(a, b)
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(x))
writer.close()

