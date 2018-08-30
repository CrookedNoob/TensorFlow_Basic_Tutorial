# Tensorflow Basic Tutorial
###### This Repository will take you through the basics of TensorFlow

**Installing TensorFlow** <br />
Visit the offical [TensorFlow](https://www.tensorflow.org/install/) webpage for installation guide <br /><br />
_Simple installation steps are: <br />_
- Go to Command Prompt <br />
- Execute the the command: ```pip3 install --upgrade tensorflow``` <br />
(Right now we are not going into the complexity of installing Tersorflow with ***GPU Support***. We will be using Tensorflow with CPU)<br/><br/>

I am assuming that you have installed _Jupyter Notebook_ already. Once Tensorflow is installed with the related dependencies, we will move to the next steps. <br />
 - Open Jupyter Notebook
 - Import _Tensorflow_ by executing 
 ```python
 import tensorflow as tf 
 ```
 - Create [symbolic variables](https://www.ibm.com/support/knowledgecenter/en/SSLTBW_2.1.0/com.ibm.zos.v2r1.ikjb800/ikjb80043.htm) called **placeholders**. We can manipulate them during program execution <br />
  ```python
a =tf.placeholder("float")
b=tf.placeholder("float")
 ``` 
 - Now we will try to implement the below Mathematical Operations:



| Operations    | Description |
|------------------|----|
| tf.multiply    | Multiply |
| tf.add         | Sum |
| tf.subtract    | Substraction |
| tf.divide      | Division |
| tf.mod        | Module |
| tf.abs        |  Returns Absolute Value  |
| tf.negative   |  Returns Negative Value  |
| tf.sign       |  Reeturns Sign  |
| tf.reciprocal |  Returns Reciprocal/Inverse  |
| tf.square     |  Returns Squared Value  |
| tf.round      |  Returns the Nearest Integer  |
| tf.sqrt       |  Returns the Square Root  |
| tf.pow         | Calculates the power |
| tf.exp        |  Calculates the Exponential Value  |
| tf.log        |  Calculate the Logarithm  |
| tf.maximum     | Compares and Returns the Maximum value |
| tf.minimum     | Compares and Returns the Minimum value |
| tf.cos        |  Calculates the cosine  |
| tf.sin        |  Calculates the Sine  |

Example- 
```python
d= tf.add(a,b)
```
- Refer to the file [Tensorflow_Basics.ipynb](https://github.com/crookednoob/Tensorflow_Basics/blob/master/Tensorflow_Basics.ipynb) for the complete code <br/>
- Once you have executed the functions for mathematical operations, Tensors for the respective functions are created e.g. 
> Tensor("Add_7:0", dtype=float32)
- Now we have to create [Session](https://www.tensorflow.org/guide/graphs) and display the final output by assigning values to the *symbolic variables* <br/>
Example-
```python
sess = tf.Session()
print(sess.run(d, feed_dict={a: 3.5, b: 2.5})
```

**Using TensorBoard with TensorFlow**<br/>
TensorBoard is a graph visualisation software which is included by default while installing TensorFlow.
For this part we will use [Tensorflow_n_Tensorboard_Basic.py](https://github.com/crookednoob/Tensorflow_Basics/blob/master/Tensorflow_n_Tensorboard_Basic.py) where we are creating two constants *a* and *b* and assigining them value of *10* and *30* respectively. <br/>
We are performing a quick addition as we did in the first code. <br/>
Now we want to visualize it using TensorBroad. If we want to visualize using TensorBoard for a code with TensorFlow running in backend, we have to create log file where we export the operations. TensorBoard creates visualizations of the graph that we created and also shares certain runtime details of the same. <br/>
Uisng TensorBoard while working on Machine Learning or Deep Learning problems is immensey helpful as it makes it easier to understand.<br/>  
In order to visualize the addition on TensorBoard, we have to add the below line to the code
```python
writer = tf.summary.FileWriter([logdir], [graph])
```
<br/>*logdir* is the path where the log files for the event will be created and the *graph* is the one that we are using for our code. The *graph* can be either user defined or created by default. Inour case it is the default one. For the default graph, we will use,<br/> 
>tf.get_default_graph() 
<br/>

Now we have to open **Terminal** and go the folder path and execute the below code:<br/>
<code class="highlighter-rouge">python Tensorflow_n_Tensorboard_Basic.py</code> 
<br/>Now to visualize using TensorBoard, we have to execute the below code from the same terminal:<br/>
<code class="highlighter-rouge">tensorboard --logdir="./graphs" --port 6006</code> 
<br/><br/>
Once executed, we will get a link like this  *http://DIN19001082:6006*. On opening the link, we will get to see the graphs as below:<br/>
![graph](https://user-images.githubusercontent.com/13174586/44649449-8651d800-aa01-11e8-8c63-3d4cf3896566.JPG)
<br/>This is how the graph looks. Now if we hover over the graphs and click on the elements of it we will get the details as below:<br/>
![graph1](https://user-images.githubusercontent.com/13174586/44649451-8651d800-aa01-11e8-9e1d-b5c7633d5320.JPG)
![graph2](https://user-images.githubusercontent.com/13174586/44649453-86ea6e80-aa01-11e8-85ca-ff56be8a8a8e.JPG)
![graph3](https://user-images.githubusercontent.com/13174586/44649448-8651d800-aa01-11e8-8a1d-7a56f9cc1bc7.JPG)

<br/>We can notice that thought we have assigned constants to *a* and *b* in the code, the TensorBoard shows them as *Const* and *Const_1*
In order to change them on the TensorBoard, we need to edit our code a bit while assigning the constants by:
```python
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = tf.add(a, b, name="sum")
```
<br/>Now, this is how it looks like:<br/>
![graph4](https://user-images.githubusercontent.com/13174586/44650239-ada9a480-aa03-11e8-8565-d94c437594aa.JPG)
<br/><br/>
 We will learn more about TensorBoard later<br/><br/><br/>


***Constant Operations*** *([Refer this file](https://github.com/crookednoob/Tensorflow_Basics/blob/master/Tensorflow_n_Tensorboard_Basic_Constant_Creation.ipynb))* ***:*** <br/>
- Create constant:<br/>
<code class="highlighter-rouge">tf.constant(value, dtype=None, shape=None, name="constant", verify_shape=False)</code>


- Constant of 1D Tensor i.e. Vector<br/>

```python
a= tf.constant([10,20], name='Vector')
```
- Constant of 2X2 tensor i.e. Matrix <br/>

```python
b= tf.constant([[10,20],[30,40]], name='Matrix')
```

- Create Tensor with specific dimension and specific values:<br/>
<code class="highlighter-rouge">tf.zeros([2,3], dtype=tf.int32, name=None)</code>

- Create a Tensor 2x3 i.e. Matrix with zero as all elements<br/>

```python
c= tf.zeros([2,3], dtype=tf.float32, name="Zero")
```

- Create a tensor of shape and type (unless specified) as tensor_shape but all the elements are zeros<br/>

```python
tensor_shape= [[0,1],[2,3],[3,4],[4,5]]
d=tf.zeros_like(tensor_shape)
```

- Create a tensor of any shape and all the elements are 1s:<br/>
<code class="highlighter-rouge">tf.ones(shape, dtype=tf.float32, name=None)</code>

```python
e= tf.ones([3,5], dtype=tf.int32, name='Ones')
```

- Create a tensor of shape and type (unless specified) as tensor_shape but all the elements are 1s:<br/>
<code class="highlighter-rouge">tf.ones_like(shape, name=None)</code>

```python
tensor_shape= [[0,1],[2,3],[3,4],[4,5]]
f=tf.ones_like(tensor_shape)
```

- Create a Tensor and fill it with any scalar value:<br/>
<code class="highlighter-rouge">tf.fill(dims, value, name=None)</code>

```python
g= tf.fill([5,4], 999)
```

- Create Tensor with sequence of constants:<br/>
<code class="highlighter-rouge">tf.lin_space(start, stop, num, name=None)</code>

```python
h= tf.lin_space(100.0, 200.0, 10, name="sequence")
```
- Create a Tensor sequence that increments by delta but does not include the limits:<br/>
<code class="highlighter-rouge">tf.range([start], limit=None, delta=delta, dtype=None, name=None)</code>

```python
i= tf.range(10, limit=20, delta=1, dtype=tf.float32, name="seq1")
j= tf.range(50, limit=10, delta=-10, dtype=tf.float32, name="seq2")
limit=5
k= tf.range(limit)
```

- Generate Random conmstants from certain distributions:<br/>
<code class="highlighter-rouge">tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)</code> - Returns a tensor of the specified shape filled with random normal values <br/>

```python
l= tf.random_normal([2,3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=1, name="norm_dist")
```
<code class="highlighter-rouge">tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)</code> - Returns A tensor of the specified shape filled with random truncated normal values<br/>

```python
m=tf.truncated_normal([3,4], mean=1.5, stddev=1.2, dtype=tf.float32, seed=123, name="trunc_norm")
```

<code class="highlighter-rouge">tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)</code> - Returns A tensor of the specified shape filled with random uniform values<br/>

```python
n= tf.random_uniform([5,5], minval=10, maxval=100, dtype=tf.float32, seed=123, name="rand_uni")
```

<br/><br/>***Mathematical Operations*** *([Refer this file](https://github.com/crookednoob/TensorFlow_Basic_Tutorial/blob/master/Tensorflow_n_Tensorboard_Basic_Math_Dtype_VarType_InteractiveSessn.ipynb)):*

<br/> We have demonstrated few mathematical operations above. Here's few more:
- Computes Python style division of *a* by *b*<br/>
```python
c= tf.divide(a,b)
```
- Returns *a / b* returns the quotient of *a* and *b*<br/>
```python
d= tf.div(a,b)
```
- Returns *a / b* evaluated in floating point. Will return error if both have different data types<br/>
```python
e= tf.truediv(a,b)
```
- Returns *a / b* rounded down (except possibly towards zero for negative integers). Returns error if inputs are complex<br/>
```python
f= tf.floordiv(a,b)
```
- Returns a Tensor. Has the same type as *a*<br/>
```python
g= tf.truncatediv(a,b)
```
- Returns a Tensor. Has the same type as *a*<br/>
```python
h= tf.floor_div(a,b) 
```
- Add *n* number of tensors mentioned as a list<br/>
```python
k= tf.add_n([a,i,j])
```
- Dot Product([Refer this site to know about axes selection](https://www.tensorflow.org/api_docs/python/tf/tensordot))<br/>
```python
m= tf.tensordot(a,i,1)
```
<br/><br/>***Data Types*** *([Refer this file](https://github.com/crookednoob/TensorFlow_Basic_Tutorial/blob/master/Tensorflow_n_Tensorboard_Basic_Math_Dtype_VarType_InteractiveSessn.ipynb)):*
<br/>In *TensorFlow* all the data are in the form of **Tensors** with **Rank**. If it is a *scalar* value, it is known as a *Tensor of Rank 0*. If it is a *vector*, in TensorFlow it is called as *Tensor of Rank 1*. In case of a *matrix*, it is known as *Tensor of Rank 2* and so on.<br/>
```python
sclr=999 
zero_0D= tf.zeros_like(sclr)
one_0D= tf.ones_like(sclr)

with tf.Session() as sess:
    print(sess.run(zero_0D))
    print("\n",sess.run(one_0D))
```
Output:<br/>
> 0 <br/>
> 1 <br>
- [Various Data Types](https://www.tensorflow.org/api_docs/python/tf/DType) used in tensorFlow:
- We can use some of the **Numpy** datatypes in *TensorFlow*
```import numpy as np
tf.ones([5,10], np.float32)

with tf.Session() as sess:
    print(sess.run(tf.ones([5,10], np.float32)))
```
Output:<br/>
> [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]<br/>
> [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]<br/>
> [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]<br/>
> [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]<br/>
> [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]<br/>

<br/><br/>***Variables*** *([Refer this file](https://github.com/crookednoob/TensorFlow_Basic_Tutorial/blob/master/Tensorflow_n_Tensorboard_Basic_Math_Dtype_VarType_InteractiveSessn.ipynb)):*
- We need variables at certain cases where a constant won't work. For example, while creating models, we we to update the *weights* and *biases* while training the data. It is not possible for a constant to do. Hence, we require a variable. We have to create instance for the class <code class="highlighter-rouge">tf.Variable()</code> 
```python
const= tf.constant([2,3], name="constant")
print(tf.get_default_graph().as_graph_def())
```
- Old way to create variables
```python
n= tf.Variable(2, name="Scalar")
o= tf.Variable([[1,2],[3,4]], name="Matrix")
p= tf.Variable(tf.ones([20,10]))
```
- Recommended way to create variables<br/>
<code class="highlighter-rouge">tf.get_variabels(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, collections=None, chaching_device=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None)</code>
<br/>example-<br/>
```python
scl1= tf.get_variable("Scalar1", initializer=tf.constant(100))
matrx1= tf.get_variable("Matrix1", initializer=tf.constant([[1,0],[0,1]]))
massv_matrx1= tf.get_variable("Massive_matrix1", shape=(1000,50), initializer=tf.ones_initializer())
```
***Initialize variables:***
- We have to initialize variables before initializing them else we will get an error 
<code class="highlighter-rouge">FailedPreconditionError: Attempting to use uninitialized value</code>
- To get the list of uninitialized variables we have to execute the below codes
```python
with tf.Session() as sess:
    print(sess.run(tf.report_uninitialized_variables()))
```
- To initialize variables we have to write
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
```
- To initialize a subset of variables
```python
with tf.Session() as sess:
    sess.run(tf.variables_initializer([scl, matrx1]))
```
- To initialize each variables independently
```python
with tf.Session() as sess:
    sess.run(massv_matrx1.initializer)
```
***Evaluate values of variables:***
- To obtain the value of any variable, we have to do it within a Session(just as we do with the Tensors for any rank)
```python
massv_matrx2= tf.get_variable("Massive_matrix2", shape=(1000,50), initializer=tf.glorot_normal_initializer())

with tf.Session() as sess:
    sess.run(massv_matrx2.initializer)
    print(sess.run(massv_matrx2))
```
- We can also fetch variables value using the below code
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(massv_matrx2.eval())
```
***Assign values to variables:***
```python
q= tf.Variable(20)
q.assign(200)
with tf.Session() as sess:
    sess.run(q.initializer)
    print(q.eval())
```
Output:<br/>
> 20

<br/>The above code creates assigns value of *20* to *q* instead of *200*<br/>
In order to assign the value of *200*, we have to do it within session<br/>
```python
with tf.Session() as sess:
    sess.run(q.assign(200))
    print(q.eval())
```
Output:<br/>
> 200 <br/>
<br/>
<code class="highlighter-rouge">assign()</code> 
<br/>itself initializes the variable *q* for us. So we do not need to do initialize it
<br/>

- Create a variable with value 5

```python
five= tf.get_variable("scalar5", initializer=tf.constant(5))
five_times_five= five.assign(five*5)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(five_times_five))
    print(sess.run(five_times_five))
    print(sess.run(five_times_five))
    print(sess.run(five_times_five))
```
Output:<br/>
> 25 <br/>
> 125 <br/>
> 625 <br/>
> 3125 
<br/>

- Different Sessions in Tensorflow store different values of the variables as defined in the graph

```python
u = tf.Variable(10)

with tf.Session() as sess:
    sess.run(u.initializer)
    print(sess.run(u.assign_add(10)))
    print(sess.run(u.assign_sub(2)))

sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(u.initializer)
sess2.run(u.initializer)
print(sess1.run(u.assign_add(10)))
print(sess2.run(u.assign_sub(2)))
print(sess1.run(u.assign_add(100)))
print(sess2.run(u.assign_sub(50)))
sess1.close()
sess2.close()
```
Output:<br/>
> 20 <br/>
> 18 <br/>
> 20 <br/>
> 8 <br/>
> 120 <br/>
> -42 <br/>

<br/>

- Variable dependent on another variable

```python
v= tf.Variable(tf.truncated_normal([100,20]))
w= tf.Variable(v*5)

with tf.Session() as sess:
    sess.run(w.initializer)
    print(sess.run(w))
 ```
 
 - We should always use initialized_value() on the independent varaible before it is used to initialize the dependent variable
 
 ```python
 w= tf.Variable(v.initialized_value()*5)

with tf.Session() as sess:
    sess.run(w.initializer)
    print(sess.run(w))
 ```
 
 <br/><br/>***Interactive Sessions*** *([Refer this file](https://github.com/crookednoob/TensorFlow_Basic_Tutorial/blob/master/Tensorflow_n_Tensorboard_Basic_Math_Dtype_VarType_InteractiveSessn.ipynb)):*
 
 - It is created as a default session where we can call the run() and eval() whithout explicitly calling the session everytime
 
 - Though this looks convenient and easey but it creates problem when we have to work on multiple sessions
 
 ```python
 sess=tf.InteractiveSession()
a=tf.constant(10)
b=tf.constant(30)
c=a*b
print(c.eval())
sess.close()
```
Output:<br/>
> 300
<br/>

<br/><br/>***Importing Data*** *([Refer this file](https://github.com/crookednoob/TensorFlow_Basic_Tutorial/blob/master/Tensorflow_n_Tensorboard_Basic_Math_Dtype_VarType_InteractiveSessn.ipynb)):*
<br/>As mentioned earlier a TensorFlow program has two parts:

- **Step1:** Create a graph

- **Step2:** Evaluate variables and execute operations using a session 

```
a= tf.placeholder(tf.float32, shape=[5])
b=tf.constant([12,13,1,23,5], tf.float32)
summation= a+b
with tf.Session() as sess:
    print(sess.run(summation, {a: [1,2,3,4,5]})) #declare the values of the placeholder
```
Output:<br/>
> [13. 15.  4. 27. 10.]

<br/><br/><br/><br/><br/><br/>***Please visit the other folders for more***  
###### Reference: [TensorFlow API Docs](https://www.tensorflow.org/api_docs/python/tf)
