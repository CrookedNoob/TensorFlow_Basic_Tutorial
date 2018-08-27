# Tensorflow Basic Tutorial
###### This Repository will take you through the basics of Tensorlow

**Installing Tensorflow** <br />
Visit the offical [Tensorflow](https://www.tensorflow.org/install/) webpage for installation guide <br /><br />
_Simple installation steps are: <br />_
- Go to Command Prompt <br />
- Execute the the command: ```pip3 install --upgrade tensorflow``` <br />
(Right now we are not going into the complexity of installing Tersorflow with ***GPU Support***. We will be using Tensorflow with CPU)<br/><br/>

I am assuming that you have installed _Jupyter Notebook_ already. Once Tensorflow is installed on your system with the related dependencies, we will move to the next steps. <br />
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
<br/>*logdir* is the path where the log files for the event will be created and the *graph* is the one taht we are using for our code. The *graph* can be either user defined or created by default. Inour case it is the default one. For the default graph, we will use,<br/> 
>tf.get_default_graph() 
<br/>

Now we have to open **Terminal** and go the foleder path and execute the below code:<br/>
<code class="highlighter-rouge">python Tensorflow_n_Tensorboard_Basic.py</code> 
<br/>Now to visualize using TensorBoard, we have to execute the below code from the same terminal:<br/>
<code class="highlighter-rouge">tensorboard --logdir="./graphs" --port 6006</code> 
<br/><br/>
Once executed, we will get a line like this  *http://DIN19001082:6006*. On opening the link, we will get to see the graphs as below:<br/>
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
<br/><br/><br/>***This repository will be updated with new codes and tutorials***   
