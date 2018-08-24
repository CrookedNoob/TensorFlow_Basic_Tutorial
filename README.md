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

***This repository will be updated with new codes and tutorials***   
