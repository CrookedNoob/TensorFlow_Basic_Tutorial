# Simple Linear Regression with TensorFlow

### Simple Linear Regression<br/>
It is a mathematical equation(generally of a straight line: **y**= *m* **x** + *c*, where **y** is *ordinate/vertical axis*, **x** is the *horizontal axis* and **m** is the *slope* and **c** is the *intercept*)<br/><br/>
In the case of Machine Learning the obove equation is represented as:<br/>
###### **y** *= w* **X** *+ b*<br/>
where **Y** is the value we want to *predict*, **X** is the *independent variable/predictor* and **w** is the *weight* and **b** is the *bias*<br/>

#### Problem Statement:<br/>
Find the relationship between Life Expectancy of a child and Birth Rate. For more description, visit the [Link](https://www.google.com/publicdata/explore?ds=d5bncppjof8f9_&ctype=b&strail=false&nselm=s&met_x=sp_dyn_le00_in&scale_x=lin&ind_x=false&met_y=sp_dyn_tfrt_in&scale_y=lin&ind_y=false&met_s=sp_pop_totl&scale_s=lin&ind_s=false&dimp_c=country:region&ifdim=country&iconSize=0.5&uniSize=0.035#!ctype=b&strail=false&bcs=d&nselm=s&met_x=sp_dyn_le00_in&scale_x=lin&ind_x=false&met_y=sp_dyn_tfrt_in&scale_y=lin&ind_y=false&met_s=sp_pop_totl&scale_s=lin&ind_s=false&dimp_c=country:region&ifdim=country&pit=1421395200000&hl=en_US&dl=en_US&ind=false)

| Data Dictionary         | Description     |
|-------------------------|-----------------|
| Predictor (X)           | Birth Rate      |
| Dependent (Y)           | Life Expectancy |
| Examples (Observations) | 190             |
| File Name               | birth_rate.txt  |


**Assumption:** The relationship between "Birth Rate" and "Life Expectancy" is linear<br/>
- In this case study, the equation will be ***Y = wX + b***, where **Y**= *Life Expectancy*, **X**= *Birth Rate*, **w**= *Weight(scalar)*, **b**= *Bias(scalar)* <br/>
- **w**, **b** will be calculated using *One Layer Neural Network with BackPropagation Technique* <br/>
- Loss to be calculated using ***MSE*** *(Mean Squared Error)*
- *MSE* will be calculated after each *epoch*
- Refer [this file](https://github.com/crookednoob/TensorFlow_Basic_Tutorial/blob/master/Regression/Tensorflow_Simple_Linear_Regression.py) for the complete code
- With tf.data, we can created a Dataset from tensors with- 
```python
tf.data.Dataset.from_tensor_slices((features, labels))
```
- **features** and **labels** are supposed to be tensors. Since TensorFlow and Numpy are seamlessly integrated, they can be NumPy arrays. We can initialize our dataset as-
```python
dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
```

![simpllinreg](https://user-images.githubusercontent.com/13174586/44777880-4b82a800-ab99-11e8-823b-d1be549d67e8.JPG)
<br/>The straight line represents the regression model on our data. As we can see that *Life Expectancy* is negatively correlated to *Birth Rate*, we can safely assume that more birth rate causes higher probaliblity for the death of the younger child.<br/>

![graph_linreg](https://user-images.githubusercontent.com/13174586/44777878-4aea1180-ab99-11e8-8f69-6258317ef322.png)
<br/>This is how the graph for our simple regression model looks like.

<br/><br/><br/><br/><br/><br/><br/><br/>
###### Reference: <br/>
[Tensorflow](https://www.tensorflow.org/)
