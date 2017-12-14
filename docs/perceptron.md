
## Perceptron

### Definission
The perceptron is an algorithm for **supervised learning** of binary classifiers. It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

![The perceptron](https://raw.githubusercontent.com/Claudiooo/DeepLearningLearning/Group2/Images/perceptron_schematic_overview.png)

### Implementation

The purpose of this example is to find if a point is above or below a defined line with an affine function. This algorithm uses the **gradient descent** algorithm associated with a perceptron to learn how to classify the data. See *[Linear Regression](https://plot.ly/~nicolasconstanty/10)* for more informations about **gradient descent**


```python
import random

def activationFunc(number):
    return 1 if number > 0 else -1
    
class Perceptron:
    def __init__(self, size, learning_rate):
        self.size = size
        self.weights = []
        self.lr = learning_rate
        for i in range(size):
            self.weights.append(random.uniform(-1.0, 1.0))
            
    def guess(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i] * self.weights[i]
        return activationFunc(sum)
    
    def train(self, inputs, target):
        value = self.guess(inputs)
        error = target - value
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.lr
            
class Point:
    def __init__(self, func):
        self.x = random.uniform(-1.0, 1.0)
        self.y = random.uniform(-1.0, 1.0)
        if (self.y > func(self.x)):
            self.label = 1
        else:
            self.label = -1
        self.bias = 1
        
```


```python
total_iteration = 0

def initialise_data(size, function):
    data = []
    for i in range(size):
        data.append(Point(function))
    return data

def run(data, number_iteration, learning_rate):
    return run_trained(data, number_iteration, Perceptron(3, learning_rate))

def run_trained(data, number_iteration, perceptron):
    global total_iteration
    total_iteration += number_iteration
    print("Training start...")
    for i in range(number_iteration):
        for point in data:
            inputs = [point.x, point.y, point.bias]
            perceptron.train(inputs, point.label)
    print("After {0} iterations".format(total_iteration))
    return perceptron

def calculate_success_percentage(data, perceptron):
    trues = 0
    for point in data:
        inputs = [point.x, point.y, point.bias]
        if (point.label - perceptron.guess(inputs) == 0):
            trues += 1
    return trues / float(len(data))

def display_success_percentage(number):
    print("Success : {0} %".format(number))

if __name__ == "__main__":
    # y = mx + b
    func = lambda x : 2 * x + 2
    size = 1000
    data = initialise_data(size, func)
    p = run(data, 1, 0.001)
    percent = calculate_error_percentage(data, p) * 100
    display_success_percentage(percent)
    while percent != 100:
        p = run_trained(data, 1, p)
        percent = calculate_error_percentage(data, p) * 100
        display_success_percentage(percent)
    
```

    Training start...
    After 1 iterations
    Success : 83.0 %
    Training start...
    After 2 iterations
    Success : 87.1 %
    Training start...
    After 3 iterations
    Success : 88.3 %
    Training start...
    After 4 iterations
    Success : 89.4 %
    Training start...
    After 5 iterations
    Success : 94.0 %
    Training start...
    After 6 iterations
    Success : 96.0 %
    Training start...
    After 7 iterations
    Success : 97.3 %
    Training start...
    After 8 iterations
    Success : 98.2 %
    Training start...
    After 9 iterations
    Success : 98.4 %
    Training start...
    After 10 iterations
    Success : 98.9 %
    Training start...
    After 11 iterations
    Success : 98.9 %
    Training start...
    After 12 iterations
    Success : 99.2 %
    Training start...
    After 13 iterations
    Success : 99.2 %
    Training start...
    After 14 iterations
    Success : 99.2 %
    Training start...
    After 15 iterations
    Success : 99.3 %
    Training start...
    After 16 iterations
    Success : 99.3 %
    Training start...
    After 17 iterations
    Success : 99.4 %
    Training start...
    After 18 iterations
    Success : 99.5 %
    Training start...
    After 19 iterations
    Success : 99.5 %
    Training start...
    After 20 iterations
    Success : 99.5 %
    Training start...
    After 21 iterations
    Success : 99.5 %
    Training start...
    After 22 iterations
    Success : 99.6 %
    Training start...
    After 23 iterations
    Success : 99.6 %
    Training start...
    After 24 iterations
    Success : 99.6 %
    Training start...
    After 25 iterations
    Success : 99.6 %
    Training start...
    After 26 iterations
    Success : 99.6 %
    Training start...
    After 27 iterations
    Success : 99.7 %
    Training start...
    After 28 iterations
    Success : 99.7 %
    Training start...
    After 29 iterations
    Success : 99.8 %
    Training start...
    After 30 iterations
    Success : 99.8 %
    Training start...
    After 31 iterations
    Success : 99.9 %
    Training start...
    After 32 iterations
    Success : 99.9 %
    Training start...
    After 33 iterations
    Success : 100.0 %
    

### Links

#### Youtube
 * [Siraj Raval - How to Make a Neural Network - Intro to Deep Learning #2](https://www.youtube.com/watch?v=p69khggr1Jo&index=3&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3)
 * [The Coding Train - 10.2: Neural Networks: Perceptron Part 1](https://www.youtube.com/watch?v=ntKn5TPHHAk&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh&index=2)
 * [The Coding Train - 10.3: Neural Networks: Perceptron Part 2](https://www.youtube.com/watch?v=DGxIcDjPzac&index=3&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)
 
#### Web
 * [Wikipedia - Perceptron](https://www.wikiwand.com/fr/Perceptron)
 * [ataspinar - The perceptron](http://ataspinar.com/2016/12/22/the-perceptron/)
