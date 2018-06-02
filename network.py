"""
Most of this code is based on the code provided in chapter 1
of Michael Nielsen's excellent book "Neural Networks and deep learning". For learning purposes I have tweaked things here and there.
"""
import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
            
    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
            
    def SGD(self, training_data, epochs, mini_batch_size, eta, val_data=None):
        n_train = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if val_data:
                print("Epoch: {0}, train err: {1:.4f} val err: {2:.4f}".format(j, self.evaluate(training_data, train=True), self.evaluate(val_data)))
            else:
                print('Epoch: {0}, train err: {1:.4f}'.format(j, self.evaluate(training_data, train=True)))
        print('Done')
    
    def update_mini_batch(self, mini_batch, eta):        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data, train=False):
        """ Function to evaluate model.

        Parameters
        ----------
        test_data : List
            test/validation data. 
        train : bool (optional)
            True if model should be evaluated on train. Default : False  
        Returns
        -------
        Accuracy : int 
            Accuracy of the network model w.r.t. to the data provided.

        """
        test_results = [(np.argmax(self.feedForward(x)), y) for (x, y) in test_data ]

        if train:
            accuracy = (sum(int(x==np.argmax(y)) for (x, y) in test_results) / len(test_data)) 
        else :
            accuracy = (sum(int(x==y) for (x, y) in test_results) / len(test_data))  
        return accuracy
    
    def predict(self, test_data):
        """Predicts class for the test data instance.

        Parameters
        ----------
        test_data: tuple
            In the form (x, y) where x is an array of pixel darkness and y is the actual output.

        Returns
        -------
        y_hat : int
            The class or number preddicted by model. 
        y : int
            Actual class. 

        """
        x, y = test_data
        y_hat = np.argmax(self.feedForward(x)) 
        return (y_hat, y) 

    def predict_proba_batch(self, test_data):  
        predictions = [{'y_hat' :self.feedForward(x), 'y': y} for (x, y) in test_data]
        return predictions

    def getn_misclassified(self, test_data, n=None):
        """ Function to get n misclassified images. 
        Since later on image data can get modified, for better performance, 
        we will try to use the images from the input test_data itself rather 
        than the whatever data that goes into the model. There is a random 
        shuffle, so that the same images are not shown every time. 

        Parameters
        ----------
        test_data : List
            data on which analysis is to be done
        n : int (optional)
            number of image indexes which need to be returned. Default : None

        Returns
        -------
        misclassified_img_x : List of array
            List of mnist image data which have been misclassified. 
        misclassified_img_pred : List of dicts
            Dicts contains the following attributes 
            y : actual class or number
            y_hat : class predicted by model
            conf : the confidence the model gave for y_hat 

        """
        random.shuffle(test_data)
        pred = self.predict_proba_batch(test_data)
        misclassified_img_pred = [{'y_hat':np.argmax(t['y_hat']), 'y':t['y'], 'conf': max(t['y_hat']),'idx':idx} 
                for idx, t in enumerate(pred) if (np.argmax(t['y_hat']) != t['y']) ]
        misclassified_img_x = [ test_data[t['idx']][0] for t in misclassified_img_pred ]

        return misclassified_img_x[:n], misclassified_img_pred[:n]

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


def sigmoid(z):
    return 1.0/ (1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
