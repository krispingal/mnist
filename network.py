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
            
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1}".format(j, self.evaluate(test_data)))
            else:
                print(f'Epoch {j} complete')
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

    def evaluate(self, test_data):
        """ Function to evaluate model.

        Parameters
        ----------
        test_data: test/validation data (List).
        
        Returns
        -------
        Accuracy: Accuracy of the network model w.r.t. to the data provided (int).

        """
        test_results = [(np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
        accuracy = (sum(int(x==y) for (x, y) in test_results) / len(test_data))  
        return accuracy
    
#    def get_test_result(self, test_data):
        """ Function to get test results including top n misclassified as well as correctly classified images.

        Parameters
        ----------
        test_data: test/validation data (List)

        Returns
        -------

        """

    def evaluate_test(self, test_data):  
        predictions = [{'y_hat' :self.feedForward(x), 'y': y} for (x, y) in test_data]
        return predictions

    def top_correctly_classified(self, test_data, n=5):
        """ Function to get the top correctly classified images. 
        Since later on image data can get modified, for better performance, 
        we will try to use the images from the input test_data itself rather than the whatever data that goes into the model.

        """
        pred = self.evaluate_test(test_data)
        correctly_classified_img = [{'y_hat':t['y_hat'], 'y':t['y'], 'idx':idx} for idx, t in enumerate(pred) if (np.argmax(t['y_hat']) == t['y'])] 
        #Create an array of their max predictions which is correct
        max_pred = np.concatenate([max(t['y_hat']) for t in correctly_classified_img]).ravel()
        #max_ids gives the idx in correctly_classified list that is a ordered set (not fully ordered though)
        max_idx = np.argpartition(max_pred, -n)
        #indexing is wrong in max_idx[:n]
        max_elem = [ correctly_classified_img[idx] for idx in max_idx [-n:] ]
        topn_idx = [ t['idx'] for t in max_elem ]
        return topn_idx 

    def top_misclassified(self, test_data, n=5):
        """ Function to get the top misclassified images. 
        Since later on image data can get modified, for better performance, 
        we will try to use the images from the input test_data itself rather than the whatever data that goes into the model.

        Parameters
        ----------
        test_data: data on which analysis is to be done (List)
        n: number of image indexes which need to be returned.

        Returns
        -------
        indices of the top n misclassified images in the given test data.
        """
        pred = self.evaluate_test(test_data)
        misclassified_img = [{'y_hat':t['y_hat'], 'y':t['y'], 'idx':idx} for idx, t in enumerate(pred) if (np.argmax(t['y_hat']) != t['y']) ] 
        #Create an array of their max predictions which is correct
        min_pred = np.concatenate([t['y_hat'][t['y']] for t in misclassified_img]).ravel()
        #max_ids gives the idx in correctly_classified list that is a ordered set (not fully ordered though)
        min_idx = np.argpartition(min_pred, n)
        #indexing is wrong in max_idx[:n]
        min_elem = [ misclassified_img[idx] for idx in min_idx [:n] ]
        topn_idx = [ t['idx'] for t in min_elem ]
        return topn_idx 

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


def sigmoid(z):
    return 1.0/ (1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
