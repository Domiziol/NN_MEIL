# Dominika Ziółkiewicz
# 313622
# Nr. zestawu 37
# Zad 1: Darken & Moody zaimplementowane jako zmiana stałej uczenia dla każdej iteracji nie epoki
# Zad 2: Momentum zaimplementowane jako dodatkowa część w strukturze sieci przechowująca ostatnie aktualizacje każdej z wag


import numpy as np
import matplotlib.pyplot as plt

np. random.seed(100)

#  Neuron class = calculation of: output, activation potential, activation functions
class activation_fcn(object):

    def __init__(self):
        pass
    
    # Claculate neuron output
    def output(self, layer, name, derivative=False):
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, str(name), "Invalid_function")
        # Call the method as we return it
        return method(layer, derivative)

    # Display error if wrong activation function name is selected
    def Invalid_function(self, *arg):
        print("Error: Invalid activation function")
        return None

    # Identity activation function
    def linear(self, layer, derivative=False):
        out = 0
        if not derivative:
            out = layer['activation_potential']
        else:
            out = np.ones(shape=np.shape(layer['activation_potential']))
        return out

    # Logistic (sigmoid) activation function
    def logistic(self, layer, derivative=False):
        out = 0
        if not derivative:
            out = 1.0 / (1.0 + np.exp(-layer['activation_potential']))
        else:
            out = layer['output'] * (1.0 - layer['output'])
        return out

    # Hyperbolic tangent activation function  
    def tanh(self, layer, derivative=False):
        out = 0
        if not derivative:
            out = (np.exp(layer['activation_potential']) - np.exp(-layer['activation_potential'])) / (
                    np.exp(layer['activation_potential']) + np.exp(-layer['activation_potential']))
        else:
            out = 1.0 - np.power(layer['output'], 2)
        return out

    #  ReLU activation function 
    def relu(self, layer, derivative=False):
        out = 0
        if not derivative:
            out = np.maximum(0, layer['activation_potential'])
        else:
            out = layer['activation_potential'] >= 0  
        return out

#  Loss function class
class loss_fcn(object):

    def __init__(self):
        pass

    # Loss/error value calculated for all input data sample
    def loss(self, loss, expected, outputs, derivative):
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, str(loss), lambda: "Invalid loss function")
        # Call the method as we return it
        return method(expected, outputs, derivative)

    # Display error if wrong loss function name is selected
    def Invalid_function(self, *arg):
        print("Error: Invalid loss function")
        return None

    # Mean Square Error loss function
    def mse(self, expected, outputs, derivative=False):
        error_value = 0
        if not derivative:
            error_value = 0.5 * np.power(expected - outputs, 2)
        else:
            error_value = -(expected - outputs)
        return error_value

    # Cross-entropy loss function
    def binary_cross_entropy(self, expected, outputs, derivative=False):
        error_value = 0
        if not derivative:
            error_value = -expected * np.log(outputs) - (1 - expected) * np.log(1 - outputs)
        else:
            error_value = -(expected / outputs - (1 - expected) / (1 - outputs))
        return error_value


class learn_fcn(object):
    def __init__(self, l_rate, tau):
        self.start = l_rate
        self.tau = tau
    
    def learn(self, learn, iter):
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, str(learn), lambda: "Invalid loss function")
        # Call the method as we return it
        return method(iter)

    # Display error if wrong loss function name is selected
    def Invalid_function(self, *arg):
        print("Error: Invalid train function")
        return None
    
    def darken_moody(self, iter):                                                                        # Darken & Moody
        return self.start/(1+iter/self.tau)


# Initialize a network
class Neural_network(object):

    def __init__(self):
        pass

                    # Momentum method
    # Została ona zaimplementowana jako dodatkowa część w strukturze nnetwork - last_update o rozmiarze równym rozmiarowi weights. 
    # Na początku last_update wypełniony zerami, z każdym przejsciem przez update_weights zostaje nadpisany również ten parametr
    def create_network(self, structure):
        self.nnetwork = [structure[0]]
        for i in range(1, len(structure)):
            new_layer = {
                'weights': np.random.randn(structure[i]['units'], structure[i-1]['units'] + structure[i]['bias']),
                'bias': structure[i]['bias'],
                'activation_function': structure[i]['activation_function'],
                'activation_potential': None,
                'delta': None,
                'output': None,
                'last_update': np.zeros(((structure[i]['units']), structure[i-1]['units']+ structure[i]['bias']), float)}
            self.nnetwork.append(new_layer)
        return self.nnetwork

    # Forward propagate input to a network output
    def forward_propagate(self, nnetwork, inputs):
        # Network input values from dataset
        af = activation_fcn()
        inp = inputs.copy()
        for i in range(1, len(nnetwork)):
            if nnetwork[i]['bias']==True:
                inp = np.append(inp, 1)
            # Storage of network outputs from present layer of network
            nnetwork[i]['activation_potential'] = np.matmul(nnetwork[i]['weights'], inp).flatten()
            nnetwork[i]['output'] = af.output(nnetwork[i], nnetwork[i]['activation_function'], derivative=False)
            inp = nnetwork[i]['output']
        return inp


    # Backpropagate error and store it in neuron
    def backward_propagate(self, loss_function, nnetwork, expected):
        # Error prooagation will start from last layer
        af = activation_fcn()
        loss = loss_fcn()
        N = len(nnetwork)-1
        for i in range(N, 0, -1):
            # Storage of error values from present layer
            errors = []
            # Calculation of error values for other layers than last layer 
            if i<N:
                weights = nnetwork[i+1]['weights']
                if nnetwork[i+1]['bias']==True:
                    weights = weights[:, :-1]
                errors = np.matmul(nnetwork[i+1]['delta'], weights)
            # Calculation of error values for last layer
            else:                
                errors = loss.loss(loss_function, expected, nnetwork[-1]['output'], derivative=True)
            
            nnetwork[i]['delta'] = np.multiply(errors, af.output(nnetwork[i], nnetwork[i]['activation_function'], derivative=True))
                

    # Update network weights with error
    def update_weights(self, nnetwork, inputs, l_rate, alfa):
        inp = inputs
        for i in range(1, len(nnetwork)):           
            if nnetwork[i]['bias']==True:
                inp = np.append(inp, 1)
            nnetwork[i]['weights'] -= (l_rate * np.matmul(nnetwork[i]['delta'].reshape(-1,1), inp.reshape(1,-1)) + alfa*nnetwork[i]['last_update'])     # Momentum
            nnetwork[i]['last_update'] = l_rate * np.matmul(nnetwork[i]['delta'].reshape(-1,1), inp.reshape(1,-1)) + alfa*nnetwork[i]['last_update']   # nadpisanie dla trzymainia ostatniej wartości aktualizacji
            inp = nnetwork[i]['output']
            

    # Train a network for a fixed number of epochs
            # Darken & Moody
    # Przekazujemy jako argument string: 'darken_moody'; działanie jest takie samo jak np. dla loss_fcn
    # Tworzymy obiekt learn_fcn i przypisujemy pierwotną wartosć stałej uczenia.
    # Następnie przed aktualizacją wag wywołujemy metodę darken_moody, która zwraca stałą uczenia odpowiednią dla danej iteracji (iter przekazujemy jako argument)
    def train(self, nnetwork, x_train, y_train, l_rate=0.01, lrate_method = 'darken_moody', tau = 100, alfa = 0.1, n_epoch=100, loss_function='mse', verbose=1):
        start_l_rate = l_rate
        lrn = learn_fcn(start_l_rate, tau)
        for epoch in range(n_epoch):
            sum_error = 0
            for iter, (x_row, y_row) in enumerate(zip(x_train, y_train)):

                self.forward_propagate(nnetwork, x_row)
                
                loss = loss_fcn()
                sum_error = np.sum(loss.loss(loss_function, y_row, nnetwork[-1]['output'], derivative=False))

                self.backward_propagate(loss_function, nnetwork, y_row)

                l_rate = lrn.learn(lrate_method, iter)  # nowe l_rate wyznaczone przez lrate_method

                self.update_weights(nnetwork, x_row, l_rate, alfa)                               

            if verbose > 0:
                print('>epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
        print('Results: epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
        return nnetwork

    # Calculate network output
    def predict(self, nnetwork, inputs):
        out = []
        for input in inputs:
            out.append(self.forward_propagate(nnetwork, input))
        return out


def generate_regression_data(n=30):
    # Generate regression dataset
    X = np.linspace(-5, 5, n).reshape(-1, 1)
    y = np.sin(2 * X) + np.cos(X) + 5
    # simulate noise
    data_noise = np.random.normal(0, 0.2, n).reshape(-1, 1)
    # Generate training data
    Y = y + data_noise

    return X.reshape(-1, 1), Y.reshape(-1, 1)


def test_regression():
    # Read data
    X, Y = generate_regression_data()

    # Create network
    model = Neural_network()
    structure = [{'type': 'input', 'units': 1},
                 {'type': 'dense', 'units': 8, 'activation_function': 'tanh', 'bias': True},
                 {'type': 'dense', 'units': 8, 'activation_function': 'tanh', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'linear', 'bias': True}]

    network = model.create_network(structure)
                                    # learn_method   tau  alfa
    model.train(network, X, Y, 0.01, 'darken_moody', 100, 0.1, 4000, 'mse', 0)

    predicted = model.predict(network, X)
    std = np.std(predicted - Y)
    print("\nStandard deviation = {}".format(std))

    X_test = np.linspace(-7, 7, 100).reshape(-1, 1)
    X_test = np.array(X_test).tolist()
    predicted = model.predict(network, X_test)

    plt.plot(X, Y, 'r--o', label="Training data")
    plt.plot(X_test, predicted, 'b--x', label="Predicted")
    plt.legend()
    plt.grid()
    plt.show()


def generate_classification_data(n=30):
    # Class 1 - samples generation
    X1_1 = 1 + 4 * np.random.rand(n, 1)
    X1_2 = 1 + 4 * np.random.rand(n, 1)
    class1 = np.concatenate((X1_1, X1_2), axis=1)
    Y1 = np.ones(n)

    # Class 0 - samples generation
    X0_1 = 3 + 4 * np.random.rand(n, 1)
    X0_2 = 3 + 4 * np.random.rand(n, 1)
    class0 = np.concatenate((X0_1, X0_2), axis=1)
    Y0 = np.zeros(n)

    X = np.concatenate((class1, class0))
    Y = np.concatenate((Y1, Y0))
    
    idx0 = [i for i, v in enumerate(Y) if v == 0]
    idx1 = [i for i, v in enumerate(Y) if v == 1]

    return X, Y, idx0, idx1


def test_classification():
    # Read data
    X, Y, idx0, idx1 = generate_classification_data()

    # Create network
    model = Neural_network()
    structure = [{'type': 'input', 'units': 2},
                 {'type': 'dense', 'units': 4, 'activation_function': 'relu', 'bias': True},
                 {'type': 'dense', 'units': 4, 'activation_function': 'relu', 'bias': True},
                 {'type': 'dense', 'units': 1, 'activation_function': 'logistic', 'bias': True}]

    network = model.create_network(structure)
                                     # learn_method    tau  alfa
    model.train(network, X, Y, 0.0001, 'darken_moody', 100, 0.1, 2000, 'binary_cross_entropy', 0)

    y = model.predict(network, X)
    t = 0
    for n, m in zip(y, Y):
        t += 1 - np.abs(np.round(np.array(n)) - np.array(m))
        print(f"pred = {n}, pred_round = {np.round(n)}, true = {m}")

    ACC = t / len(X)
    print(f"\nClassification accuracy = {ACC * 100}%")

    # Plotting decision regions
    xx, yy = np.meshgrid(np.arange(0, 8, 0.1),
                         np.arange(0, 8, 0.1))

    X_vis = np.c_[xx.ravel(), yy.ravel()]

    h = model.predict(network, X_vis)
    h = np.array(h) >= 0.5
    h = np.reshape(h, (len(xx), len(yy)))

    plt.contourf(xx, yy, h, cmap='jet')
    plt.scatter(X[idx1, 0], X[idx1, 1], marker='^', c="red", edgecolors="white", label="class 1")
    plt.scatter(X[idx0, 0], X[idx0, 1], marker='o', c="blue", edgecolors="white", label="class 0")
    plt.show()


generate_classification_data()
test_classification()

generate_regression_data(30)
test_regression()
