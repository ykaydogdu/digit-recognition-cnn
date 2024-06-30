#include "network.hpp"
#include <cstdlib>
#include <bits/stdc++.h>
#include <iostream>
#include <chrono>
#include <sstream>

double sigmoid(double z) { return 1 / (1 + exp(-z));};
double sigmoid_prime(double z) { return sigmoid(z) * (1 - sigmoid(z));};

Network::Network(int* sizes, int n)
{
    numLayers = n;
    this->sizes = sizes;

    biases = new ygzVector<double>[numLayers];
    weights = new ygzMatrix<double>[numLayers];

    // initialize random weights and biases for each layer
    biases[0] = ygzVector<double>(sizes[0]); // not used
    weights[0] = ygzMatrix<double>(sizes[1], sizes[0]); // not used
    for (int l = 1; l < numLayers; l++)
    {
        biases[l] = ygzVector<double>::randn(sizes[l]);
        weights[l] = ygzMatrix<double>::randn(sizes[l], sizes[l - 1]);  
    }
}

Network::Network(std::ifstream &weightsFile, std::ifstream &biasesFile)
{
    weightsFile >> numLayers;
    sizes = new int[numLayers];
    for (int i = 0; i < numLayers; i++)
    {
        weightsFile >> sizes[i];
    }

    // check for biases and weights file mismatch
    int x;
    biasesFile >> x;
    int* biasSizes = new int[x];
    for (int i = 0; i < x; i++)
    {
        biasesFile >> biasSizes[i];
    }
    if (x != numLayers)
    {
        throw std::invalid_argument("Biases file and weights file mismatch\n");
        return;
    }
    bool control = true;
    for (int i = 0; i < numLayers; i++)
    {
        if (sizes[i] != biasSizes[i])
        {
            control = false;
            break;
        }
    }
    if (!control)
    {
        throw std::invalid_argument("Biases file and weights file mismatch\n");
        return;
    }
    delete[] biasSizes;


    biases = new ygzVector<double>[numLayers];
    weights = new ygzMatrix<double>[numLayers];

    biases[0] = ygzVector<double>(sizes[0]); // not used
    weights[0] = ygzMatrix<double>(sizes[1], sizes[0]); // not used
    std::string row, el;
    getline(biasesFile, row); // skip the first line (sizes of the layers)
    getline(weightsFile, row); // skip the first line (sizes of the layers)
    for (int l = 1; l < numLayers; l++)
    {
        // bias read
        double* biasArr = new double[sizes[l]];
        row.clear();
        getline(biasesFile, row);
        std::stringstream ss(row);
        int i = 0;
        while (getline(ss, el, ','))
        {
            biasArr[i++] = std::stod(el);
        }
        biases[l] = ygzVector<double>(sizes[l], biasArr);
        delete[] biasArr;

        // weight read
        double* weightArr = new double[sizes[l] * sizes[l - 1]];
        for (int i = 0; i < sizes[l]; i++)
        {
            row.clear();
            getline(weightsFile, row);
            std::stringstream ss(row);
            int j = 0;
            while (getline(ss, el, ','))
            {
                weightArr[i * sizes[l - 1] + j] = std::stod(el);
                j++;
            }
        }
        weights[l] = ygzMatrix<double>(sizes[l], sizes[l - 1], weightArr);
        delete[] weightArr;
    }

    weightsFile.close();
    biasesFile.close();
}

ygzVector<double> Network::feedforward(ygzVector<double> a)
{
    for (int i = 0; i < numLayers - 1; i++)
    {
        // Activation of the next layer is calculated as sigmoid(weights * activation + biases)
        a = ((weights[i + 1] * a) + biases[i + 1]).map(sigmoid);
    }

    return a;
}

void Network::SGD(ygzVector<double>** trainingData, int numTraining, int miniBatchSize, double eta, int epochs, ygzVector<double>* testData, int numTest)
{
    // start time
    auto t1 = std::chrono::high_resolution_clock::now();
    // for each epoch
    for (int i = 0; i < epochs; i++)
    {
        // shuffle training data
        std::shuffle(trainingData, trainingData + numTraining, std::default_random_engine());
        
        // iterate through training data in mini-batches
        for (int j = 0; j < numTraining; j += miniBatchSize)
        {
            // split into mini-batch
            ygzVector<double>* miniBatch = new ygzVector<double>[2 * miniBatchSize];
            for (int k = 0; k < miniBatchSize; k++)
            {
                miniBatch[2 * k] = trainingData[j + k][0];
                miniBatch[2 * k + 1] = trainingData[j + k][1];
            }

            // update weights and biases using mini-batch
            updateMiniBatch(miniBatch, miniBatchSize, eta);
            delete[] miniBatch;
        }

        // if any test data is provided, evaluate the network
        if (testData != nullptr)
        {
            int numCorrect = evaluate(testData, numTest);

            // end time
            auto t2 = std::chrono::high_resolution_clock::now();
            auto sec = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
            std::cout << "Epoch " << i << ": " << numCorrect << " / " << numTest << " correct. Time elapsed: " << (int)(sec / 60) << ":" << sec % 60 << std::endl;
        } else {
            std::cout << "Epoch " << i << " complete\n";
        }
    }
}

// Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch
// The "eta" parameter is the learning rate
void Network::updateMiniBatch(ygzVector<double>* miniBatch, int miniBatchSize, double eta)
{
    ygzVector<double>* nabla_b = new ygzVector<double>[numLayers];
    ygzMatrix<double>* nabla_w = new ygzMatrix<double>[numLayers];

    nabla_b[0] = ygzVector<double>(sizes[0]); // not used
    nabla_w[0] = ygzMatrix<double>(sizes[1], sizes[0]); // not used

    for (int i = 1; i < numLayers; i++)
    {
        nabla_b[i] = ygzVector<double>(sizes[i]);
        nabla_w[i] = ygzMatrix<double>(sizes[i], sizes[i - 1]);
    }
    // now for each training example in the mini batch, we will calculate the gradient of the cost function
    for (int i = 0; i < miniBatchSize; i++)
    {
        ygzVector<double> x = miniBatch[2 * i];
        ygzVector<double> y = miniBatch[2 * i + 1];

        ygzVector<double>* delta_nabla_b = new ygzVector<double>[numLayers];
        ygzMatrix<double>* delta_nabla_w = new ygzMatrix<double>[numLayers];
        backprop(x, y, delta_nabla_b, delta_nabla_w);
        // add the gradient of the cost function for this training example to the total gradient
        for (int j = 1; j < numLayers; j++)
        {
            nabla_b[j] = nabla_b[j] + delta_nabla_b[j];
            nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
        }
        delete[] delta_nabla_b;
        delete[] delta_nabla_w;
    }
    // update weights and biases
    for (int i = 1; i < numLayers; i++)
    {
        biases[i] = biases[i] - ((eta / miniBatchSize) * nabla_b[i]);
        weights[i] = weights[i] - ((eta / miniBatchSize) * nabla_w[i]);
    }

    // free memory
    delete[] nabla_b;
    delete[] nabla_w;
}

void Network::backprop(ygzVector<double> x, ygzVector<double> y, ygzVector<double>* nabla_b, ygzMatrix<double>* nabla_w)
{
    nabla_b[0] = ygzVector<double>(sizes[0]); // not used
    nabla_w[0] = ygzMatrix<double>(sizes[0], 1); // not used
    for (int i = 1; i < numLayers; i++)
    {
        nabla_b[i] = ygzVector<double>(sizes[i]);
        nabla_w[i] = ygzMatrix<double>(sizes[i], sizes[i - 1]);
    }

    // feedforward
    ygzVector<double>* activations = new ygzVector<double>[numLayers]; // store activations layer by layer
    activations[0] = ygzVector<double>(x); // input layer
    ygzVector<double>* zs = new ygzVector<double>[numLayers]; // store z ygzVector<double>s layer by layer
    for (int i = 0; i < numLayers - 1; i++)
    {
        zs[i + 1] = (weights[i + 1] * activations[i]) + biases[i + 1];
        activations[i + 1] = zs[i + 1].map(sigmoid);
    }

    // finding delta of the output layer
    ygzVector<double> delta = ygzVector<double>::hadamardProduct(costDerivative(activations[numLayers - 1], y), zs[numLayers - 1].map(sigmoid_prime));
    nabla_b[numLayers - 1] = delta;
    nabla_w[numLayers - 1] = delta * activations[numLayers - 2].toMatrix(false);
    // Backpropagate the error and calculate the gradient of the cost function using delta
    for (int l = numLayers - 2; l > 0; l--)
    {
        delta = ygzVector<double>::hadamardProduct(weights[l + 1].transpose() * delta, zs[l].map(sigmoid_prime));
        nabla_b[l] = delta;
        nabla_w[l] = delta * activations[l - 1].toMatrix(false);
    }
    // free memory
    delete[] activations;
    delete[] zs;
}

ygzVector<double> Network::costDerivative(ygzVector<double> outputActivations, ygzVector<double> y)
{
    // Cost function is 1/2 * (outputActivations - y)^2 so derivative wrt to outputactivations is outputActivations - y
    return outputActivations - y;
}

// Evaluate the network on test data and return the number of correct predictions
int Network::evaluate(ygzVector<double>* testData, int testDataSize)
{
    int numCorrect = 0;

    for (int i = 0; i < testDataSize; i++)
    {
        ygzVector<double> x = testData[2 * i];
        ygzVector<double> y = testData[2 * i + 1];

        ygzVector<double> output = feedforward(x);
        if (output.argmax() == y.argmax())
        {
            numCorrect++;
        }
    }

    return numCorrect;
}

// Save the biases to a CSV file
void Network::saveBiases(std::string filename)
{
    std::ofstream file(filename);
    file << numLayers << "\n";
    for (int i = 0; i < numLayers; i++)
    {
        file << sizes[i] << " ";
    }
    file << "\n";
    for (int i = 1; i < numLayers; i++)
    {
        file << biases[i].toCSV();
    }
    file.close();
}

// Save the weights to a CSV file
void Network::saveWeights(std::string filename)
{
    std::ofstream file(filename);
    file << numLayers << "\n";
    for (int i = 0; i < numLayers - 1; i++)
    {
        file << sizes[i] << " ";
    }
    file << sizes[numLayers - 1] << "\n";
    for (int i = 1; i < numLayers; i++)
    {
        file << weights[i].toCSV();
    }
    file.close();
}