#ifndef NETWORK_H
#define NETWORK_H

#include "ygzlinalg.hpp"
#include <math.h>

class Network {
public:
    Network(int* sizeArr, int n);
    Network(std::ifstream &weightsFile, std::ifstream &biasesFile);
    void SGD(ygzVector<double>** trainingData, int numTraining, int miniBatchSize, double eta, int epochs, ygzVector<double>* testData = nullptr, int numTest = 0);
    ygzVector<double> feedforward(ygzVector<double> a);

    void saveBiases(std::string filename);
    void saveWeights(std::string filename);
private:
    int numLayers;
    int* sizes;
    ygzVector<double>* biases;
    ygzMatrix<double>* weights;

    void updateMiniBatch(ygzVector<double>* miniBatch, int miniBatchSize, double eta);
    void backprop(ygzVector<double> x, ygzVector<double> y, ygzVector<double>* nabla_b, ygzMatrix<double>* nabla_w);
    int evaluate(ygzVector<double>* testData, int testDataSize);
    ygzVector<double> costDerivative(ygzVector<double> outputActivations, ygzVector<double> y);
};

#endif