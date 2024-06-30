#include <iostream>
#include <cstdint> // Include the header file for 'uchar' type
#include <string> // Include the header file for 'string' type
#include "network.hpp"
#include <fstream>

using std::cout;
using std::string;
using uchar = unsigned char; // Define 'uchar' as an alias for 'unsigned char'

uchar** readMNISTImages(string fullPath, int &numImages, int &imageSize)
{
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(fullPath, std::ios::binary);

    if (file.is_open())
    {
        int magicNumber = 0, nRows = 0, nCols = 0;

        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        if (magicNumber != 2051)
        {
            cout << "Incorrect MNIST image file magic number: " << magicNumber << "\n";
            return nullptr;
        }

        file.read((char *)&numImages, sizeof(numImages)), numImages = reverseInt(numImages);
        file.read((char *)&nRows, sizeof(nRows)), nRows = reverseInt(nRows);
        file.read((char *)&nCols, sizeof(nCols)), nCols = reverseInt(nCols);

        imageSize = nRows * nCols;

        uchar **images = new uchar *[numImages];
        for (int i = 0; i < numImages; i++)
        {
            images[i] = new uchar[imageSize];
            file.read((char *)images[i], imageSize);
        }
        return images;
    } else
    {
        cout << "Unable to open file: " << fullPath << "\n";
        return nullptr;
    }
}

uchar* readMNISTLabels(string fullPath, int &numImages)
{
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(fullPath, std::ios::binary);

    if (file.is_open())
    {
        int magicNumber = 0;

        file.read((char *)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        if (magicNumber != 2049)
        {
            cout << "Incorrect MNIST label file magic number: " << magicNumber << "\n";
            return nullptr;
        }

        file.read((char *)&numImages, sizeof(numImages)), numImages = reverseInt(numImages);

        uchar *labels = new uchar[numImages];
        for (int i = 0; i < numImages; i++)
        {
            file.read((char *)&labels[i], 1);
        }
        return labels;
    } else
    {
        cout << "Unable to open file: " << fullPath << "\n";
        return nullptr;
    }
}

ygzVector<double> convertPPMToVector(uchar* ppm, int imageSize)
{
    ygzVector<double> imageVector = ygzVector<double>(imageSize);
    for (int i = 0; i < imageSize; i++)
    {
        double val = (double)(ppm[3 * i] + ppm[3 * i + 1] + ppm[3 * i + 1]) / 765.0;
        imageVector.setElement(i, val);
    }
    return imageVector;
}

ygzVector<double> convertImageToVector(uchar* image, int imageSize)
{
    ygzVector<double> imageVector = ygzVector<double>(imageSize);
    for (int i = 0; i < imageSize; i++)
    {
        imageVector.setElement(i, (double)image[i] / 255.0);
    }
    return imageVector;
}

ygzVector<double> convertLabelToVector(uchar label)
{
    if ((int) label > 9 || (int) label < 0)
    {
        throw std::invalid_argument("Label must be between 0 and 9");
    }
    int lbl = (int) label;
    ygzVector<double> labelVector = ygzVector<double>(10);
    for (int i = 0; i < 10; i++)
    {
        labelVector.setElement(i, i == lbl ? 1.0 : 0.0);
    }
    return labelVector;
}   

int main(int argc, char** argv)
{
    if (argc == 1)
    {
        cout << "Usage: ./digit-recognition <train | eval> <args>\n";
        return 1;
    }
    string cmd = argv[1];
    if (cmd.compare("train") == 0)
    {
        srand(time(0)); // Seed the random number generator

        cout << "Reading MNIST data...\n";
        // read MNIST data
        int numTrainingImages = 60000, trainingImageSize = 28 * 28;
        string trainingImagesPath = "C:/Users/y_kaa/Desktop/digit-recognition/data/";
        uchar** trainingImages = readMNISTImages(trainingImagesPath + "train-images.idx3-ubyte", numTrainingImages, trainingImageSize);
        uchar* trainingLabels = readMNISTLabels(trainingImagesPath + "train-labels.idx1-ubyte", numTrainingImages);

        int numTestImages = 10000, testImageSize = 28 * 28;
        uchar** testImages = readMNISTImages(trainingImagesPath + "t10k-images.idx3-ubyte", numTestImages, testImageSize);
        uchar* testLabels = readMNISTLabels(trainingImagesPath + "t10k-labels.idx1-ubyte", numTestImages);

        if (trainingImages == nullptr || trainingLabels == nullptr || testImages == nullptr || testLabels == nullptr)
        {
            return 1;
        }

        cout << "Converting data into vectors...\n";
        // convert images to vectors
        ygzVector<double>** trainingVectors = new ygzVector<double>*[numTrainingImages];
        for (int i = 0; i < numTrainingImages; i++)
        {
            trainingVectors[i] = new ygzVector<double>[2];
            trainingVectors[i][0] = convertImageToVector(trainingImages[i], trainingImageSize);
            trainingVectors[i][1] = convertLabelToVector(trainingLabels[i]);
        }

        ygzVector<double>* testVectors = new ygzVector<double>[2 * numTestImages];
        for (int i = 0; i < numTestImages; i++)
        {
            testVectors[2 * i] = convertImageToVector(testImages[i], trainingImageSize);
            testVectors[2 * i + 1] = convertLabelToVector(testLabels[i]);
        }

        cout << "Creating network...\n";
        // create network
        int sizes[3] = {784, 30, 10};
        Network network(sizes, 3);
        cout << "Training network...\n";
        network.SGD(trainingVectors, numTrainingImages, 10, 3.0, 30, testVectors, numTestImages);

        cout << "Saving weights and biases...\n";
        // save weights and biases to csv files
        network.saveWeights("weights.csv");
        network.saveBiases("biases.csv");
        cout << "Training complete\n";
    } else if (cmd.compare("eval") == 0)
    {
        if (argc == 2)
        {
            cout << "Usage: ./digit-recognition eval <image_path>\n";
            return 1;
        }
        std::ifstream weightsFile("weights.csv", std::ios::in);
        std::ifstream biasesFile("biases.csv", std::ios::in);

        if (!weightsFile.is_open() || !biasesFile.is_open())
        {
            cout << "Unable to open weights or biases file\n";
            return 1;
        }

        // process image
        string command = "python image-process.py ";
        command += argv[2];
        system(command.c_str());

        std::ifstream file("image.ppm", std::ios::binary);
        if (!file.is_open())
        {
            cout << "Unable to open image\n";
            return 1;
        }

        // vectorise image
        uchar* image = new uchar[3 * 28 * 28];
        file.seekg(13); // first 13 bytes are header
        file.read((char*) image, 3 * 28 * 28);
        ygzVector<double> imageVector = convertPPMToVector(image, 28 * 28);
        // create network using weights and biases
        Network network(weightsFile, biasesFile);
        ygzVector<double> output = network.feedforward(imageVector);
        // get the digit
        int digit = output.argmax();
        double confidence = output.getElement(digit);
        cout << "The digit is: " << digit << " with confidence: " << confidence << "\n";
    } else {
        cout << "Usage: ./digit-recognition <train | eval> <args>\n";
        return 1;
    }
    
    return 0;
}