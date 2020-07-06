#ifndef CNETWORK_H
#define CNETWORK_H

#include <iostream>
#include <chrono>
#include <random>
#include <string.h>
#include <algorithm>
#include "utility.h"
#include "Cmnist_loader.h"


/***********************************************************
 *  The list ``sizes`` contains the number of neurons in the
 *  respective layers of the network.  For example, if the list
 *  was [2, 3, 1] then it would be a three-layer network, with the
 *  first layer containing 2 neurons, the second layer 3 neurons,
 *  and the third layer 1 neuron.  The biases and weights for the
 *  network are initialized randomly, using a Gaussian
 *  distribution with mean 0, and variance 1.  Note that the first
 *  layer is assumed to be an input layer, and by convention we
 *  won't set any biases for those neurons, since biases are only
 *  ever used in computing the outputs from later layers.
***********************************************************/

class CNetwork
{
/*Data*/
private:
    int iNumLayers;
    std::vector<int> vSizes;
    std::vector<std::vector<float64>> vBiases;
    std::vector<std::vector<std::vector<float64>>> vWeights;

/*Methods*/
public:
    CNetwork();    /*default construct*/
    ~CNetwork();
    CNetwork(std::vector<int> & aVector);
    CNetwork(const CNetwork& x) : iNumLayers(x.iNumLayers), vSizes(x.vSizes), \
    vBiases(x.vBiases), vWeights(x.vWeights) {}   /*Copy Constructor*/
    CNetwork& operator= (const CNetwork& x);   /*Copy Assignment operator*/

    std::vector<float64> feedforward(std::vector<float64> aVector);

/*  void StochasticGradientDescent(SUPPERVISED_DATA * prtTrainData, int epochs, \
                int mini_batch_size, float64 eta, SUPPERVISED_DATA * prtTestData);  */

    void StochasticGradientDescent(ANN_DATA * prtTrainData, int epochs, \
                int mini_batch_size, float64 eta, ANN_DATA * prtTestData);  

    /*void update_mini_batch(SUPPERVISED_DATA * pMiniBatch, int mini_batch_size, float64 fEta);*/
    void update_mini_batch(ANN_DATA * pMiniBatch, int mini_batch_size, float64 fEta);


    /*
    void backPropogation(SUPPERVISED_DATA * pSomeMiniBatch, std::vector<std::vector<float64>> &\
                    , std::vector<std::vector<std::vector<float64>>> &);
    */
    void backPropogation(ANN_DATA * pSomeMiniBatch, std::vector<std::vector<float64>> &\
                    , std::vector<std::vector<std::vector<float64>>> &);

    std::vector<float64> sigmoid(std::vector<float64> z);
    std::vector<float64> sigmoid_prime(std::vector<float64> zVec);
    std::vector<float64> cost_derivative(std::vector<float64> vOutPutActivation, std::vector<float64> vY);
    /*    int evaluate(SUPPERVISED_DATA * ptrTestData);*/
    int evaluate(ANN_DATA * ptrTestData);
    
};

#endif
