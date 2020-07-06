#include "CNetwork.h"


CNetwork::CNetwork()
{

}

CNetwork::CNetwork(std::vector<int> & aVector)
{
   this->iNumLayers = aVector.size();
   this->vSizes = aVector;

   // construct a trivial random generator engine from a time-based seed:
   unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
   std::default_random_engine generator (seed);
   std::normal_distribution<double> distribution (0.0,1.0);

   std::vector<int> leftSlicedVec = slice(aVector,1,aVector.size()-1);
   std::vector<int> rightSlicedVec = slice(aVector,0,aVector.size()-2);

   for(auto itr : leftSlicedVec)
   {
     std::cout << itr << std::endl;
     std::vector<float64> vGaussianDist;
     float64 fGaussianDistribution = 0.0;
     for (int iCnt=0; iCnt < itr; ++iCnt)
     {
#ifndef component_testing_on  
       fGaussianDistribution = distribution(generator);
#else
         fGaussianDistribution = 0.2;
#endif
       std::cout << iCnt << ":"<< fGaussianDistribution << std::endl;
       vGaussianDist.push_back(fGaussianDistribution);
     }
     (this->vBiases).push_back(vGaussianDist);
   }

   for (int iCnt = 0 ; iCnt < aVector.size()-1 ; iCnt++)
   {
      int iCol = rightSlicedVec[iCnt];
      int iRow = leftSlicedVec[iCnt];
      std::vector<std::vector<float64>> vIntermidateVec;

      for (int jCnt = 0 ; jCnt < iRow ; jCnt++)
      {
         std::vector<float64> vGaussianDist;
         for (int kCnt = 0; kCnt < iCol ; kCnt++)
         {
            float64 fGaussianDistribution = 0.0;
            //Added for debug
#ifndef component_testing_on  
          fGaussianDistribution = distribution(generator);
#else
          fGaussianDistribution = 0.3;
#endif
            std::cout << kCnt << ":"<< fGaussianDistribution << std::endl;
            vGaussianDist.push_back(fGaussianDistribution);
         }
         vIntermidateVec.push_back(vGaussianDist);
      }
      //std::cout << "add here" << std::endl;
      this->vWeights.push_back(vIntermidateVec);
   }
}

CNetwork::~CNetwork()
{

}

CNetwork& CNetwork::operator= (const CNetwork& x)   /*Copy Assignment operator*/
{
   this->vBiases = x.vBiases;
   this->iNumLayers = x.iNumLayers;
   this->vSizes = x.vSizes;
   this->vWeights = x.vWeights;
  
   return *this;
}

/***********************************************
 * Train the neural network using mini-batch stochastic
 * gradient descent.  The ``training_data`` is a list of tuples
 * ``(x, y)`` representing the training inputs and the desired
 * outputs.  The other non-optional parameters are
 * self-explanatory.  If ``test_data`` is provided then the
 * network will be evaluated against the test data after each
 * epoch, and partial progress printed out.  This is useful for
 * tracking progress, but slows things down substantially.
 *  
***********************************************/
/*void CNetwork::StochasticGradientDescent(SUPPERVISED_DATA * prtTrainData, int epochs, \
                     int mini_batch_size, float64 eta, SUPPERVISED_DATA * ptrTestData=nullptr)*/

void CNetwork::StochasticGradientDescent(ANN_DATA * pTrainData, int epochs, \
                     int mini_batch_size, float64 eta, ANN_DATA * pTestData=nullptr)
{
    ANN_DATA * pMiniBatchsize = new ANN_DATA[mini_batch_size];
   
    for (int iEpochs = 0; iEpochs < epochs; iEpochs++)
    {
         memset( pMiniBatchsize, 0, mini_batch_size*sizeof(ANN_DATA));
         
         /***************************
          *********Shuffling********* 
         ***************************/
         std::vector <long> aShuffeledVect;
         shuffle(aShuffeledVect);

         ANN_DATA * ptrSingleElementTrainingData = new ANN_DATA;
         for (int jCnt = 0; jCnt < aShuffeledVect.size() ; jCnt += 2)
         {
            memset(ptrSingleElementTrainingData,0,sizeof(ANN_DATA)*1);

            long lFirst = aShuffeledVect[jCnt];
            long lSecond = aShuffeledVect[jCnt+1];
            memcpy(ptrSingleElementTrainingData,pTrainData + lFirst,sizeof(ANN_DATA)*1);
            memcpy(pTrainData + lFirst, pTrainData + lSecond,sizeof(ANN_DATA)*1);
            memcpy(pTrainData + lSecond, ptrSingleElementTrainingData,sizeof(ANN_DATA)*1);
         
         }
         delete ptrSingleElementTrainingData;
         /***************************/

         int iCnt = 0;
         int iEnd = (int)(TraingDataSize/mini_batch_size);
         while (iCnt < iEnd) 
         {
            //pMiniBatchsize[iCnt];     // increment pointer
            int iNextBatch = iCnt * mini_batch_size;
            memcpy(pMiniBatchsize,pTrainData + iNextBatch,sizeof(ANN_DATA)*mini_batch_size);
            update_mini_batch(pMiniBatchsize, mini_batch_size, eta);

            //std::cout << iCnt << " : " << iEnd << std::endl;
            iCnt++;
            //break;   //for testing only - must be removed.
         }
         
         if (pTestData != nullptr)
         {
            int iEval = evaluate(pTestData);
            int iSize = TestDataSize;
            std::cout << "Epoch-" << iEpochs << " : " << iEval << " / 10000" << std::endl;
         }
         else
         {
            /* code */
         }
    }
   
    delete[] pMiniBatchsize;

}

/***********************************************
  * Update the network's weights and biases by applying
  * gradient descent using backpropagation to a single mini batch.
  * The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
  * is the learning rate. 
  *  
***********************************************/
/*
void CNetwork::update_mini_batch(SUPPERVISED_DATA * pMiniBatch, int mini_batch_size, float64 fEta)
*/

void CNetwork::update_mini_batch(ANN_DATA * pMiniBatch, int mini_batch_size, float64 fEta)
{
   std::vector<std::vector<float64>> aNableBiasedVec;
   std::vector<std::vector<std::vector<float64>>> aNableWeightVec;

   std::vector<std::vector<float64>> vNablaBiases;
   std::vector<std::vector<std::vector<float64>>> vNablaWeights;
   std::vector<int> leftSlicedVec = slice(this->vSizes,1,this->vSizes.size()-1);
   std::vector<int> rightSlicedVec = slice(this->vSizes,0,this->vSizes.size()-2);
   float64 fZeroDist = 0.0;

   for(auto itr : leftSlicedVec)
   {
     std::vector<float64> vZeroDist;
     for (int iCnt=0; iCnt < itr; ++iCnt) {vZeroDist.push_back(fZeroDist);}
     (vNablaBiases).push_back(vZeroDist);
   }

   for (int iCnt = 0 ; iCnt < (this->vSizes).size()-1 ; iCnt++)
   {
      int iCol = rightSlicedVec[iCnt];
      int iRow = leftSlicedVec[iCnt];
      std::vector<std::vector<float64>> vIntermidateVec;

      for (int jCnt = 0 ; jCnt < iRow ; jCnt++)
      {
         std::vector<float64> vZeroDist;
         for (int kCnt = 0; kCnt < iCol ; kCnt++) {vZeroDist.push_back(fZeroDist);}
         vIntermidateVec.push_back(vZeroDist);
      }
      vNablaWeights.push_back(vIntermidateVec);
   }
   
   for (int lCnt = 0; lCnt < mini_batch_size; lCnt++)
   {
      backPropogation(pMiniBatch+lCnt, aNableBiasedVec, aNableWeightVec);

      vNablaBiases = baseElementWiseAdd(vNablaBiases , aNableBiasedVec);
      vNablaWeights = weightElementWiseAdd(vNablaWeights , aNableWeightVec);
   }

   float64 fEtaLen = fEta/mini_batch_size;
	/* self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] */
   vWeights = weightElementWisetSubtractMultiply(vWeights, vNablaWeights, fEtaLen);

	/* self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]   */
   vBiases  = baseElementWiseSubtractMultiply(vBiases, vNablaBiases, fEtaLen);
   //std::cout << "Do nothing" << std::endl;
}

/*****************************************
 * Return a tuple ``(nabla_b, nabla_w)`` representing the
 * gradient for the cost function C_x.  ``nabla_b`` and
 * ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
 * to ``self.biases`` and ``self.weights``."""
*****************************************/
/*
void CNetwork::backPropogation(SUPPERVISED_DATA * pSomeMiniBatch, \
                  std::vector<std::vector<float64>> & aNableBiasedVec, \
                  std::vector<std::vector<std::vector<float64>>> & aNableWeightVec)
*/
void CNetwork::backPropogation(ANN_DATA * pSomeMiniBatch, \
                  std::vector<std::vector<float64>> & aNableBiasedVec, \
                  std::vector<std::vector<std::vector<float64>>> & aNableWeightVec)
{
   std::vector<std::vector<float64>> vNablaBiases;
   std::vector<std::vector<std::vector<float64>>> vNablaWeights;
   std::vector<int> leftSlicedVec = slice(this->vSizes,1,this->vSizes.size()-1);
   std::vector<int> rightSlicedVec = slice(this->vSizes,0,this->vSizes.size()-2);
   float64 fZeroDist = 0.0;

   /*****************************************
    * Produce an empty vector for biases data
   ******************************************/
   for(auto itr : leftSlicedVec)
   {
     std::vector<float64> vZeroDist;
     for (int iCnt=0; iCnt < itr; ++iCnt) {vZeroDist.push_back(fZeroDist);}
     (vNablaBiases).push_back(vZeroDist);
   }

   /*****************************************
    * Produce an empty vector for Weight data
   ******************************************/
   for (int iCnt = 0 ; iCnt < (this->vSizes).size()-1 ; iCnt++)
   {
      int iCol = rightSlicedVec[iCnt];
      int iRow = leftSlicedVec[iCnt];
      std::vector<std::vector<float64>> vIntermidateVec;

      for (int jCnt = 0 ; jCnt < iRow ; jCnt++)
      {
         std::vector<float64> vZeroDist;
         for (int kCnt = 0; kCnt < iCol ; kCnt++) {vZeroDist.push_back(fZeroDist);}
         vIntermidateVec.push_back(vZeroDist);
      }
      vNablaWeights.push_back(vIntermidateVec);
   }

   /********************************************/
   /******************feedforward***************/
   /********************************************/
   int n = sizeof(pSomeMiniBatch->dImageArr) / sizeof(pSomeMiniBatch->dImageArr[0]); 
   int len = sizeof(pSomeMiniBatch->dLabelArr) / sizeof(pSomeMiniBatch->dLabelArr[0]);
   std::vector<float64> vX(pSomeMiniBatch->dImageArr, pSomeMiniBatch->dImageArr + n);
   std::vector<float64> vY(pSomeMiniBatch->dLabelArr, pSomeMiniBatch->dLabelArr + len);

   std::vector<float64> vActivation = vX;
   std::vector<std::vector<float64>> vActivations;
   std::vector<std::vector<float64>> vZS;
   vActivations.push_back(vActivation);

   //int m = sizeof(vBiases)/sizeof(vBiases[0]) + 1;
   int m = vBiases.size();
   for (int iCnt = 0 ; iCnt < m; iCnt++)
   {
      std::vector<float64> b;
      std::vector<std::vector<float64>> w;

      b = vBiases[iCnt];
      w = vWeights[iCnt];
      std::vector<float64> z;

      for (int jCnt = 0 ;jCnt < b.size() ; jCnt ++)
      {
         float64 fRow = 0.0;
         fRow = dot_product(vActivation.begin(), vActivation.end(), w[jCnt].begin(), 0.0);
         fRow = fRow  + b[jCnt];
         z.push_back(fRow);
      }
      vZS.push_back(z);
      vActivation = sigmoid(z);
      vActivations.push_back(vActivation);
   }

   /********************************************/
   /****************backward pass***************/
   /********************************************/
   std::vector<float64> tempRight = sigmoid_prime(vZS.back());
   std::vector<float64> tempLeft = cost_derivative(vActivations.back() , vY);
   std::vector<float64> delta = multiply(tempLeft, tempRight);
   vNablaBiases.pop_back();  vNablaBiases.push_back(delta);
   std::vector<float64> vActivationSecondLast = *(vActivations.rbegin() + 1);

   std::vector<std::vector<float64>> vTmp = vectorMultiply(delta, vActivationSecondLast);
   vNablaWeights.pop_back(); vNablaWeights.push_back(vTmp);

  /*********************************************
   * Note that the variable l in the loop below is used a little
   * differently to the notation in Chapter 2 of the book.  Here,
   * l = 1 means the last layer of neurons, l = 2 is the
   * second-last layer, and so on.  It's a renumbering of the
   * scheme in the book, used here to take advantage of the fact
   * that Python can use negative indices in lists.
  **********************************************/
   for (int l = 2; l < (this->iNumLayers); l++)
   {
      std::vector<float64> z = *(vZS.rbegin()+1);  //return the last.
      std::vector<float64> sp = sigmoid_prime(z);
      //std::vector<std::vector<float64>> vWeightsTemp = *(vWeights.rbegin() + (-l+2));
      std::vector<std::vector<float64>> vWeightsTemp = *(vWeights.rbegin() + (l-2));

      std::vector<std::vector<float64>> vDelta;
      vDelta.push_back(delta);
      std::vector<std::vector<float64>> vTemp = baseMultiply(vDelta,vWeightsTemp);
      std::vector<float64> vtmp = *(vTemp.rbegin());
      delta = multiply(vtmp,sp);

      /*replace the one element at -l position*/
      int iPos = (vNablaBiases.size() -1) - (l - 1)  ;
      for (int i = 0 ; i < vNablaBiases.size()-1 ; i++)
      {         
         if (i == iPos)
         {
              vNablaBiases[i] = delta;
              break;
         }
      }

      vTemp = vectorMultiply(delta, *(vActivations.rbegin()+l));
      for (int i = 0 ; i < vNablaWeights.size()-1 ; i++)
      {         
         if (i == iPos)
         {
              vNablaWeights[i] = vTemp;
              break;
         }
      }
   }

   aNableBiasedVec = vNablaBiases;
   aNableWeightVec = vNablaWeights;
}

/******************************
 * Return the vector of partial 
 * derivatives \partial C_x for 
 * the output activations.
 * return (output_activations-y)
******************************/
std::vector<float64> CNetwork::cost_derivative(std::vector<float64> vOutPutActivation, std::vector<float64> vY)
{
   std::vector<float64> vResult;

   for (int i = 0; i < vOutPutActivation.size() ; i++)
   {
      float64 fValue = vOutPutActivation.at(i) - vY.at(i);
      vResult.push_back(fValue);
   }
   
   return vResult;
}

/******************************
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
******************************/
std::vector<float64> CNetwork::sigmoid(std::vector<float64> zVec)
{
   std::vector<float64> tempZ;
   for (std::vector<float64>::iterator it = zVec.begin() ; it != zVec.end(); ++it)
   {
      //std::cout << ' ' << *it << std::endl;
      float64 fVal = 0.0;
      fVal = 1.0/(1.0+exp(-1*(*it)));
      tempZ.push_back(fVal);
   }
   return tempZ;
}


/******************************
def sigmoid_prime(z):
    Derivative of the sigmoid function.
    return sigmoid(z)*(1-sigmoid(z)) 
******************************/
std::vector<float64> CNetwork::sigmoid_prime(std::vector<float64> zVec)
{
   std::vector<float64> result;
   std::vector<float64> vRet = sigmoid(zVec);

   for(int i = 0; i < zVec.size(); i++)
	{
      float64 fVal = 0.0;
      fVal = vRet[i] * (1- vRet[i]);
      result.push_back(fVal);
	}

   return result;
}

/******************************
 * Return the number of test inputs for which the neural
 * network outputs the correct result. Note that the neural
 * network's output is assumed to be the index of whichever
 * neuron in the final layer has the highest activation.
 * 
 * test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
 * return sum(int(x == y) for (x, y) in test_results)
******************************/
/*int CNetwork::evaluate(SUPPERVISED_DATA * ptrTestData=nullptr)*/
int CNetwork::evaluate(ANN_DATA * ptrTestData=nullptr)
{
   ptrTestData->dImageArr;  //x
   ptrTestData->dLabelArr;  //y
   int iSum = 0;

   ANN_DATA  * current = ptrTestData;
   int n = sizeof(ptrTestData->dImageArr) / sizeof(ptrTestData->dImageArr[0]); 
   int m = sizeof(ptrTestData->dLabelArr ) / sizeof(ptrTestData->dLabelArr[0]); 
   
   for (int iCnt = 0 ; iCnt < TestDataSize; iCnt++)
   {
      std::vector<float64> vX(current[iCnt].dImageArr, current[iCnt].dImageArr + n);
      std::vector<float64> vY(current[iCnt].dLabelArr, current[iCnt].dLabelArr + m);

      std::vector<float64> test_result = feedforward(vX);
      int iMax = returnIndexMax(test_result);
      int jMax = returnIndexMax(vY);
      iSum += int(iMax == jMax);
   }
   
   return iSum;
}


/******************************
 * Return the output of the network if ``a`` is input.
 * def feedforward(self, a):
 *    for b, w in zip(self.biases, self.weights):
 *       a = sigmoid(np.dot(w, a)+b)
 *    return a
******************************/
std::vector<float64> CNetwork::feedforward(std::vector<float64> vA)
{
   //int m = sizeof(vBiases)/sizeof(vBiases[0]) + 1;
   int m = vBiases.size();
   for (int iCnt = 0 ; iCnt < m; iCnt++)
   {
      std::vector<float64> b = vBiases[iCnt];
      std::vector<std::vector<float64>> w = vWeights[iCnt];
      std::vector<float64> z;

      for (int jCnt = 0 ;jCnt < b.size() ; jCnt ++)
      {
         float64 fRow = 0.0;
         fRow = dot_product(vA.begin(), vA.end(), w[jCnt].begin(), 0.0);
         fRow = fRow  + b[jCnt];
         z.push_back(fRow);
      }
      vA = sigmoid(z);
   }

   return vA;
}

