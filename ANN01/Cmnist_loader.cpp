#include "Cmnist_loader.h"

CMnistLoader::CMnistLoader()
{
  this->ptrTrainData = new SUPPERVISED_DATA[TraingDataSize];
  this->ptrValidationData = new SUPPERVISED_DATA[ValidationDataSize];
  this->ptrTestData = new SUPPERVISED_DATA[TestDataSize];

  this->pTrainData = new ANN_DATA[TraingDataSize];
  this->pValidationData = new ANN_DATA[ValidationDataSize];
  this->pTestData = new ANN_DATA[TestDataSize];

}

CMnistLoader::~CMnistLoader()
{
  if (ptrTrainData != nullptr)
  {
    delete[] ptrTrainData;
    ptrTrainData = nullptr;
  }

  if (ptrTrainData != nullptr)
  {
    delete[] ptrValidationData;
    ptrValidationData = nullptr;
  }

  if (ptrTrainData != nullptr)
  {
    delete[] ptrTestData;
    ptrTestData = nullptr;
  }

  if (pTrainData != nullptr)
  {
    delete[] pTrainData;
    pTrainData = nullptr;
  }

  if (pTrainData != nullptr)
  {
    delete[] pValidationData;
    pValidationData = nullptr;
  }

  if (pTrainData != nullptr)
  {
    delete[] pTestData;
    pTestData = nullptr;
  }


}
    
/*
 Return the MNIST data as a tuple containing the training data,
 the validation data, and the test data.
 The ``training_data`` is returned as a tuple with two entries.
 The first entry contains the actual training images.  This is a
 numpy ndarray with 50,000 entries.  Each entry is, in turn, a
 numpy ndarray with 784 values, representing the 28 * 28 = 784
 pixels in a single MNIST image.
 The second entry in the ``training_data`` tuple is a numpy ndarray
 containing 50,000 entries.  Those entries are just the digit
 values (0...9) for the corresponding images contained in the first
 entry of the tuple.
 The ``validation_data`` and ``test_data`` are similar, except
 each contains only 10,000 images.
 This is a nice data format, but for use in neural networks it's
 helpful to modify the format of the ``training_data`` a little.
 That's done in the wrapper function ``load_data_wrapper()``, see
 below.
  */
int CMnistLoader::load_data()
{
  std::cout << "CMnistLoader::load_data()" << std::endl;

#ifndef component_testing_on

  std::ifstream rf_td("/home/faramarz/Desktop/ArtNeuralNet/dataset/training_data.bin", std::ios::out | std::ios::binary);
  if(!rf_td) {
    std::cout << "Cannot open file!" << std::endl;
    return 1;
  }
  
  for(long i = 0; i < TraingDataSize; i++)
  {
    rf_td.read((char *) &(ptrTrainData[i]) , sizeof(SUPPERVISED_DATA));

    for(int iCnt = 0; iCnt < ImageSize; ++iCnt) 
    {
      ((pTrainData[i]).dImageArr)[iCnt] = (float64)((ptrTrainData[i]).fImageArr)[iCnt];
    }

    for(int iCnt = 0; iCnt < LabelSize; ++iCnt) 
    {
      ((pTrainData[i]).dLabelArr)[iCnt] = (float64)((ptrTrainData[i]).fLabelArr)[iCnt];
    }
  }

  rf_td.close();


  std::ifstream rf_vd("/home/faramarz/Desktop/ArtNeuralNet/dataset/validation_data.bin", std::ios::out | std::ios::binary);
  if(!rf_vd) {
    std::cout << "Cannot open file!" << std::endl;
    return 1;
  }
  for(long i = 0; i < ValidationDataSize; i++)
    rf_vd.read((char *) &(ptrValidationData[i]) , sizeof(SUPPERVISED_DATA));

  rf_vd.close();

  std::ifstream rf_testd("/home/faramarz/Desktop/ArtNeuralNet/dataset/test_data.bin", std::ios::out | std::ios::binary);
  if(!rf_testd) {
    std::cout << "Cannot open file!" << std::endl;
    return 1;
  }
  for(long i = 0; i < TestDataSize; i++)
  {
    rf_testd.read((char *) &(ptrTestData[i]) , sizeof(SUPPERVISED_DATA));

    for(int iCnt = 0; iCnt < ImageSize; ++iCnt) 
    {
      ((pTestData[i]).dImageArr)[iCnt] = (float64)((ptrTestData[i]).fImageArr)[iCnt];
    }

    for(int iCnt = 0; iCnt < LabelSize; ++iCnt) 
    {
      ((pTestData[i]).dLabelArr)[iCnt] = (float64)((ptrTestData[i]).fLabelArr)[iCnt];
    }

  }

  rf_testd.close();

#else

  /*Populate the training data for component testing*/
  SUPPERVISED_DATA aSupperVisedData;
  for (int i = 0 ; i < ImageSize ; i++)
    aSupperVisedData.fImageArr[i] = 0.52;

  for (int i = 0 ; i < LabelSize ; i++)
    aSupperVisedData.fLabelArr[i] = 0.52;

   //do something.
  for(long i = 0; i < TraingDataSize; i++)
    memcpy( &(ptrTrainData[i]) , &aSupperVisedData ,sizeof(SUPPERVISED_DATA));

  /*Populate the test data for component testing*/
  for (int i = 0 ; i < ImageSize ; i++)
    aSupperVisedData.fImageArr[i] = 0.7;

  for (int i = 0 ; i < LabelSize ; i++)
    aSupperVisedData.fLabelArr[i] = 0.7;

  for(long i = 0; i < TestDataSize; i++)
    memcpy( &(ptrTestData[i]) , &aSupperVisedData ,sizeof(SUPPERVISED_DATA));
  
  std::cout << "do nothing." << std::endl;

#endif

  /*
  delete[] ptrTrainData;
  delete[] ptrValidationData;
  delete[] ptrTestData;

  ptrTrainData = nullptr;
  ptrValidationData = nullptr;
  ptrTestData = nullptr;
  */
  
}

ANN_DATA * CMnistLoader::returnTraingDataPtr()
{
  return (this->pTrainData);
}

ANN_DATA * CMnistLoader::returnTestDataPtr()
{
  return (this->pTestData);
}