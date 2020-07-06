#ifndef CMNIST_LOADER_H
#define CMNIST_LOADER_H

#include "CDataType.h"
#include <iostream>
#include <fstream>
#include <string.h>

class  CMnistLoader
{
private:
    /* data */
    SUPPERVISED_DATA * ptrTrainData;
    SUPPERVISED_DATA * ptrValidationData;
    SUPPERVISED_DATA * ptrTestData;

    ANN_DATA * pTrainData;
    ANN_DATA * pValidationData;
    ANN_DATA * pTestData;

public:
    /* Methods*/
    CMnistLoader();
    ~CMnistLoader();
    int load_data();
    ANN_DATA * returnTraingDataPtr();
    ANN_DATA * returnTestDataPtr();
};

#endif