#ifndef CDATA_TYPE_H
#define CDATA_TYPE_H

//#define component_testing_on 

typedef double float64;
typedef float float32;


const int ImageSize = 784;
const int LabelSize = 10;

const int TraingDataSize = 50000;
const int ValidationDataSize = 10000;
const int TestDataSize = 10000;

struct SUPPERVISED_DATA
{
    /* data */
    float32 fImageArr[ImageSize];
    float32 fLabelArr[LabelSize];
};

struct ANN_DATA
{
    /* data */
    float64 dImageArr[ImageSize];
    float64 dLabelArr[LabelSize];
};


#endif
