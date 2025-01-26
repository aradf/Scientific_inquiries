#pragma once
#include "opencv2/imgproc.hpp"

class Helpers
{
public:
    /*
	 * 'static' implies the storage type is static.
	 * standard template library (STL) has vector container holding floats.
	 */
	static std::vector< float > loadImage;
	static std::vector< std::string > loadLabels;
};

