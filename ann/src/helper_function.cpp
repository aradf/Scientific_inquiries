#include "../include/helper_funtion.hpp"

void debug_print(std::string msg)
{
    if (DEBUG_01)
        printf("%s \n", msg.c_str());
}
