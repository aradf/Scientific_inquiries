#ifndef MYPAIR_H
#define MYPAIR_H

template < class T>
class mypair 
{
   T a, b;
   public:
    mypair(T first, T second)
    {
        a = first;
        b = second;
    }
    T get_max();

};

template < class T>
T mypair<T>::get_max()
{
    T return_value;
    return_value = a > b ? a : b;
    return return_value;
}


#endif