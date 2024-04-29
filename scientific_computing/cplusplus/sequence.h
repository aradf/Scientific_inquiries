#ifndef SEQUENCE_H
#define SEQUENCE_H

template < class A, int N>
class mysequence
{
    A memblock [N];
    public:
       void set_member(int x, A value);
       A get_member(int x);
};

template<class A, int N>
void mysequence<A, N>::set_member(int x, A value)
{
    memblock[x] = value;
}

template<class T, int N>
T mysequence<T, N>::get_member(int x)
{
    return memblock[x];
}

#endif