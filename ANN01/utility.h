#ifndef UTILITITY_H
#define UTILITITY_H

#include <vector>
#include <algorithm>
#include <ctime>
#include "CDataType.h"

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
   auto first = v.cbegin() + m;
   auto last = v.cbegin() + n + 1;
   std::vector<T> vec(first,last);
   return vec;
}

template <class InputIterator1, class InputIterator2, class T>
T dot_product (InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, T init)
{
  while (first1!=last1) 
  {
    init = init + (*first1)*(*first2);
    // or: init = binary_op1 (init, binary_op2(*first1,*first2));
    ++first1; ++first2;
  }
  return init;
}

template<typename T>
std::vector<T> multiply(std::vector<T> const &left, std::vector<T> const &right)
{
   std::vector<T> result;
   T TValue;

   for (int iCnt = 0 ; iCnt < left.size(); iCnt++)
   {
      TValue = left[iCnt]*right[iCnt];
      result.push_back(TValue);
   }

   return result;
}

template<typename T>
std::vector<std::vector<T>> vectorMultiply(std::vector<T> const &left, std::vector<T> const &right)
{
   std::vector<std::vector<T>> result;

    for (int i = 0; i < left.size(); i++) 
    { 
        std::vector<T> row;
        for (int j = 0; j < right.size(); j++) 
        { 
           T temp = left[i] * right[j];
           row.push_back(temp);
        }
        result.push_back(row);
    } 

   return result;
}

template <class T>
std::vector <std::vector<T>> baseMultiply(std::vector <std::vector<T>> &left, std::vector <std::vector<T>> &right)
{
    const int n = left.size();     // a rows
    const int m = left[0].size();  // a cols
    const int p = right[0].size();  // b cols

    std::vector <std::vector<T>> result(n, std::vector<T>(p, 0));
    for (auto j = 0; j < p; ++j)
    {
        for (auto k = 0; k < m; ++k)
        {
            for (auto i = 0; i < n; ++i)
            {
                result[i][j] += left[i][k] * right[k][j];
            }
        }
    }
    return result;
}

template <class T>
std::vector<std::vector<T>> baseElementWiseAdd(std::vector <std::vector<T>> &left, std::vector <std::vector<T>> &right)
{
    std::vector <std::vector<T>> result;
    const int n = left.size();     // a rows

    for (auto i = 0; i < n; ++i)
    {
        const int k = left[i].size();
        std::vector<T> vRow;
        for (auto j = 0; j < k; ++j)
        {
            T value = left[i][j] + right[i][j];
            vRow.push_back(value);
            //std::cout << " - " << std::endl;
        }
        result.push_back(vRow);
    }
    return result;
}

template <class T>
std::vector<std::vector<std::vector<T>>> weightElementWiseAdd( \
                    std::vector<std::vector<std::vector<T>>> &left, \
                    std::vector<std::vector<std::vector<T>>> &right)
{
    std::vector<std::vector<std::vector<T>>> result;
    const int n = left.size();     // a rows

    for (auto i = 0; i < n; ++i)
    {
        const int k = left[i].size();
        std::vector<std::vector<T>> iRow;
        for (auto j = 0; j < k; ++j)
        {
            const int l = left[i][j].size();
            std::vector<T> jRow;
            for (auto m = 0 ; m < l; ++m)
            {
                T value = left[i][j][m] + right[i][j][m]; 
                jRow.push_back(value);
            }
            iRow.push_back(jRow);
            //std::cout << " - " << std::endl;
        }
        result.push_back(iRow);
    }

    return result;
}

/***************************
 * self.biases = [b-(eta/len(mini_batch))*nb
 * for b, nb in zip(self.biases, nabla_b)]   
 *********************************************/
template <class T>
std::vector<std::vector<T>> baseElementWiseSubtractMultiply( \
                            std::vector <std::vector<T>> &left, \
                            std::vector <std::vector<T>> &right, T fEtaLen)
{
    std::vector <std::vector<T>> result;
    const int n = left.size();     // a rows

    for (auto i = 0; i < n; ++i)
    {
        const int k = left[i].size();
        std::vector<T> vRow;
        for (auto j = 0; j < k; ++j)
        {
            T value = (left[i][j] - (fEtaLen * right[i][j]));
            vRow.push_back(value);
            //std::cout << " - " << std::endl;
        }
        result.push_back(vRow);
    }
    return result;
}

/**********************************************
 * self.weights = [w-(eta/len(mini_batch))*nw
 * for w, nw in zip(self.weights, nabla_w)]
**********************************************/
template <class T>
std::vector<std::vector<std::vector<T>>> weightElementWisetSubtractMultiply( \
                    std::vector<std::vector<std::vector<T>>> &left, \
                    std::vector<std::vector<std::vector<T>>> &right, T fEtaLen)
{
    std::vector<std::vector<std::vector<T>>> result;
    const int n = left.size();     // a rows

    for (auto i = 0; i < n; ++i)
    {
        const int k = left[i].size();
        std::vector<std::vector<T>> iRow;
        for (auto j = 0; j < k; ++j)
        {
            const int l = left[i][j].size();
            std::vector<T> jRow;
            for (auto m = 0 ; m < l; ++m)
            {
                T value = (left[i][j][m] - (fEtaLen * right[i][j][m])); 
                jRow.push_back(value);
            }
            iRow.push_back(jRow);
            //std::cout << " - " << std::endl;
        }
        result.push_back(iRow);
    }

    return result;
}

template <class T>
int returnIndexMax(std::vector<T> nums)
{

    int minPos = 0, maxPos = 0;
    for (unsigned i = 0; i < nums.size(); ++i)
    {
        if (nums[i] < nums[minPos]) // Found a smaller min
            minPos = i;
        if (nums[i] > nums[maxPos]) // Found a bigger max
            maxPos = i;
    }
    //std::cout << "Min is " << nums[minPos] << " at position " << minPos;
    //std::cout << "\nMax is " << nums[maxPos] << " at position " << maxPos;
    
    return maxPos;
}

template <class T>
void shuffle(std::vector<T> & aVector)
{
    //std::vector <T> aVector;
    for (long i=0; i< TraingDataSize; ++i) aVector.push_back(i);
    std::random_shuffle(aVector.begin(), aVector.end());
}

#endif