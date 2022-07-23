/*
 * helper.h
 *
 *  Created on: 8 Apr 2017
 *      Author: jkiesele
 */

#ifndef DEEPJET_MODULES_INTERFACE_HELPER_H_
#define DEEPJET_MODULES_INTERFACE_HELPER_H_

#include <dirent.h>
#include <stdlib.h>
#include <sstream>
#include <string>


#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <algorithm>
#include <random>


template <class T>
std::vector<T> GenerateRandomVector(int NumberCount,T minimum=0, T maximum=1) {
    std::random_device rd;
    std::mt19937 gen(rd()); // these can be global and/or static, depending on how you use random elsewhere

    std::vector<T> values(NumberCount);
    std::uniform_real_distribution<T> dis(minimum, maximum);
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}

#include <iostream>
template <typename T>
std::ostream& operator<<(std::ostream& output, std::vector<T> const& values)
{
    for (auto const& value : values)
    {
        output << value << ' ';
    }
    output << std::endl;
    return output;
}

template<class T>
pybind11::array_t<T, pybind11::array::c_style> STLToNumpy(const T * data, const std::vector<int>& shape, const size_t& size, bool copy=true);

template<class T>
pybind11::array_t<T, pybind11::array::c_style> STLToNumpy(const T * data, const std::vector<int>& shape, const size_t& size, bool copy){

    //this seems to always copy
    pybind11::array_t<T, pybind11::array::c_style> arr(shape, data);
    return arr;

}




#endif /* DEEPJET_MODULES_INTERFACE_HELPER_H_ */
