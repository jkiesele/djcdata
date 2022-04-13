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



#endif /* DEEPJET_MODULES_INTERFACE_HELPER_H_ */
