// empty
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "simpleArray.h"

namespace py = pybind11;
using namespace djc;

template<class T, class M>
void makeArr(M& m, std::string name){
py::class_<simpleArray<T> >(m, name.data())
    .def(py::init())

    .def(py::self == py::self)
    .def(py::self != py::self)

    .def("readDtypeFromFile", &simpleArray<T>::readDtypeFromFile)

    .def("setName", &simpleArray<T>::setName)
    .def("name", &simpleArray<T>::name)

    .def("dtypeI", &simpleArray<T>::dtypeI)

    .def("readFromFile", &simpleArray<T>::readFromFile)
    .def("writeToFile", &simpleArray<T>::writeToFile)

    .def("setFeatureNames", &simpleArray<T>::setFeatureNames)
    .def("featureNames", &simpleArray<T>::featureNames)

    .def("hasNanOrInf", &simpleArray<T>::hasNanOrInf)

    .def("isRagged", &simpleArray<T>::isRagged)
    .def("split", &simpleArray<T>::split)
    .def("getSlice", &simpleArray<T>::getSlice)
   // .def<void (simpleArray<T>::*)(const simpleArray_float32&)>("append", &simpleArray_float32::append)
    .def("cout", &simpleArray<T>::cout)
    .def("size", &simpleArray<T>::isize)
    .def("shape", &simpleArray<T>::shapePy)


    .def("set", static_cast<void (simpleArray<T>::*)(size_t, T)>(&simpleArray<T>::set))
    .def("set", static_cast<void (simpleArray<T>::*)(size_t,size_t, T)>(&simpleArray<T>::set))
    .def("set", static_cast<void (simpleArray<T>::*)(size_t,size_t,size_t, T)>(&simpleArray<T>::set))
    .def("set", static_cast<void (simpleArray<T>::*)(size_t,size_t,size_t,size_t, T)>(&simpleArray<T>::set))
    .def("set", static_cast<void (simpleArray<T>::*)(size_t,size_t,size_t,size_t,size_t, T)>(&simpleArray<T>::set))

    //don't expose the const ones, anyway protected through python function call
    .def("at", static_cast<T& (simpleArray<T>::*)(size_t)>(&simpleArray<T>::at))
    .def("at", static_cast<T& (simpleArray<T>::*)(size_t,size_t)>(&simpleArray<T>::at))
    .def("at", static_cast<T& (simpleArray<T>::*)(size_t,size_t,size_t)>(&simpleArray<T>::at))
    .def("at", static_cast<T& (simpleArray<T>::*)(size_t,size_t,size_t,size_t)>(&simpleArray<T>::at))
    .def("at", static_cast<T& (simpleArray<T>::*)(size_t,size_t,size_t,size_t,size_t)>(&simpleArray<T>::at))

    //only go for explicit same-type append in python
    .def("append", static_cast<void (simpleArray<T>::*)(const simpleArray<T> &)>(&simpleArray<T>::append))
    //numpy bindings

    .def("assignFromNumpy", &simpleArray<T>::assignFromNumpy)
    .def("transferToNumpy", &simpleArray<T>::transferToNumpy, py::arg("pad_rowsplits")=false)
    .def("createFromNumpy", &simpleArray<T>::createFromNumpy)
    .def("copyToNumpy", &simpleArray<T>::copyToNumpy, py::arg("pad_rowsplits")=false)


    ;
}




//warp it up
PYBIND11_MODULE(compiled, m) {

    makeArr<float>(m,"simpleArrayF");
    makeArr<int32_t>(m,"simpleArrayI");

}

//return_value_policy::take_ownership

