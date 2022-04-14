// empty
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "simpleArray.h"
#include "trainData.h"
#include "trainDataGenerator.h"

namespace py = pybind11;


template<class T, class M>
void makeArr(M& m, std::string name){
    using namespace djc;
    py::class_<simpleArray<T> >(m, name.data())
            .def(py::init())

            .def(py::self == py::self)
            .def(py::self != py::self)

            //.def("__copy__",  [](const simpleArray<T> &self) {
            //     return simpleArray<T>(self);
            // })
            // .def("__deepcopy__", [](const simpleArray<T> &self, py::dict) {
            //     return simpleArray<T>(self);
            // })


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
            .def("copyToNumpy", &simpleArray<T>::copyToNumpy, py::arg("pad_rowsplits")=false);
}


template<class M>
void makeTD(M & m, std::string name){
    using namespace djc;
    py::class_<trainData>(m, name.data())
        .def(py::init())

        .def(py::self == py::self)
        .def(py::self != py::self)

        //.def("__copy__",  [](const trainData &self) {
        //return trainData(self);
        //})
        //.def("__deepcopy__", [](const trainData &self, py::dict) {
        //    return trainData(self);
        //})

        //no data being copied
        //.def(py::pickle(
        //        [](const trainData &p) { // __getstate__
        //            /* Return a tuple that fully encodes the state of the object */
        //            return py::make_tuple(0,0);
        //        },
        //        [](py::tuple t) { // __setstate__
        //
        //            return trainData();
        //        }
        //    ))

        .def("storeFeatureArray", static_cast<int (trainData::*)(simpleArray_float32 &)>(&trainData::storeFeatureArray))
        .def("storeFeatureArray", static_cast<int (trainData::*)(simpleArray_int32 &)>(&trainData::storeFeatureArray))

        .def("storeTruthArray", static_cast<int (trainData::*)(simpleArray_float32 &)>(&trainData::storeTruthArray))
        .def("storeTruthArray", static_cast<int (trainData::*)(simpleArray_int32 &)>(&trainData::storeTruthArray))

        .def("storeWeightArray", static_cast<int (trainData::*)(simpleArray_float32 &)>(&trainData::storeWeightArray))
        .def("storeWeightArray", static_cast<int (trainData::*)(simpleArray_int32 &)>(&trainData::storeWeightArray))


        .def("nFeatureArrays", &trainData::nFeatureArrays)
        .def("nTruthArrays", &trainData::nTruthArrays)
        .def("nWeightArrays", &trainData::nWeightArrays)

        .def("truncate", &trainData::truncate)
        .def("append", &trainData::append)
        .def("split", &trainData::split)
        .def("nElements", &trainData::nElements)
        .def("readMetaDataFromFile", &trainData::readMetaDataFromFile)

        .def("readFromFile", &trainData::readFromFile)
        .def("readFromFileBuffered", &trainData::readFromFileBuffered)
        .def("writeToFile", &trainData::writeToFile)
        .def("addToFile", &trainData::addToFile)


        .def("copy", &trainData::copy)
        .def("clear", &trainData::clear)
        .def("skim", &trainData::skim)
        .def("getSlice", &trainData::getSlice)

        .def("getNumpyFeatureShapes", &trainData::getNumpyFeatureShapes)
        .def("getNumpyTruthShapes", &trainData::getNumpyTruthShapes)
        .def("getNumpyWeightShapes", &trainData::getNumpyWeightShapes)

        .def("getNumpyFeatureDTypes", &trainData::getNumpyFeatureDTypes)
        .def("getNumpyTruthDTypes", &trainData::getNumpyTruthDTypes)
        .def("getNumpyWeightDTypes", &trainData::getNumpyWeightDTypes)

        .def("getNumpyFeatureArrayNames", &trainData::getNumpyFeatureArrayNames)
        .def("getNumpyTruthArrayNames", &trainData::getNumpyTruthArrayNames)
        .def("getNumpyWeightArrayNames", &trainData::getNumpyWeightArrayNames)

        .def("getTruthRaggedFlags", &trainData::getTruthRaggedFlags)
        .def("transferFeatureListToNumpy", &trainData::transferFeatureListToNumpy)
        .def("transferTruthListToNumpy", &trainData::transferTruthListToNumpy)
        .def("transferWeightListToNumpy", &trainData::transferWeightListToNumpy)


        .def("copyFeatureListToNumpy", &trainData::copyFeatureListToNumpy)
        .def("copyTruthListToNumpy", &trainData::copyTruthListToNumpy)
        .def("copyWeightListToNumpy", &trainData::copyWeightListToNumpy);
}

template<class M>
void makeTDG(M & m, std::string name){
    using namespace djc;
       py::class_<trainDataGenerator>(m, name.data())
           .def(py::init())
           .def("setBatchSize", &trainDataGenerator::setBatchSize)

           .def("setFileList", &trainDataGenerator::setFileList)
           .def("shuffleFileList", &trainDataGenerator::shuffleFileList)

           .def("setBuffer", &trainDataGenerator::setBuffer)


           .def("setFileTimeout", &trainDataGenerator::setFileTimeout)
           .def("setSquaredElementsLimit", &trainDataGenerator::setSquaredElementsLimit)
           .def("setSkipTooLargeBatches", &trainDataGenerator::setSkipTooLargeBatches)

           .def("clear", &trainDataGenerator::clear)
           .def("getNBatches", &trainDataGenerator::getNBatches)

           .def("lastBatch", &trainDataGenerator::lastBatch)
           .def("isEmpty", &trainDataGenerator::isEmpty)

           .def("prepareNextEpoch", &trainDataGenerator::prepareNextEpoch)
           .def("getBatch", &trainDataGenerator::getBatch)

           .def("getNTotal", &trainDataGenerator::getNTotal)

           .def_readwrite("debuglevel", &trainDataGenerator::debuglevel);

}

//warp it up
PYBIND11_MODULE(compiled, m) {
    makeArr<float>(m,"simpleArrayF");
    makeArr<int32_t>(m,"simpleArrayI");

    makeTD(m,"trainData");

    makeTDG(m, "trainDataGenerator");

}

//return_value_policy::take_ownership

