#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <map>
#include "medianFilter3D.h"
#include "morphology.h"
namespace py = pybind11;
using std::cout;
using std::endl;

void print_type(py::array arr)
{
  py::scoped_ostream_redirect stream(std::cout, py::module::import("sys").attr("stdout"));
}

namespace
{
  std::map<std::string, cudaTextureAddressMode> mode_map = {
  {"clamp", cudaAddressModeClamp},
  {"border", cudaAddressModeBorder},
  {"mirror", cudaAddressModeMirror},
  {"wrap", cudaAddressModeWrap},
  };
  cudaTextureAddressMode get_address_mode(const std::string &mode)
  {
    auto it = mode_map.find(mode);
    if (mode_map.count(mode) == 0) {
      throw(std::invalid_argument("Invalid address mode : " + mode));
    }
    return mode_map[mode];
  }
}
template <typename S, typename T> // S is a dummy type
py::array_t<T, py::array::c_style> medianFilter3D(py::array_t<T, py::array::c_style> arr, int filter_size, std::string address_mode)
{
  if (arr.ndim() != 3) {
    throw(std::invalid_argument("Invalid ndim. 3 is supposed."));
  }
  auto shape = arr.shape();
  py::array::ShapeContainer sc(shape, shape + 3);
  py::array_t<T, py::array::c_style> result(sc);
  auto dims = make_int3(shape[2], shape[1], shape[0]); // reversed order
  auto mode = get_address_mode(address_mode);
  medianFilter3D<T>(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, filter_size, mode);
  return result;
}
void medianFilter3D(py::array arr)
{
  throw(std::invalid_argument("Invalid array type."));
}
enum class PyMOP {Dilate,Erode,Close,Open};

template <PyMOP OP, typename T>
py::array_t<T, py::array::c_style> pyoperate2D(py::array_t<T, py::array::c_style> arr, py::array_t<int, py::array::c_style | py::array::forcecast> footprint, int n_iter, std::string address_mode)
{
  if (arr.ndim() != 2) {
    throw(std::invalid_argument("Invalid ndim. 2 is supposed."));
  }
  auto shape = arr.shape();
  py::array::ShapeContainer sc(shape, shape + 2);
  py::array_t<T, py::array::c_style> result(sc);
  auto dims = make_int2(shape[1], shape[0]); // reversed order
  auto footprint_dims = make_int2(footprint.shape()[1], footprint.shape()[0]);
  if (footprint_dims.x % 2 == 0 || footprint_dims.y % 2 == 0) {
    throw(std::runtime_error("Invalid footprint size"));
  }
  auto mode = get_address_mode(address_mode);
  switch (OP) {
  case (PyMOP::Dilate):
    dilate2D(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, footprint.data(), footprint_dims, n_iter, mode);
    break;
  case (PyMOP::Erode):
    erode2D(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, footprint.data(), footprint_dims, n_iter, mode);
    break;
  case (PyMOP::Close):
    close2D(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, footprint.data(), footprint_dims, n_iter, mode);
    break;
  case (PyMOP::Open):
    open2D(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, footprint.data(), footprint_dims, n_iter, mode);
    break;
  }
  return result;
}
template <PyMOP OP, typename T>
py::array_t<T, py::array::c_style> pyoperate3D(py::array_t<T, py::array::c_style> arr, py::array_t<int, py::array::c_style | py::array::forcecast> footprint, int n_iter, std::string address_mode)
{
  if (arr.ndim() != 3) {
    throw(std::invalid_argument("Invalid ndim. 3 is supposed."));
  }
  auto shape = arr.shape();
  py::array::ShapeContainer sc(shape, shape + 3);
  py::array_t<T, py::array::c_style> result(sc);
  auto dims = make_int3(shape[2], shape[1], shape[0]); // reversed order
  auto footprint_dims = make_int3(footprint.shape()[2], footprint.shape()[1], footprint.shape()[0]);
  if (footprint_dims.x % 2 == 0 || footprint_dims.y % 2 == 0 || footprint_dims.z % 2 == 0) {
    throw(std::runtime_error("Invalid footprint size"));
  }
  auto mode = get_address_mode(address_mode);
  switch (OP) {
  case (PyMOP::Dilate):
    dilate3D(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, footprint.data(), footprint_dims, n_iter, mode);
    break;
  case (PyMOP::Erode):
    erode3D(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, footprint.data(), footprint_dims, n_iter, mode);
    break;
  case (PyMOP::Close):
    close3D(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, footprint.data(), footprint_dims, n_iter, mode);
    break;
  case (PyMOP::Open):
    open3D(static_cast<const T*>(arr.data()), static_cast<T*>(result.mutable_data()), dims, footprint.data(), footprint_dims, n_iter, mode);
    break;
  }
  return result;
}
#define template_macro(exp) \
{typedef int8_t TT;exp;}\
{typedef int16_t TT;exp;}\
{typedef int32_t TT;exp;}\
{typedef uint8_t TT;exp;}\
{typedef uint16_t TT;exp;}\
{typedef uint32_t TT;exp;}\
{typedef float TT;exp;}
//{typedef double TT;exp;}\
//{typedef int64_t TT;exp;}\
//{typedef uint64_t TT;exp;}\

#include "fillhole.h"
py::array_t<uint8_t, py::array::c_style> fillhole2(py::array_t<uint8_t, py::array::c_style> arr)
{
  if (arr.ndim() == 2) {
    auto shape = arr.shape();
    py::array::ShapeContainer sc(shape, shape + 2);
    py::array_t<uint8_t, py::array::c_style> result(sc);
    auto dims = make_int2(shape[1], shape[0]); // reversed order
    fillhole2D(arr.data(), result.mutable_data(), dims);
    return result;
  } else if (arr.ndim() == 3) {
    auto shape = arr.shape();
    py::array::ShapeContainer sc(shape, shape + 3);
    py::array_t<uint8_t, py::array::c_style> result(sc);
    auto dims = make_int3(shape[2], shape[1], shape[0]); // reversed order
    fillhole2D_consecutive(arr.data(), result.mutable_data(), dims);
    return result;
  } else {
    throw(std::invalid_argument("Invalid ndim. 2 or 3 is supported."));
  }
}

PYBIND11_MODULE(cuda_filters, m)
{
  m.doc() = "Image filters boosted by CUDA.";
  m.def("medfilt3",&medianFilter3D<float, float>,"3D median filter.");
  m.def("medfilt3",&medianFilter3D<float, uint8_t>,"3D median filter.");
  m.def("medfilt3",&medianFilter3D<float, int16_t>,"3D median filter.");
  m.def("medfilt3",&medianFilter3D,"3D median filter.");
  m.def("fillhole2",&fillhole2,"2D binary fill hole.");
  template_macro(m.def("dilate2",&pyoperate2D<PyMOP::Dilate,TT>,"2D dilate filter.",py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>()))
  template_macro(m.def("erode2",&pyoperate2D<PyMOP::Erode,TT>,"2D erode filter.",py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>()))
  template_macro(m.def("close2",&pyoperate2D<PyMOP::Close,TT>,"2D close filter.",py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>()))
  template_macro(m.def("open2",&pyoperate2D<PyMOP::Open,TT>,"2D open filter.",py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>()))
  template_macro(m.def("dilate3",&pyoperate3D<PyMOP::Dilate,TT>,"3D dilate filter.",py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>()))
  template_macro(m.def("erode3",&pyoperate3D<PyMOP::Erode,TT>,"3D erode filter.",py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>()))
  template_macro(m.def("close3",&pyoperate3D<PyMOP::Close,TT>,"3D close filter.",py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>()))
  template_macro(m.def("open3",&pyoperate3D<PyMOP::Open,TT>,"3D open filter.",py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>()))
}
