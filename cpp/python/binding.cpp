#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <sstream>

#include "klygo/tensor.h"
#include "klygo/tensor_impl.h"
#include "klygo/dtype.h"
#include "klygo/tensor_factory.h"
#include "klygo/device.h"
#include "klygo/tensor_printer.h"

namespace py = pybind11;

namespace {

void parse_python_data(py::handle data, std::vector<std::size_t>& shape, std::vector<double>& flat_data, std::size_t depth) {
    if (py::isinstance<py::sequence>(data) && !py::isinstance<py::str>(data)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(data);
        std::size_t len = seq.size();
        if (shape.size() <= depth) {
            shape.push_back(len);
        } else {
            if (shape[depth] != len) {
                throw std::runtime_error("ValueError: Gator shape mismatch in nested lists");
            }
        }
        for (std::size_t i = 0; i < len; ++i) {
            parse_python_data(seq[i], shape, flat_data, depth + 1);
        }
    } else {
        double val = py::cast<double>(data);
        flat_data.push_back(val);
    }
}

std::vector<std::size_t> parse_shape(py::args args) {
    std::vector<std::size_t> shape;
    if (args.size() == 1) {
        py::handle first = args[0];
        if (py::isinstance<py::sequence>(first) && !py::isinstance<py::str>(first)) {
            py::sequence seq = py::reinterpret_borrow<py::sequence>(first);
            for (auto item : seq) {
                shape.push_back(py::cast<std::size_t>(item));
            }
            return shape;
        }
    }
    for (auto item : args) {
        shape.push_back(py::cast<std::size_t>(item));
    }
    return shape;
}

std::pair<std::vector<std::size_t>, DType> parse_shape_and_dtype(py::args args, DType default_dtype) {
    DType dtype = default_dtype;
    std::vector<py::handle> shape_items;
    for (auto item : args) {
        shape_items.push_back(item);
    }
    
    if (!shape_items.empty()) {
        py::handle last = shape_items.back();
        if (py::isinstance<DType>(last)) {
            dtype = py::cast<DType>(last);
            shape_items.pop_back();
        }
    }
    
    std::vector<std::size_t> shape;
    if (shape_items.size() == 1) {
        py::handle first = shape_items[0];
        if (py::isinstance<py::sequence>(first) && !py::isinstance<py::str>(first)) {
            py::sequence seq = py::reinterpret_borrow<py::sequence>(first);
            for (auto item : seq) {
                shape.push_back(py::cast<std::size_t>(item));
            }
            return {shape, dtype};
        }
    }
    
    for (auto item : shape_items) {
        shape.push_back(py::cast<std::size_t>(item));
    }
    return {shape, dtype};
}

Tensor empty_wrapper(py::args args, py::kwargs kwargs) {
    DType dtype = DType::Float32;
    if (kwargs.contains("dtype")) {
        dtype = py::cast<DType>(kwargs["dtype"]);
    }
    auto parsed = parse_shape_and_dtype(args, dtype);
    return empty(parsed.first, parsed.second);
}

Tensor zeros_wrapper(py::args args, py::kwargs kwargs) {
    DType dtype = DType::Float32;
    if (kwargs.contains("dtype")) {
        dtype = py::cast<DType>(kwargs["dtype"]);
    }
    auto parsed = parse_shape_and_dtype(args, dtype);
    return zeros(parsed.first, parsed.second);
}

Tensor ones_wrapper(py::args args, py::kwargs kwargs) {
    DType dtype = DType::Float32;
    if (kwargs.contains("dtype")) {
        dtype = py::cast<DType>(kwargs["dtype"]);
    }
    auto parsed = parse_shape_and_dtype(args, dtype);
    return ones(parsed.first, parsed.second);
}

Tensor full_wrapper(py::object shape_obj, py::object value, DType dtype) {
    std::vector<std::size_t> shape;
    if (py::isinstance<py::sequence>(shape_obj) && !py::isinstance<py::str>(shape_obj)) {
        for (auto item : py::cast<py::sequence>(shape_obj)) {
            shape.push_back(py::cast<std::size_t>(item));
        }
    } else {
        shape.push_back(py::cast<std::size_t>(shape_obj));
    }
    double val = py::cast<double>(value);
    return full(shape, val, dtype);
}

Tensor rand_wrapper(py::args args, py::kwargs kwargs) {
    DType dtype = DType::Float32;
    if (kwargs.contains("dtype")) {
        dtype = py::cast<DType>(kwargs["dtype"]);
    }
    auto parsed = parse_shape_and_dtype(args, dtype);
    return rand(parsed.first, parsed.second);
}

Tensor randn_wrapper(py::args args, py::kwargs kwargs) {
    DType dtype = DType::Float32;
    if (kwargs.contains("dtype")) {
        dtype = py::cast<DType>(kwargs["dtype"]);
    }
    auto parsed = parse_shape_and_dtype(args, dtype);
    return randn(parsed.first, parsed.second);
}

Tensor randint_wrapper(py::args args, py::kwargs kwargs) {
    DType dtype = DType::Int64;
    if (kwargs.contains("dtype")) {
        dtype = py::cast<DType>(kwargs["dtype"]);
    }
    if (args.empty()) throw std::invalid_argument("randint requires at least high or (low, high)");
    
    int64_t low = 0;
    int64_t high = 0;
    std::vector<std::size_t> shape;
    
    if (args.size() == 1) {
        throw std::invalid_argument("randint requires size parameter");
    }
    
    if (py::isinstance<py::sequence>(args[1]) && !py::isinstance<py::str>(args[1])) {
        low = 0;
        high = py::cast<int64_t>(args[0]);
        py::sequence seq = py::reinterpret_borrow<py::sequence>(args[1]);
        for (auto item : seq) shape.push_back(py::cast<std::size_t>(item));
    } else if (args.size() >= 3) {
        low = py::cast<int64_t>(args[0]);
        high = py::cast<int64_t>(args[1]);
        if (py::isinstance<py::sequence>(args[2]) && !py::isinstance<py::str>(args[2])) {
            py::sequence seq = py::reinterpret_borrow<py::sequence>(args[2]);
            for (auto item : seq) shape.push_back(py::cast<std::size_t>(item));
        } else {
            for (std::size_t i = 2; i < args.size(); ++i) {
                shape.push_back(py::cast<std::size_t>(args[i]));
            }
        }
    } else {
        low = 0;
        high = py::cast<int64_t>(args[0]);
        for (std::size_t i = 1; i < args.size(); ++i) shape.push_back(py::cast<std::size_t>(args[i]));
    }
    return randint(low, high, shape, dtype);
}

Tensor empty_like_wrapper(const Tensor& input, py::object dtype_opt) {
    DType dtype = input.dtype();
    if (!dtype_opt.is_none()) dtype = py::cast<DType>(dtype_opt);
    return empty_like(input, dtype);
}

Tensor zeros_like_wrapper(const Tensor& input, py::object dtype_opt) {
    DType dtype = input.dtype();
    if (!dtype_opt.is_none()) dtype = py::cast<DType>(dtype_opt);
    return zeros_like(input, dtype);
}

Tensor ones_like_wrapper(const Tensor& input, py::object dtype_opt) {
    DType dtype = input.dtype();
    if (!dtype_opt.is_none()) dtype = py::cast<DType>(dtype_opt);
    return ones_like(input, dtype);
}

Tensor full_like_wrapper(const Tensor& input, double value, py::object dtype_opt) {
    DType dtype = input.dtype();
    if (!dtype_opt.is_none()) dtype = py::cast<DType>(dtype_opt);
    return full_like(input, value, dtype);
}

Tensor tensor_getitem(const Tensor& self, py::object index) {
    if (py::isinstance<py::int_>(index)) {
        int64_t idx = py::cast<int64_t>(index);
        int64_t dim0 = static_cast<int64_t>(self.shape()[0]);
        if (idx < 0) idx += dim0;
        if (idx < 0 || idx >= dim0) throw std::out_of_range("index out of range");
        std::vector<Tensor> slices = split(self, 1, 0);
        return slices[idx].squeeze(0);
    }
    if (py::isinstance<py::slice>(index)) {
        py::slice s = py::cast<py::slice>(index);
        std::size_t start, stop, step, slicelength;
        if (!s.compute(self.shape()[0], &start, &stop, &step, &slicelength)) {
            throw py::error_already_set();
        }
        std::vector<Tensor> slices = split(self, 1, 0);
        std::vector<Tensor> chosen;
        for (std::size_t i = start; i < stop; i += step) {
            chosen.push_back(slices[i]);
        }
        if (chosen.empty()) {
            std::vector<std::size_t> empty_shape = self.shape();
            empty_shape[0] = 0;
            return empty(empty_shape, self.dtype());
        }
        return cat(chosen, 0);
    }
    if (py::isinstance<py::tuple>(index)) {
        py::tuple t = py::cast<py::tuple>(index);
        Tensor current = self;
        std::size_t current_dim = 0;
        for (auto item : t) {
            if (current_dim >= current.dim()) break;
            py::handle h = item;
            if (py::isinstance<py::int_>(h)) {
                int64_t idx = py::cast<int64_t>(h);
                int64_t dim_len = static_cast<int64_t>(current.shape()[current_dim]);
                if (idx < 0) idx += dim_len;
                std::vector<Tensor> slices = split(current, 1, current_dim);
                current = slices[idx].squeeze(current_dim);
            } else if (py::isinstance<py::slice>(h)) {
                py::slice s = py::reinterpret_borrow<py::slice>(h);
                std::size_t start, stop, step, slicelength;
                if (!s.compute(current.shape()[current_dim], &start, &stop, &step, &slicelength)) {
                    throw py::error_already_set();
                }
                std::vector<Tensor> slices = split(current, 1, current_dim);
                std::vector<Tensor> chosen;
                for (std::size_t i = start; i < stop; i += step) chosen.push_back(slices[i]);
                if (chosen.empty()) {
                    std::vector<std::size_t> es = current.shape();
                    es[current_dim] = 0;
                    current = empty(es, current.dtype());
                } else {
                    current = cat(chosen, current_dim);
                    current_dim++;
                }
            }
        }
        return current;
    }
    throw std::invalid_argument("Unsupported index type");
}

void tensor_setitem(Tensor& self, py::object index, py::object value) {
    std::vector<int64_t> indices;
    if (py::isinstance<py::int_>(index)) {
        indices.push_back(py::cast<int64_t>(index));
    } else if (py::isinstance<py::tuple>(index)) {
        py::tuple t = py::cast<py::tuple>(index);
        bool all_ints = true;
        for (auto h : t) {
            if (py::isinstance<py::int_>(h)) {
                indices.push_back(py::cast<int64_t>(h));
            } else {
                all_ints = false;
                break;
            }
        }
        if (!all_ints) {
            Tensor sub = tensor_getitem(self, index);
            if (py::isinstance<Tensor>(value)) {
                Tensor val_t = py::cast<Tensor>(value);
                std::memcpy(sub.data(), val_t.contiguous().data(), sub.numel() * sub.element_size());
            } else {
                sub.fill_(py::cast<double>(value));
            }
            return;
        }
    } else {
        Tensor sub = tensor_getitem(self, index);
        if (py::isinstance<Tensor>(value)) {
            Tensor val_t = py::cast<Tensor>(value);
            std::memcpy(sub.data(), val_t.contiguous().data(), sub.numel() * sub.element_size());
        } else {
            sub.fill_(py::cast<double>(value));
        }
        return;
    }
    
    std::size_t offset = self.offset();
    for (std::size_t d = 0; d < indices.size(); ++d) {
        int64_t idx = indices[d];
        int64_t dim_len = static_cast<int64_t>(self.shape()[d]);
        if (idx < 0) idx += dim_len;
        if (idx < 0 || idx >= dim_len) throw std::out_of_range("index out of range");
        offset += idx * self.stride()[d];
    }
    
    double val = py::cast<double>(value);
    switch (self.dtype()) {
        case DType::Float32: static_cast<float*>(self.data())[offset] = static_cast<float>(val); break;
        case DType::Float64: static_cast<double*>(self.data())[offset] = val; break;
        case DType::Int32: static_cast<int32_t*>(self.data())[offset] = static_cast<int32_t>(val); break;
        case DType::Int64: static_cast<int64_t*>(self.data())[offset] = static_cast<int64_t>(val); break;
        case DType::Bool: static_cast<bool*>(self.data())[offset] = static_cast<bool>(val != 0.0); break;
    }
}

py::object tensor_tolist(const Tensor& self) {
    Tensor c = self.contiguous();
    std::size_t nd = c.dim();
    if (nd == 0) {
        return py::cast(c.item_double());
    }
    if (nd == 1) {
        py::list res;
        std::vector<Tensor> slices = split(c, 1, 0);
        for (std::size_t i = 0; i < c.shape()[0]; ++i) {
            res.append(slices[i].squeeze(0).item_double());
        }
        return res;
    }
    py::list res;
    std::vector<Tensor> slices = split(c, 1, 0);
    for (const auto& s : slices) {
        res.append(tensor_tolist(s.squeeze(0)));
    }
    return res;
}

py::object tensor_item(const Tensor& self) {
    double v = self.item_double();
    if (self.dtype() == DType::Bool) return py::cast(static_cast<bool>(v != 0.0));
    if (self.dtype() == DType::Int32 || self.dtype() == DType::Int64) return py::cast(static_cast<int64_t>(v));
    return py::cast(v);
}

Tensor view_wrapper(const Tensor& self, py::args args) {
    std::vector<int64_t> shape;
    if (args.size() == 1) {
        py::handle first = args[0];
        if (py::isinstance<py::sequence>(first) && !py::isinstance<py::str>(first)) {
            py::sequence seq = py::reinterpret_borrow<py::sequence>(first);
            for (auto item : seq) {
                shape.push_back(py::cast<int64_t>(item));
            }
            return self.view(shape);
        }
    }
    for (auto item : args) {
        shape.push_back(py::cast<int64_t>(item));
    }
    return self.view(shape);
}

Tensor create_tensor_from_data(py::object data, py::object dtype_obj) {
    std::vector<std::size_t> shape;
    std::vector<double> flat_data;
    parse_python_data(data, shape, flat_data, 0);

    DType dtype = DType::Float32;
    if (!dtype_obj.is_none()) {
        dtype = py::cast<DType>(dtype_obj);
    }

    Tensor t = empty(shape, dtype);
    void* ptr = t.data();
    std::size_t numel = t.numel();
    switch (dtype) {
        case DType::Float32: {
            float* p = static_cast<float*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<float>(flat_data[i]);
            break;
        }
        case DType::Float64: {
            double* p = static_cast<double*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = flat_data[i];
            break;
        }
        case DType::Int32: {
            int32_t* p = static_cast<int32_t*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<int32_t>(flat_data[i]);
            break;
        }
        case DType::Int64: {
            int64_t* p = static_cast<int64_t*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<int64_t>(flat_data[i]);
            break;
        }
        case DType::Bool: {
            bool* p = static_cast<bool*>(ptr);
            for (std::size_t i = 0; i < numel; ++i) p[i] = static_cast<bool>(flat_data[i]);
            break;
        }
    }
    return t;
}

} // namespace

PYBIND11_MODULE(klygo, m)
{
    m.doc() = "Klygo C++ Tensor Backend";


    // ======================
    // dtype
    // ======================

    py::enum_<DType>(m, "dtype")

        .value("float32", DType::Float32)
        .value("float64", DType::Float64)

        .value("int32", DType::Int32)
        .value("int64", DType::Int64)

        .value("bool", DType::Bool)

        .export_values();

    m.attr("float") = DType::Float32;
    m.attr("double") = DType::Float64;
    m.attr("int") = DType::Int32;
    m.attr("long") = DType::Int64;



    // ======================
    // device
    // ======================

    py::class_<Device>(m, "device")
        .def(py::init<const std::string&>())
        .def("__repr__", [](const Device& self) {
            return "device(type='" + self.toString() + "')";
        })
        .def("__eq__", [](const Device& self, const Device& other) {
            return self == other;
        })
        .def("__eq__", [](const Device& self, const std::string& other_str) {
            try {
                return self == Device(other_str);
            } catch (...) {
                return false;
            }
        });

    // ======================
    // Tensor
    // ======================

    py::class_<Tensor>(m, "Tensor")
        .def("__repr__", [](const Tensor& self) {
            std::ostringstream ss;
            ss << self;
            return ss.str();
        })
        .def("__str__", [](const Tensor& self) {
            std::ostringstream ss;
            ss << self;
            return ss.str();
        })


        .def(
            "numel",
            &Tensor::numel
        )


        .def_property_readonly(
            "shape",
            [](const Tensor& self)
            {
                const auto& sh = self.impl()->shape();
                py::tuple t(sh.size());
                for (size_t i = 0; i < sh.size(); ++i) {
                    t[i] = sh[i];
                }
                return t;
            }
        )


        .def(
            "stride",
            [](const Tensor& self)
            {
                const auto& st = self.impl()->stride();
                py::tuple t(st.size());
                for (size_t i = 0; i < st.size(); ++i) {
                    t[i] = st[i];
                }
                return t;
            }
        )


        .def(
            "offset",
            [](const Tensor& self)
            {
                return self.impl()->offset();
            }
        )


        .def_property_readonly(
            "dtype",
            [](const Tensor& self)
            {
                return self.impl()->dtype();
            }
        )


        .def_property_readonly(
            "ndim",
            [](const Tensor& self)
            {
                return self.impl()->dim();
            }
        )

        .def_property_readonly(
            "device",
            [](const Tensor& self)
            {
                return self.device();
            }
        )

        .def("to", [](const Tensor& self, const Device& dev) { return self.to(dev); })
        .def("to", [](const Tensor& self, const std::string& dev_str) { return self.to(Device(dev_str)); })

        // Layout & Memory Properties
        .def_property_readonly("is_contiguous", &Tensor::is_contiguous)
        .def("contiguous", &Tensor::contiguous)
        .def("clone", &Tensor::clone)
        .def("element_size", &Tensor::element_size)
        .def("item", &tensor_item)
        .def("tolist", &tensor_tolist)
        .def("__len__", [](const Tensor& self) {
            if (self.dim() == 0) throw py::type_error("len() of a 0-d tensor");
            return self.shape()[0];
        })

        // Subscript Indexing & Slicing
        .def("__getitem__", &tensor_getitem)
        .def("__setitem__", &tensor_setitem)

        // Shape Manipulation & .T property
        .def_property_readonly("T", [](const Tensor& self) {
            if (self.dim() <= 2) return self.t();
            std::vector<int64_t> dims;
            for (int64_t i = static_cast<int64_t>(self.dim()) - 1; i >= 0; --i) dims.push_back(i);
            return self.permute(dims);
        })
        .def("view", &view_wrapper)
        .def("reshape", &view_wrapper)
        .def("transpose", &Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("t", &Tensor::t)
        .def("squeeze", py::overload_cast<>(&Tensor::squeeze, py::const_))
        .def("squeeze", py::overload_cast<int64_t>(&Tensor::squeeze, py::const_), py::arg("dim"))
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim"))
        .def("permute", [](const Tensor& self, py::args args) {
            std::vector<int64_t> dims;
            if (args.size() == 1 && py::isinstance<py::sequence>(args[0])) {
                for (auto item : py::cast<py::sequence>(args[0])) dims.push_back(py::cast<int64_t>(item));
            } else {
                for (auto item : args) dims.push_back(py::cast<int64_t>(item));
            }
            return self.permute(dims);
        })
        .def("flatten", &Tensor::flatten, py::arg("start_dim") = 0, py::arg("end_dim") = -1)

        // Out-of-place Arithmetic
        .def("add", py::overload_cast<const Tensor&>(&Tensor::add, py::const_), py::arg("other"))
        .def("add", py::overload_cast<double>(&Tensor::add, py::const_), py::arg("other"))
        .def("sub", py::overload_cast<const Tensor&>(&Tensor::sub, py::const_), py::arg("other"))
        .def("sub", py::overload_cast<double>(&Tensor::sub, py::const_), py::arg("other"))
        .def("mul", py::overload_cast<const Tensor&>(&Tensor::mul, py::const_), py::arg("other"))
        .def("mul", py::overload_cast<double>(&Tensor::mul, py::const_), py::arg("other"))
        .def("div", py::overload_cast<const Tensor&>(&Tensor::div, py::const_), py::arg("other"))
        .def("div", py::overload_cast<double>(&Tensor::div, py::const_), py::arg("other"))

        // In-place Operations
        .def("add_", py::overload_cast<const Tensor&>(&Tensor::add_), py::arg("other"))
        .def("add_", py::overload_cast<double>(&Tensor::add_), py::arg("other"))
        .def("sub_", py::overload_cast<const Tensor&>(&Tensor::sub_), py::arg("other"))
        .def("sub_", py::overload_cast<double>(&Tensor::sub_), py::arg("other"))
        .def("mul_", py::overload_cast<const Tensor&>(&Tensor::mul_), py::arg("other"))
        .def("mul_", py::overload_cast<double>(&Tensor::mul_), py::arg("other"))
        .def("div_", py::overload_cast<const Tensor&>(&Tensor::div_), py::arg("other"))
        .def("div_", py::overload_cast<double>(&Tensor::div_), py::arg("other"))
        .def("fill_", &Tensor::fill_, py::arg("value"))
        .def("zero_", &Tensor::zero_)

        // Unary Math & Clamp
        .def("pow", py::overload_cast<double>(&Tensor::pow, py::const_), py::arg("exponent"))
        .def("pow", py::overload_cast<const Tensor&>(&Tensor::pow, py::const_), py::arg("exponent"))
        .def("sqrt", &Tensor::sqrt)
        .def("exp", &Tensor::exp)
        .def("log", &Tensor::log)
        .def("abs", &Tensor::abs)
        .def("neg", &Tensor::neg)
        .def("clamp", &Tensor::clamp, py::arg("min") = -1e300, py::arg("max") = 1e300)

        // Comparisons
        .def("eq", py::overload_cast<const Tensor&>(&Tensor::eq, py::const_), py::arg("other"))
        .def("eq", py::overload_cast<double>(&Tensor::eq, py::const_), py::arg("other"))
        .def("ne", py::overload_cast<const Tensor&>(&Tensor::ne, py::const_), py::arg("other"))
        .def("ne", py::overload_cast<double>(&Tensor::ne, py::const_), py::arg("other"))
        .def("lt", py::overload_cast<const Tensor&>(&Tensor::lt, py::const_), py::arg("other"))
        .def("lt", py::overload_cast<double>(&Tensor::lt, py::const_), py::arg("other"))
        .def("le", py::overload_cast<const Tensor&>(&Tensor::le, py::const_), py::arg("other"))
        .def("le", py::overload_cast<double>(&Tensor::le, py::const_), py::arg("other"))
        .def("gt", py::overload_cast<const Tensor&>(&Tensor::gt, py::const_), py::arg("other"))
        .def("gt", py::overload_cast<double>(&Tensor::gt, py::const_), py::arg("other"))
        .def("ge", py::overload_cast<const Tensor&>(&Tensor::ge, py::const_), py::arg("other"))
        .def("ge", py::overload_cast<double>(&Tensor::ge, py::const_), py::arg("other"))

        // Python Operator Overloads
        .def("__add__", [](const Tensor& self, const Tensor& other) { return self.add(other); })
        .def("__add__", [](const Tensor& self, double other) { return self.add(other); })
        .def("__radd__", [](const Tensor& self, double other) { return self.add(other); })
        .def("__iadd__", [](Tensor& self, const Tensor& other) { return self.add_(other); })
        .def("__iadd__", [](Tensor& self, double other) { return self.add_(other); })

        .def("__sub__", [](const Tensor& self, const Tensor& other) { return self.sub(other); })
        .def("__sub__", [](const Tensor& self, double other) { return self.sub(other); })
        .def("__rsub__", [](const Tensor& self, double other) { return Tensor(self.impl()).mul(-1.0).add(other); })
        .def("__isub__", [](Tensor& self, const Tensor& other) { return self.sub_(other); })
        .def("__isub__", [](Tensor& self, double other) { return self.sub_(other); })

        .def("__mul__", [](const Tensor& self, const Tensor& other) { return self.mul(other); })
        .def("__mul__", [](const Tensor& self, double other) { return self.mul(other); })
        .def("__rmul__", [](const Tensor& self, double other) { return self.mul(other); })
        .def("__imul__", [](Tensor& self, const Tensor& other) { return self.mul_(other); })
        .def("__imul__", [](Tensor& self, double other) { return self.mul_(other); })

        .def("__truediv__", [](const Tensor& self, const Tensor& other) { return self.div(other); })
        .def("__truediv__", [](const Tensor& self, double other) { return self.div(other); })
        .def("__itruediv__", [](Tensor& self, const Tensor& other) { return self.div_(other); })
        .def("__itruediv__", [](Tensor& self, double other) { return self.div_(other); })

        .def("__pow__", [](const Tensor& self, double exp) { return self.pow(exp); })
        .def("__pow__", [](const Tensor& self, const Tensor& exp) { return self.pow(exp); })
        .def("__neg__", [](const Tensor& self) { return self.neg(); })

        .def("__eq__", [](const Tensor& self, const Tensor& other) { return self.eq(other); })
        .def("__eq__", [](const Tensor& self, double other) { return self.eq(other); })
        .def("__ne__", [](const Tensor& self, const Tensor& other) { return self.ne(other); })
        .def("__ne__", [](const Tensor& self, double other) { return self.ne(other); })
        .def("__lt__", [](const Tensor& self, const Tensor& other) { return self.lt(other); })
        .def("__lt__", [](const Tensor& self, double other) { return self.lt(other); })
        .def("__le__", [](const Tensor& self, const Tensor& other) { return self.le(other); })
        .def("__le__", [](const Tensor& self, double other) { return self.le(other); })
        .def("__gt__", [](const Tensor& self, const Tensor& other) { return self.gt(other); })
        .def("__gt__", [](const Tensor& self, double other) { return self.gt(other); })
        .def("__ge__", [](const Tensor& self, const Tensor& other) { return self.ge(other); })
        .def("__ge__", [](const Tensor& self, double other) { return self.ge(other); })
        .def("__matmul__", &Tensor::matmul, py::arg("other"))

        // Reductions & Stats
        .def("sum", py::overload_cast<>(&Tensor::sum, py::const_))
        .def("sum", py::overload_cast<int64_t, bool>(&Tensor::sum, py::const_), py::arg("dim"), py::arg("keepdim") = false)
        .def("mean", py::overload_cast<>(&Tensor::mean, py::const_))
        .def("mean", py::overload_cast<int64_t, bool>(&Tensor::mean, py::const_), py::arg("dim"), py::arg("keepdim") = false)
        .def("max", py::overload_cast<>(&Tensor::max, py::const_))
        .def("min", py::overload_cast<>(&Tensor::min, py::const_))
        .def("prod", py::overload_cast<>(&Tensor::prod, py::const_))
        .def("var", py::overload_cast<bool>(&Tensor::var, py::const_), py::arg("unbiased") = true)
        .def("std", py::overload_cast<bool>(&Tensor::std, py::const_), py::arg("unbiased") = true)
        .def("all", py::overload_cast<>(&Tensor::all, py::const_))
        .def("any", py::overload_cast<>(&Tensor::any, py::const_))
        .def("matmul", &Tensor::matmul, py::arg("other"));

    // ======================
    // Factory & Module Functions
    // ======================
    m.def("manual_seed", &manual_seed, py::arg("seed"));
    m.def("cat", &cat, py::arg("tensors"), py::arg("dim") = 0);
    m.def("stack", &stack, py::arg("tensors"), py::arg("dim") = 0);
    m.def("split", &split, py::arg("tensor"), py::arg("split_size_or_sections"), py::arg("dim") = 0);
    m.def("chunk", &chunk, py::arg("tensor"), py::arg("chunks"), py::arg("dim") = 0);
    m.def("where", &where, py::arg("condition"), py::arg("input"), py::arg("other"));

    m.def("empty", &empty_wrapper);
    m.def("zeros", &zeros_wrapper);
    m.def("ones", &ones_wrapper);
    m.def(
        "full",
        &full_wrapper,
        py::arg("shape"),
        py::arg("fill_value"),
        py::arg("dtype") = DType::Float32
    );

    m.def(
        "eye",
        [](std::size_t n, py::object m_opt, DType dtype) {
            std::size_t m = n;
            if (!m_opt.is_none()) {
                m = py::cast<std::size_t>(m_opt);
            }
            return eye(n, m, dtype);
        },
        py::arg("n"),
        py::arg("m") = py::none(),
        py::arg("dtype") = DType::Float32
    );

    m.def(
        "arange",
        [](double start, py::object end_opt, py::object step_opt, DType dtype) {
            double end = start;
            double real_start = 0.0;
            double step = 1.0;
            if (!end_opt.is_none()) {
                real_start = start;
                end = py::cast<double>(end_opt);
            }
            if (!step_opt.is_none()) {
                step = py::cast<double>(step_opt);
            }
            return arange(real_start, end, step, dtype);
        },
        py::arg("start"),
        py::arg("end") = py::none(),
        py::arg("step") = py::none(),
        py::arg("dtype") = DType::Float32
    );

    m.def(
        "tensor",
        &create_tensor_from_data,
        py::arg("data"),
        py::arg("dtype") = py::none()
    );

    m.def(
        "linspace",
        &linspace,
        py::arg("start"),
        py::arg("end"),
        py::arg("steps") = 100,
        py::arg("dtype") = DType::Float32
    );

    m.def("rand", &rand_wrapper);
    m.def("randn", &randn_wrapper);
    m.def("randint", &randint_wrapper);

    m.def(
        "empty_like",
        &empty_like_wrapper,
        py::arg("input"),
        py::arg("dtype") = py::none()
    );

    m.def(
        "zeros_like",
        &zeros_like_wrapper,
        py::arg("input"),
        py::arg("dtype") = py::none()
    );

    m.def(
        "ones_like",
        &ones_like_wrapper,
        py::arg("input"),
        py::arg("dtype") = py::none()
    );

    m.def(
        "full_like",
        &full_like_wrapper,
        py::arg("input"),
        py::arg("fill_value"),
        py::arg("dtype") = py::none()
    );
}