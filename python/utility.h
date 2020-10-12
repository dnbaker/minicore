#pragma once
#include "pyfgc.h"

minicore::distance::DissimilarityMeasure obj2m(py::object o);
minicore::coresets::SensitivityMethod obj2sm(py::object o);
