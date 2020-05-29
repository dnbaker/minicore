#pragma once
#include "pyfgc.h"

minocore::distance::DissimilarityMeasure obj2m(py::object o);
minocore::coresets::SensitivityMethod obj2sm(py::object o);
