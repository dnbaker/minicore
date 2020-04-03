#ifndef FGC_GENERALIZED_K_MEDIAN_H
#define FGC_GENERALIZED_K_MEDIAN_H
#include "fgc/jsd.h"

namespace fgc {

namespace jsd {

// 1.  Directional k-means for asymmetric measures.

template<typename ContainerType, typename MatrixType, typename...Args>
void directional_kmeans(const ProbDivApplicator<MatrixType> &) {
    static_assert(is_finished, "This must be finished before I let it pass CI.");
    //
}

// 2. 

template<typename Measure, typename ContainerType>
void optimize_parameters_soft() {
    throw std::runtime_error("NotImplemented yet. This should take a set of points or indices and assignments and select the new centroids");
    // This will
}
template<typename ContainerType>
void optimize_parameters_soft() {
    switch(measure) {
        case measure: optimize_parameters_soft<measure>();
    }
}
template<typename Measure, typename ContainerType>
void optimize_parameters_soft() {
}

template<typename ContainerType>
void optimize_parameters_hard() {
    switch(measure) {
        case measure: optimize_parameters_soft<measure>();
    }
}

} // namespace jsd

} // namespace fgc

#endif /* FGC_GENERALIZED_K_MEDIAN_H */
