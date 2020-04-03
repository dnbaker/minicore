#ifndef FGC_GENERALIZED_K_MEDIAN_H
#define FGC_GENERALIZED_K_MEDIAN_H
#include "fgc/jsd.h"

namespace fgc {

namespace jsd {


#if 0
// 1.  Directional k-means for asymmetric measures.

template<typename ContainerType, typename MatrixType, typename...Args>
void directional_kmeans(const ProbDivApplicator<MatrixType> &) {
}

// 2. 

template<typename Measure, typename ContainerType>
void optimize_parameters_soft() {
    throw std::runtime_error("NotImplemented yet. This should take a set of points or indices and assignments and select the new centroids");
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
#endif

} // namespace jsd

} // namespace fgc

#endif /* FGC_GENERALIZED_K_MEDIAN_H */
