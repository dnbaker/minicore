#ifndef FGC_CLUSTERING_H__
#define FGC_CLUSTERING_H__
#include "fgc/distance.h"
#include "fgc/applicator.h"

namespace fgc {

namespace clustering {

enum ClusteringAssignmentType: size_t {
    HARD
    SOFT
};

}

} // namespace fgc

#endif /* FGC_CLUSTERING_H__ */
