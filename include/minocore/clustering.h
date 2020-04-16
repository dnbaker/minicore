#ifndef FGC_CLUSTERING_H__
#define FGC_CLUSTERING_H__
#include "minocore/distance.h"
#include "minocore/applicator.h"

namespace minocore {

namespace clustering {

enum ClusteringAssignmentType: size_t {
    HARD
    SOFT
};

}

} // namespace minocore

#endif /* FGC_CLUSTERING_H__ */
