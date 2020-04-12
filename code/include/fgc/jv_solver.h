#ifndef JV_SOLVER_H__
#define JV_SOLVER_H__
#include "blaze_adaptor.h"
#include "jv_util.h"
#include <chrono>

namespace fgc {

namespace jv {


template<typename MatrixType, typename FT=blaze::ElementType_t<MatrixType>, typename IT=uint32_t>
struct JVSolver {

    static_assert(std::is_floating_point<FT>::value, "FT must be floating-point");
    static_assert(std::is_integral<IT>::value, "IT must be integral");

    using payment_t = packed::pair<FT, IT>;
    using edge_type = jvutil::edgetup<FT, IT>;
    using this_type = JVSolver<MatrixType, FT, IT>;

    // Helper structure
    struct pay_compare_t {
        bool operator()(const payment_t lhs, const payment_t rhs) const {
            return lhs.first < rhs.first || lhs.second < rhs.second;
        }
    };
    struct payment_queue: public std::set<payment_t, pay_compare_t> {
        void push(payment_t payment) {
            this->insert(payment);
        }
        template<typename IterT>
        void push(IterT start, IterT end) {
            this->insert(start, end);
        }
        auto top() const {
            if(this->empty()) throw std::runtime_error("Attempting to access an empty structure");
            return *this->begin();
        }
        auto pop() {
            auto ret = top();
            this->erase(this->begin());
            return ret;
        }
        void update(FT oldc, FT newc, IT idx) {
            // Remove old payment, add new payment
            if(oldc == newc) return;
            assert(oldc != PAID_IN_FULL);
            assert(newc != PAID_IN_FULL);
            payment_t tmp = {oldc, idx};
            auto it = this->find(tmp);
            if(it != this->end()) {
                this->erase(it);
                tmp.first = newc;
                assert(this->find(tmp) == this->end() || !std::fprintf(stderr, "Just removed %g:%u, and just found it here with value %g:%u\n", tmp.first, tmp.second, this->find(tmp)->first, this->find(tmp)->second));
                this->insert(tmp);
            }
        }
    };

private:
    // Private members
    // Distance matrix: values are infinite for those missing (e.g., sparse)
    const MatrixType *distmatp_;
    blaze::DynamicMatrix<FT> client_w_;  // W matrix: Willingness of each client to pay for each facility
    std::shared_ptr<edge_type[]> edges_; // list of all edges, sorted by cost (shared ptr so that it can be shared by multiple instances)
    std::vector<payment_t> client_v_;    // List of coverage by each facility

    blaze::DynamicVector<FT> facility_cost_;
    // Costs for facilities. Of size 1 if uniform, of size # fac otherwise.

    // List of facilities assigned to
    std::vector<std::vector<IT>> clients_cpy_; //List of facilities each client is assigned to
    std::unique_ptr<std::vector<IT>[]> working_open_facilities_; // temporarily open facilities

    // List of final open facilities
    std::vector<IT> final_open_facilities_;

    // List of final facility assignments
    std::vector<std::vector<IT>> final_open_facility_assignments_;

    std::unique_ptr<FT[]> contribution_time_; // Time when contributions are made to a facility
    std::unique_ptr<FT[]> fac_contributions_; // Track of contributions to a given facility

    std::vector<FT> pay_schedule_;

    payment_queue next_paid_;

    size_t n_open_clients_;
    size_t nedges_;
    size_t ncities_;
    size_t nfac_;


    // Private code

    FT final_phase1_loop(FT time) {
        while(!next_paid_.empty()) {
            payment_t next_fac = next_facility();
            if(next_fac.first > 0) {
                time = next_fac.first;
            }
            n_open_clients_ = update_facilities(next_fac.second, working_open_facilities_[next_fac.second], time);
            if(n_open_clients_ == 0) break;
        }
        return time;
    }

    // TODO: consider replacing with blaze::SmallArray to avoid heap allocation/access
    IT update_facilities(IT gfid, const std::vector<IT> &update_facilities, const FT cost) {
#if VERBOSE_AF
        std::fprintf(stderr, "About to update facility %zu with update_facilities of size %zu with cost %g\n",
                     size_t(gfid), update_facilities.size(), cost);
#endif
        if(!open_client(update_facilities)) {
            return n_open_clients_;
        }
        for(const IT cid: update_facilities) {
            if(cid == EMPTY) break;

            auto &assigned_facilities = clients_cpy_[cid];
            if(!open_client(assigned_facilities)) continue;

            for(const IT fid: assigned_facilities) {
                auto &open_fac = working_open_facilities_[fid];
                if(open_client(open_fac) && fid != gfid) {
                    auto &fac_pay = pay_schedule_[fid];
                    if(fac_pay != PAID_IN_FULL) {
                        if(client_w_(fid, cid) != PAID_IN_FULL) {
                            FT nclients_fid = working_open_facilities_[fid].size();
                            FT update_pay = nclients_fid * (cost - contribution_time_[fid]);
                            FT oldv = fac_pay;
                            FT current_contrib = fac_contributions_[fid] + update_pay;
                            fac_contributions_[fid] = current_contrib;
                            fac_pay -= current_contrib;
                            contribution_time_[fid] = cost;
                            FT remaining_time;
                            FT opening_cost = get_fac_cost(fid);
                            if(!working_open_facilities_[fid].empty()) {
                                auto &ofr = working_open_facilities_[fid];
                                if(std::find(ofr.begin(), ofr.end(), cid) != ofr.end()) {
                                    remaining_time = ofr.size() > 1 ? (opening_cost - current_contrib) / (ofr.size() - 1)
                                                                    :  opening_cost - cost + EPS;
                                } else {
                                    remaining_time = (opening_cost - current_contrib) / ofr.size();
                                }
                            } else {
                                remaining_time = opening_cost - cost + EPS;
                            }
                            FT newv = cost + remaining_time;
                            fac_pay = newv;
                            //std::fprintf(stderr, "About to update in update_facilities. fid = %u. cid = %u. Under gfid %u\n", fid, cid, gfid);
                            next_paid_.update(oldv, newv, fid);
                        }
                    }
                }
            }
            clients_cpy_[cid].push_back(EMPTY);
            client_v_[cid] = {cost, gfid};
            --n_open_clients_;
        }
        if(FT oldv = pay_schedule_[gfid]; oldv != PAID_IN_FULL) {
            next_paid_.erase(payment_t{oldv, gfid});
            pay_schedule_[gfid] = PAID_IN_FULL;
        }
        return n_open_clients_;
    }

    FT get_fac_cost(size_t ind) const {
        if(facility_cost_.size()) {
            if(facility_cost_.size() == 1) return facility_cost_[0];
            return facility_cost_[ind];
        }
        throw std::invalid_argument("Facility cost must be set");
    }

    void cluster_results() {
        std::vector<IT> temporarily_open;
        for(size_t i = 0; i < pay_schedule_.size(); ++i) {
            if(pay_schedule_[i] == PAID_IN_FULL)
                temporarily_open.push_back(i);
        }
        std::vector<std::vector<IT>> open_facility_assignments;
        std::vector<IT> open_facilities;
        shared::flat_hash_set<IT> assigned_clients;
        if(!distmatp_) {
            throw std::runtime_error("distmatp must be set");
        }
        const MatrixType &distmat(*distmatp_);
        // Close temporary facilities
        while(!temporarily_open.empty()) {
            IT cfid = temporarily_open.back();
            temporarily_open.pop_back();
            std::vector<IT> facility_assignment;
            for(IT cid = 0; cid < ncities_; ++cid) {
                payment_t client_data = client_v_[cid];
                IT witness = client_data.second; // WITNESS ME
                FT witness_cost = client_data.first - client_w_(witness, cid) + distmat(witness, cid);
                FT current_cost = client_data.first - client_w_(cfid, cid) + distmat(cfid, cid);
                if(current_cost <= witness_cost && client_w_(cfid, cid) != PAID_IN_FULL) {
                    assigned_clients.insert(cid);
                    facility_assignment.push_back(cid);
                    std::vector<IT> facilities_to_rm;
                    const FT cwc = client_w_(cfid, cid);
                    if(cwc > 0 && cwc != PAID_IN_FULL) {
                        for(const auto f2rm: temporarily_open) {
                            FT c2c = client_w_(f2rm, cid);
                            if(c2c > 0 && c2c != PAID_IN_FULL) {
                                facilities_to_rm.push_back(f2rm);
                            }
                        }
                    } // TODO: speed up this removal
                    for(const auto f2rm: facilities_to_rm) {
                        auto it = std::find(temporarily_open.begin(), temporarily_open.end(), f2rm);
                        if(it != temporarily_open.end()) {
                            std::swap(*it, temporarily_open.back());
                            temporarily_open.pop_back();
                        } else throw std::runtime_error("Error in facility removal");
                    }
                }
            }
            if(facility_assignment.size()) {
                open_facility_assignments.push_back(std::move(facility_assignment));
                open_facilities.push_back(cfid);
            }
        }

#if VERBOSE_AF
        std::fprintf(stderr, "Got here, assigning the rest of unassigned\n");
#endif

        // Assign all unassigned
        for(IT cid = 0; cid < ncities_; ++cid) {
            if(assigned_clients.find(cid) != assigned_clients.end()) continue;
            // cout << "Assigning client " << j << endl;
            IT best_fid = 0;
            FT mindist = distmat(open_facilities.front(), cid), cdist;
            for(size_t i = 1; i < open_facilities.size(); ++i) {
                if((cdist = distmat(open_facilities[i], cid)) < mindist)
                    mindist = cdist, best_fid = i;
            }
            open_facility_assignments[best_fid].push_back(cid);
        }
        final_open_facilities_ = std::move(open_facilities);
        final_open_facility_assignments_ = std::move(open_facility_assignments);
        std::fprintf(stderr, "%zu open facilities\n", final_open_facilities_.size());
    }

    IT service_tight_edge(edge_type edge) {
        //std::fprintf(stderr, "About to service tight edge\n");
        payment_t oldp, newp;
        IT fid = edge.fi(), cid = edge.di(); // Facility id, city id
        auto cost = edge.cost();
        std::vector<IT> updated_clients;
        //std::fprintf(stderr, "edge cost: %g\n", cost);

        //Set when cid starts contributing to fid
        const bool open_cid = open_client(clients_cpy_[cid]);
        //std::fprintf(stderr, "open_cid? %d\n", open_cid);
        if(open_cid) {
            client_w_(fid, cid) = cost;
        }

        //
        if(pay_schedule_[fid] == PAID_IN_FULL) {
            //std::fprintf(stderr, "Facility fid = %u is paid in full\n", fid);
            working_open_facilities_[fid].push_back(cid);
            updated_clients.push_back(cid);
            n_open_clients_ = update_facilities(fid, std::vector<IT>{cid}, cost);
        } else {
            if(open_cid) {
                clients_cpy_[cid].push_back(fid);
                //std::fprintf(stderr, "City %d now has facility %u as its newest with total number of facilities %zu\n", cid, fid, clients_cpy_[cid].size());
            }
            auto &fac_clients = working_open_facilities_[fid];
            const FT current_facility_cost = get_fac_cost(fid);
            if(!open_client(fac_clients)) {
                //std::fprintf(stderr, "No current contributing clients. Now add empty to fac clients.\n");
                assert(fac_clients.size() == 1);
#if 0
                fac_clients[0] = cid;
#else
                std::replace(fac_clients.begin(), fac_clients.end(), EMPTY, cid);
                contribution_time_[fid] = cost;
#endif
                next_paid_.update(pay_schedule_[fid], current_facility_cost, fid);
            } else {
                // facility already has some clients
                size_t nclients_fid = fac_clients.size();
                //assert(nclients_fid);
                FT update_client_pay = nclients_fid * (cost - contribution_time_[fid]);
                fac_contributions_[fid] += update_client_pay;
                FT current_pay = fac_contributions_[fid];
                if(open_cid) {
                    working_open_facilities_[fid].push_back(cid);
                    ++nclients_fid;
                }
                contribution_time_[fid] = cost;
                if(current_pay >= current_facility_cost) {
                    // Open facility
                    n_open_clients_ = update_facilities(fid, working_open_facilities_[fid], cost);
                } else {
                    FT oldc = pay_schedule_[fid];
                    FT remaining_time = nclients_fid ? (current_facility_cost - current_pay) / nclients_fid
                                                     : current_facility_cost - cost + EPS;
                    FT newc = cost + remaining_time;
                    pay_schedule_[fid] = newc;
                    next_paid_.update(oldc, newc, fid);
                }
            }
        }
        return n_open_clients_;
    }

    static constexpr IT EMPTY    = std::numeric_limits<IT>::max();
    static constexpr FT PAID_IN_FULL = std::numeric_limits<FT>::max();
    static constexpr FT EPS = 1e-10;

    INLINE static bool open_client(const std::vector<IT> &client) {
        return client.empty() || std::find(client.begin(), client.end(), EMPTY) == client.end();
    }


public:
    JVSolver(): distmatp_(nullptr), nedges_(0), ncities_(0), nfac_(0) {
    }

    template<typename CostType>
    JVSolver(const this_type &o, const CostType &cost):
        distmatp_(o.distmatp_),
        client_w_(o.client_w_.rows(), o.client_w_.columns()),
        edges_(o.edges_), // Note: this is copying a reference to the shared ptr of edges_.
        client_v_(o.client_w_.columns(), payment_t{PAID_IN_FULL, EMPTY}),
        clients_cpy_(o.distmatp_->columns(), std::vector<IT>()),
        working_open_facilities_(new std::vector<IT>[o.distmatp_->rows()]),
        contribution_time_(new FT[o.distmatp_->rows()]()),
        fac_contributions_(new FT[o.distmatp_->rows()]()),
        n_open_clients_(o.distmatp_->columns()),
        nedges_(o.nedges_),
        ncities_(o.ncities_),
        nfac_(o.nfac_)
    {
        client_w_ = FT(0);
        set_fac_cost(cost);
        pay_schedule_.resize(nfac_);
        next_paid_.clear();
        for(size_t i = 0; i < nfac_; ++i) {
            auto cost = get_fac_cost(i);
            next_paid_.push({cost, i});
            pay_schedule_[i] = cost;
        }
    }

    template<typename CostType>
    JVSolver(const MatrixType &mat, const CostType &cost): JVSolver() {
        setup(mat, cost);
    }
    JVSolver(const MatrixType &mat): JVSolver(mat, blz::max(mat)) {
    }

    template<typename CostType>
    void reset_cost(const CostType &cost) {
        set_fac_cost(cost);
        client_w_ = static_cast<FT>(0);
        for(size_t i = 0; i < ncities_; ++i) clients_cpy_[i].clear();
            for(size_t i = 0; i < nfac_; ++i)
                working_open_facilities_[i] = {EMPTY};
        std::memset(contribution_time_.get(), 0, sizeof(contribution_time_[0]) * nfac_);
        std::memset(fac_contributions_.get(), 0, sizeof(fac_contributions_[0]) * nfac_);
        n_open_clients_ = ncities_;
        pay_schedule_.resize(nfac_);
        next_paid_.clear();
        for(size_t i = 0; i < nfac_; ++i) {
            auto cost = get_fac_cost(i);
            next_paid_.push({cost, i});
            pay_schedule_[i] = cost;
        }
        assert(next_paid_.find({get_fac_cost(0), 0}) != next_paid_.end());
        assert(next_paid_.size() == pay_schedule_.size());
        assert(client_w_.rows() == distmatp_->rows());
        assert(client_w_.columns() == distmatp_->columns());
    }

    template<typename CostType>
    this_type clone_with_cost(const CostType &cost) {
        return this_type(*this, cost);
    }

    template<typename CostType>
    void setup(const MatrixType &mat, const CostType &cost) {
        distmatp_ = &mat;
        set_fac_cost(cost);

        // Initialize W and edge vector
        client_w_.resize(mat.rows(), mat.columns());
        client_w_ = static_cast<FT>(0);
        const size_t edges_to_use = blaze::IsDenseMatrix_v<MatrixType> ? mat.rows() * mat.columns(): blaze::nonZeros(mat);
        if(nedges_ != edges_to_use) {
            edges_.reset(new edge_type[edges_to_use]);
        }
        nedges_ = edges_to_use;
        // Set edge values, then sort by cost
        if constexpr(blaze::IsDenseMatrix_v<MatrixType>) {
            OMP_PFOR
            for(size_t i = 0; i < mat.rows(); ++i) {
                const size_t nc = mat.columns();
                edge_type *const eptr = &edges_[i * mat.columns()];
                auto matptr = row(mat, i, blaze::unchecked);
                size_t j = 0;
                for(;j + 8 <= nc; j += 8) {
                    eptr[j] =     {matptr[j], i, j};
                    eptr[j + 1] = {matptr[j + 1], i, j + 1};
                    eptr[j + 2] = {matptr[j + 2], i, j + 2};
                    eptr[j + 3] = {matptr[j + 3], i, j + 3};
                    eptr[j + 4] = {matptr[j + 4], i, j + 4};
                    eptr[j + 5] = {matptr[j + 5], i, j + 5};
                    eptr[j + 6] = {matptr[j + 6], i, j + 6};
                    eptr[j + 7] = {matptr[j + 7], i, j + 7};
                }
                for(;j < nc; eptr[j] = {matptr[j], i, j}, ++j);
            }
        } else if constexpr(blaze::IsSparseMatrix_v<MatrixType>) {
#ifdef _OPENMP
            std::unique_ptr<IT[]> offsets(new IT[mat.rows()]);
            offsets[0] = 0;
            for(size_t i = 1; i < mat.rows(); ++i) {
                offsets[i] = offsets[i - 1] + blaze::nonZeros(row(mat, i, blaze::unchecked));
            }
            OMP_PFOR
            for(size_t i = 0; i < mat.rows(); ++i) {
                auto eptr = edges_.get() + offsets[i];
                if(offsets[i] != offsets[i + 1]) // If non-empty
                    for(const auto &pair: row(mat, i, blaze::unchecked))
                        *eptr++ = {pair->value(), i, pair->index()};
            }
#else
            auto eptr = edges_.get();
            for(size_t i = 0; i < mat.rows(); ++i)
                for(const auto &pair: row(mat, i, blaze::unchecked))
                    *eptr++ = {pair->value(), i, pair->index()};
            assert(eptr == &edges_[nedges_]);
#endif
        } else {
            throw std::runtime_error("Not currently supported: non-blaze matrices");
        }
        shared::sort(edges_.get(), edges_.get() + nedges_, [](edge_type x, edge_type y) {
            return x.cost() < y.cost();
        });

        // Initialize V, T, and S
        if(ncities_ < mat.columns()) {
            client_v_.clear();
            client_v_.resize(mat.columns());
            clients_cpy_.clear();
            clients_cpy_.resize(mat.columns());
        } else {
            for(size_t i = 0; i < mat.columns(); ++i) clients_cpy_[i].clear();
        }
        std::fill(client_v_.data(), &client_v_[mat.columns()], payment_t{PAID_IN_FULL, EMPTY});
        if(nfac_ < mat.rows()) {
            working_open_facilities_.reset(new std::vector<IT>[mat.rows()]);
            nfac_ = mat.rows();
            contribution_time_.reset(new FT[nfac_]());
            fac_contributions_.reset(new FT[nfac_]());
        } else {
            for(size_t i = 0; i < nfac_; working_open_facilities_[i++] = {EMPTY});
            std::memset(contribution_time_.get(), 0, sizeof(contribution_time_[0]) * nfac_);
            std::memset(fac_contributions_.get(), 0, sizeof(fac_contributions_[0]) * nfac_);
        }
        ncities_ = mat.columns();
        nfac_ = mat.rows();
        n_open_clients_ = ncities_;
        pay_schedule_.resize(nfac_);
        next_paid_.clear();
        for(size_t i = 0; i < nfac_; ++i) {
            auto cost = get_fac_cost(i);
            next_paid_.push({cost, i});
            pay_schedule_[i] = cost;
        }
    }

    payment_t next_facility() {
        return next_paid_.pop();
    }

    auto &run() {
        open_candidates();
        cluster_results();
        return final_open_facilities_;
    }

    FT open_candidates() {
        IT edge_idx = 0;
        FT time = 0.;
        while(n_open_clients_) {
            if(edge_idx == nedges_) {
                //std::fprintf(stderr, "All edges processed, now starting final loop\n");
                time = final_phase1_loop(time);
                break;
            }
            edge_type current_edge = edges_[edge_idx];
            auto current_edge_cost = current_edge.cost();
            payment_t next_fac = next_paid_.empty() ? payment_t{current_edge_cost + 1e10, 0}
                                                    : next_facility();

            if(current_edge_cost <= next_fac.first) {
                n_open_clients_ = service_tight_edge(current_edge);
                time = current_edge_cost;
                ++edge_idx;
            } else {
                n_open_clients_ = update_facilities(next_fac.second, working_open_facilities_[next_fac.second], time);
                time = next_fac.first;
            }
        }
        return time;
    }

    template<typename VT, bool TF>
    void set_fac_cost(const blaze::Vector<VT, TF> &val) {
        if((~val).size() != client_w_.rows()) throw std::invalid_argument("Val has wrong number of rows");
        facility_cost_.resize((~val).size());
        facility_cost_ = (~val);
    }
    template<typename CostType, typename=std::enable_if_t<std::is_convertible_v<CostType, FT> > >
    void set_fac_cost(CostType val) {
        facility_cost_.resize(1);
        facility_cost_[0] = val;
    }
    size_t nedges() const {
        return client_w_.rows() * client_w_.columns();
    }
    FT calculate_cost(bool including_costs=true) {
        FT sum = 0.;
        for(size_t i = 0; i < final_open_facilities_.size(); ++i) {
            const auto fid = final_open_facilities_[i];
            const auto &clients = final_open_facility_assignments_[i];
            for(const auto cid: clients) {
                sum += (*distmatp_)(fid, cid);
            }
            if(including_costs) {
                sum += get_fac_cost(fid);
            }
        }
        return sum;
    }
    std::pair<std::vector<IT>, std::vector<std::vector<IT>>>
    kmedian(unsigned k, unsigned maxrounds=500, double maxcost_starting=0.) {
        auto kmed_start = std::chrono::high_resolution_clock::now();
        auto &dm = *distmatp_;
        double maxcost;
        if(maxcost_starting) maxcost = maxcost_starting;
        else {
            maxcost = max(dm);
            if(std::isinf(maxcost)) {
                maxcost = 0.;
                for(auto r: blz::rowiterator(dm))
                    for(auto v: r)
                        if(std::isfinite(v) && v > maxcost)
                            maxcost = v;
            }
            maxcost *= dm.columns();
        }
        double mincost = 0.;
        double medcost = maxcost / 2;
        reset_cost(medcost);
        run();
        DBG_ONLY(std::fprintf(stderr, "##first solution for cost %g: %zu (want k %u)\n", medcost, final_open_facilities_.size(), k);)
        size_t roundnum = 0;
        for(;;) {
#if !NDEBUG
            std::fprintf(stderr, "##round %zu. current size: %zu\n", ++roundnum, final_open_facilities_.size());
#endif
            size_t nopen = final_open_facilities_.size();
            if(nopen == k) break;
            if(nopen > k) {
                mincost = medcost; // med has too many, increase cost.
                //std::fprintf(stderr, "Assigning mincost to current cost. New lower bound on cost: %g. k: %u. current sol size %zu\n", mincost, k, nopen);
            } else {
                maxcost = medcost; // med has too few, lower cost.
                //std::fprintf(stderr, "Assigning maxcost to current cost. New upper bound on cost: %g because we have too few items (%zu instead of %u)\n", maxcost, nopen, k);
            }
            //std::fprintf(stderr, "##mincost: %g. maxcost: %g\n", mincost, maxcost);
            double rat = double(nopen) / k;
            double lam = 1. - rat / (1. + rat);
            medcost = lam * mincost + (1. - lam) * maxcost;
#ifndef NDEBUG
            auto start = std::chrono::high_resolution_clock::now();
#endif
            reset_cost(medcost);
            run();
#ifndef NDEBUG
            auto stop = std::chrono::high_resolution_clock::now();
            std::fprintf(stderr, "##Solution cost: %f. size: %zu. Time in ms: %g. Dimensions: %zu/%zu\n", calculate_cost(false), final_open_facilities_.size(), (stop - start).count() * 0.000001, client_w_.rows(), client_w_.columns());
#endif
            if(roundnum > maxrounds) {
                break;
            }
        }
        auto kmed_stop = std::chrono::high_resolution_clock::now();
#ifndef NDEBUG
        std::fprintf(stderr, "Solution cost with %zu centers: %g. Time to perform clustering: %g\n", final_open_facilities_.size(), calculate_cost(false),
                     (kmed_stop - kmed_start).count() * 1.e-6);
#else
        std::fprintf(stderr, "Solution with %u centers took %gms\n", k, (kmed_stop - kmed_start).count() * 1.e-6);
#endif
        return std::make_pair(final_open_facilities_, final_open_facility_assignments_);
    }
};

} // namespace jv

} // namespace fgc

#endif
