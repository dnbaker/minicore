#ifndef JV_SOLVER_H__
#define JV_SOLVER_H__
#include "blaze_adaptor.h"
#include "jv_util.h"
#include <chrono>

namespace fgc {

namespace jv {

#ifndef NDEBUG
#define __access operator[]
#else
#define __access at
#endif


template<typename MatrixType, typename FT=blaze::ElementType_t<MatrixType>, typename IT=uint32_t>
struct JVSolver {

    static_assert(std::is_floating_point<FT>::value, "FT must be floating-point");
    static_assert(std::is_integral<IT>::value, "IT must be integral");

    using payment_t = packed::pair<FT, IT>;
    using edge_type = jvutil::edgetup<FT, IT>;

private:
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
            if(oldc == newc) return;
            assert(oldc != PAID_IN_FULL);
            assert(newc != PAID_IN_FULL);
            // Remove old payment, add new payment
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

    const MatrixType *distmatp_;
    blz::DM<FT> client_w_;
    // W matrix
    // Willingness of each client to pay for each facility
    std::unique_ptr<edge_type[]> edges_;
    // list of all edges, sorted by cost

    std::vector<payment_t> client_v_;
    // List of coverage by each facility

    blz::DV<FT> facility_cost_;
    // Costs for facilities. Of size 1 if uniform, of size # fac otherwise.

    // List of facilities assigned to
    std::vector<std::vector<IT>> clients_cpy_;

    // List of temporarily open facilities
    std::unique_ptr<std::vector<IT>[]> working_open_facilities_;

    // List of final open facilities
    std::vector<IT> final_open_facilities_;

    // List of final facility assignments
    std::vector<std::vector<IT>> final_open_facility_assignments_;

    // Time when contributions are made to a facility
    std::unique_ptr<FT[]> contribution_time_;
    // Track of contributions to a given facility
    std::unique_ptr<FT[]> fac_contributions_;

    std::vector<payment_t> pay_schedule_;

    payment_queue next_paid_;

    size_t n_open_clients_;

    size_t nedges_;
    size_t ncities_;
    size_t nfac_;

    FT time_;


    // Private code
    void final_phase1_loop() {
        while(!next_paid_.empty()) {
            payment_t next_fac = next_facility();
            if(next_fac.first > 0) {
                time_ = next_fac.first;
            }
            n_open_clients_ = update_facilities(next_fac.second, working_open_facilities_[next_fac.second], time_);
            if(n_open_clients_ == 0) break;
        }
        //std::fprintf(stderr, "Finished final_phase1_loop\n");
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
                    if(fac_pay.first != PAID_IN_FULL) {
                        if(client_w_(fid, cid) != PAID_IN_FULL) {
                            FT nclients_fid = working_open_facilities_[fid].size();
                            FT update_pay = nclients_fid * (cost - contribution_time_[fid]);
                            FT oldv = fac_pay.first;
                            FT current_contrib = fac_contributions_[fid] + update_pay;
                            fac_contributions_[fid] = current_contrib;
                            fac_pay.first -= current_contrib;
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
                            fac_pay.first = newv;
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
        if(FT oldv = pay_schedule_[gfid].first; oldv != PAID_IN_FULL) {
            next_paid_.erase(payment_t{oldv, gfid});
            pay_schedule_[gfid].first = PAID_IN_FULL;
        }
        return n_open_clients_;
    }

    FT get_fac_cost(size_t ind) const {
        if(facility_cost_.size()) {
            if(facility_cost_.size() == 1) return facility_cost_[0];
            return facility_cost_.__access(ind);
        }
        throw std::invalid_argument("Facility cost must be set");
    }



    void cluster_results() {
        std::vector<IT> temporarily_open;
        for(size_t i = 0; i < pay_schedule_.size(); ++i) {
            if(pay_schedule_[i].first == PAID_IN_FULL)
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
        if(pay_schedule_.__access(fid).first == PAID_IN_FULL) {
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
                if(fac_clients.size() != 1) throw std::runtime_error("Testing if this assumption is true");
#if 0
                fac_clients[0] = cid;
#else
                std::replace(fac_clients.begin(), fac_clients.end(), EMPTY, cid);
                contribution_time_[fid] = cost;
#endif
                next_paid_.update(pay_schedule_[fid].first, current_facility_cost, fid);
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
                    FT oldc = pay_schedule_[fid].first;
                    FT remaining_time = nclients_fid ? (current_facility_cost - current_pay) / nclients_fid
                                                     : current_facility_cost - cost + EPS;
                    FT newc = cost + remaining_time;
                    pay_schedule_[fid].first = newc;
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
        const bool ret = client.empty() || std::find(client.begin(), client.end(), EMPTY) == client.end();
#ifndef NDEBUG
        {
            bool oret = !client.empty() && std::find(client.begin(), client.end(), EMPTY) != client.end()
                        ? false: true;
            assert(oret == ret);
        }
#endif
        return ret;
    }
    

public:
    JVSolver(): distmatp_(nullptr), nedges_(0), ncities_(0), nfac_(0) {
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
        for(size_t i = 0; i < client_w_.columns(); ++i) clients_cpy_[i].clear();
            for(size_t i = 0; i < nfac_; ++i)
                working_open_facilities_[i] = {EMPTY};
        std::memset(contribution_time_.get(), 0, sizeof(contribution_time_[0]) * nfac_);
        std::memset(fac_contributions_.get(), 0, sizeof(fac_contributions_[0]) * nfac_);
        n_open_clients_ = ncities_;
        pay_schedule_.resize(nfac_);
        for(size_t i = 0; i < nfac_; ++i) {
            pay_schedule_[i] = {get_fac_cost(i), i};
        }
        next_paid_.clear();
        next_paid_.push(pay_schedule_.begin(), pay_schedule_.end());
        assert(next_paid_.find({get_fac_cost(0), 0}) != next_paid_.end());
        assert(next_paid_.size() == pay_schedule_.size());
        assert(client_w_.rows() == distmatp_->rows());
        assert(client_w_.columns() == distmatp_->columns());
    }

    template<typename CostType>
    void setup(const MatrixType &mat, const CostType &cost) {
        //std::fprintf(stderr, "starting setup\n");
        distmatp_ = &mat;
        // Set facility cost (either a single value or one per facility)
        //std::fprintf(stderr, "setting facility cost\n");
        set_fac_cost(cost);

        // Initialize W and edge vector
        //std::fprintf(stderr, "resizing client w\n");
        client_w_.resize(mat.rows(), mat.columns());
        //std::fprintf(stderr, "setting client w\n");
        client_w_ = static_cast<FT>(0);
        if(nedges_ < client_w_.rows() * client_w_.columns()) {
            edges_.reset(new edge_type[client_w_.rows() * client_w_.columns()]);
        }
        nedges_ = client_w_.rows() * client_w_.columns();
        // Set edge values, then sort by cost
        //DBG_ONLY(edge_type *const total_eptr = &edges_[nedges_];)
        OMP_PFOR
        for(size_t i = 0; i < client_w_.rows(); ++i) {
            edge_type *const eptr = &edges_[i * client_w_.columns()];
            auto matptr = row(mat, i, blaze::unchecked);
            const size_t nc = client_w_.columns();
            size_t j = 0;
#if 0
            for(;j + 8 <= nc;j += 8) {
                eptr[j] =     {matptr[j], i, j};
                eptr[j + 1] = {matptr[j + 1], i, j + 1};
                eptr[j + 2] = {matptr[j + 2], i, j + 2};
                eptr[j + 3] = {matptr[j + 3], i, j + 3};
                eptr[j + 4] = {matptr[j + 4], i, j + 4};
                eptr[j + 5] = {matptr[j + 5], i, j + 5};
                eptr[j + 6] = {matptr[j + 6], i, j + 6};
                eptr[j + 7] = {matptr[j + 7], i, j + 7};
                j += 8;
            }
            assert(j <= matptr.size()); 
#endif
            while(j < nc) {
                assert(j < matptr.size());
                eptr[j] = {matptr[j], i, j}, ++j;
            }
        }
        shared::sort(edges_.get(), edges_.get() + nedges_, [](edge_type x, edge_type y) {
            return x.cost() < y.cost();
        });

        // Initialize V, T, and S
        if(ncities_ < client_w_.columns()) {
            client_v_.clear();
            client_v_.resize(client_w_.columns());
            clients_cpy_.clear();
            clients_cpy_.resize(client_w_.columns());
        } else {
            for(size_t i = 0; i < client_w_.columns(); ++i) clients_cpy_[i].clear();
        }
        std::fill(client_v_.data(), &client_v_[client_w_.columns()], payment_t{PAID_IN_FULL, EMPTY});
        if(nfac_ < client_w_.rows()) {
            working_open_facilities_.reset(new std::vector<IT>[client_w_.rows()]);
            nfac_ = client_w_.rows();
            contribution_time_.reset(new FT[nfac_]());
            fac_contributions_.reset(new FT[nfac_]());
        } else {
            for(size_t i = 0; i < nfac_; ++i)
                working_open_facilities_[i] = {EMPTY};
            std::memset(contribution_time_.get(), 0, sizeof(contribution_time_[0]) * nfac_);
            std::memset(fac_contributions_.get(), 0, sizeof(fac_contributions_[0]) * nfac_);
        }
        ncities_ = client_w_.columns();
        nfac_ = client_w_.rows();
        n_open_clients_ = ncities_;
        pay_schedule_.resize(nfac_);
        for(size_t i = 0; i < nfac_; ++i) {
            pay_schedule_[i] = {get_fac_cost(i), i};
        }
        next_paid_.clear();
        next_paid_.push(pay_schedule_.begin(), pay_schedule_.end());
        assert(next_paid_.find({get_fac_cost(0), 0}) != next_paid_.end());
        assert(next_paid_.size() == pay_schedule_.size());
    }

    payment_t next_facility() {
        return next_paid_.pop();
    }

    auto &run() {
        open_candidates();
        return prune_candidates();
    }
    void open_candidates() {
        IT edge_idx = 0;
        time_ = 0.;
        while(n_open_clients_) {
            if(edge_idx == nedges_) {
                //std::fprintf(stderr, "All edges processed, now starting final loop\n");
                final_phase1_loop();
                break;
            }
            edge_type current_edge = edges_[edge_idx];
            auto current_edge_cost = current_edge.cost();
            payment_t next_fac;
            if(!next_paid_.empty())
                next_fac = next_facility();
            else {
                next_fac = {current_edge_cost + 1e10, 0};
            }

            if(current_edge_cost <= next_fac.first) {
                n_open_clients_ = service_tight_edge(current_edge);
                time_ = current_edge_cost;
                ++edge_idx;
            } else {
                n_open_clients_ = update_facilities(next_fac.second, working_open_facilities_[next_fac.second], time_);
                time_ = next_fac.first;
            }
        }
    }
    auto &prune_candidates() {
        //std::fprintf(stderr, "Starting pruning\n");
        cluster_results();
        return final_open_facilities_;
    }

    template<typename VT, bool TF>
    void set_fac_cost(const blaze::Vector<VT, TF> &val) {
        if((~val).size() != client_w_.rows()) throw std::invalid_argument("Val has wrong number of rows");
        //std::cerr << ~val << '\n';
        facility_cost_.resize((~val).size());
        facility_cost_ = (~val);
    }
    template<typename CostType, typename=std::enable_if_t<std::is_convertible_v<CostType, FT> > >
    void set_fac_cost(CostType val) {
        facility_cost_.resize(1);
        facility_cost_[0] = val;
        //std::fprintf(stderr, "facility cost: %g\n", val);
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
            if(rat > 1.1 || 1. / rat > 1.1) {
            // TODO: This binary search can be tweaked a little, I'm sure
                double lam = 1. - rat / (1. + rat);
                medcost = lam * mincost + (1. - lam) * maxcost;
#if VERBOSE_AF
                std::fprintf(stderr, "Because of spaced out stuff, I'm trying an imbalanced binary search.\n"
                                     "I would have used %g, but instead because %zu open with k = %u, using %g\n",
                             (mincost + maxcost) / 2., nopen, k, medcost);
#endif
            } else medcost = (mincost + maxcost) / 2.;
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

namespace dontuse {

template<typename FT>
struct NaiveJVSolver {
    // Complexity: something like F^2N + N^2F
    // Much slower than it should be
    using DefIT = unsigned int;
    using edge_type = jvutil::edgetup<FT, DefIT>;
    blz::DM<FT> w_;
    blz::DV<FT> v_;
    blz::DV<uint32_t> numconnected_, numtight_;
    std::vector<edge_type> edges_;
    size_t edgeindex_ = 0;
    double facility_cost_, maxalph_;
    std::unordered_set<uint32_t> S_, tempopen_, nottempopen_;
    NaiveJVSolver(size_t nf, size_t nc, double fc=1.):
        w_(nf, nc, 0), v_(nc, 0), numconnected_(nf, 0), numtight_(nf, 0), edges_(nf * nc), facility_cost_(fc), maxalph_(0)
    {
    }
    void reset(double newfacility_cost) {
        if(newfacility_cost) facility_cost_ = newfacility_cost;
        w_ = FT(0);
        //((blaze::DynamicMatrix<FT> &)w_) = FT(0);
        v_ = FT(0);
        numconnected_ = 0;
        numtight_ = 0;
        maxalph_ = 0;
        nottempopen_.clear();
        for(size_t i = 0; i < w_.rows(); ++i) nottempopen_.insert(i);
        tempopen_.clear();
    }
    template<typename MatType>
    void setup(const MatType &mat) {
        auto start = std::chrono::high_resolution_clock::now();
        if(mat.rows() != w_.rows() || mat.columns() != w_.columns()) {
            char buf[256];
            std::sprintf(buf, "Wrong number of rows or columns: received %zu/%zu, expected %zu/%zu\n", mat.rows(), mat.columns(), w_.rows(), w_.columns());
            throw std::runtime_error(buf);
        }
        OMP_PFOR
        for(size_t i = 0; i < mat.rows(); ++i) {
            auto p = &edges_[i * mat.columns()];
            auto r = row(mat, i);
            for(size_t j = 0; j < mat.columns(); ++j) {
                *p++ = {r[j], i, j};
            }
        }
        shared::sort(&edges_[0], &edges_[edges_.size()], [](const auto x, const auto y) {return x.cost() < y.cost();});
        tempopen_.clear();
        for(size_t i = 0; i < mat.rows(); ++i) nottempopen_.insert(i);
        auto stop = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "Setup took %g\n", 0.000001 * (stop - start).count());
    }
    template<typename MatType, typename IType=DefIT>
    std::vector<IType> phase2() { // Electric Boogaloo
        std::fprintf(stderr, "tos: ntos: %zu/%zu\n", tempopen_.size(), nottempopen_.size());
        double sum = blaze::sum(w_);
        uint64_t seed;
        std::memcpy(&seed, &sum, sizeof(seed));
        wy::WyRand<uint32_t, 2> rng(seed);
        std::vector<uint32_t> tov(tempopen_.begin(), tempopen_.end());
        std::vector<uint32_t> to_remove;
        auto lai = rng() % tov.size();
        std::swap(tov[lai], tov.back());
        auto la = tov.back();
        tov.pop_back();
        std::vector<IType> ret{la};
        while(tov.size()) {
            auto r = row(w_, la);
            for(size_t i = 0; i < w_.columns(); ++i) {
                if(r[i] > 0.) {
                    for(size_t j = 0; j < w_.rows(); ++j)
                        if(w_(j, i) > 0.)
                            to_remove.push_back(j);
                }
            }
            for(const auto item: to_remove)
                tempopen_.erase(item);
            tov.assign(tempopen_.begin(), tempopen_.end());
            if(tempopen_.empty()) break;
            to_remove.clear();
            auto ci = rng() % tov.size();
            la = tov[ci];
            std::swap(tov[ci], tov.back());
            tov.pop_back();
            ret.push_back(la);
        }
        return ret;
    }
    template<typename MatType, typename CT>
    double calculate_cost(const MatType &mat, const CT &open_facilities) const {
        if(open_facilities.empty()) return std::numeric_limits<double>::max();
        double faccost = open_facilities.size() * facility_cost_;
        double citycost = blz::sum(blz::min<blz::columnwise>(rows(mat, open_facilities, blaze::unchecked)));
        return citycost + faccost;
    }
    template<typename MatType, typename IType=DefIT>
    std::vector<IType> ufl(const MatType &mat, double faccost) {
        // Uncapacited Facility Location problem with facility cost = faccost
        this->reset(faccost);
        assert(nottempopen_.size() == w_.rows());
        assert(tempopen_.size() == 0);
        std::fprintf(stderr, "##Starting phase1 with faccost %.12g\n", faccost);
        phase1(mat);
        return phase2<MatType, IType>();
    }
    template<typename MatType, typename IType=DefIT>
    std::vector<IType> kmedian(const MatType &mat, unsigned k, unsigned maxrounds=500, double maxcost_starting=0.) {
        setup(mat);
        double maxcost = maxcost_starting ? maxcost_starting: double(mat.columns() * max(mat));
        if(std::isinf(maxcost)) {
            maxcost = std::numeric_limits<double>::min();
            for(const auto r: blz::rowiterator(mat)) {
                for(const auto v: r)
                    if(!std::isinf(v) && v > maxcost)
                        maxcost = v;
            }
        }
        double mincost = 0.;
        double medcost = maxcost / 2;
        //auto ubound = ufl(mat, maxcost);
        //auto lbound = ufl(mat, mincost);
        auto med = ufl(mat, medcost);
        std::fprintf(stderr, "##first solution: %zu (want k %u)\n", med.size(), k);
        size_t roundnum = 0;
        while(med.size() != k) {
            std::fprintf(stderr, "##round %zu. current size: %zu\n", ++roundnum, med.size());
            if(med.size() == k) break;
            if(med.size() > k)
                mincost = medcost; // med has too many, increase cost.
            else
                maxcost = medcost; // med has too few, lower cost.
            medcost = (mincost + maxcost) / 2.;
            auto start = std::chrono::high_resolution_clock::now();
            med = ufl<MatType, IType>(mat, medcost);
            auto stop = std::chrono::high_resolution_clock::now();
            std::fprintf(stderr, "Solution cost: %f. size: %zu. Time in ms: %g. Dimensions: %zu/%zu\n", calculate_cost(mat, med), med.size(), (stop - start).count() * 0.000001, w_.rows(), w_.columns());
            if(roundnum > maxrounds) {
                break;
            }
        }
        return med;
    }
    std::pair<uint32_t, double> min_tightening_cost() const {
        if(edgeindex_ == edges_.size()) return std::make_pair(uint32_t(-1), std::numeric_limits<double>::max());
        auto edge = edges_[edgeindex_];
        return std::make_pair(edge.di(), edge.cost() - maxalph_);
    }
    std::pair<uint32_t, double> min_opening_cost() const {
        double mincost = std::numeric_limits<double>::max();
        uint32_t ind = -1u;
        for(const auto fid: nottempopen_) {
            auto nsupport = std::accumulate(row(w_, fid).begin(), row(w_, fid).end(), size_t(0), [](auto x, auto y) {return x + y >= 0.;});
            if(nsupport == 0) return std::make_pair(-1u, std::numeric_limits<double>::max());
            auto availsum = std::accumulate(row(w_, fid).begin(), row(w_, fid).end(), 0.);
            //std::fprintf(stderr, "rowsum: %f. facility cost: %f\n", availsum, facility_cost_);
            auto diff = facility_cost_ - availsum;
            auto cost = nsupport ? diff / nsupport: std::numeric_limits<double>::max();
            //std::fprintf(stderr, "diff: %g. cost: %g\n", diff, cost);
            if(cost < mincost) mincost = cost, ind = fid;
        }
        return std::make_pair(ind, mincost);
    }
    template<typename MatType>
    void perform_increment(double inc,  std::vector<uint32_t> &to_remove, const MatType &mat) {
        maxalph_ += inc;
        for(const auto item: S_) {
            v_[item] = maxalph_;
            for(size_t fi = 0; fi < w_.rows(); ++fi) {
                if(maxalph_ >= mat(fi,item)) {
                    if(tempopen_.find(fi) != tempopen_.end()) // && std::find(to_remove.begin(), to_remove.end(), item) != to_remove.end())
                        to_remove.push_back(item);
                }
                w_(fi, item) = std::max(0., maxalph_ - mat(fi, item));
            }
        }
    }
    template<typename MatType>
    void phase1(const MatType &mat) {
        auto &S(S_);
        S.clear(); S.reserve(v_.size());
        for(size_t i = 0; i < v_.size(); S.insert(i++));
        assert(nottempopen_.size() == w_.rows());
        assert(w_.rows());
        std::vector<uint32_t> to_remove;
        size_t nz = 0;
        edgeindex_ = std::find_if(edges_.begin(), edges_.end(), [](auto x) {return x.cost() > 0.;})
                     - edges_.begin();
        // Skip over 0 indices
        std::fprintf(stderr, "nz: %zu\n", nz);
        while(S.size()) {
            //std::fprintf(stderr, "Size of S: %zu. nto size: %zu. tos: %zu\n", S.size(), nottempopen_.size(), tempopen_.size());
            //std::fprintf(stderr, "getting min tight cost\n");
            auto [bestedge, tightinc] = min_tightening_cost();
            //std::fprintf(stderr, "got min tight cost\n");
            auto [bestfac, openinc]   = min_opening_cost();
            //std::fprintf(stderr, "got min opening cost\n");
            bool tighten = true;
            if(tightinc < openinc) {
                auto bec = edges_[edgeindex_].cost();
                do ++edgeindex_; while(edges_[edgeindex_].cost() == bec);
                // Skip over identical weights
            } else tighten = false;
            const double inc = std::min(tightinc, openinc);
            //std::fprintf(stderr, "inc: %g. open: %g. tighten: %g\n", inc, openinc, tightinc);
            perform_increment(inc, to_remove, mat);
            //std::fprintf(stderr, "new alpha: %g\n", maxalph_);
            if(!tighten) {
                //auto fc = facility_cost_;
                tempopen_.insert(bestfac);
                nottempopen_.erase(bestfac);
                //std::fprintf(stderr, "Inserting bestfac %u. nto size %zu. tempopen size: %zu\n", bestfac, nottempopen_.size(), tempopen_.size());
                for(const auto item: S) {
                    assert(v_.size() && v_.size() > item);
                    if(v_.at(item) >= mat(bestfac, item)) // && std::find(to_remove.begin(), to_remove.end(), s) != to_remove.end())
                        to_remove.push_back(item);
                }
            }
            for(const auto item: to_remove) {
                S.erase(item);
            }
        }
    }
};

} // namespace dontuse

} // namespace jv

} // namespace fgc

#endif
