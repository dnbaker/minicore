#ifndef JV_SOLVER_H__
#define JV_SOLVER_H__
#include "minocore/util/blaze_adaptor.h"
#include "minocore/util/packed.h"
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>

namespace minocore {

namespace shared {

template<typename T1, typename T2>
using packed_pair = packed::pair<T1, T2>;

template<typename... Types>
using packed_triple = packed::triple<Types...>;

} // namespace shared

namespace jvutil {

template<typename FT, typename IT=uint32_t>
struct edgetup: public packed::triple<FT, IT, IT> {
    // Consists of:
    // 1. Cost of edge
    // 2. Facility index
    // 3 Distance index.
    // Can be easily accessed with these member functions:
    using super = packed::triple<FT, IT, IT>;
    template<typename...A> edgetup(A &&...args): super(std::forward<A>(args)...) {}
    auto cost() const {return this->first;}
    //auto &cost()      {return this->first;}
    auto fi()   const {return this->second;}
    //auto &fi()        {return this->second;}
    auto di()   const {return this->third;}
    //auto &di()        {return this->third;}
    auto sprintf(char *buf) const {
        return std::sprintf(buf, "%f:%u:%u", cost(), fi(), di());
    }
    auto printf(std::FILE *ofp=stderr) const {
        return std::fprintf(ofp, "%f:%u:%u", cost(), fi(), di());
    }
};

} // namespace jvutil

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
        void pop_top() {
            this->erase(this->begin());
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
                assert(this->find(tmp) == this->end() || !std::fprintf(stderr, "Just removed %0.12g:%u, and just found it here with value %0.12g:%u\n", tmp.first, tmp.second, this->find(tmp)->first, this->find(tmp)->second));
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

    bool verbose = false;


    // Private code

    FT final_phase1_loop(FT time) {
        while(!next_paid_.empty()) {
            if(next_paid_.top().first > 0) {
                time = next_paid_.top().first;
            }
            n_open_clients_ = update_facilities(next_paid_.top().second, working_open_facilities_[next_paid_.top().second], time);
            //next_paid_.pop_top();
            std::fprintf(stderr, "n open clients: %zu. facilities size: %zu\n", n_open_clients_, next_paid_.size());
            if(n_open_clients_ == 0) break;
        }
        return time;
    }

    // TODO: consider replacing with blaze::SmallArray to avoid heap allocation/access
    IT update_facilities(IT gfid, const std::vector<IT> &update_facilities, const FT cost) {
#if VERBOSE_AF
        std::fprintf(stderr, "About to update facility %zu with update_facilities of size %zu with cost %0.12g\n",
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

    void cluster_results(std::atomic<int> *early_terminate=nullptr) {
        std::vector<IT> temporarily_open;
        for(size_t i = 0; i < pay_schedule_.size(); ++i) {
            if(pay_schedule_[i] == PAID_IN_FULL)
                temporarily_open.push_back(i);
        }
        if(verbose) std::fprintf(stderr, "%zu temporarily open\n", temporarily_open.size());

        std::vector<std::vector<IT>> open_facility_assignments;
        std::vector<IT> open_facilities;
        shared::flat_hash_set<IT> unassigned_clients;
        for(size_t i = 0; i < ncities_; ++i)
            unassigned_clients.insert(i);
        if(!distmatp_) {
            throw std::runtime_error("distmatp must be set");
        }
        const MatrixType &distmat(*distmatp_);
        // Close temporary facilities
        while(!temporarily_open.empty()) {
            if(early_terminate && early_terminate->load()) return;
            IT cfid = temporarily_open.back();
            temporarily_open.pop_back();
            std::vector<IT> facility_assignment;
            for(IT cid = 0; cid < ncities_; ++cid) {
                payment_t client_data = client_v_[cid];
                IT witness = client_data.second; // WITNESS ME
                FT witness_cost = client_data.first - client_w_(witness, cid) + distmat(witness, cid);
                FT current_cost = client_data.first - client_w_(cfid, cid) + distmat(cfid, cid);
                if(current_cost <= witness_cost && client_w_(cfid, cid) != PAID_IN_FULL) {
                    unassigned_clients.erase(cid);
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

        if(early_terminate && early_terminate->load()) return;
        // Assign all unassigned
        if(open_facilities.empty()) {
            blaze::DynamicVector<FT> fac_costs = blaze::sum<blaze::rowwise>(distmat);
            open_facilities.push_back(std::min_element(fac_costs.begin(), fac_costs.end()) - fac_costs.begin());
        }
        for(const IT cid: unassigned_clients) {
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
        DBG_ONLY(if(verbose) std::fprintf(stderr, "%zu open facilities\n", final_open_facilities_.size());)
    }
    void reassign() {
        final_open_facility_assignments_.resize(final_open_facilities_.size());
        for(auto &f: final_open_facility_assignments_) f.clear();
        for(IT cid = 0; cid < ncities_; ++cid) {
            IT best_fid = 0;
            FT mindist = (*distmatp_)(final_open_facilities_.front(), cid), cdist;
            for(size_t i = 1; i < final_open_facilities_.size(); ++i) {
                if((cdist = (*distmatp_)(final_open_facilities_[i], cid)) < mindist)
                    mindist = cdist, best_fid = i;
            }
            final_open_facility_assignments_[best_fid].push_back(cid);
        }
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

    void make_verbose() {
        verbose = true;
    }

    template<typename CostType>
    JVSolver(const MatrixType &mat, const CostType &cost): JVSolver() {
        setup(mat, cost);
    }
    JVSolver(const MatrixType &mat): JVSolver(mat, blaze::max(mat)) {
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
        if(verbose) std::fprintf(stderr, "Edges set\n");
        shared::sort(edges_.get(), edges_.get() + nedges_, [](edge_type x, edge_type y) {
            return x.cost() < y.cost();
        });
        if(verbose) std::fprintf(stderr, "Edges sorted\n");

        // Initialize V, T, and S
        if(ncities_ < mat.columns()) {
            client_v_.clear();
            client_v_.resize(mat.columns());
            clients_cpy_.clear();
            clients_cpy_.resize(mat.columns());
        } else {
            OMP_PFOR
            for(size_t i = 0; i < mat.columns(); ++i) clients_cpy_[i].clear();
        }
        OMP_PRAGMA("omp parallel for schedule(static,256)")
        for(size_t i = 0; i < mat.columns(); ++i)
            client_v_[i] = payment_t{PAID_IN_FULL, EMPTY};
        if(nfac_ < mat.rows()) {
            working_open_facilities_.reset(new std::vector<IT>[mat.rows()]);
            nfac_ = mat.rows();
            contribution_time_.reset(new FT[nfac_]());
            fac_contributions_.reset(new FT[nfac_]());
        } else {
            OMP_PRAGMA("omp parallel for schedule(static,512)")
            for(size_t i = 0; i < nfac_; ++i)
                working_open_facilities_[i] = {EMPTY};

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

    auto &run(std::atomic<int> *early_terminate=nullptr) {
        if(early_terminate && early_terminate->load()) return final_open_facilities_;
        open_candidates(early_terminate);
        if(early_terminate && early_terminate->load()) return final_open_facilities_;
        cluster_results(early_terminate);
        return final_open_facilities_;
    }

    FT open_candidates(std::atomic<int> *early_terminate=nullptr) {
        IT edge_idx = 0;
        FT time = 0.;
        DBG_ONLY(const size_t edge_log_num = nedges_ / 10;)
        while(n_open_clients_) {
            if(early_terminate && early_terminate->load()) return time;
            if(edge_idx == nedges_) {
                //std::fprintf(stderr, "All edges processed, now starting final loop\n");
                time = final_phase1_loop(time);
                break;
            }
            edge_type current_edge = edges_[edge_idx];
            auto current_edge_cost = current_edge.cost();
            if(next_paid_.size() && next_paid_.top().first > current_edge_cost) {
                const auto next_fac = next_paid_.top();
                size_t current_n = next_paid_.size();
                //DBG_ONLY(std::fprintf(stderr, "Trying to update by removing the next facility. Current in next_paid_ %zu\n", next_paid_.size());)
                n_open_clients_ = update_facilities(next_fac.second, working_open_facilities_[next_fac.second], time);
                time = next_fac.first;
                if(current_n == next_paid_.size()) // If it wasn't removed
                    next_paid_.pop_top();
                //DBG_ONLY(std::fprintf(stderr, "n open: %zu. time: %0.12g. Now facilities left to pay: %zu\n", size_t(n_open_clients_), time, next_paid_.size());)
            } else {
                n_open_clients_ = service_tight_edge(current_edge);
                time = current_edge_cost;
                ++edge_idx;
                DBG_ONLY(if(verbose && edge_idx % edge_log_num == 0) std::fprintf(stderr, "Processed %zu/%zu edges\n", size_t(edge_idx), nedges_);)
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
    static void run_loop(this_type &solver, double mycost, std::atomic<double> &maxcost, std::atomic<double> &mincost,
                         std::set<double> &current_running, std::mutex &mut, unsigned &maxk, unsigned &mink,
                         std::atomic<int> &terminate, uint64_t seed, unsigned nthreads, std::atomic<uint32_t> &rounds_completed, uint32_t max_rounds, unsigned k)
    {
        wy::WyRand<uint32_t, 0> rng(seed);
        std::uniform_real_distribution<double> urd;
        //const double spacing = 1. / nthreads;
        const size_t my_id = std::hash<std::thread::id>()(std::this_thread::get_id());
        for(;;) {
            solver.reset_cost(mycost);
            if(terminate.load()) break;
            solver.run(&terminate);
            if(terminate.load()) break;
            ++rounds_completed;
            if(rounds_completed.load() >= max_rounds) {
                terminate.store(1);
                break;
            }
            size_t nf = solver.final_open_facilities_.size();
            if(nf == k) {
                std::fprintf(stderr, "[%zu] nf == k, setting to terminate\n", my_id);
                terminate.store(1);
                maxcost.store(mycost);
                mincost.store(mycost);
                {
                    std::lock_guard<std::mutex> lock(mut);
                    mink = maxk = nf;
                }
                break;
            } else if(nf > k) {
                std::fprintf(stderr, "[%zu] nf > k, setting mincost from %g to %g\n", my_id, mincost.load(), mycost);
                double lastmincost = mincost;
                while(mycost > lastmincost && !std::atomic_compare_exchange_weak(
                      &mincost,
                      &lastmincost,
                      mycost))
                {
                     std::fprintf(stderr, "Doing compare/weak with mincost = %g and last = %g\n", mincost.load(), lastmincost);
                }
                if(nf < maxk) {
                    std::fprintf(stderr, "[%zu] nf > k, setting maxk from %u to %zu at %g \n", my_id, maxk, nf, mycost);
                    std::lock_guard<std::mutex> lock(mut);
                    maxk = nf;
                }
                std::fprintf(stderr, "[%zu] released lock, nf > k[%u] (csize: %zu at %g) Old min cost: %g\n", my_id, k, nf, mycost, lastmincost);
            } else {
                double lastmaxcost = maxcost;
                while(mycost < lastmaxcost && !std::atomic_compare_exchange_weak(
                      &maxcost,
                      &lastmaxcost,
                      mycost));
                if(nf > mink) {
                    std::lock_guard<std::mutex> lock(mut);
                    mink = nf;
                }
                std::fprintf(stderr, "[%zu] released lock, nf < k [%u] (csize: %zu at %g). Old max cost: %g\n", my_id, k,  nf, mycost, lastmaxcost);
            }
            double cost;
            typename std::set<double>::const_iterator it;
            // do
            {
                auto randval = urd(rng);
                if(mincost.load()) {
                    double exp = std::pow(maxcost.load() / mincost.load(), 1. / nthreads);
                    cost = std::pow(exp, randval) * mincost.load();
                } else {
                    cost = (maxcost.load() - mincost.load()) * std::pow(randval, 3) + mincost.load();
                }
                std::fprintf(stderr, "[%zu] randomly selected cost: %g\n", my_id, cost);
                // TODO: Consider random selection in geometric mean
            }
            //while(maxcost.load() != mincost.load() &&
            //      myspacing / (maxcost.load() - mincost.load()) < spacing * .1 &&
            //      !terminate.load());
            std::fprintf(stderr, "Selected new cost: %g\n", cost);
            {
                std::lock_guard<std::mutex> lock(mut);
                current_running.erase(mycost);
                current_running.insert(cost);
            }
            mycost = cost;
            if(terminate.load()) break;
            std::fprintf(stderr, "thread %zu viewing min/max costs of %g/%g\n", my_id,
                                mincost.load(), maxcost.load());
        }
    }
    std::pair<std::vector<IT>, std::vector<std::vector<IT>>>
    kmedian_parallel(int num_threads, unsigned k, unsigned maxrounds, double maxcost=0., double mincost=0., uint64_t seed = 0) {
        auto fstart = std::chrono::high_resolution_clock::now();
        if(num_threads <= 1)
            return kmedian(k, maxrounds, maxcost, mincost);
        std::vector<this_type> solvers;
        auto &dm = *distmatp_;
        if(maxcost == 0.) {
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
        std::unique_ptr<double[]> assigned_costs(new double[num_threads]);
        if(mincost == 0) {
            while(solvers.size() < size_t(num_threads)) {
                double frac = (1. + solvers.size()) / (num_threads + 1);
                double assigned_cost = maxcost * std::pow(frac, 4);
                assigned_costs[solvers.size()] = assigned_cost;
                solvers.emplace_back(clone_with_cost(assigned_cost));
            }
        } else {
            double mul = std::pow(maxcost / mincost, 1. / (num_threads + 1));
            while(solvers.size() < unsigned(num_threads)) {
                double assigned_cost = mincost * std::pow(mul, 4 * (solvers.size() + 1));
                assigned_costs[solvers.size()] = assigned_cost;
                solvers.emplace_back(clone_with_cost(assigned_cost));
            }
        }
        // One randomly goes to a small cost to ensure we have a non-zero lower bound
        // to then use the geometric mean rather than arithmetic
        assigned_costs[wy::WyRand<uint32_t, 0>(seed)() % num_threads] = 1.e-6 * maxcost;
        std::atomic<double> amaxcost, amincost;
        amaxcost.store(maxcost);
        amincost.store(mincost);
        std::mutex mut;
        unsigned maxk = std::numeric_limits<unsigned>::max(), mink = 0;
        std::atomic<int> terminate;
        terminate.store(0);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        std::set<double> current_running_costs(assigned_costs.get(), assigned_costs.get() + num_threads);
        std::atomic<uint32_t> rounds_completed;
        rounds_completed.store(0);
        while(threads.size() < size_t(num_threads)) {
            unsigned ind = threads.size();
            threads.emplace_back(
                run_loop, std::ref(solvers[ind]), assigned_costs[ind],
                          std::ref(amaxcost), std::ref(amincost), std::ref(current_running_costs),
                          std::ref(mut), std::ref(maxk), std::ref(mink),
                          std::ref(terminate), seed + ind, num_threads, std::ref(rounds_completed), maxrounds, k
            );
        }
        std::fprintf(stderr, "Threads started [%zu]\n", threads.size());
        for(auto &t: threads) t.join();
        maxcost = amaxcost.load();
        mincost = amincost.load();
        std::pair<std::vector<IT>, std::vector<std::vector<IT>>> ret;
        if(maxcost == mincost) { // Success!
            auto it = std::find_if(solvers.begin(), solvers.end(), [k](const auto &x) {
                return x.final_open_facilities_.size() == k;
            });
            if(it == solvers.end()) throw 1;
            ret = std::make_pair(std::move(it->final_open_facilities_), std::move(it->final_open_facility_assignments_));
        } else {
            auto it = std::min_element(solvers.begin(), solvers.end(),[k](const auto &x, const auto &y) {
                std::ptrdiff_t lhdiff = x.final_open_facilities_.size() > k ? x.final_open_facilities_.size() - k: k - x.final_open_facilities_.size();
                std::ptrdiff_t rhdiff = y.final_open_facilities_.size() > k ? y.final_open_facilities_.size() - k: k - y.final_open_facilities_.size();
                return lhdiff < rhdiff;
            });
            // Got close. Greedy local search to desired count.
            auto &lsolver = *it;
            while(lsolver.final_open_facilities_.size() < k) {
                lsolver.final_open_facilities_.push_back(lsolver.local_best_to_add());
            }
            while(lsolver.final_open_facilities_.size() > k) {
                IT to_rm = lsolver.local_best_to_rm();
                auto lit = std::find(lsolver.final_open_facilities_.begin(), lsolver.final_open_facilities_.end(), to_rm);
                lsolver.final_open_facilities_.erase(lit);
                lsolver.final_open_facility_assignments_.erase(lsolver.final_open_facility_assignments_.begin() + (lit - lsolver.final_open_facilities_.begin()));
            }
            lsolver.reassign();
            ret = std::make_pair(std::move(lsolver.final_open_facilities_), std::move(lsolver.final_open_facility_assignments_));
        }
        auto fstop = std::chrono::high_resolution_clock::now();
        if(verbose) std::fprintf(stderr, "Solution of size %zu took %u rounds and %0.12gms\n",
                                 ret.first.size(), rounds_completed.load(), (fstop - fstart).count() * 1.e-6);
        return ret;
    }
    std::pair<std::vector<IT>, std::vector<std::vector<IT>>>
    kmedian(unsigned k, unsigned maxrounds=100, double maxcost=0., double mincost=0.)
    {
        auto kmed_start = std::chrono::high_resolution_clock::now();
        auto &dm = *distmatp_;
        if(maxcost == 0.) {
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
        double medcost = (maxcost - mincost) / (dm.columns()) + mincost;
        if(verbose) std::fprintf(stderr, "First iteration, medcost = %0.12g, mincost = %0.12g, maxcost = %0.12g\n", medcost, mincost, maxcost);
        auto fstart = std::chrono::high_resolution_clock::now();
        reset_cost(medcost);
        run();
        auto fstop = std::chrono::high_resolution_clock::now();
        if(verbose) std::fprintf(stderr, "[V|%s:%d:%s] first solution for cost %0.12g took %0.6gms and had %zu facilities (want k %u)\n",
                                 __PRETTY_FUNCTION__, __LINE__, __FILE__, medcost, (fstop - fstart).count() *1.e-6, final_open_facilities_.size(), k);
        size_t roundnum = 0;
        unsigned lower_k_ = -1, upper_k_ = -1;
        if(final_open_facilities_.size() > k) upper_k_ = final_open_facilities_.size();
        else                                  lower_k_ = final_open_facilities_.size();
        while(final_open_facilities_.size() != k) {
            size_t nopen = final_open_facilities_.size();
            if(nopen > k) {
                mincost = medcost; // med has too many, increase cost.
                upper_k_ = nopen;
                if(verbose) std::fprintf(stderr, "Assigning mincost to current cost. New lower bound on cost: %0.12g. k: %u. current sol size %zu. upper k: %u\n", mincost, k, nopen, upper_k_);
            } else {
                maxcost = medcost; // med has too few, lower cost.
                lower_k_ = nopen;
                if(verbose) std::fprintf(stderr, "Assigning maxcost to current cost. New upper bound on cost: %0.12g because we have too few items (%zu instead of %u). k lower: %u\n", maxcost, nopen, k, lower_k_);
            }
            double ratio = double(nopen) / k;
#if 0
            // Old filter to fall back on arithmetic mean, but it might be okay to remve.
            if(ratio < 1.05 && ratio > 0.95) {
                medcost = (maxcost + mincost) * .5;
            } else
#endif
            if(mincost != 0) {
                medcost = std::sqrt(maxcost) * std::sqrt(mincost); // Geometric mean instead of arithmetic
                if(verbose) std::fprintf(stderr, "%0.12g = sqrt(%0.12g [mincost] * [maxcost] %0.12g)\n", medcost, mincost, maxcost);
            } else {
                double lam = 1. - ratio / (1. + ratio);
                medcost = lam * mincost + (1. - lam) * maxcost;
                if(verbose) std::fprintf(stderr, "%0.12g = %0.12g * (mincost/%g) + %0.12g * (maxcost/%g)\n", medcost, lam, 1. - lam, mincost, maxcost);
            }
            fstart = std::chrono::high_resolution_clock::now();
            reset_cost(medcost);
            run();
            fstop = std::chrono::high_resolution_clock::now();
            if(verbose) std::fprintf(stderr, "[Round %zu] Facility cost: %0.12g. Size: %zu. Time in ms: %g. \n",
                                     roundnum, medcost, final_open_facilities_.size(), (fstop - fstart).count() * 0.000001);
            if(++roundnum > maxrounds || std::abs(mincost - maxcost) < 1e-16 * medcost) {
                std::fprintf(stderr, "Failed to find exact solution using JV in %zu rounds. Now using local search from current solution of %zu points to desired k = %u\n",
                             roundnum, final_open_facilities_.size(), k);
                while(final_open_facilities_.size() < k) {
                    final_open_facilities_.push_back(local_best_to_add());
                }
                while(final_open_facilities_.size() > k) {
                    IT to_rm = local_best_to_rm();
                    auto it = std::find(final_open_facilities_.begin(), final_open_facilities_.end(), to_rm);
                    final_open_facilities_.erase(it);
                    final_open_facility_assignments_.erase(final_open_facility_assignments_.begin() + (it - final_open_facilities_.begin()));
                }
                reassign();
            }
        }
        auto kmed_stop = std::chrono::high_resolution_clock::now();
        FT sol_cost = calculate_cost(false);
        std::fprintf(stderr, "Solution cost with %zu centers: %g. Time to perform clustering: %g\n", final_open_facilities_.size(), sol_cost,
                     (kmed_stop - kmed_start).count() * 1.e-6);
        return std::make_pair(final_open_facilities_, final_open_facility_assignments_);
    }
    IT local_best_to_add() const {
        blaze::DynamicVector<FT,blaze::rowVector> current_costs = blaze::min<blaze::columnwise>(blaze::rows(*distmatp_, final_open_facilities_.data(), final_open_facilities_.size()));
        FT max_improvement = -std::numeric_limits<FT>::max();
        IT bestind = -1;
        for(size_t i = 0; i < nfac_; ++i) {
            if(std::find(final_open_facilities_.begin(), final_open_facilities_.end(), i) !=  final_open_facilities_.end())
                continue;
            FT improvement = 0.;
            auto lrow = row(*distmatp_, i);
            for(size_t j = 0; j < lrow.size(); ++j) {
                FT cost = lrow[j];
                if(cost < current_costs[j])
                    improvement += (current_costs[j] - cost);
            }
            if(improvement > max_improvement) improvement = max_improvement, bestind = i;
        }
        return bestind;
    }
    IT local_best_to_rm() const {
        blaze::DynamicVector<FT, blaze::rowVector> current_costs = blaze::min<blaze::columnwise>(blaze::rows(*distmatp_, final_open_facilities_.data(), final_open_facilities_.size()));
        FT min_loss = std::numeric_limits<FT>::max();
        IT bestind = -1;
        std::unique_ptr<IT[]> min_counters(new IT[ncities_]());
        for(size_t i = 0; i < ncities_; ++i)
            for(const auto fid: final_open_facilities_)
                if(current_costs[i] == (*distmatp_)(fid, i))
                    ++min_counters[i];
        for(const auto fid: final_open_facilities_) {
            FT loss = 0.;
            for(size_t i = 0; i < ncities_; ++i) {
                if(current_costs[i] == (*distmatp_)(fid, i) && min_counters[i] == 1) {
                    FT minv = std::numeric_limits<FT>::max(); // Get the next-lowest cost in this set
                    for(const auto fid2: final_open_facilities_)
                        if(fid != fid2)
                            minv = std::min(minv, (*distmatp_)(fid2, i));
                    loss += current_costs[i] - minv; // Should be nonpositive
                }
            }
            if(loss < min_loss) min_loss = loss, bestind = fid;
        }
        return bestind;
    }
};

template<typename MT, typename FT=blaze::ElementType_t<MT>, typename IT=uint32_t>
auto make_jv_solver(const MT &mat) {
    return JVSolver<MT, FT, IT>(mat);
}


} // namespace jv

} // namespace minocore

#endif
