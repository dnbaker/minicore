#ifndef JV_SOLVER2_H__
#define JV_SOLVER2_H__
#include "blaze_adaptor.h"

namespace fgc {
namespace jv {
struct edgetup: public std::tuple<double, uint32_t, uint32_t> {
    // Consists of:
    // 1. Cost of edge
    // 2. Facility index
    // 3 Distance index.
    // Can be easily accessed with these member functions:
    template<typename...A> edgetup(A &&...args): std::tuple<double, uint32_t, uint32_t>(std::forward<A>(args)...) {}
    auto cost() const {return std::get<0>(*this);}
    auto &cost() {return std::get<0>(*this);}
    auto fi() const {return std::get<1>(*this);}
    auto &fi() {return std::get<1>(*this);}
    auto di() const {return std::get<2>(*this);}
    auto &di() {return std::get<2>(*this);}
    auto sprintf(char *buf) const {
        return std::sprintf(buf, "%f:%u:%u", cost(), fi(), di());
    }
};

#ifndef NDEBUG
#define __access operator[]
#else
#define __access at
#endif


template<typename MatrixType, typename FT=float, typename IT=uint32_t>
struct JVSolver {

    static_assert(std::is_floating_point<FT>::value, "FT must be floating-point");
    static_assert(std::is_integral<IT>::value, "IT must be integral");

    using payment_t = shared::packed_pair<FT, IT>;
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
        auto top() const {
            return *this->begin();
        }
        auto pop() {
            auto ret = top();
            this->erase(this->begin());
            return ret;
        }
        void update(FT oldc, FT newc, IT idx) {
            // Remove old payment, add new payment
            payment_t tmp = {oldc, idx};
            auto it = this->find(tmp);
            if(unlikely(it == this->end()))
                throw std::runtime_error("Payment missing from structure");
            this->erase(it);
            tmp.first = newc;
            this->insert(tmp);
        }
    };

    const MatrixType *distmatp_;
    blz::DM<FT> client_w_;
    // W matrix
    // Willing of each client to pay for each facility
    std::unique_ptr<edgetup[]> edges_;
    // list of all edges, sorted by cost
    std::unique_ptr<payment_t[]> client_v_;
    // List of coverage by each facility

    blz::DV<FT> facility_cost_;
    // Costs for facilities. Of size 1 if uniform, of size # fac otherwise.

    // List of facilities assigned to
    std::unique_ptr<std::vector<IT>[]> clients_cpy_;

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
            shared::packed_pair<FT, IT> next_fac = next_facility();
            if(next_fac.first > 0) {
                time_ = next_fac.first;
            }
            n_open_clients_ = update_facilities(next_fac.second, working_open_facilities_[next_fac.second], time_);
            if(n_open_clients_ == 0) break;
        }
    }

    // TODO: consider replacing with blaze::SmallArray to avoid heap allocation/access
    IT update_facilities(IT gfid, const std::vector<IT> &update_facilities, const FT cost) {
        std::fprintf(stderr, "About to update facility %zu with update_facilities of size %zu with cost %g\n",
                     size_t(gfid), update_facilities.size(), cost);
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
    }

    IT service_tight_edge(edgetup edge) {
        payment_t oldp, newp;
        IT fid = edge.fi(), cid = edge.di(); // Facility id, city id
        auto cost = edge.cost();
        std::vector<IT> updated_clients;

        //Set when cid starts contributing to fid
        const bool open_cid = open_client(clients_cpy_[cid]);
        if(open_cid) {
            client_w_(fid, cid) = cost;
        }

        //
        if(pay_schedule_.__access(fid).first == PAID_IN_FULL) {
            working_open_facilities_[fid].push_back(cid);
            updated_clients.push_back(cid);
            n_open_clients_ = update_facilities(fid, std::vector<IT>{cid}, cost);
        } else {
            if(open_cid) {
                clients_cpy_[cid].push_back(fid);
            }
            auto &fac_clients = working_open_facilities_[fid];
            const FT current_facility_cost = get_fac_cost(fid);
            if(!open_client(fac_clients)) {
                std::replace(fac_clients.begin(), fac_clients.end(), EMPTY, cid);
                contribution_time_[fid] = cost;
                next_paid_.update(pay_schedule_[fid].first, current_facility_cost, fid);
            } else {
                // facility already has some clients
                size_t nclients_fid = fac_clients.size();
                assert(nclients_fid);
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
                    FT remaining_time = fac_clients.size() ? (current_facility_cost - current_pay) / nclients_fid
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

    INLINE static constexpr bool open_client(const std::vector<IT> &client) {
        return client.empty() || std::find(client.begin(), client.end(), EMPTY) == client.end();
    }

public:
    JVSolver(): distmatp_(nullptr), nedges_(0), ncities_(0), nfac_(0) {
    }

    template<typename CostType>
    void set(const MatrixType &mat, const CostType &cost) {
        distmatp_ = &mat;
        // Set facility cost (either a single value or one per facility)
        set_fac_cost(cost);

        // Initialize W and edge vector
        client_w_.resize(mat.rows(), mat.columns());
        if(nedges_ < client_w_.rows() * client_w_.columns()) {
            nedges_ = client_w_.rows() * client_w_.columns();
            edges_.reset(new edgetup[nedges_]);
        }
        // Set edge values, then sort by cost
        OMP_PFOR
        for(size_t i = 0; i < client_w_.rows(); ++i) {
            edgetup *const eptr = &edges_[i * client_w_.columns()];
            auto matptr = row(mat, i, blaze::unchecked);
            _Pragma("GCC unroll 8")
            for(size_t j = 0; j < client_w_.columns(); ++j) {
                eptr[j] = {matptr[j], i, j};
            }
        }
        shared::sort(edges_.get(), edges_.get() + nedges_, [](edgetup x, edgetup y) {
            return x.cost() < y.cost();
        });

        // Initialize V, T, and S
        if(ncities_ < client_w_.columns()) {
            client_v_.reset(new shared::packed_pair<FT, IT>[client_w_.columns()]);
            clients_cpy_.reset(new std::vector<IT>[client_w_.columns()]);
        } else {
            for(size_t i = 0; i < client_w_.columns(); ++i) clients_cpy_[i].clear();
        }
        std::fill(client_v_.get(), &client_v_[client_w_.columns()], shared::packed_pair<FT, IT>{-1.,EMPTY});
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
        for(const auto payment: pay_schedule_) next_paid_.push(payment);
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
                final_phase1_loop();
                break;
            }
            edgetup current_edge = edges_[edge_idx];
            auto current_edge_cost = current_edge.cost();
            shared::packed_pair<FT, IT> next_fac;
            if(!next_paid_.empty())
                next_fac = next_facility();
            else
                next_fac = {current_edge_cost + EPS, 0};

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
        cluster_results();
        return final_open_facilities_;
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
};

} // namespace jv2

} // namespace fgc

#endif
