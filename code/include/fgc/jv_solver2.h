#ifndef JV_SOLVER2_H__
#define JV_SOLVER2_H__
#include "blaze_adaptor.h"

namespace fgc {
namespace jv2 {
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


template<typename FT=float, typename IT=uint32_t>
struct JVSolver {

    using payment_t = shared::packed_pair<FT, IT>;
    struct pay_compare_t {
        bool operator()(const payment_t lhs, const payment_t rhs) const {
            return lhs.first > rhs.first;
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
            if(it == this->end()) throw 1;
            this->erase(it);
            tmp.first = newc;
            this->insert(tmp);
        }
    };

private:
    blz::DM<FT> w_;
    // W matrix
    // Willing of each client to pay for each facility
    std::unique_ptr<edgetup[]> edges_;
    // list of all edges, sorted by cost
    std::unique_ptr<shared::packed_pair<FT, IT>[]> client_v_;
    // List of coverage by each facility

    blz::DV<FT> facility_cost_;
    // Costs for facilities. Of size 1 if uniform, of size # fac otherwise.

    // List of facilities assigned to
    std::unique_ptr<std::vector<IT>[]> clients_cpy_;

    // List of temporarily open facilities
    std::unique_ptr<std::vector<IT>[]> open_facilities_;

    // List of temporarily open facilities
    std::vector<std::vector<IT>> final_open_facilities_;

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
            n_open_clients_ = update_facilities(next_fac.second, open_facilities_[next_fac.second], time_);
            if(n_open_clients_ == 0) break;
        }
    }

    // TODO: consider replacing with blaze::SmallArray to avoid heap allocation/access
    IT update_facilities(IT f_id, const std::vector<IT> &update_facilities, FT cost) {
        std::fprintf(stderr, "About to update facility %zu with update_facilities of size %zu with cost %g\n",
                     size_t(f_id), update_facilities.size(), cost);
        throw std::runtime_error("Not implemented");
        return 0;
    }

    FT get_fac_cost(size_t ind) const {
        if(facility_cost_.size()) {
            if(facility_cost_.size() == 1) return facility_cost_[0];
            return facility_cost_.__access(ind);
        }
        throw std::invalid_argument("Facility cost must be set");
    }

    void cluster_results() {
        throw std::runtime_error("Not implemented");
    }

    IT service_tight_edge(edgetup edge) {
        payment_t oldp, newp;
        IT fid = edge.fi(), cid = edge.di(); // Facility id, city id
        auto cost = edge.cost();
        std::vector<IT> updated_clients;

        //Set when cid starts contributing to fid
        const bool open_cid = open_client(clients_cpy_[cid]);
        if(open_cid) {
            w_(fid, cid) = cost;
        }

        //
        if(pay_schedule_.__access(fid).first == PAID_IN_FULL) {
            open_facilities_[fid].push_back(cid);
            updated_clients.push_back(cid);
            n_open_clients_ = update_facilities(fid, std::vector<IT>{cid}, cost);
        } else {
            if(open_cid) {
                clients_cpy_[cid].push_back(fid);
            }
            auto &fac_clients = open_facilities_[fid];
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
                    open_facilities_[fid].push_back(cid);
                    ++nclients_fid;
                }
                contribution_time_[fid] = cost;
                if(current_pay >= current_facility_cost) {
                    // Open facility
                    n_open_clients_ = update_facilities(fid, open_facilities_[fid], cost);
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
    JVSolver(): nedges_(0), ncities_(0), nfac_(0) {
    }

    template<typename Mat, typename CostType>
    void set(const Mat &mat, const CostType &cost) {
        // Set facility cost (either a single value or one per facility)
        set_fac_cost(cost);

        // Initialize W and edge vector
        w_.resize(mat.rows(), mat.columns());
        if(nedges_ < w_.rows() * w_.columns()) {
            nedges_ = w_.rows() * w_.columns();
            edges_.reset(new edgetup[nedges_]);
        }
        // Set edge values, then sort by cost
        OMP_PFOR
        for(size_t i = 0; i < w_.rows(); ++i) {
            edgetup *const eptr = &edges_[i * w_.columns()];
            auto matptr = row(mat, i, blaze::unchecked);
            _Pragma("GCC unroll 8")
            for(size_t j = 0; j < w_.columns(); ++j) {
                eptr[j] = {matptr[j], i, j};
            }
        }
        shared::sort(edges_.get(), edges_.get() + nedges_, [](edgetup x, edgetup y) {
            return x.cost() < y.cost();
        });

        // Initialize V, T, and S
        if(ncities_ < w_.columns()) {
            client_v_.reset(new shared::packed_pair<FT, IT>[w_.columns()]);
            clients_cpy_.reset(new std::vector<IT>[w_.columns()]);
        } else {
            for(size_t i = 0; i < w_.columns(); ++i) clients_cpy_[i].clear();
        }
        std::fill(client_v_.get(), &client_v_[w_.columns()], shared::packed_pair<FT, IT>{-1.,EMPTY});
        if(nfac_ < w_.rows()) {
            open_facilities_.reset(new std::vector<IT>[w_.rows()]);
            nfac_ = w_.rows();
            contribution_time_.reset(new FT[nfac_]());
            fac_contributions_.reset(new FT[nfac_]());
        } else {
            for(size_t i = 0; i < nfac_; ++i)
                open_facilities_[i] = {EMPTY};
            std::memset(contribution_time_.get(), 0, sizeof(contribution_time_[0]) * nfac_);
            std::memset(fac_contributions_.get(), 0, sizeof(fac_contributions_[0]) * nfac_);
        }
        ncities_ = w_.columns();
        nfac_ = w_.rows();
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
                n_open_clients_ = update_facilities(next_fac.second, open_facilities_[next_fac.second], time_);
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
        if((~val).size() != w_.rows()) throw std::invalid_argument("Val has wrong number of rows");
        facility_cost_.resize((~val).size());
        facility_cost_ = (~val);
    }
    template<typename CostType, typename=std::enable_if_t<std::is_convertible_v<CostType, FT> > >
    void set_fac_cost(CostType val) {
        facility_cost_.resize(1);
        facility_cost_[0] = val;
    }
    size_t nedges() const {
        return w_.rows() * w_.columns();
    }
};

} // namespace jv2

} // namespace fgc

#endif
