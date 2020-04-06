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


template<typename FT=float, typename IT=uint32_t>
struct JVSolver2 {
private:
    blz::DM<FT> client_w_;
    // W matrix
    // Willing of each client to pay for each facility
    std::unique_ptr<edgetup[]> edges_;
    // list of all edges, sorted by cost
    std::unique_ptr<shared::packed_pair<FT, IT>[]> client_v_;
    // List of coverage by each facility

    blz::DV<FT> facility_cost_;
    // Costs for facilities. Of size 1 if uniform, of size # fac otherwise.

    // List of facilities assigned to
    std::unique_ptr<std::vector<IT>> clients_cpy_;

    // List of temporarily open facilities
    std::unique_ptr<std::vector<IT>> open_facilities_;

    // List of temporarily open facilities
    std::vector<std::vector<IT>> final_open_facilities_;

    // Time when contributions are made to a facility
    std::unique_ptr<FT> contribution_time_;
    // Track of contributions to a given facility
    std::unique_ptr<FT> fac_contributions_;

    std::vector<shared::packed_pair<FT, IT>> pay_schedule_, next_paid_;
    
    size_t n_open_clients_;

    size_t nedges_;
    size_t ncities_;
    size_t nfac_;

    FT time_;


    // Private code
    void final_phase1_loop() {
        while(!next_paid_.empty()) {
            shared::packed_pack<FT, IT> next_fac = next_paid_.back();
            next_paid_.pop_back();
            if(next_fac.first > 0) {
                time_ = next_fac.first;
            }
            n_open_clients_ = update_facilities(next_fac.second);
            if(n_open_clients_ == 0) break;
        }
    }


    IT update_facilities(IT f_id) {
        throw std::runtime_error("Not implemented");
        return 0;
    }

    FT get_fac_cost(size_t ind) const {
        if(facility_cost_.size()) {
            if(facility_cost_.size() == 1) return facility_cost_[0];
            return facility_cost_[ind];
        }
        throw std::invalid_argument("Facility cost must be set");
    }

    void cluster_results() {
        throw std::runtime_error("Not implemented");
    }

public:
    JVSolver2(): nedges_(0), ncities_(0), nfac_(0) {
    }

    template<typename Mat>
    void set(const Mat &mat) {
        w_.resize(mat.rows(), mat.columns());
        if(nedges_ < w_.rows() * w_.columns()) {
            nedges_ = w_.rows() * w_.columns();
            edges_.reset(new edgetup[nedges_]);
        }
        if(ncities_ < w_.columns()) {
            client_v_.reset(new shared::packed_pair<FT, IT>[w_.columns()]);
            clients_cpy_.reset(new std::vector<IT>[w_.columns()]);
        } else {
            for(size_t i = 0; i < w_.columns(); ++i) clients_cpy_[i].clear();
        }
        std::fill(client_v_.get(), &client_v_[w_.columns()], shared::packed_pair<FT, IT>{-1,-1});
        if(nfac_ < w_.rows()) {
            open_facilities_.reset(new std::vector<IT>[w_.rows()]);
            nfac_ = w_.rows();
            contribution_time_.reset(new FT[nfac_]());
            fac_contributions_.reset(new FT[nfac_]());
        } else {
            for(size_t i = 0; i < nfac_; ++i)
                open_facilities_[i] = {-1.};
            std::fill(contribution_time_.get(), contribution_time_.get() + nfac_, FT(0));
            std::fill(fac_contributions_.get(), fac_contributions_.get() + nfac_, FT(0));
        }
        ncities_ = w_.columns();
        nfac_ = w_.rows();
        n_open_clients_ = ncities_;
        pay_schedule_.resize(nfac_);
        for(size_t i = 0; i < nfac_; ++i) {
            pay_schedule_[i] = {get_fac_cost(i), i};
        }
        next_paid_ = pay_schedule_;
    }

    auto &run() {
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
                next_fac = next_paid_.back(), next_paid_.pop_back();
            else next_fac = {current_edge_cost + 1., 0};

            if(current_edge_cost <= next_fac.first) {
                n_open_clients_ = service_tight_edge();
                time_ = current_edge_cost;
                ++edge_idx;
            } else {
                n_open_clients_ = update_facilities(next_fac.second);
                time_ = next_fac.first;
            }
        }
        cluster_results();
        return final_open_facilities_;
    }

    template<typename VT, bool TF>
    void set_fac_cost(const blaze::Vector<VT, TF> &val) {
        if((~val).size() != w_.rows()) throw std::invalid_argument("Val has wrong number of rows");
        facility_cost_.resize((~val).size());
        facility_cost_ = (~val);
    }
    void set_fac_cost(FT val) {
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
