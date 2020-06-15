#ifndef TIMER_H__
#define TIMER_H__
#include <chrono>
#include <cstdint>
#include <iostream>

namespace minocore {
using std::uint32_t;

namespace util {
using hrc = std::chrono::high_resolution_clock;

template<typename Clock>
static inline double timediff2ms(std::chrono::time_point<Clock> start, std::chrono::time_point<Clock> stop) {
    if(stop < start) std::swap(stop, start);
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

struct Timer {

    std::string msg_;
    std::chrono::time_point<hrc> start_, stop_;

    static auto now() {return hrc::now();}

    Timer(std::string msg=std::string()): msg_(msg), start_(now()) {}

    void report() {
        stop_ = now();
        display();
    }
    void restart(std::string msg) {if(!msg.empty()) msg_ = msg; reset(); start();}
    void display() const {
        if(start_.time_since_epoch().count())
            std::cerr << "[---Timer---] " << msg_ << " " << timediff2ms(start_, stop_) << "ms\n";
    }
    void reset() {
        start_ = stop_ =  std::chrono::time_point<hrc>();
    }
    void start() {
        if(start_.time_since_epoch().count()) {
            stop_ = now();
            display();
        }
        start_ = now();
    }

    uint64_t diff() const {return timediff2ms(start_, stop_);}

    void stop() {
        stop_ = now();
    }

    ~Timer() {
        stop();
        display();
    }
};

struct TimeStamper {
    using value_type = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using v_t = value_type;
    struct Event {
        std::string label;
        v_t time;
        Event(std::string l, v_t t): label(l), time(t) {}
    };
    static auto now() {return std::chrono::high_resolution_clock::now();}
    std::vector<Event> events;
    bool emit_on_close_ = false;
    TimeStamper(std::string msg, bool emit_on_close=true): events({Event{msg, now()}}), emit_on_close_(emit_on_close) {}
    TimeStamper() {}
    void restart(std::string label) {
        events = {Event{label, now()}};
    }
    void add_event(std::string label) {
        events.emplace_back(label, now());
    }
    ~TimeStamper() {
        if(emit_on_close_) {
            emit();
        }
    }
    std::vector<std::pair<std::string, double>> to_intervals() const {
        auto t = now();
        std::vector<std::pair<std::string, double>> ret(events.size());
        for(size_t i = 0; i < events.size(); ++i) {
            auto nt = i == events.size() - 1 ? t: events[i + 1].time;
            ret[i] = {events[i].label, timediff2ms(events[i].time, nt)};
        }
        return ret;
    }
    void emit() const {
        auto ivls = to_intervals();
        auto total_time = std::accumulate(ivls.begin(), ivls.end(), 0., [](auto x, const auto &y) {return x + y.second;});
        auto prod = 100. / total_time;
        for(const auto &ivl: ivls) {
            std::fprintf(stderr, "___Event '%s' took %gms___%%%0.12g of total___\n", ivl.first.data(), ivl.second, total_time);
        }
        std::vector<unsigned> idx(ivls.size());
        std::iota(idx.data(), idx.data() + ivls.size(), 0);
        std::sort(idx.begin(), idx.end(), [&](auto x, auto y) {return ivls[x].second > ivls[y].second;});
        for(size_t i = 0; i < ivls.size(); ++i) {
            std::fprintf(stderr, "%zu{st/th/nd} most expensive has id %u with %%%0.12g of total time\n",
                         i + 1, idx[i], ivls[idx[i]].second * prod);
        }
    }
};

} // util

} // minocore

#endif /*TIMER_H__ */
