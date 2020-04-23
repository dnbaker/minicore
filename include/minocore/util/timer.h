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
static inline uint32_t timediff2ms(std::chrono::time_point<Clock> start, std::chrono::time_point<Clock> stop) {
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

} // util

} // minocore

#endif /*TIMER_H__ */
