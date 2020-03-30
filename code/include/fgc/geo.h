#ifndef FGC_GEO_H__
#define FGC_GEO_H__
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <string>

namespace fgc {
struct latlon_t: public std::pair<double, double> {
    using super = std::pair<double, double>;
    template<typename...Args>
    latlon_t(Args &&...args): super(std::forward<Args>(args)...) {}
    template<typename T>
    latlon_t &operator=(const T &x) {
        return super::operator=(x);
    }
    std::string to_string() const {return std::to_string(lon()) + ',' + std::to_string(lat());}

    double lat() const {return this->first;}
    double lon() const {return this->second;}
    double &lat() {return this->first;}
    double &lon() {return this->second;}
};


struct BoundingBoxData;
static inline BoundingBoxData parse_bbdata(const char *s);
struct BoundingBoxData {
    double latlo = 0., lathi = 0., lonlo = 0., lonhi = 0.;
    double p_box = 0.99, p_nobox = 0.01;
    BoundingBoxData &operator=(const char *s) {
        assert(s);
        return *this = parse_bbdata(s);
    }
    bool set() const {return latlo || lathi || lonlo || lonhi;}
    bool valid() const {
        return lathi >= latlo && lonhi >= lonlo
            && p_box <= 1. && p_box >= 0.
            && p_nobox <= 1. && p_nobox >= 0.;
    }
    std::string to_string() const {
        char buf[256];
        return std::string(buf, std::sprintf(buf, "lat (%0.12g->%0.12g), lon (%0.12g->%0.12g), probabilities: %0.12g/%0.12g", latlo, lathi, lonlo, lonhi, p_box, p_nobox));
    }
    void print(std::FILE *fp=stderr) const {
        const std::string str = to_string();
        if(std::fwrite(str.data(), 1, str.size(), fp) != str.size())
            throw std::runtime_error("Failed to write to file");
    }
    bool contains(latlon_t pt) const {
        return (pt.lat() <= lathi && pt.lat() >= latlo)
            && (pt.lon() <= lonhi && pt.lon() >= lonlo);
    }
    static BoundingBoxData parse_bbdata(const char *s) {
        /*
         * lon,lat,lon,lat
         * %f,%f,%f,%f[,%f][,%f]
         */
        std::fprintf(stderr, "parsing %s\n", s);
        double llon, llat, ulon, ulat, highprob = 0.99, loprob=0.01;
        llon = std::strtod(s, const_cast<char **>(&s));
        llat = std::strtod(++s, const_cast<char **>(&s));
        ulon = std::strtod(++s, const_cast<char **>(&s));
        ulat = std::strtod(++s, const_cast<char **>(&s));
    
        if(*s == ',') {
            highprob = std::strtod(++s, const_cast<char **>(&s));
            if(*s == ',') {
                loprob = std::strtod(++s, const_cast<char **>(&s));
            }
        }
        assert(loprob < highprob);
        BoundingBoxData ret{llat, ulat, llon, ulon, highprob, loprob};
        ret.print(stderr);
        std::fputc('\n', stderr);
        return ret;
    }
};

} // namespace fgc

#endif /* FGC_GEO_H__ */
