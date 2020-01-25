#include <iostream>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <cinttypes>
#include <memory>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

#ifndef likely
#  if defined(__GNUC__)
#    define likely(x) __builtin_expect(!!(x), 1)
#  else
#    define likely(x) (x)
#  endif
#endif
static constexpr double mile2meter = .0254 * 5280 * 12;


void usage(char **argv) {
    std::fprintf(stderr, "Usage: %s <in.gr> <in.csv> [out.gr]\n"
                          "If out.gr is omitted, emit to stdout\n", *argv);
    std::exit(1);
}

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    if(argc < 3) usage(argv);
    std::ifstream ifs(argv[1]);
    std::ifstream icsv(argv[2]);
    std::string opath("/dev/stdout");
    if(argc > 3) opath = argv[3];

    std::unique_ptr<char[]> obuf(new char[65536]);
    std::ofstream ofs(opath);
    ofs.rdbuf()->pubsetbuf(obuf.get(), 65536);
    ofs << "c Produced by adding a connection from all nodes in in.gr (in meters)\n"
        << "c to New York city in.csv (meant for splicing the NYC osm map)\n"
        << "c into the NBER database of intercity distances (in miles).\n";
    std::string iline;
    for(;std::getline(ifs, iline) && (!iline.empty() && iline.front() != 'p'););
    if(iline.empty() || iline.front() != 'p') {
        std::cerr << iline << '\n';
        throw std::runtime_error("Ill-formatted input .gr file");
    }
    char *s = std::strchr(iline.data() + 2, ' ');
    if(!s) throw 1;
    ++s;
    double max_intracity_distance = 0.;
    std::vector<std::string> nyclines;
    std::unordered_set<uint64_t> nycids;
    std::fprintf(stderr, "Parsing header\n");
    //ofs << iline << '\n';
    do {
        if(!std::getline(ifs, iline)) throw std::runtime_error("Ill-formatted input .gr file");
    } while(iline[0] != 'a');
    do {
        if(likely(!iline.empty() && iline.front() == 'a')) {
            //std::cerr << iline << '\n';
            char *p = &(*iline.end());
            while(!std::isspace(p[-1])) --p;
            double cdist = std::atof(p);
            max_intracity_distance = std::max(cdist, max_intracity_distance);
            const char *lhptr = iline.data() + 2;
            nycids.insert(std::strtoull(iline.data() + 2, &p, 10)); // First id
            const char *rhptr = p + 1;
            nycids.insert(std::strtoull(p + 1, nullptr, 10));       // Second id
            //std::fprintf(stderr, "lh ptr: %s. rh ptr: %s. distance: %g\n", lhptr, rhptr, cdist);
            nyclines.push_back(std::move(iline));
        }
    } while(std::getline(ifs, iline));
    std::fprintf(stderr, "Parsed input gr. num lines: %zu. Num ids: %zu. max intracity distance so far: %g\n", nyclines.size(), nycids.size(), max_intracity_distance);
    const size_t newnode_nyc = nycids.size() + 1;
    const uint64_t nycmapid = (36ull << 32) | 51000u;
    for(const auto id: nycids) {
        char buf[256];
        nyclines.emplace_back(buf, std::sprintf(buf, "a %" PRIu64 " %zu %f", id, newnode_nyc, max_intracity_distance * 3));
    }
    // At this point, we have a fully connected graph for NYC, with a synthetic node that's far but connected to everything
    std::unordered_map<uint64_t, uint64_t> cityid2gidmap{{nycmapid, newnode_nyc}};
    typename std::unordered_map<uint64_t, uint64_t>::iterator lhit, rhit;
    char buf[400];
    std::unordered_set<uint64_t> connected_to_nyc;
    std::fprintf(stderr, "Started things, about to parse icsv\n");
    std::getline(icsv, iline); // skip header
    while(std::getline(icsv, iline)) {
        if(iline.empty() || iline.front() == '#') continue;
        //std::cerr << "Line in csv: " << iline << '\n';
        //iline.erase(std::remove(iline.begin(), iline.end(), '"'));
        assert(iline.data());
        char *s = std::strchr(iline.data(), ',');
        assert(s);
        //std::fprintf(stderr, "s: %s\n", s);
        //if(!s) throw 2;
        ++s;
        uint64_t lhid = (uint64_t(std::atoi(iline.data())) << 32) | std::strtoul(s, &s, 10);
        //std::fprintf(stderr, "lhid: %zu\n", size_t(lhid));
        const double distance = std::strtod(s + 1, &s) * mile2meter;
        //std::fprintf(stderr, "dist: %g\n", distance);
        uint64_t rhid = std::strtoull(s + 1, &s, 10) << 32;
        //std::fprintf(stderr, "rhid: %zu\n", size_t(rhid));
        rhid |= std::strtoul(s + 1, nullptr, 10);
        if((lhit = cityid2gidmap.find(lhid)) == cityid2gidmap.end()) {
            lhit = cityid2gidmap.emplace(lhid, cityid2gidmap.size() + newnode_nyc).first;
        }
        if((rhit = cityid2gidmap.find(rhid)) == cityid2gidmap.end()) {
            rhit = cityid2gidmap.emplace(rhid, cityid2gidmap.size() + newnode_nyc).first;
        }
        assert(rhit != cityid2gidmap.end() && lhit != cityid2gidmap.end());
        nyclines.emplace_back(buf, std::sprintf(buf, "a %" PRIu64 " %" PRIu64 " %g", lhit->second, rhit->second, distance));
        if(lhid == nycmapid)
            connected_to_nyc.insert(rhid);
        else if(rhid == nycmapid)
            connected_to_nyc.insert(lhid);
    }
    connected_to_nyc.insert(nycmapid);
    const double maxdist = 3200 * mile2meter;
    for(const auto pair: cityid2gidmap) {
        if(connected_to_nyc.find(pair.first) == connected_to_nyc.end())
             nyclines.emplace_back(buf, std::sprintf(buf, "a %" PRIu64 " %" PRIu64 " %g", newnode_nyc, pair.second, maxdist));
    }
    const size_t nv = nycids.size() + cityid2gidmap.size();
    const size_t ne = nyclines.size();
    ofs << "p sp " << nv << ' ' << ne << '\n';
    for(const auto &line: nyclines)
        ofs << line << '\n';
    // Add a connection from nyc to all cities so that the graph is connected
    // but make the distance longer than nyc to any city in the country.
}
