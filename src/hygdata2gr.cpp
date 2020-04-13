#include <iostream>
#include <cstring>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <array>

struct NeighborDist: public std::pair<double, uint32_t> {
    template<typename...Args>
    NeighborDist(Args &&...args): std::pair<double, uint32_t>(std::forward<Args>(args)...) {}
};
struct xyz_t: public std::array<float, 3> {};

inline double dist(xyz_t lh, xyz_t rh) {
    double v1 = (lh[0] - rh[0]);
    double v2 = (lh[1] - rh[1]);
    double v3 = (lh[2] - rh[2]);
    return std::sqrt(v1 * v1 + v2 * v2 + v3 * v3);
}

struct pq_t: public std::priority_queue<NeighborDist> {
    auto &getc() {return this->c;}
};
struct pqrev_t: public std::priority_queue<NeighborDist, std::vector<NeighborDist>, std::greater<>> {
    auto &getc() {return this->c;}
};

int main(int argc, char **argv) {
    std::vector<xyz_t> coords;
    unsigned k = argc < 3 ? 3: std::atoi(argv[2]);
    coords.reserve(120000);
    std::ios_base::sync_with_stdio(false);
    std::ifstream ifs(argv[1]);
    for(std::string line;std::getline(ifs, line);) {
        const char *p = std::strchr(line.data(), '\t') + 1;
        xyz_t x;
        x[0] = std::atof(p);
        p = std::strchr(p + 1, '\t');
        x[1] = std::atof(p);
        p = std::strchr(p + 1, '\t');
        x[2] = std::atof(p);
        coords.push_back(x);
    }
    const size_t np = coords.size();
    size_t nedges = np * k;
    std::cout << "c Auto-generated astronomy data to graph\n";
    std::cout << "p sp " << np << ' ' << nedges << '\n';
    for(size_t i = 0; i < np; ++i) {
        pq_t pq;
        pqrev_t pqr;
        for(size_t j = 0; j < np; ++j) {
            if(i == j) continue;
            const auto d = dist(coords[i], coords[j]);
            pq.push(NeighborDist{d, j});
            if(pq.size() > k) pq.pop();
            pqr.push(NeighborDist{d, j});
            if(pqr.size() > k) pq.pop();
        }
        auto c = std::move(pq.getc());
        for(const auto item: c) {
            std::cout << "a " << i + 1 << ' ' << item.second + 1 << ' ' << item.first << '\n';
        }
    }
}
