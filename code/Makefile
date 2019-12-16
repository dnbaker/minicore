
INCLUDE_PATHS=. include /usr/local/Cellar/boost/1.71.0/include/ blaze
INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
CXX?=g++
STD?=c++17
WARNINGS+=-Wall -Wextra
CXXFLAGS+=-O0 -std=$(STD) -march=native $(WARNINGS) $(INCLUDE) #-ftemplate-backtrace-limit=0

EX=$(patsubst src/%.cpp,%,$(wildcard src/*.cpp))
all: $(EX)

ifdef OMP
CXXFLAGS += -fopenmp
endif

OSDIR=/usr/local/Cellar/libosmium/2.15.4
# Replace this for your machine
OSINC=-I$(OSDIR)/include -I$(OSDIR)/libexec/include

ifdef TBBDIR
CXXFLAGS += -DUSE_TBB -L$(TBBDIR)/lib -I$(TBBDIR)/include -ltbb
endif

%: src/%.cpp $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@

%pg: src/%.cpp $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) $< -pg -o $@

%pgf: src/%.cpp $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) $< -pg -fopenmp -o $@

%f: src/%.cpp $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) $< -fopenmp -o $@

osm2dimacs: src/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) \
        $< -lz -lbz2 -lexpat -o $@

clean:
	rm -f $(EX)
