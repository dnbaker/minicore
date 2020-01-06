
INCLUDE_PATHS=. include include/fgc /usr/local/Cellar/boost/1.71.0/include/ blaze libosmium/include protozero/include
ifdef BOOST_DIR
INCLUDE_PATHS+= $(BOOST_DIR)
endif
INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
CXX?=g++
STD?=c++17
WARNINGS+=-Wall -Wextra
CXXFLAGS+=-O3 -std=$(STD) -march=native $(WARNINGS) $(INCLUDE) #-ftemplate-backtrace-limit=0

EX=$(patsubst src/%.cpp,%,$(wildcard src/*.cpp))
all: $(EX)

ifdef OMP
CXXFLAGS += -fopenmp
endif

ifdef TBBDIR
CXXFLAGS += -DUSE_TBB -L$(TBBDIR)/lib -I$(TBBDIR)/include -ltbb
endif

%: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@

%pg: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -pg -o $@

%pgf: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -pg -fopenmp -o $@

%f: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -fopenmp -o $@

osm2dimacs: src/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@

clean:
	rm -f $(EX)
