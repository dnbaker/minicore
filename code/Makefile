INCLUDE_PATHS=. include include/fgc blaze libosmium/include protozero/include pdqsort
LIBPATHS+=

ifdef BOOST_DIR
INCLUDE_PATHS += $(BOOST_DIR)
endif


ifdef SLEEF_DIR
INCLUDE_PATHS+= $(SLEEF_DIR)/include
LIBPATHS+= $(SLEEF_DIR)/lib
LDFLAGS+= -lsleef
endif

INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
LIBS=$(patsubst %,-L%,$(LIBPATHS))
CXX?=g++
STD?=c++17
WARNINGS+=-Wall -Wextra -Wpointer-arith -Wformat -pedantic -Wunused-variable -Wno-attributes -Wno-ignored-qualifiers
OPT?=O3
LDFLAGS+=-lz
CXXFLAGS+=-$(OPT) -std=$(STD) -march=native $(WARNINGS) $(INCLUDE) $(LIBS) $(LDFLAGS) -flto


EX=$(patsubst src/%.cpp,%dbg,$(wildcard src/*.cpp))
FEX=$(patsubst src/%.cpp,%,$(wildcard src/*.cpp))
PGEX=$(patsubst %,%pg,$(EX))
PGFEX=$(patsubst %,%pgf,$(EX))
all: $(EX)
fall: $(FEX)
pgall: $(PGEX)
pgfall: $(PGFEX)
that:
ALL: pgall pgfall \
    all that fall

ifneq (,$(findstring clang++,$(CXX)))
    OMP_STR:=-lomp
else
    OMP_STR:=-fopenmp#-simd
endif

ifdef TBBDIR
CXXFLAGS += -DUSE_TBB -L$(TBBDIR)/lib -I$(TBBDIR)/include -ltbb
endif

TESTS=tbmdbg coreset_testdbg bztestdbg btestdbg osm2dimacsdbg dmlsearchdbg diskmattestdbg graphtestdbg jvtestdbg kmpptestdbg tbasdbg \
      jsdtestdbg jsdkmeanstestdbg

clust: kzclustexpdbg kzclustexp kzclustexpf

tests: $(TESTS)
print_tests:
	@echo "Tests: " $(TESTS)

%dbg: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ # -fsanitize=address

graphrun: src/graphtest.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR)

dmlrun: src/dmlsearch.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR)

%: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3


HDFLAGS=-L$(HDFPATH)/lib -I$(HDFPATH)/include -lhdf5_cpp -lhdf5 -lhdf5_hl -lhdf5_hl_cpp
hdf2dm: src/hdf2dm.cpp $(wildcard include/fgc/*.h)
	ls $(HDFPATH)/lib && \
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 $(HDFLAGS)

mpi%: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -Ofast -DUSE_BOOST_PARALLEL=1

%pg: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -pg -o $@

%pgf: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< -pg $(OMP_STR) -o $@

%f: src/%.cpp $(wildcard include/fgc/*.h)
	$(CXX) $(CXXFLAGS) $< $(OMP_STR) -o $@


osm2dimacsdbg: src/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@

osm2dimacs: src/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@ -O3 $(OMP_STR) -DNDEBUG

osm2dimacspgf: src/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@ -O3 -lbz2 -lexpat -pg -DNDEBUG $(OMP_STR)

osm2dimacspg: src/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@ -O3 -lbz2 -lexpat -pg


clean:
	rm -f $(EX) graphrun dmlrun $(FEX) $(PGEX) $(PGFEX) $(EX)
