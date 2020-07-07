INCLUDE_PATHS=. include include/minocore blaze libosmium/include protozero/include pdqsort third_party
LIBPATHS+=

ifdef BOOST_DIR
INCLUDE_PATHS += $(BOOST_DIR)
endif

ifdef HDFPATH
INCLUDE_PATHS+= $(HDFPATH)/include
LIBPATHS+= $(HDFPATH)/lib
LINKS+=-lhdf5  -lhdf5_hl  -lhdf5_hl_cpp -lhdf5_cpp
endif


ifdef SLEEF_DIR
INCLUDE_PATHS+= $(SLEEF_DIR)/include
LIBPATHS+= $(SLEEF_DIR)/lib
LINKS+= -lsleef
CXXFLAGS+=-DBLAZE_USE_SLEEF=1
endif

ifdef CBLASFILE
DEFINES+= -DCBLASFILE='${CBLASFILE}'
endif

INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
LIBS=$(patsubst %,-L%,$(LIBPATHS))
CXX?=g++
STD?=c++17
WARNINGS+=-Wall -Wextra -Wpointer-arith -Wformat -Wunused-variable -Wno-attributes -Wno-ignored-qualifiers -Wno-unused-function \
    -Wno-deprecated-copy # Because of Boost.Fusion
OPT?=O3
LDFLAGS+=$(LIBS) -lz $(LINKS)
EXTRA?=
DEFINES+= #-DBLAZE_RANDOM_NUMBER_GENERATOR='wy::WyHash<uint64_t, 2>'
CXXFLAGS+=-$(OPT) -std=$(STD) -march=native $(WARNINGS) $(INCLUDE) $(DEFINES) $(BLAS_LINKING_FLAGS) \
    -DBOOST_NO_AUTO_PTR -fvisibility=hidden -lz


DEX=$(patsubst src/%.cpp,%dbg,$(wildcard src/*.cpp))
EX=$(patsubst src/utils/%.cpp,%,$(wildcard src/utils/*.cpp))
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

ifdef LZMA_ARCHIVE
CXXFLAGS += $(LZMA_ARCHIVE) -llzma -DHAVE_LZMA
endif

ifdef TBBDIR
INCLUDE_PATHS+= $(TBBDIR)/include
LIBPATHS+= $(TBBDIR)/lib
CXXFLAGS += -DUSE_TBB
LINKS += -ltbb
endif

TESTS=tbmdbg coreset_testdbg bztestdbg btestdbg osm2dimacsdbg dmlsearchdbg diskmattestdbg graphtestdbg jvtestdbg kmpptestdbg tbasdbg \
      jsdtestdbg jsdkmeanstestdbg jsdhashdbg fgcinctestdbg geomedtestdbg oracle_thorup_ddbg sparsepriortestdbg istestdbg msvdbg knntestdbg

tests: $(TESTS)
print_tests:
	@echo "Tests: " $(TESTS)

ifdef USESAN
CXXFLAGS += -fsanitize=address
endif
CXXFLAGS += $(EXTRA)

CXXFLAGS += $(LDFLAGS)

HEADERS=$(shell find include -name '*.h')

%dbg: src/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread -lz $(LDFLAGS)

%dbg: src/tests/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread -lz $(LDFLAGS)

%: src/tests/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread -DNDEBUG $(OMP_STR)

printlibs:
	echo $(LIBPATHS)


#graphrun: src/graphtest.cpp $(wildcard include/minocore/*.h)
#	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR)

%: src/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3

%: src/utils/%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3

mtx%: src/mtx%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 $(LDFLAGS) -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 -DNDEBUG # -fsanitize=undefined -fsanitize=address

mtx%: src/utils/mtx%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 $(LDFLAGS) -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 -DNDEBUG # -fsanitize=undefined -fsanitize=address

mtx%dbg: src/mtx%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 $(LDFLAGS) -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0  # -fsanitize=undefined -fsanitize=address

mtx%dbg: src/utils/mtx%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 $(LDFLAGS) -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0  # -fsanitize=undefined -fsanitize=address

alphaest: src/utils/alphaest.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3

dae: src/utils/alphaest.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 -DDENSESUB

jsdkmeanstest: src/tests/jsdkmeanstest.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 -lz $(LDFLAGS)

jsdkmeanstestdbg: src/tests/jsdkmeanstest.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 -lz $(LDFLAGS)


HDFLAGS=-L$(HDFPATH)/lib -I$(HDFPATH)/include -lhdf5_cpp -lhdf5 -lhdf5_hl -lhdf5_hl_cpp
hdf2dm: src/utils/hdf2dm.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 $(HDFLAGS)

mpi%: src/%.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -Ofast -DUSE_BOOST_PARALLEL=1

%pg: src/%.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< -pg -o $@

%pgf: src/%.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< -pg $(OMP_STR) -o $@

%f: src/%.cpp $(wildcard include/minocore/*.h)
	$(CXX) $(CXXFLAGS) $< $(OMP_STR) -o $@


osm2dimacsdbg: src/utils/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@

osm2dimacs: src/utils/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@ -O3 $(OMP_STR) -DNDEBUG

osm2dimacspgf: src/utils/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lbz2 -lexpat -o $@ -O3 -lbz2 -lexpat -pg -DNDEBUG $(OMP_STR)

osm2dimacspg: src/utils/osm2dimacs.cpp
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lbz2 -lexpat -o $@ -O3 -lbz2 -lexpat -pg


libsleef.a:
	+cd sleef && mkdir -p __build && cd __build && cmake .. -DBUILD_SHARED_LIBS=0 && $(MAKE) && cp lib/libsleef.a lib/libsleefdft.a ../.. && cd .. && rm -r __build



clean:
	rm -f $(EX) graphrun dmlrun $(FEX) $(PGEX) $(PGFEX) $(EX)
