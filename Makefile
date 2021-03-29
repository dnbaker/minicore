.PHONY: all clean fall pgall pgfall

INCLUDE_PATHS=. include include/minicore blaze libosmium/include protozero/include pdqsort include/thirdparty

ifdef BOOST_DIR
INCLUDE_PATHS += $(BOOST_DIR)
endif


LIBKL?=./libkl
LIBPATHS+=$(LIBKL)

LINKS+=-lz


ifdef HDFPATH
INCLUDE_PATHS+= $(HDFPATH)/include
LIBPATHS+= $(HDFPATH)/lib
LINKS+=-lhdf5  -lhdf5_hl  -lhdf5_hl_cpp -lhdf5_cpp
endif



# Handle SLEEF

CXXFLAGS+=-DBLAZE_USE_SLEEF=1

ifdef SLEEF_DIR
INCLUDE_PATHS+= $(SLEEF_DIR)/include
LIBPATHS+=$(SLEEF_DIR)/lib
else
INCLUDE_PATHS+=sleef/build/include
INCLUDE_PATHS+=sleef/dynbuild/include
LIBPATHS+=sleef/build/lib
LIBPATHS+=sleef/dynbuild/lib
endif

LINKS+= -lsleef

ifdef TBBDIR
INCLUDE_PATHS+= $(TBBDIR)/include
LIBPATHS+= $(TBBDIR)/lib
CXXFLAGS += -DUSE_TBB
LINKS += -ltbb
endif

ifdef CBLASFILE
DEFINES+= -DCBLASFILE='${CBLASFILE}'
endif

CMAKE?=cmake

INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
LIBS=$(patsubst %,-L%,$(LIBPATHS))
CXX?=g++
CC?=gcc
STD?=c++17
WARNINGS+=-Wall -Wextra -Wpointer-arith -Wformat -Wunused-variable -Wno-attributes -Wno-ignored-qualifiers -Wno-unused-function -Wdeprecated -Wno-deprecated-declarations \
    -Wno-deprecated-copy # Because of Boost.Fusion
OPT?=O3
LDFLAGS+=$(LIBS) $(LINKS)
EXTRA?=
DEFINES+= #-DBLAZE_RANDOM_NUMBER_GENERATOR='wy::WyHash<uint64_t, 2>'
CXXFLAGS+=-$(OPT) -std=$(STD) -march=native $(WARNINGS) $(INCLUDE) $(DEFINES) $(BLAS_LINKING_FLAGS) \
    -DBOOST_NO_AUTO_PTR -lz # -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0

EX=$(patsubst src/utils/%.cpp,%,$(wildcard src/utils/*.cpp)) $(patsubst src/%.cpp,%,$(wildcard src/*.cpp))


ifneq (,$(findstring clang++,$(CXX)))
    OMP_STR:=-lomp
else
    OMP_STR:=-fopenmp#-simd
endif

ifdef LZMA_ARCHIVE
CXXFLAGS += $(LZMA_ARCHIVE) -llzma -DHAVE_LZMA
endif


TESTS=tbmdbg coreset_testdbg bztestdbg btestdbg osm2dimacsdbg dmlsearchdbg diskmattestdbg graphtestdbg jvtestdbg kmpptestdbg tbasdbg \
      jsdtestdbg jsdkmeanstestdbg jsdhashdbg fgcinctestdbg geomedtestdbg oracle_thorup_ddbg sparsepriortestdbg istestdbg msvdbg knntestdbg \
        fkmpptestdbg mergetestdbg solvetestdbg testmsrdbg testmsrcsrdbg test_centroiddbg

all: $(EX)
ex: $(EX)

LIBPATHS+=libkl

tests: $(TESTS)
print_tests:
	@echo "Tests: " $(TESTS)

ifdef USESAN
CXXFLAGS += -fsanitize=address
endif
CXXFLAGS += $(EXTRA)

CXXFLAGS += $(LDFLAGS)

HEADERS=$(shell find include -name '*.h')
STATIC_LIBS=libsleef.a

#libsimdsampling/libsimdsampling.a: libsimdsampling/simdsampling.cpp libsimdsampling/simdsampling.h libsleef.dyn.gen
#ls libsimdsampling/libsimdsampling.a 2>/dev/null || (cd libsimdsampling && $(MAKE) libsimdsampling.a INCLUDE_PATHS="../sleef/build/include" LINK_PATHS="../sleef/build/lib" && cd ..)

%: src/tests/%.o $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread -DNDEBUG $(LDFLAGS) $(OMP_STR) $(STATIC_LIBS)

src/tests/%.o: src/tests/%.cpp $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) -c $< -o $@ -pthread -DNDEBUG $(LDFLAGS) $(OMP_STR) $(STATIC_LIBS)

%.dbgo: %.cpp $(HEADERS) $(STATIC_LIBS)  libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) -c $< -o $@ -pthread  $(LDFLAGS) $(OMP_STR) $(STATIC_LIBS)  -O1
%dbg: src/tests/%.dbgo $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread  $(LDFLAGS) $(OMP_STR) $(STATIC_LIBS)  -O1

printlibs:
	echo $(LIBPATHS)


#graphrun: src/graphtest.cpp $(wildcard include/minicore/*.h)
#	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR)

%: src/%.cpp $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 $(STATIC_LIBS)

%: src/utils/%.cpp $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 $(STATIC_LIBS)

mtx%: src/mtx%.cpp $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 $(LDFLAGS)  -DNDEBUG $(STATIC_LIBS) # -fsanitize=undefined -fsanitize=address

mtx%: src/utils/mtx%.cpp $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 $(LDFLAGS) -DNDEBUG $(STATIC_LIBS) # -fsanitize=undefined -fsanitize=address

mtx%dbg: src/mtx%.cpp $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 $(LDFLAGS) $(STATIC_LIBS)  # -fsanitize=undefined -fsanitize=address

mtx%dbg: src/utils/mtx%.cpp $(HEADERS) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 $(LDFLAGS) $(STATIC_LIBS)

alphaest: src/utils/alphaest.cpp $(wildcard include/minicore/*.h) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 $(STATIC_LIBS)

dae: src/utils/alphaest.cpp $(wildcard include/minicore/*.h) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 -DDENSESUB $(STATIC_LIBS)

jsdkmeanstest: src/tests/jsdkmeanstest.cpp $(wildcard include/minicore/*.h) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 -lz $(LDFLAGS) $(STATIC_LIBS)

jsdkmeanstestdbg: src/tests/jsdkmeanstest.cpp $(wildcard include/minicore/*.h) $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ $(OMP_STR) -O3 -lz $(LDFLAGS) $(STATIC_LIBS)


HDFLAGS=-L$(HDFPATH)/lib -I$(HDFPATH)/include -lhdf5_cpp -lhdf5 -lhdf5_hl -lhdf5_hl_cpp
hdf2dm: src/utils/hdf2dm.cpp $(wildcard include/minicore/*.h) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -O3 $(HDFLAGS)

mpi%: src/%.cpp $(wildcard include/minicore/*.h) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -o $@ -DNDEBUG $(OMP_STR) -Ofast -DUSE_BOOST_PARALLEL=1

%pg: src/%.cpp $(wildcard include/minicore/*.h) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -pg -o $@

%pgf: src/%.cpp $(wildcard include/minicore/*.h) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< -pg $(OMP_STR) -o $@

%f: src/%.cpp $(wildcard include/minicore/*.h) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) $< $(OMP_STR) -o $@


osm2dimacsdbg: src/utils/osm2dimacs.cpp $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@

osm2dimacs: src/utils/osm2dimacs.cpp $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lz -lbz2 -lexpat -o $@ -O3 $(OMP_STR) -DNDEBUG

osm2dimacspgf: src/utils/osm2dimacs.cpp $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lbz2 -lexpat -o $@ -O3 -lbz2 -lexpat -pg -DNDEBUG $(OMP_STR)

osm2dimacspg: src/utils/osm2dimacs.cpp $(STATIC_LIBS) libsleef.dyn.gen
	$(CXX) $(CXXFLAGS) \
        $(OSINC) -pthread \
        $< -lbz2 -lexpat -o $@ -O3 -lbz2 -lexpat -pg


libsimdsampling/libsimdsampling.a: libsimdsampling/simdsampling.cpp libsimdsampling/simdsampling.h
	cd libsimdsampling && $(MAKE) libsimdsampling.a SLEEF_DIR=../sleef && cp libsimdsampling.a ..
libsimdsampling/libsimdsampling.so: libsimdsampling/simdsampling.cpp libsimdsampling/simdsampling.h
	cd libsimdsampling && $(MAKE) libsimdsampling.so SLEEF_DIR=../sleef && cp libsimdsampling.so ..
libkl/libkl.so: libkl/libkl.c libkl/libkl.h sleef.h
	ls $@ libkl/libkl.dylib 2>/dev/null || (cd libkl && $(MAKE) SLEEF_DIR=../sleef/build)

libkl.o: libkl/libkl.c libkl/libkl.h libsleef.a
	$(CC) $< -o $@ -c $(INCLUDE) $(WARNINGS) $(EXTRA) -std=c11 $(ND)
libkl.a: libkl.o
	$(AR) rcs $@ $<
libsimdsampling.a: libsimdsampling/libsimdsampling.a
	cp $< $@
libsimdsampling/libsimdsampling.a:
	cd libsimdsampling && $(MAKE) libsimdsampling.a

libsleef.dyn.gen:
	+(ls libsleef.so || ls libsleef.dylib) 2>/dev/null || (cd sleef && mkdir -p dynbuild && cd dynbuild && $(CMAKE) .. -DBUILD_SHARED_LIBS=1 && $(MAKE) && cp lib/* ../.. && cd ../.. && touch libsleef.dyn.gen)

sleef/sleefbuild:
	mkdir $@

libsleef.a: sleef/sleefbuild
	cd sleef/sleefbuild && rm -f CMakeCache.txt && $(CMAKE) .. -DBUILD_SHARED_LIBS=0 && $(MAKE) && cp lib/libsleef.a lib/libsleefdft.a  ../.. && cd ..
sleef/sleefbuild/include/sleef.h: libsleef.a sleef/sleefbuild
sleef.h: sleef/sleefbuild/include/sleef.h
	cp $< $@


soft: solvetestdbg solvetest solvesoft solvesoftdbg
ssoft: solvesoft solvesoftdbg
hsoft: solvetest solvetestdbg


clean:
	rm -f $(EX) graphrun dmlrun
