
INCLUDE_PATHS=. include /usr/local/Cellar/boost/1.71.0/include/ blaze
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

%: src/%.cpp $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(EX)
