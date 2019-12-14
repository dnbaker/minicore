
INCLUDE_PATHS=. include /usr/local/Cellar/boost/1.71.0/include/ blaze
INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
CXX?=g++
STD?=c++17
CXXFLAGS+=-O3 -std=$(STD) -march=native $(INCLUDE) -ftemplate-backtrace-limit=0

EX=$(patsubst src/%.cpp,%,$(wildcard src/*.cpp))
all: $(EX)

ifdef OMP
CXXFLAGS += -fopenmp
endif

%: src/%.cpp $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(EX)
