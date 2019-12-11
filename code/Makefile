
INCLUDE_PATHS=. include /usr/local/Cellar/boost/1.71.0/include/
INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
CXX?=g++
CXXFLAGS+=-O3 -std=c++17 -march=native $(INCLUDE) -ftemplate-backtrace-limit=0

all: graphtest

%: src/%.cpp $(wildcard include/*.h)
	$(CXX) $(CXXFLAGS) $< -o $@
