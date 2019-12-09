

INCLUDE=-I. -I//usr/local/Cellar/boost/1.71.0/include/ -Iinclude
CXX?=g++
CXXFLAGS+=-O3 -std=c++17 -march=native $(INCLUDE)

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
