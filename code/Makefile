

INCLUDE=-I. -I//usr/local/Cellar/boost/1.71.0/include/ -Iinclude
CXX?=g++
CXXFLAGS+=-O3 -std=c++17

%: src/%.cpp
	$(CXX)	-O3	$(INCLUDE)	$<	-o	$@
