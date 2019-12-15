## Fast Generic Coresets

fgc is a library for generic coresets on graphs and other metric spaces.
It includes methods for approximate, bicriteria solutions as well as sampling algorithms.


### Dependencies

1. Boost, specifically the Boost Graph Library.
2. A compiler supporting C++17. We could remove this requirement without much work.
3. We conditionally use OpenMP. This is enabled with the setting of the OMP variable.
