#!/bin/bash
set -e
EX=/net/langmead-bigmem-ib.bluecrab.cluster/storage/dnb/code2/fgc/code/kzclustexp

for sample_frac in 0.1 0.175 0.25 0.375 0.5 0.75 0.9 0.99
do
sf=$(echo "((122761 * $sample_frac) + 23680) / 1097400" | bc -l)
$EX -ouniform_nyc.exp.k25.sample.UNBIASED.${sample_frac}.${sf} -p20 -k25 -T8 -K15 -K10 -K5 -t500 -i8 \
    -c3 -c5 -c6 -c9 -c10 -c15 -c18 -c20 -c25 -c27 -c50 -c54 -c75 -c81 -c100 -c125 -c162 \
    -c243 -c250 -c375 -c486 -c500 -c625 -c729 -c1250 -c1458 -c1875 -c2187 -c2500 -c3125 -c3750 -c4374 -c6561 -c13122 -c19683\
    -N5 -B-80,10,-60,80,${sf},0.025 ../data/nyc-1-16.annotated.gr  \
    &> uniform_nyc.exp.k25.frac.UNBIASED.${sample_frac}.${sf}.log
done
