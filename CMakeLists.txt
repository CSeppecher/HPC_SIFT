cmake_minimum_required(VERSION 2.8)


project(vlfeat_slic_example)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${MPI_INCLUDE_PATH})
add_subdirectory(vlfeat_slic_cli)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_library(vlfeat_slic lib_vlfeat/vl/host.c
lib_vlfeat/vl/random.c
lib_vlfeat/vl/generic.c
lib_vlfeat/vl/slic.c
lib_vlfeat/vl/quickshift.c
lib_vlfeat/vl/random.c
lib_vlfeat/vl/mathop.c
lib_vlfeat/vl/stringop.c
lib_vlfeat/vl/imopv.c
lib_vlfeat/vl/pgm.c
lib_vlfeat/vl/rodrigues.c
lib_vlfeat/vl/sift.c
lib_vlfeat/vl/fisher.c
lib_vlfeat/vl/gmm.c
lib_vlfeat/vl/svm.c
lib_vlfeat/vl/kmeans.c
lib_vlfeat/vl/aib.c
lib_vlfeat/vl/hog.c
lib_vlfeat/vl/lbp.c
lib_vlfeat/vl/liop.c
lib_vlfeat/vl/mser.c
lib_vlfeat/vl/vlad.c
lib_vlfeat/vl/array.c
lib_vlfeat/vl/dsift.c
lib_vlfeat/vl/covdet.c
lib_vlfeat/vl/kdtree.c
lib_vlfeat/vl/ikmeans.c
lib_vlfeat/vl/hikmeans.c
lib_vlfeat/vl/stringop.c
lib_vlfeat/vl/homkermap.c
lib_vlfeat/vl/imopv_sse2.c
#lib_vlfeat/vl/mathop_avx.c
lib_vlfeat/vl/scalespace.c
lib_vlfeat/vl/svmdataset.c
lib_vlfeat/vl/getopt_long.c
lib_vlfeat/vl/mathop_sse2.c)












