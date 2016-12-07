include_directories(../lib_vlfeat/)

find_package(OpenCV REQUIRED)
find_package(Threads)


#fonction slic qui fonctionne
#add_executable(SIFT_exe main.cpp)
add_executable(SIFT_exe main_sift.cpp)

#openmp
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp)

#mpi
target_link_libraries(SIFT_exe ${MPI_LIBRARIES})
target_link_libraries(SIFT_exe ${OpenCV_LIBS} vlfeat_slic pthread)



