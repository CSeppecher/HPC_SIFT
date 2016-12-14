include_directories(../lib_vlfeat/)

find_package(OpenCV REQUIRED)
find_package(Threads)

find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

find_package(MPI REQUIRED)
if (MPI_FOUND)
	SET(CMAKE_C_COMPILER mpicc)
	SET(CMAKE_CXX_COMPILER mpicxx)
endif()


include_directories(MPI_INCLUDE_PATH)

#fonction slic qui fonctionne
#add_executable(SIFT_exe main.cpp)
add_executable(SIFT_exe main_sift.cpp)


# Indique que l'on installe l'exécutable dans un dossier bin
#install (TARGETS SIFT_exe DESTINATION bin)

#openmp
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp)

#mpi
target_link_libraries(SIFT_exe ${MPI_LIBRARIES})
target_link_libraries(SIFT_exe ${OpenCV_LIBS} vlfeat_slic pthread)



