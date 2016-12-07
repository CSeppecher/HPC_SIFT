include_directories(../lib_vlfeat/)

find_package(OpenCV REQUIRED)
find_package(Threads)
#fonction slic qui fonctionne
#add_executable(vlfeat_slic_example main.cpp)
add_executable(vlfeat_slic_example main_sift.cpp)
target_link_libraries(vlfeat_slic_example ${OpenCV_LIBS} vlfeat_slic pthread)
