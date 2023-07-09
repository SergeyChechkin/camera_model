cmake_minimum_required(VERSION 3.8)
project(camera_model_lib)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})

set(SOURCES
    src/CameraCalibration.cpp
)

set(HEADERS
)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${EIGEN3_LIBS} Ceres::ceres)
target_link_libraries(${PROJECT_NAME} utils_lib)

install(
  DIRECTORY include/camera_model
  DESTINATION include
)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

add_subdirectory(test)