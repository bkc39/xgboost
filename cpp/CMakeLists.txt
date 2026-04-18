cmake_minimum_required(VERSION 3.20)
project(xgbcompat VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(xgboost CONFIG REQUIRED)

add_library(xgbcompat SHARED
  src/xgbcompat.cpp
  src/examples.cpp
)
add_library(xgbcompat::xgbcompat ALIAS xgbcompat)

target_include_directories(xgbcompat
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(xgbcompat
  PUBLIC
    xgboost::xgboost
)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)

  add_executable(xgbcompat_test
    tests/xgbcompat_test.cpp
  )

  target_link_libraries(xgbcompat_test
    PRIVATE
      xgbcompat
      GTest::gtest_main
  )

  gtest_discover_tests(xgbcompat_test)
endif()

install(TARGETS xgbcompat
  EXPORT xgbcompatTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT xgbcompatTargets
  FILE xgbcompatTargets.cmake
  NAMESPACE xgbcompat::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xgbcompat
)
