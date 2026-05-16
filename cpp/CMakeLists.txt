cmake_minimum_required(VERSION 3.20)
project(xgbcompat VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/XgbcompatWarnings.cmake)
include(cmake/XgbcompatTools.cmake)

find_package(xgboost CONFIG REQUIRED)

add_library(xgbcompat SHARED)
add_library(xgbcompat::xgbcompat ALIAS xgbcompat)

target_sources(xgbcompat
  PRIVATE
    src/xgbcompat/booster/attributes.cpp
    src/xgbcompat/booster/dump.cpp
    src/xgbcompat/booster/evaluation.cpp
    src/xgbcompat/booster/feature_info.cpp
    src/xgbcompat/booster/lifecycle.cpp
    src/xgbcompat/booster/prediction.cpp
    src/xgbcompat/booster/serialization.cpp
    src/xgbcompat/booster/training.cpp
    src/xgbcompat/detail/buffer.cpp
    src/xgbcompat/detail/error.cpp
    src/xgbcompat/dmatrix/create.cpp
    src/xgbcompat/dmatrix/info.cpp
    src/xgbcompat/dmatrix/io.cpp
    src/xgbcompat/dmatrix/query.cpp
    src/xgbcompat/global.cpp
)

target_include_directories(xgbcompat
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(xgbcompat
  PUBLIC
    xgboost::xgboost
)

xgbcompat_enable_warnings(xgbcompat)
xgbcompat_apply_nix_cflags(xgbcompat)
xgbcompat_add_format_target(format-check)
xgbcompat_add_tidy_target(tidy xgbcompat)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)

  add_library(xgbcompat_c_api_compile_test OBJECT
    tests/xgbcompat/c_api_compile_test.c
  )
  target_include_directories(xgbcompat_c_api_compile_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  xgbcompat_apply_nix_cflags(xgbcompat_c_api_compile_test)

  add_executable(xgbcompat_tests
    tests/xgbcompat/booster/attributes_test.cpp
    tests/xgbcompat/booster/evaluation_test.cpp
    tests/xgbcompat/booster/feature_info_test.cpp
    tests/xgbcompat/booster/lifecycle_test.cpp
    tests/xgbcompat/booster/prediction_test.cpp
    tests/xgbcompat/booster/serialization_test.cpp
    tests/xgbcompat/booster/training_test.cpp
    tests/xgbcompat/dmatrix/create_test.cpp
    tests/xgbcompat/dmatrix/info_test.cpp
    tests/xgbcompat/dmatrix/io_test.cpp
    tests/xgbcompat/dmatrix/query_test.cpp
    tests/xgbcompat/global_test.cpp
    $<TARGET_OBJECTS:xgbcompat_c_api_compile_test>
  )

  target_include_directories(xgbcompat_tests
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tests
  )

  target_link_libraries(xgbcompat_tests
    PRIVATE
      xgbcompat
      GTest::gtest_main
  )

  xgbcompat_enable_warnings(xgbcompat_tests)
  xgbcompat_apply_nix_cflags(xgbcompat_tests)
  gtest_discover_tests(xgbcompat_tests)
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
