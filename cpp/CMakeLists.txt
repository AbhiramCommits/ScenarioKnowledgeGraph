cmake_minimum_required(VERSION 3.12)
project(skg_geom)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, trying to fetch from pip")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip show pybind11
        RESULT_VARIABLE PYBIND11_FOUND
        OUTPUT_VARIABLE PYBIND11_OUTPUT
    )
    if(PYBIND11_FOUND EQUAL 0)
        string(REGEX MATCH "Location: ([^\n]*)" _ ${PYBIND11_OUTPUT})
        set(pybind11_DIR ${CMAKE_MATCH_1}/pybind11/share/cmake/pybind11)
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found and could not be installed")
    endif()
endif()

pybind11_add_module(skg_geom geometry.cpp)
target_link_libraries(skg_geom PRIVATE m)
