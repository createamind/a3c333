cmake_minimum_required(VERSION 3.5)
project(rpcio)
set(CMAKE_AUTOMOC ON)
include(cmake/cotire.cmake)

set(FILES_SLICE ${CMAKE_SOURCE_DIR}/slice/data.ice)
set(MODULE_SLICE_DIR ${CMAKE_SOURCE_DIR}/slice)

INCLUDE(cmake/slice.cmake)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

#message("${SOURCES_SLICE}")
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_C_FLAGS_DEBUG "-g -ggdb3 -O0 -fPIC")
set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -O3 -march=native -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -ggdb3 -O0 -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -march=native -fPIC")

set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")

set(PYTHON_DIR $ENV{HOME}/anaconda3)
set(PYTHON_INCLUDE_DIR ${PYTHON_DIR}/include/python3.6m)
set(PYTHON_LIB python3.6m)
set(PYTHON_SITE_DIR ${PYTHON_DIR}/lib/python3.6/site-packages)

set(QT_INCLUDE_DIR ${PYTHON_DIR}/include/qt)
set(QT_LIB_DIR ${PYTHON_DIR}/lib)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE "Debug")
        #        set(CMAKE_BUILD_TYPE "Release")
    endif()
    add_definitions(-DLINUX)
    # to fit for the offical tensorflow build, also need rebuild the atari-py module
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
#    set(IPP_LIBRARY
#            /opt/intel/ipp/lib/intel64/libippi.a
#            /opt/intel/ipp/lib/intel64/libipps.a
#            /opt/intel/ipp/lib/intel64/libippcore.a
#            )
#    LINK_DIRECTORIES(/lib/x86_64-linux-gnu)
    link_libraries(:_pywrap_tensorflow_internal.so)
    link_libraries(bsd crypt
#            IceUtil++11
            Ice
            /lib/x86_64-linux-gnu/libcrypto.so.1.0.0
            )
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-DOSX)
    add_definitions(-DMACOSX)

#    set(IPP_LIBRARY
#            /opt/intel/ipp/lib/libippi.a
#            /opt/intel/ipp/lib/libipps.a
#            /opt/intel/ipp/lib/libippcore.a
#            )
    link_libraries(
#            IceUtil
            Ice
    )
    link_libraries(${PYTHON_SITE_DIR}/tensorflow/python/_pywrap_tensorflow_internal.so)
    #    link_directories(${PYTHON_SITE_DIR}/tensorflow/python)
endif()
#add_definitions(-DICE_CPP11_MAPPING)
add_definitions(-DLIBCLI_EXPORTS -DLIBCOMMON_EXPORTS -DMSPDEBUG_EXPORTS)

link_libraries(
        ${PYTHON_LIB}
#        ${IPP_LIBRARY}
        Qt5Core
        pthread
)



INCLUDE_DIRECTORIES(
        .
        include
        src
        ${QT_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIR}
        ${PYTHON_SITE_DIR}/numpy/core/include
        ${PYTHON_SITE_DIR}/tensorflow/include
#        /opt/intel/ipp/include
        src/msp/libmspdebug/include
        src/msp/libcommon/include
        src/msp/libcli/include
        src/msp/msp-cli-lib/src
        src/msp/msp-cli-lib/cli-frameworks/dparrish-libcli-a61be59
        src/msp/libcli/src
)

LINK_DIRECTORIES(
        ${QT_LIB_DIR}
        ${PYTHON_DIR}/lib
        ${PYTHON_SITE_DIR}/tensorflow/core
        ${PYTHON_SITE_DIR}/tensorflow/python
#        /opt/intel/ipp/lib
)
file(GLOB_RECURSE LIBCOMMON_SOURCE src/msp/libcommon/*.cpp src/msp/msp-cli-lib/*.cpp)
file(GLOB_RECURSE LIBCLI_SOURCE src/msp/libcli/*.cpp)
file(GLOB_RECURSE LIBMSPDEBUG_SOURCE src/msp/libmspdebug/*.cpp)

set(SOURCE_FILES
        src/utils/liblog.cpp
        src/utils/mini_log.cpp
        slice/data.cpp
        slice/data.h
        ${SOURCES_SLICE}
        src/dataio_recv.cc
        src/dataio_send.cc
        src/stdafx.h
        src/stdafx.cpp
        src/utils/recycle_queue.cpp
        src/utils/recycle_queue.h
        src/utils/utils_tensor.h
        src/utils/utils_tensor.cpp
        src/utils/tf_tensor_utils.cpp
        src/utils/tf_tensor_utils.h
        src/utils/utils_numpy.cpp
        src/utils/utils_numpy.h
        src/utils/utils_stack_trace.cpp
        src/utils/utils_stack_trace.h
        src/utils/utils_std.cpp
        src/utils/utils_std.h
        ${LIBCOMMON_SOURCE}
        ${LIBCLI_SOURCE}
        ${LIBMSPDEBUG_SOURCE}
        include/common_def.h src/dataio_cli.cpp src/dataio.cpp src/dataio.h src/base_op.cpp src/base_op.h src/cdataio_ice.cpp src/cdataio.h src/cdataio.cpp src/streamio.cpp src/streamio.h)

add_custom_target(GEN_SLICE_SOURCE ${SOURCES_SLICE})
add_library(rpcio_ops SHARED ${SOURCE_FILES})
set_target_properties(rpcio_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#if (APPLE)
#    set_target_properties(rpcio_ops PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "src/stdafx.h")
#    set_target_properties(rpcio_ops PROPERTIES COTIRE_ADD_UNITY_BUILD FALSE)
#    cotire(rpcio_ops GEN_SLICE_SOURCE)
#endif()


set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_SOURCE_DIR})
SET_SOURCE_FILES_PROPERTIES(cdataio.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(cmodule.i PROPERTIES SWIG_FLAGS "-E")
#SET_SOURCE_FILES_PROPERTIES(cmodule.i PROPERTIES SWIG_FLAGS "-E -builtin")
SWIG_ADD_MODULE(cdataio python cdataio.i)
set_target_properties(_cdataio PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SWIG_OUTDIR})
SWIG_LINK_LIBRARIES(cdataio python2.7 rpcio_ops)
