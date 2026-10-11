# Install script for directory: /home/embed/gids_all/bam

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RELEASE")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/home/embed/gids_all/bam/include/bafs_ptr.h"
    "/home/embed/gids_all/bam/include/buffer.h"
    "/home/embed/gids_all/bam/include/ctrl.h"
    "/home/embed/gids_all/bam/include/event.h"
    "/home/embed/gids_all/bam/include/host_util.h"
    "/home/embed/gids_all/bam/include/nvm_admin.h"
    "/home/embed/gids_all/bam/include/nvm_aq.h"
    "/home/embed/gids_all/bam/include/nvm_cmd.h"
    "/home/embed/gids_all/bam/include/nvm_ctrl.h"
    "/home/embed/gids_all/bam/include/nvm_dma.h"
    "/home/embed/gids_all/bam/include/nvm_error.h"
    "/home/embed/gids_all/bam/include/nvm_io.h"
    "/home/embed/gids_all/bam/include/nvm_parallel_queue.h"
    "/home/embed/gids_all/bam/include/nvm_queue.h"
    "/home/embed/gids_all/bam/include/nvm_rpc.h"
    "/home/embed/gids_all/bam/include/nvm_types.h"
    "/home/embed/gids_all/bam/include/nvm_util.h"
    "/home/embed/gids_all/bam/include/page_cache copy.h"
    "/home/embed/gids_all/bam/include/page_cache.h"
    "/home/embed/gids_all/bam/include/queue.h"
    "/home/embed/gids_all/bam/include/util.h"
    "/home/embed/gids_all/bam/include/window_buffer.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/embed/gids_all/bam/build/lib/libnvm.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libnvm.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/embed/gids_all/bam/build/benchmarks/array/cmake_install.cmake")
  include("/home/embed/gids_all/bam/build/benchmarks/iodepth-block/cmake_install.cmake")
  include("/home/embed/gids_all/bam/build/benchmarks/block/cmake_install.cmake")
  include("/home/embed/gids_all/bam/build/benchmarks/block-cache/cmake_install.cmake")
  include("/home/embed/gids_all/bam/build/benchmarks/readwrite/cmake_install.cmake")
  include("/home/embed/gids_all/bam/build/benchmarks/readwrite_stripe/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/embed/gids_all/bam/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
