""" some utility for call C++ code"""

from __future__ import absolute_import

import os
import ctypes
import platform
import multiprocessing


def _load_lib(env_name):
    """ Load library in build/lib. """
    cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(cur_path, "./build/")
    if platform.system() == 'Darwin':
        path_to_so_file = os.path.join(lib_path, f"lib{env_name}Env.dylib")
        lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    elif platform.system() == 'Linux':
        path_to_so_file = os.path.join(lib_path, f"lib{env_name}Env.so")
        lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    elif platform.system() == 'Windows':
        path_to_so_file = os.path.join(lib_path, f"lib{env_name}Env.dll")
        # On Windows, add the directory to DLL search path and load without RTLD_GLOBAL
        # Add the build directory to PATH for dependency resolution
        if hasattr(os, 'add_dll_directory'):
            # Python 3.8+ on Windows
            os.add_dll_directory(lib_path)
            # Also add MinGW/MSYS2 bin directory for runtime libraries
            mingw_paths = [
                r'C:\msys64\mingw64\bin',
                r'C:\msys64\ucrt64\bin',
                r'C:\mingw64\bin',
                r'C:\MinGW\bin'
            ]
            for mingw_path in mingw_paths:
                if os.path.exists(mingw_path):
                    os.add_dll_directory(mingw_path)
                    break
        else:
            # Fallback: add to PATH
            os.environ['PATH'] = lib_path + os.pathsep + os.environ.get('PATH', '')
        lib = ctypes.CDLL(path_to_so_file)
    else:
        raise BaseException("unsupported system: " + platform.system())
    return lib


def as_double_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def as_int32_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def as_bool_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))


if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count() // 2)

