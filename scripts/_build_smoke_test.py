"""Sanity test: can we compile + link + load a trivial DLL via setuptools' MSVC?"""
import os
import sys
import tempfile
import ctypes
from setuptools._distutils import _msvccompiler


def main():
    cc = _msvccompiler.MSVCCompiler()
    cc.initialize()
    print(f"cl.exe: {cc.cc}")

    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "hello.c")
    with open(src, "w") as f:
        f.write("__declspec(dllexport) int add(int a, int b) { return a + b; }\n")

    objs = cc.compile([src], output_dir=tmp)
    print(f"objs: {objs}")
    out_dll = os.path.join(tmp, "hello.dll")
    cc.link_shared_object(objs, out_dll)
    print(f"linked: {out_dll}")

    lib = ctypes.CDLL(out_dll)
    lib.add.restype = ctypes.c_int
    lib.add.argtypes = [ctypes.c_int, ctypes.c_int]
    print(f"add(3,4) = {lib.add(3, 4)}")


if __name__ == "__main__":
    sys.exit(main())
