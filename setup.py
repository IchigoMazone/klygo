import os
import shutil
import subprocess
import sys
import glob

import pybind11
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


class CMakeBuild(build_ext):
    def run(self):
        # Source tree (works both in-tree and from sdist temp dir)
        here = os.path.dirname(os.path.abspath(__file__))
        cpp_src_dir = os.path.join(here, "cpp")

        build_dir = os.path.join(here, "_cmake_build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=True)

        python_exe = sys.executable
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        cmake_args = [
            "cmake",
            "-B", build_dir,
            "-S", cpp_src_dir,
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
            f"-DPython3_EXECUTABLE={python_exe}",
            f"-DPYTHON_EXECUTABLE={python_exe}",
        ]
        if "CMAKE_ARGS" in os.environ:
            cmake_args.extend(os.environ["CMAKE_ARGS"].split())
        subprocess.check_call(cmake_args)
        subprocess.check_call(["cmake", "--build", build_dir])

        # Find the built .pyd / .so (search recursively — MSVC puts it in Release/ subdir)
        pyd_found = None
        for root, dirs, files in os.walk(build_dir):
            for fname in files:
                if fname.startswith("klygo.") and (fname.endswith(".pyd") or fname.endswith(".so")):
                    pyd_found = os.path.join(root, fname)
                    break
            if pyd_found:
                break

        if pyd_found is None:
            raise RuntimeError(
                f"Built extension module not found in {build_dir}. "
                "Ensure CMake successfully compiled the pybind11 extension."
            )

        # Copy into source package dir
        pkg_dir = os.path.join(here, "klygo")
        dst_src = os.path.join(pkg_dir, os.path.basename(pyd_found))
        shutil.copyfile(pyd_found, dst_src)
        print(f"[klygo] Installed native extension: {dst_src}")

        # On Windows: bundle MinGW runtime DLLs so users don't need MinGW installed
        if sys.platform == "win32":
            mingw_dll_dirs = [
                os.path.join(os.environ.get("MSYS2_ROOT", r"C:\msys64"), "ucrt64", "bin"),
                os.path.join(os.environ.get("MSYS2_ROOT", r"C:\msys64"), "mingw64", "bin"),
            ]
            mingw_dlls = [
                "libgcc_s_seh-1.dll",
                "libstdc++-6.dll",
                "libgomp-1.dll",
                "libwinpthread-1.dll",
            ]
            for dll in mingw_dlls:
                for dll_dir in mingw_dll_dirs:
                    dll_src = os.path.join(dll_dir, dll)
                    if os.path.exists(dll_src):
                        dll_dst = os.path.join(pkg_dir, dll)
                        shutil.copyfile(dll_src, dll_dst)
                        print(f"[klygo] Bundled runtime DLL: {dll}")
                        break

        # ALSO copy into build/lib so bdist_wheel bundles it into the .whl
        build_lib = self.build_lib
        if build_lib:
            dst_lib = os.path.join(build_lib, "klygo", os.path.basename(pyd_found))
            os.makedirs(os.path.dirname(dst_lib), exist_ok=True)
            shutil.copyfile(pyd_found, dst_lib)
            # Copy DLLs to build_lib too
            if sys.platform == "win32":
                for dll in mingw_dlls:
                    dll_local = os.path.join(pkg_dir, dll)
                    if os.path.exists(dll_local):
                        shutil.copyfile(dll_local, os.path.join(build_lib, "klygo", dll))
            print(f"[klygo] Installed native extension to build_lib: {dst_lib}")

        self.extensions = []


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="klygo",
    version="2.0.6",
    packages=find_packages(exclude=["cpp*", "test*", "build*", "_cmake_build*"]),
    distclass=BinaryDistribution,
    cmdclass={"build_ext": CMakeBuild},
    package_data={"klygo": ["*.pyd", "*.so", "*.dll"]},
    include_package_data=True,
    zip_safe=False,
)
