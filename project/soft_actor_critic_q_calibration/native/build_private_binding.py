#!/usr/bin/env python3
"""Build a private copy of the Drive binding with a read-only diagnostics API."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import sysconfig

import numpy
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


NATIVE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = NATIVE_DIR.parent
REPOSITORY_ROOT = CALIBRATION_DIR.parents[1]
DRIVE_DIR = REPOSITORY_ROOT / "pufferlib" / "ocean" / "drive"
OCEAN_DIR = REPOSITORY_ROOT / "pufferlib" / "ocean"
EXTERNAL_DIR = REPOSITORY_ROOT / "extern"
RAYLIB_DIR = EXTERNAL_DIR / "raylib-5.5_linux_amd64"
INIH_DIR = EXTERNAL_DIR / "inih-r62"
GENERATED_BINDING = NATIVE_DIR / "diagnostic_binding.c"
GENERATED_ENV_BINDING = NATIVE_DIR / "diagnostic_env_binding.h"
BUILD_STAMP = NATIVE_DIR / ".diagnostic_binding.sha256"


def _generated_source() -> str:
    source = (DRIVE_DIR / "binding.c").read_text()
    markers = (
        "#define MY_METHODS \\\n",
        '#include "../env_binding.h"\n',
    )
    if any(source.count(marker) != 1 for marker in markers):
        raise RuntimeError("Drive binding changed; inspect and update the private-binding generator markers")

    source = source.replace(
        "// clang-format off\n#define MY_METHODS \\\n",
        "static PyObject *vec_get_diagnostic(PyObject *self, PyObject *args);\n\n"
        "// clang-format off\n#define MY_METHODS \\\n"
        '    {"vec_get_diagnostic", vec_get_diagnostic, METH_VARARGS, "Read pre-action diagnostic arrays."}, \\\n',
        1,
    )
    source = source.replace(
        '#include "../env_binding.h"\n',
        '#include "diagnostic_env_binding.h"\n\n#include "diagnostic_export.h"\n',
        1,
    )
    return source


def _generated_env_binding() -> str:
    source = (OCEAN_DIR / "env_binding.h").read_text()
    module_marker = 'static PyModuleDef module = {PyModuleDef_HEAD_INIT, "binding", NULL, -1, methods};'
    init_marker = "PyMODINIT_FUNC PyInit_binding(void) {"
    if source.count(module_marker) != 1 or source.count(init_marker) != 1:
        raise RuntimeError("PufferLib env binding changed; inspect private module-name substitutions")
    source = source.replace(
        module_marker,
        'static PyModuleDef module = {PyModuleDef_HEAD_INIT, "diagnostic_binding", NULL, -1, methods};',
        1,
    )
    source = source.replace(init_marker, "PyMODINIT_FUNC PyInit_diagnostic_binding(void) {", 1)
    return source


def _input_digest(generated_source: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(generated_source)
    digest.update(GENERATED_ENV_BINDING.read_bytes())
    tracked_paths = [
        NATIVE_DIR / "build_private_binding.py",
        NATIVE_DIR / "diagnostic_export.h",
        OCEAN_DIR / "env_binding.h",
        INIH_DIR / "ini.h",
        INIH_DIR / "ini.c",
    ]
    tracked_paths.extend(sorted(DRIVE_DIR.glob("*.h")))
    for path in tracked_paths:
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _extension_path() -> Path:
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python did not report an extension-module suffix")
    return NATIVE_DIR / f"diagnostic_binding{extension_suffix}"


def build_private_binding(force_rebuild: bool = False) -> Path:
    """Build the private extension in this folder and return its shared-object path."""
    generated_source = _generated_source().encode("utf-8")
    generated_env_binding = _generated_env_binding().encode("utf-8")
    if GENERATED_BINDING.exists() and GENERATED_BINDING.read_bytes() != generated_source:
        GENERATED_BINDING.write_bytes(generated_source)
    elif not GENERATED_BINDING.exists():
        GENERATED_BINDING.write_bytes(generated_source)
    if GENERATED_ENV_BINDING.exists() and GENERATED_ENV_BINDING.read_bytes() != generated_env_binding:
        GENERATED_ENV_BINDING.write_bytes(generated_env_binding)
    elif not GENERATED_ENV_BINDING.exists():
        GENERATED_ENV_BINDING.write_bytes(generated_env_binding)

    expected_digest = _input_digest(generated_source)
    extension_path = _extension_path()
    if (
        not force_rebuild
        and extension_path.is_file()
        and BUILD_STAMP.is_file()
        and BUILD_STAMP.read_text().strip() == expected_digest
    ):
        return extension_path

    raylib_archive = RAYLIB_DIR / "lib" / "libraylib.a"
    ini_source = INIH_DIR / "ini.c"
    required_paths = (raylib_archive, RAYLIB_DIR / "include" / "raylib.h", ini_source, INIH_DIR / "ini.h")
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Private diagnostic build needs existing local dependencies: " + ", ".join(missing_paths))

    extra_compile_args = [
        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
        "-DPLATFORM_DESKTOP",
        "-O2",
        "-flto",
        "-Wno-alloc-size-larger-than",
        "-Wno-implicit-function-declaration",
        "-fmax-errors=3",
        "-ffp-contract=off",
        "-D_GNU_SOURCE",
        '-DINI_START_COMMENT_PREFIXES="#"',
        '-DINI_INLINE_COMMENT_PREFIXES="#"',
    ]
    extra_link_args = ["-fwrapv", "-O2", "-Bsymbolic-functions"]
    if Path("/usr/include/EGL/egl.h").is_file():
        extra_link_args.extend(["-lEGL", "-lGL", "-ldl"])

    extension = Extension(
        "diagnostic_binding",
        sources=[str(GENERATED_BINDING), str(ini_source)],
        include_dirs=[
            numpy.get_include(),
            str(RAYLIB_DIR / "include"),
            str(INIH_DIR),
            str(DRIVE_DIR),
            str(OCEAN_DIR),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        extra_objects=[str(raylib_archive)],
    )
    distribution = Distribution({"ext_modules": [extension]})
    command = build_ext(distribution)
    command.build_lib = str(NATIVE_DIR)
    command.build_temp = str(NATIVE_DIR / "build" / "temp")
    command.force = True
    command.ensure_finalized()
    command.run()
    if not extension_path.is_file():
        raise RuntimeError(f"Private diagnostic build did not create {extension_path}")
    BUILD_STAMP.write_text(expected_digest + "\n")
    return extension_path


if __name__ == "__main__":
    print(build_private_binding(force_rebuild="--force" in sys.argv[1:]))
