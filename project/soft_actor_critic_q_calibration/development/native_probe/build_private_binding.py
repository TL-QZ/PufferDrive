"""Build an isolated Drive binding with boundary holding and probe exports."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import sysconfig

import numpy
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


NATIVE_DIR = Path(__file__).resolve().parent
BUILD_DIR = NATIVE_DIR / "_build"
CALIBRATION_DIR = NATIVE_DIR.parents[1]
REPOSITORY_ROOT = CALIBRATION_DIR.parents[1]
DRIVE_DIR = REPOSITORY_ROOT / "pufferlib" / "ocean" / "drive"
OCEAN_DIR = REPOSITORY_ROOT / "pufferlib" / "ocean"
EXTERNAL_DIR = REPOSITORY_ROOT / "extern"
RAYLIB_DIR = EXTERNAL_DIR / "raylib-5.5_linux_amd64"
INIH_DIR = EXTERNAL_DIR / "inih-r62"
GENERATED_BINDING = BUILD_DIR / "probe_binding.c"
GENERATED_ENV_BINDING = BUILD_DIR / "probe_env_binding.h"
GENERATED_DRIVE_HEADER = BUILD_DIR / "drive.h"
GENERATED_EXPORT_HEADER = BUILD_DIR / "probe_export.h"
GENERATED_HORIZON_HEADER = BUILD_DIR / "probe_horizon.h"
BUILD_STAMP = BUILD_DIR / ".probe_binding.sha256"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one {label} marker; inspect Drive source before updating the probe builder")
    return source.replace(old, new, 1)


def _generated_drive_header() -> str:
    source = (DRIVE_DIR / "drive.h").read_text()
    source = _replace_once(
        source,
        "    int eval_episode_done;\n",
        "    int eval_episode_done;\n    int probe_hold_boundary;\n    int probe_at_fault_enabled;\n"
        "    int probe_horizon_started;\n",
        "probe flags",
    )
    source = _replace_once(
        source,
        "if (env->compute_eval_metrics && is_at_fault_collision(env, agent_idx, car_collided_with_index))",
        "if ((env->compute_eval_metrics || env->probe_at_fault_enabled)\n"
        "            && is_at_fault_collision(env, agent_idx, car_collided_with_index))",
        "at-fault collision gate",
    )
    source = _replace_once(
        source,
        "        agent->metrics_array[AT_FAULT_COLLISION_IDX] = 0.0f;\n",
        "        if (!env->probe_at_fault_enabled) {\n"
        "            agent->metrics_array[AT_FAULT_COLLISION_IDX] = 0.0f;\n"
        "        }\n",
        "at-fault metric preservation",
    )
    source = _replace_once(
        source,
        "    if (env->eval_mode && env->eval_episode_done) {\n",
        "    if ((env->eval_mode || env->probe_hold_boundary) && env->eval_episode_done) {\n",
        "held-boundary step guard",
    )
    source = _replace_once(
        source,
        "        if (env->eval_mode) {\n            env->eval_episode_done = 1;\n",
        "        if (env->eval_mode || env->probe_hold_boundary) {\n            env->eval_episode_done = 1;\n",
        "held-boundary return",
    )
    return source


def _generated_binding() -> str:
    source = (DRIVE_DIR / "binding.c").read_text()
    source = _replace_once(
        source,
        "static void my_reseed(Env *env, uint64_t seed) {\n    env->init_seed = seed;\n",
        "static void my_reseed(Env *env, uint64_t seed) {\n"
        "    env->init_seed = seed;\n"
        "    if (env->probe_hold_boundary) {\n"
        "        env->eval_episode_done = 0;\n"
        "    }\n",
        "private held-scene reseed reset",
    )
    methods_marker = (
        "// clang-format off\n"
        "#define MY_METHODS \\\n"
        '    {"map_cache_size", map_cache_size_py, METH_NOARGS, "Map cache slot count."}, \\\n'
        '    {"map_cache_live_count", map_cache_live_count_py, METH_NOARGS, "Map cache live count."}\n'
        "// clang-format on"
    )
    replacement = (
        "static PyObject *vec_set_probe_mode(PyObject *self, PyObject *args);\n"
        "static PyObject *vec_get_probe_metadata(PyObject *self, PyObject *args);\n"
        "static PyObject *vec_get_probe_fingerprints(PyObject *self, PyObject *args);\n\n"
        "static PyObject *vec_start_probe_horizon(PyObject *self, PyObject *args);\n\n"
        "static PyObject *vec_begin_probe_metrics(PyObject *self, PyObject *args);\n"
        "static PyObject *vec_get_probe_metrics(PyObject *self, PyObject *args);\n\n"
        "// clang-format off\n"
        "#define MY_METHODS \\\n"
        '    {"map_cache_size", map_cache_size_py, METH_NOARGS, "Map cache slot count."}, \\\n'
        '    {"map_cache_live_count", map_cache_live_count_py, METH_NOARGS, "Map cache live count."}, \\\n'
        '    {"vec_set_probe_mode", vec_set_probe_mode, METH_VARARGS, "Enable deterministic probe boundaries."}, \\\n'
        '    {"vec_get_probe_metadata", vec_get_probe_metadata, METH_VARARGS, "Read compact probe metadata."}, \\\n'
        '    {"vec_get_probe_fingerprints", vec_get_probe_fingerprints, METH_VARARGS, "Hash meaningful native state."}, \\\n'
        '    {"vec_start_probe_horizon", vec_start_probe_horizon, METH_VARARGS, "Extend a verified probe scene horizon."}, \\\n'
        '    {"vec_begin_probe_metrics", vec_begin_probe_metrics, METH_VARARGS, "Snapshot per-scene ego metrics."}, \\\n'
        '    {"vec_get_probe_metrics", vec_get_probe_metrics, METH_VARARGS, "Read per-scene interval metrics."}\n'
        "// clang-format on"
    )
    source = _replace_once(source, methods_marker, replacement, "binding method table")
    source = _replace_once(
        source,
        '#include "../env_binding.h"\n',
        '#include "probe_env_binding.h"\n\n#include "probe_export.h"\n#include "probe_horizon.h"\n',
        "private probe exports include",
    )
    return source


def _generated_env_binding() -> str:
    source = (OCEAN_DIR / "env_binding.h").read_text()
    source = _replace_once(
        source,
        'static PyModuleDef module = {PyModuleDef_HEAD_INIT, "binding", NULL, -1, methods};',
        'static PyModuleDef module = {PyModuleDef_HEAD_INIT, "probe_binding", NULL, -1, methods};',
        "private module name",
    )
    source = _replace_once(
        source,
        "PyMODINIT_FUNC PyInit_binding(void) {",
        "PyMODINIT_FUNC PyInit_probe_binding(void) {",
        "private module initializer",
    )
    return source


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    generated_binding = _generated_binding().encode("utf-8")
    generated_env = _generated_env_binding().encode("utf-8")
    generated_drive = _generated_drive_header().encode("utf-8")
    export = (NATIVE_DIR / "probe_export.h").read_bytes()
    horizon = (NATIVE_DIR / "probe_horizon.h").read_bytes()
    for name, content in (
        ("probe_binding.c", generated_binding),
        ("probe_env_binding.h", generated_env),
        ("drive.h", generated_drive),
        ("probe_export.h", export),
        ("probe_horizon.h", horizon),
        ("build_private_binding.py", Path(__file__).read_bytes()),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(content)
    for path in (OCEAN_DIR / "env_binding.h", INIH_DIR / "ini.h", INIH_DIR / "ini.c"):
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    for path in sorted(DRIVE_DIR.glob("*.h")):
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update((DRIVE_DIR / "binding.c").read_bytes())
    return digest.hexdigest()


def _extension_path() -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError("Python did not report an extension-module suffix")
    return BUILD_DIR / f"probe_binding{suffix}"


def build_private_binding(force_rebuild: bool = False) -> Path:
    """Generate and compile the private extension under ``native_probe/_build``."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    generated = (
        (GENERATED_BINDING, _generated_binding().encode("utf-8")),
        (GENERATED_ENV_BINDING, _generated_env_binding().encode("utf-8")),
        (GENERATED_DRIVE_HEADER, _generated_drive_header().encode("utf-8")),
        (GENERATED_EXPORT_HEADER, (NATIVE_DIR / "probe_export.h").read_bytes()),
        (GENERATED_HORIZON_HEADER, (NATIVE_DIR / "probe_horizon.h").read_bytes()),
    )
    for path, contents in generated:
        if not path.is_file() or path.read_bytes() != contents:
            path.write_bytes(contents)

    fingerprint = source_fingerprint()
    extension_path = _extension_path()
    if (
        not force_rebuild
        and extension_path.is_file()
        and BUILD_STAMP.is_file()
        and BUILD_STAMP.read_text().strip() == fingerprint
    ):
        return extension_path

    raylib_archive = RAYLIB_DIR / "lib" / "libraylib.a"
    ini_source = INIH_DIR / "ini.c"
    required_paths = (raylib_archive, RAYLIB_DIR / "include" / "raylib.h", ini_source, INIH_DIR / "ini.h")
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Private probe build needs local dependencies: " + ", ".join(missing_paths))

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
    extra_link_args = ["-fwrapv", "-O2", "-Wl,-Bsymbolic-functions"]
    if Path("/usr/include/EGL/egl.h").is_file():
        extra_link_args.extend(["-lEGL", "-lGL", "-ldl"])
    extension = Extension(
        "probe_binding",
        sources=[str(GENERATED_BINDING), str(ini_source)],
        include_dirs=[
            numpy.get_include(),
            str(RAYLIB_DIR / "include"),
            str(INIH_DIR),
            str(BUILD_DIR),
            str(DRIVE_DIR),
            str(OCEAN_DIR),
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        extra_objects=[str(raylib_archive)],
    )
    distribution = Distribution({"ext_modules": [extension]})
    command = build_ext(distribution)
    command.build_lib = str(BUILD_DIR)
    command.build_temp = str(BUILD_DIR / "temp")
    command.force = True
    command.ensure_finalized()
    command.run()
    if not extension_path.is_file():
        raise RuntimeError(f"Private probe build did not create {extension_path}")
    BUILD_STAMP.write_text(fingerprint + "\n")
    return extension_path


if __name__ == "__main__":
    print(build_private_binding(force_rebuild="--force" in sys.argv[1:]))
