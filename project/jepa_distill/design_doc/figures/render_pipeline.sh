#!/usr/bin/env bash
# Graphviz owns layout; LaTeX typesets labels; both outputs keep vector glyphs.
set -euo pipefail
FIGURE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${FIGURE_DIR}/../../../.." && pwd)"
DOT2TEX="${REPO_ROOT}/.venv/bin/dot2tex"
for tool in dot latex pdflatex pdftocairo; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "Required diagram renderer is missing: ${tool}" >&2
        exit 1
    fi
done
if [[ ! -x "${DOT2TEX}" ]]; then
    echo "Install the diagram dependency: .venv/bin/python -m pip install -r project/jepa_distill/design_doc/figures/requirements.txt" >&2
    exit 1
fi
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
cd "${BUILD_DIR}"
"${DOT2TEX}" --format=pgf --autosize --crop --margin=10pt \
    --docpreamble='\usepackage{amsmath}' --figpreamble='\sffamily' \
    "${FIGURE_DIR}/condition_b_pipeline.dot" -o condition_b_pipeline.tex
if ! pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape \
    condition_b_pipeline.tex > build.log 2>&1; then
    tail -n 60 build.log >&2
    exit 1
fi
pdftocairo -svg condition_b_pipeline.pdf condition_b_pipeline.svg
cp condition_b_pipeline.pdf condition_b_pipeline.svg "${FIGURE_DIR}/"
printf 'Generated %s\n' "${FIGURE_DIR}/condition_b_pipeline.svg" "${FIGURE_DIR}/condition_b_pipeline.pdf"
