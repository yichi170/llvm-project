#!/bin/bash
set -e

LLVM_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${LLVM_PROJECT_DIR}/../llvm-build"

RUN_SOURCE=false
RUN_MIR=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--source) RUN_SOURCE=true; shift ;;
        -m|--mir) RUN_MIR=true; shift ;;
        -h|--help)
            echo "Usage: ./run_taint_tests.sh [OPTIONS]"
            echo "Options:"
            echo "  -s, --source    Run only source-level tests"
            echo "  -m, --mir       Run only MIR-level tests"
            echo "  -h, --help      Show this help message"
            echo "If no options are provided, both test suites are run."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# If no flags are provided, run all tests.
if [[ "$RUN_SOURCE" == false && "$RUN_MIR" == false ]]; then
    RUN_SOURCE=true
    RUN_MIR=true
fi

echo "=== Building components ==="
cmake --build "${BUILD_DIR}" --target llc opt clang -j$(sysctl -n hw.ncpu)

TARGET_DIRS=""
if [[ "$RUN_SOURCE" == true ]]; then
    TARGET_DIRS="$TARGET_DIRS ${LLVM_PROJECT_DIR}/llvm/test/CodeGen/TaintAnalysis/source"
fi
if [[ "$RUN_MIR" == true ]]; then
    TARGET_DIRS="$TARGET_DIRS ${LLVM_PROJECT_DIR}/llvm/test/CodeGen/TaintAnalysis/mir"
fi

echo "=== Running Taint Analysis Tests ==="
python3 "${BUILD_DIR}/bin/llvm-lit" -v $TARGET_DIRS
