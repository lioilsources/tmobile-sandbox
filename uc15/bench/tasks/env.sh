# Source before running UC15 task tests on the VM:  . uc15/bench/tasks/env.sh
export UC15_ROOT=${UC15_ROOT:-/srv/work/w2/uc15}
export UC15_TOOLS=${UC15_TOOLS:-$UC15_ROOT/tools}
export PATH="$UC15_TOOLS/venv/bin:$UC15_TOOLS/dart-sdk/bin:/usr/local/go/bin:$PATH"
export GOPATH="$UC15_TOOLS/gopath" GOCACHE="$UC15_TOOLS/gocache" GOFLAGS=-mod=mod
export PUB_CACHE="$UC15_TOOLS/pub-cache"
export PYTHONDONTWRITEBYTECODE=1
