# UC3: lc0 (CUDA backend + Python bindings) + python-chess + Stockfish.
#   docker build -t tms/lc0 -f uc3/Dockerfile /srv/work/uc3/lc0
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip meson ninja-build zlib1g-dev git stockfish ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --break-system-packages chess zstandard numpy pybind11
COPY . /src/lc0
WORKDIR /src/lc0
RUN meson setup build/release --buildtype=release -Dgtest=false -Dpython_bindings=true -Dplain_cuda=true -Dcudnn=false \
 && ninja -C build/release -j8 \
 && cp build/release/lc0 /usr/local/bin/ \
 && find build/release -name "backends*.so" -exec cp {} /usr/local/lib/python3.12/dist-packages/ \;
ENV PATH=/usr/games:$PATH
WORKDIR /work
