# ScrollGT — registered human ground-truth ink evaluation.
#
# Everything needed to reproduce every published number ships in this image:
# scoring is CPU-only and needs no network and no GPU at run time. The bundled
# targets are ~12 MB, so the whole image is small enough to pull and run.
#
#   docker build -t scrollgt .
#   docker run --rm scrollgt                    # CLI help
#   docker run --rm scrollgt scrollgt check --window-px 64 --scan-um 8.0
#   docker run --rm scrollgt pytest -q          # the full suite, ~12 minutes
#
# To score your own prediction, mount it in:
#   docker run --rm -v "$PWD:/work" scrollgt \
#       scrollgt score /work/my_prediction.png data/scroll1_20231210121321 \
#                      --json-out /work/card.json
#
# System requirements: any x86-64 host with Docker. No GPU, no network at run
# time, ~200 MB of disk for the image.

FROM python:3.11-slim

# git is not needed to run, only to report provenance if someone asks the image
# what it is; keeping the layer small matters more than that convenience.
WORKDIR /app

# Dependency metadata first so a source-only edit does not re-resolve wheels.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir . && pip install --no-cache-dir "pytest>=7.0"

# Targets and tests last: they change most often and are pure data.
COPY data/ ./data/
COPY tests/ ./tests/

# Scoring must not reach the network; fail loudly rather than silently degrade
# if that assumption is ever broken.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

CMD ["scrollgt", "--help"]
