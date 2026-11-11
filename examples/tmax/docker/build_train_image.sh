#!/usr/bin/env bash
# Builds the TMax training image from this repository's docker/Dockerfile at HEAD and uploads it to Beaker as
# <account>/tmax-miles-train-<commit>, where it stays until deleted. A config picks its image by that name.
#
# The build context is `git archive HEAD`, so the image holds exactly that commit: its Dockerfile with the SGLang
# and Megatron pins, docker/pip_pins.txt, and its Miles at /root/miles. Jobs still run the commit they are
# submitted from (PYTHONPATH), so a new image is needed only when the environment changes.
#
#   bash examples/tmax/docker/build_train_image.sh              # build HEAD's image and upload it
#   bash examples/tmax/docker/build_train_image.sh --no-upload  # build localhost/tmax-miles-train:<commit> only
#
# Refuses uncommitted changes to tracked files (the image would silently leave them out) and an image name that
# already exists on Beaker. `beaker image create` talks to a Docker API: with podman, start
# `podman system service --time=0 unix:///run/podman/podman.sock &` and set DOCKER_HOST=unix:///run/podman/podman.sock.
set -euo pipefail

workspace=ai2/oe-agents-holmes
budget=ai2/oe-omai
upload=1
while [ $# -gt 0 ]; do
    case "$1" in
        --no-upload) upload=0; shift ;;
        --workspace) workspace=$2; shift 2 ;;
        --budget) budget=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
    echo "uncommitted changes to tracked files; commit them first, the image is built from HEAD:" >&2
    git -C "$repo" status --short --untracked-files=no >&2
    exit 1
fi
commit=$(git -C "$repo" rev-parse HEAD)
name=tmax-miles-train-${commit:0:10}
tag=localhost/tmax-miles-train:${commit:0:10}

if [ "$upload" = 1 ]; then
    account=$(beaker account whoami --format json \
        | python3 -c "import json, sys; d = json.load(sys.stdin); print((d[0] if isinstance(d, list) else d)['name'])")
    if beaker image get "$account/$name" > /dev/null 2>&1; then
        echo "Beaker image $account/$name already exists" >&2
        exit 1
    fi
fi

context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
git -C "$repo" archive HEAD | tar -x -C "$context"
# The Dockerfile's `COPY wheel[s]/` matches nothing without a wheels/ directory: BuildKit skips it, podman fails.
mkdir "$context/wheels"

docker build "$context" -f "$context/docker/Dockerfile" -t "$tag" \
    --platform linux/amd64 --build-arg TARGETARCH=amd64 --label tmax.commit="$commit"

if [ "$upload" = 1 ]; then
    beaker image create --name "$name" --workspace "$workspace" --budget "$budget" \
        --description "TMax training image: docker/Dockerfile and Miles at $commit" "$tag"
    echo "config: image: beaker:$account/$name"
fi
