#!/usr/bin/env python3
"""Dump the safetensors headers of every pinned Idol weight file (HTTP Range, no weights downloaded).

Usage: python3 -I Scripts/reference/dump_headers.py [out_dir]   (default .local-runs/headers)
Prints one line per file: tensor count and dtypes. The counts are the loader contracts in
Sources/IdolCore/Configuration/IdolWeights.swift; rerun after any revision bump.
"""
import json
import os
import struct
import sys
import urllib.request

FILES = [  # (repo, revision, path)
    ("speridlabs/idol", "2731c2291fc65a797c5cb627a5e5526fc1490902", "denoiser.safetensors"),
    ("speridlabs/idol", "2731c2291fc65a797c5cb627a5e5526fc1490902", "decoder.safetensors"),
    ("speridlabs/idol", "2731c2291fc65a797c5cb627a5e5526fc1490902", "decoder_gs.safetensors"),
    ("speridlabs/idol", "2731c2291fc65a797c5cb627a5e5526fc1490902", "birefnet.safetensors"),
    ("speridlabs/idol-turbo", "45cdd41f1f4d6d70496daa7d619f57ad79307a37", "idol_turbo_lora.safetensors"),
    ("Comfy-Org/Idol", "e907d8dca6c465735eaf4ce9b3a3c2c6401d5705", "clip_vision/dino_v3_vit_h_bf16.safetensors"),
    ("Comfy-Org/Idol", "e907d8dca6c465735eaf4ce9b3a3c2c6401d5705", "vae/flux2-vae.safetensors"),
]


def read_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(request).read()


def header(url):
    length = struct.unpack("<Q", read_range(url, 0, 7))[0]
    return json.loads(read_range(url, 8, 7 + length))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else ".local-runs/headers"
    os.makedirs(out_dir, exist_ok=True)
    for repo, revision, path in FILES:
        h = header(f"https://huggingface.co/{repo}/resolve/{revision}/{path}")
        name = f"{repo}/{path}".replace("/", "_") + ".json"
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(h, f, indent=0)
        h.pop("__metadata__", None)
        dtypes = {}
        for entry in h.values():
            dtypes[entry["dtype"]] = dtypes.get(entry["dtype"], 0) + 1
        print(f"{repo}/{path}: {len(h)} tensors {dtypes}")


if __name__ == "__main__":
    main()
