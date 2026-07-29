#!/usr/bin/env python3
"""Concatenate .smp sample files (format written by tools/train.c --dump)."""
import struct
import sys

MAGIC = 0x4C435344

def read_header(f):
    magic, size, pik, _ = struct.unpack("<4I", f.read(16))
    (count,) = struct.unpack("<Q", f.read(8))
    if magic != MAGIC:
        sys.exit("bad magic")
    return size, pik, count

def main(out, ins):
    size = pik = None
    total = 0
    with open(out, "wb") as o:
        o.write(struct.pack("<4I", MAGIC, 0, 0, 0))
        o.write(struct.pack("<Q", 0))
        for path in ins:
            with open(path, "rb") as f:
                s, p, c = read_header(f)
                if size is None:
                    size, pik = s, p
                elif (s, p) != (size, pik):
                    sys.exit(f"{path}: incompatible sample layout")
                copied = 0
                while True:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        break
                    o.write(chunk)
                    copied += len(chunk)
                if copied != c * size:
                    sys.exit(f"{path}: expected {c*size} bytes, got {copied}")
                total += c
                print(f"{path}: {c} samples")
        o.seek(0)
        o.write(struct.pack("<4I", MAGIC, size, pik, 0))
        o.write(struct.pack("<Q", total))
    print(f"{out}: {total} samples")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: merge_samples.py OUT IN1 IN2 [IN...]")
    main(sys.argv[1], sys.argv[2:])
