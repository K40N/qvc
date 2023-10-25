import matplotlib.pyplot as plt
import numpy as np

from box import Box, EncodedChunk

import pickle as pkl

CHUNK_TIME = 16 # Frames per chunk
WIDTH = 32
HEIGHT = 32

VIDEO_NAME = "bad-apple"

def decode_chunk_classical(chunk: EncodedChunk) -> list[np.array]:
    frames = [ np.zeros(WIDTH, HEIGHT, dtype=np.bool) for _ in CHUNK_TIME ]
    for box in chunk.boxes:
        for x, y, t in box.member_coords():
            frames[t][x, y] = True
    return frames

with open(f"../encoded_{VIDEO_NAME}.pkl", "rb") as f:
    encoded_chunks = pkl.load(f)

all_frames = []
for chunk in encode_chunks:
    all_frames.append(decode_chunk_classical(chunk))
print(all_frames)
