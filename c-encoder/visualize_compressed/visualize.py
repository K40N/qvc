import cv2
import numpy as np

from box import Box, BoxCfg, EncodedChunk

import pickle as pkl

from constants import *

def decode_chunk_classical(chunk: EncodedChunk) -> list[np.array]:
    frames = [ np.zeros((WIDTH, HEIGHT), dtype=np.uint8) for _ in range(CHUNK_TIME) ]
    for box_repr in chunk:
        box = eval(box_repr) # YES THIS IS STUPID I AM AWARE ITS 午前零時半 SORRY LOL
        if box is None:
            continue
        for x, y, t in box.member_coords():
            frames[t][x, y] = 255
    return frames

print("Loading encoded chunks...")
with open(f"../encoded_{VIDEO_NAME}.pkl", "rb") as f:
    encoded_chunks = pkl.load(f)

print(f"Decoding {CHUNK_TIME*len(encoded_chunks)} frames ({len(encoded_chunks)} chunks)...")
all_frames = []
for chunk in encoded_chunks:
    all_frames.extend(decode_chunk_classical(chunk))

print("Writing decoded frames...")
for i, frame in enumerate(all_frames):
    cv2.imwrite(f"vis_frames/frame-{i}.jpg", frame)

print("All done.")
