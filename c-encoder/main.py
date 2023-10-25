import cv2
import numpy as np

from box import Box, EncodedChunk
from encoder import ChunkEncoder

from dataclasses import dataclass
import pickle as pkl
import sys

from constants import *

N_CHUNKS = (7 * LEN_SECONDS) // CHUNK_TIME

FRAMES_PATH = lambda n: f"media_frames/{VIDEO_NAME}/frame-{n}.jpg"

def get_chunk_array(chunk_i: int) -> np.array:
    frames = []
    start = chunk_i * CHUNK_TIME
    for i in range(start, start + CHUNK_TIME):
        rgb = cv2.imread(FRAMES_PATH(i))
        img = rgb[:,:,0] > (255//2)
        if img.sum() > ((WIDTH*HEIGHT)//2):
            img ^= True
        frames.append(img.reshape(WIDTH, HEIGHT, 1))
    result = np.concatenate(frames, axis=2)
    t = CHUNK_TIME // 3
    for x in range(WIDTH // 2):
        for y in range(HEIGHT // 2):
            assert result[x,y,t] == frames[t][x,y]
    return result

def encode_chunk(chunk_i: int) -> EncodedChunk:
    print(f"Chunk #{chunk_i+1:03}/{N_CHUNKS:03}: ", end="")
    sys.stdout.flush()
    encoder = ChunkEncoder(get_chunk_array(chunk_i))
    print(f"{chunk_i*100/(N_CHUNKS-1):>5.1f}% [", end="")
    sys.stdout.flush()
    boxes = []
    for _ in range(BOXES_PER_CHUNK):
        print("#", end="")
        sys.stdout.flush()
        boxes.append(encoder.emit_box())
    print("]")
    return EncodedChunk(boxes)

print("Encoding...")
encoded = [ encode_chunk(chunk_i) for chunk_i in range(N_CHUNKS) ]
print("Saving...")
with open(f"encoded_{VIDEO_NAME}.pkl", "wb+") as f:
    pkl.dump(encoded, f)
print("All done.")
