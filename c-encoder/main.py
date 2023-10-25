import cv2
import numpy as np

from box import Box
from encoder import ChunkEncoder

from dataclasses import dataclass
import sys

CHUNK_TIME = 16 # Frames per chunk
WIDTH = 32
HEIGHT = 32
BOXES_PER_CHUNK = 16
FRAMES_PATH = lambda n: f"media_frames/bad-apple/frame-{n}.jpg"
N_CHUNKS = 14

def get_chunk_array(chunk_i: int) -> np.array:
    frames = []
    start = chunk_i * CHUNK_TIME
    for i in range(start, start + CHUNK_TIME):
        rgb = cv2.imread(FRAMES_PATH(i))
        img = rgb[:,:,0] > (255//2)
        frames.append(img.reshape(WIDTH, HEIGHT, 1))
    result = np.concatenate(frames, axis=2)
    t = CHUNK_TIME // 3
    for x in range(WIDTH // 2):
        for y in range(HEIGHT // 2):
            assert result[x,y,t] == frames[t][x,y]
    return result

@dataclass(frozen=True)
class EncodedChunk:
    boxes: list[Box]

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

encoded = [ encode_chunk(chunk_i) for chunk_i in range(N_CHUNKS) ]
