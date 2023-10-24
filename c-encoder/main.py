import cv2
import numpy as np

CHUNK_TIME = 16 # Frames per chunk
WIDTH = 32
HEIGHT = 32
FRAMES_PATH = lambda n: f"media_frames/bad-apple/frame-{n}.jpg"

def get_chunk_array(chunk_i: int):
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

print(get_chunk_array(6))
