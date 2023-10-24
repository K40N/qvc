import cv2

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
        frames.append(img)
    return frames

print(get_chunk_array(6))
