import matplotlib.pyplot as plt

from box import Box, EncodedChunk

CHUNK_TIME = 16 # Frames per chunk
WIDTH = 32
HEIGHT = 32

def decode_chunk_classical(chunk: EncodedChunk) -> list[np.array]:
    frames = [ np.zeros(WIDTH, HEIGHT) for _ in CHUNK_TIME ]
    for box in chunk.boxes:
        
    return frames