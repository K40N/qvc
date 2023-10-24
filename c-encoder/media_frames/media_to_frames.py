import cv2
import os

def process_video(video_name: str, video_path_rel: str):
	video_path = f"../media/{video_path_rel}"
	print(f"Processing video {video_name} at {video_path}...")
	os.mkdir(video_name)
	cap = cv2.VideoCapture(video_path)
	current_frame = 0
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(f"Extracting {total_frames} frames...")
	while current_frame < total_frames:
		_, frame = cap.read()
		frame_filename = f"{video_name}/frame-{current_frame}.jpg"
		cv2.imwrite(frame_filename, frame)
		print(f"Saved {video_name} frame #{current_frame + 1}...")
		current_frame += 1
	cap.release()
	assert current_frame == total_frames
	print(f"Done processing {video_name}.")

for i in os.listdir("../media/"):
	process_video(input(f'<!> Enter a name for file {i}: '), i)
print("All done.")
