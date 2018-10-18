import os, argparse, tqdm
from skvideo.io import FFmpegReader
from PIL import Image

def extract(video_file, output_dir, save_every=100):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  video = FFmpegReader(video_file)
  length = video.getShape()[0]
  for i, frame in zip(tqdm.trange(length), video.nextFrame()):
    if not i % save_every:
      filename = os.path.join(output_dir, str(i).zfill(10)+'.jpg')
      Image.fromarray(frame).convert('RGB').save(filename)

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('video_file', type=str, help='video file from which to extract frames')
  p.add_argument('output_dir', type=str, help='directory in which to put extracted frames')
  kwargs = p.parse_args().__dict__
  extract(**kwargs)
