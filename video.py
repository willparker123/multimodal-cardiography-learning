from config import load_config
import cv2
import glob
import re
import config
import os
import torchaudio
from moviepy.editor import *


def convert_video_to_audio_moviepy(video_file):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    audioclip = AudioFileClip(video_file)
    audioclip.write_audiofile(video_file[:len(video_file)-3]+'.wav')
    audio, sr = torchaudio.load(video_file[:len(video_file)-3]+'.wav')
    return audio, sr
    
def create_video(imagespath, outpath, filename, images=None, framerate=config.global_opts.fps):
    #out, _ = (
    #    ffmpeg.input(imagespath+'*.png', pattern_type='glob', framerate=framerate)
    #    .output(outpath+filename+'.mp4')
    #    .run()
    #)
    img_array = []
    size = (0,0)
    if images is not None:
        img_array = images
    else:
        imgs = [None] * len(glob.glob(imagespath+'*.png'))
        for ind, fn in enumerate(glob.glob(imagespath+'*.png')):
            i=0
            m = re.search('seg_(.+?)_e', fn)
            if m:
                i = int(m.group(1).split("seg_",1)[1])

            img = cv2.imread(fn)
            height, width, layers = img.shape
            if ind == 0:
                size = (width,height)
            else:
                assert size == (width,height)
            imgs[i] = img
        img_array = imgs
    
    out = cv2.VideoWriter(outpath+filename+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), framerate, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return img_array

def resample_video(videopath, fps):
    cap = cv2.VideoCapture(videopath)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(videopath.replace('.mp4', '')+f"_{fps}fps.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    #command = (f'ffmpeg -i {videopath+f"_{fps}fps_.mp4"} -vcodec copy -acodec copy -movflags faststart {videopath+f"_{fps}fps.mp4"}')
    #from subprocess import call
    #cmd = command.split(' ')
    #call(cmd, shell=True)

def load_video(filepath):
    cap = cv2.VideoCapture(filepath)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    img_array = []
    hasFrames,image = cap.read()
    if hasFrames:
        img_array.append(image)
    return img_array, fps, size
    
