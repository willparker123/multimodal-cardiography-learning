from helpers import seg_factor
import cv2
import glob
import re

def create_video(imagespath, outpath, filename, images=None, framerate=seg_factor):
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
            print(fn)
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
    
    out = cv2.VideoWriter(outpath+filename+'.mp4',cv2.VideoWriter_fourcc(*'DIVX'), framerate, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return img_array