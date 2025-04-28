import numpy as np
import cv2
import random
import os
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#import functions as fnc
from itertools import product
import gc
import torch

min_radi=int(sys.argv[1])

try:
    max_radi=int(sys.argv[2])
except:
    max_radi=3000

try:
    shadow_strength=float(sys.argv[3])
except:
    shadow_strength=0.2

azimuth_l=[0, 90, 180, 270]
inclination_l=[60, 70, 80]
image_size=10000
num_circles = 5000

from numba import cuda 
from math import sqrt
@cuda.jit("float32[:, :], float32[:, :], int64, int64, int64, float64, float64, float64")
def cuda_kern_cast_shadow(s, z, ys, xs, thr, slope, azim, incl):
    i, j = cuda.grid(2)
    if i < ys and j < xs:
        b = i - slope * j
        aazim = abs(azim)
        klen = 0
        slen = 0
        k = 0
        o = 0
        if 45 <= aazim and aazim <= 135:
            # steep steps, each row one pixel
            if 0 <= azim:
                for y in range(i, ys):
                    x = int((y - b) / slope)
                    if xs <= x:
                        break
                    if x < 0:
                        break
                    dy, dx = y - i, x - j
                    r = sqrt(dx*dx + dy*dy)
                    dz = z[y, x] - z[i, j] - r * incl
                    if dz > 0:
                        if k == o + 1:
                            klen += 1
                        else:
                            klen = 0
                        if klen > slen:
                            slen = klen
                        o = k
                    if slen > thr:
                        break
                    k += 1
            else:
                for y in range(i, 0, -1):
                    x = int((y - b) / slope)
                    if xs <= x:
                        break
                    if x < 0:
                        break
                    dy, dx = y - i, x - j
                    r = sqrt(dx*dx + dy*dy)
                    dz = z[y, x] - z[i, j] - r * incl
                    if dz > 0:
                        if k == o + 1:
                            klen += 1
                        else:
                            klen = 0
                        if klen > slen:
                            slen = klen
                        o = k
                    if slen > thr:
                        break
                    k += 1
        else:
            # shallow steps, each col one pixel
            if abs(azim) <= 90:
                for x in range(j, xs):
                    y = int(slope * x + b)
                    if ys <= y:
                        break
                    if y < 0:
                        break
                    dy, dx = y - i, x - j
                    r = sqrt(dx*dx + dy*dy)
                    dz = z[y, x] - z[i, j] - r * incl
                    if dz > 0:
                        if k == o + 1:
                            klen += 1
                        else:
                            klen = 0
                        if klen > slen:
                            slen = klen
                        o = k
                    if slen > thr:
                        break
                    k += 1
            else:
                for x in range(j, 0, -1):
                    y = int(slope * x + b)
                    if ys <= y:
                        break
                    if y < 0:
                        break
                    dy, dx = y - i, x - j
                    r = sqrt(dx*dx + dy*dy)
                    dz = z[y, x] - z[i, j] - r * incl
                    if dz > 0:
                        if k == o + 1:
                            klen += 1
                        else:
                            klen = 0
                        if klen > slen:
                            slen = klen
                        o = k
                    if slen > thr:
                        break
                    k += 1
        if slen > thr:
            s[i, j] = 1


def cast_shadow(z, azimuth, inclination):
    thr = 5 # minimum light blocking thickness
    assert inclination < 90
    incli = np.tan((90 - inclination) * np.pi / 180)
    slope = np.tan(azimuth * np.pi / 180)
    ys, xs = z.shape
    d_z = cuda.to_device(z.astype("float32"))
    d_s = cuda.device_array((ys, xs), np.float32)
    nthreads = (16, 16)
    nblocksy = ys // nthreads[0] + 1
    nblocksx = xs // nthreads[0] + 1
    nblocks = (nblocksy, nblocksx)
    cuda_kern_cast_shadow[nblocks, nthreads](d_s, d_z, ys, xs, thr, slope, np.float64(azimuth), incli)
    s = d_s.copy_to_host()
    return s

def add_gaussian_noise_to_circle(array, mean ,std , mask=None, edge_std=None):
    '''
    add gaussian noise to the input image. if mask is given noise will not be added to the area outside the circle. if edge_std is given, different noise will be applied to the edge. mask is required for that.
    '''
    gaussian_noise = np.random.normal(mean, std, array.shape)
    if np.any(mask):
        mask=mask>0
        gaussian_noise=gaussian_noise*mask[:, :, np.newaxis]
        if edge_std:
            gaussian_noise_ed = np.random.normal(mean, edge_std, array.shape)
            gaussian_noise+=gaussian_noise_ed*~mask[:, :, np.newaxis]
    noisy_image = array.astype(float) + gaussian_noise
    #noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image
Dir=f'/DATA/vito/data/ran_synth_{min_radi:02}_{max_radi}_shadow_{str(shadow_strength).replace(".", "_")}/'

if not os.path.exists(Dir[:-1]):
    os.makedirs(Dir[:-1])
    os.makedirs(Dir+'img')
    os.makedirs(Dir + 'shd')
image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
mask = np.zeros((image_size, image_size), dtype=np.uint16)
height_image = np.zeros((image_size, image_size), dtype=np.float32)

circles = []

for circle_id in range(1, num_circles + 1):
    max_attempts = 100  # Limit the number of attempts to find a non-overlapping position
    for attempt in range(max_attempts):
        radius = random.randint(min_radi, max_radi)
        center_x = random.randint(radius, image_size - radius)
        center_y = random.randint(radius, image_size - radius)
        center = (center_x, center_y)

        # Check for overlap with existing circles
        overlap = False
        for (existing_center, existing_radius) in circles:
            dist = np.sqrt((center_x - existing_center[0]) ** 2 + (center_y - existing_center[1]) ** 2)
            if dist < radius + existing_radius + 1:#+1 so that no contact between circles
                overlap = True
                break

        # If no overlap, add the circle and break out of the attempt loop
        if not overlap:
            circles.append((center, radius))
            
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            cv2.circle(image, center, radius, color, -1)
            
            cv2.circle(mask, center, radius, circle_id, -1)

            #DEM
            y1, y2 = center_y - radius, center_y + radius
            x1, x2 = center_x - radius, center_x + radius

            if x1 < 0 or y1 < 0 or x2 >= image_size or y2 >= image_size:
                continue

            yy, xx = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing='ij')
            dx = xx - center_x
            dy = yy - center_y
            dist_sq = dx**2 + dy**2
            mask_circle = dist_sq <= radius**2

            z = np.zeros_like(dist_sq, dtype=np.float32)
            z[mask_circle] = np.sqrt(radius**2 - dist_sq[mask_circle])

            height_image[y1:y2, x1:x2] = np.maximum(height_image[y1:y2, x1:x2], z)
            break  
noisy_height=add_gaussian_noise_to_circle(height_image,0,3)
noisy_height = np.clip(noisy_height, a_min=0,a_max=None)
np.save(Dir+f'msk',mask)
np.save(Dir+f'dem',noisy_height)
np.save(Dir+f'org_rgb',image)

for i,ang in enumerate(list(product(azimuth_l, inclination_l))):
    gc.collect()
    torch.cuda.empty_cache()
    azimuth = ang[0]
    inclination = ang[1]
    shadow_image = cast_shadow(noisy_height, azimuth, inclination)
    shadow_image-=1
    shadow_image=np.abs(shadow_image)
    shadow_image=np.clip(shadow_image,shadow_strength, None)
    shadowed_rgb = (image * shadow_image[:, :, np.newaxis]).astype(np.uint8)

    np.save(Dir+f'img/img_{i:02}_{inclination}_{azimuth:03}',shadowed_rgb)
    np.save(Dir+f'shd/shd_{i:02}_{inclination}_{azimuth:03}',shadow_image)