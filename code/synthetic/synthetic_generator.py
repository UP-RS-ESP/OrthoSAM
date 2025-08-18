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
from numba import cuda 
from math import sqrt

def generator(min_radi=1, max_radi=3000, image_size=10000, num_circles=5000, Dir=None, max_attempts = 100, overlap = False, ran_color=True, generate_dem=True):
    '''
    Generates a synthetic scene with circles of random radii and positions.
    Parameters:
    - min_radi(int): Minimum radius of the circles. Default is 1.
    - max_radi(int): Maximum radius of the circles. Default is 3000.
    - image_size(int): Size of the generated image (image_size x image_size). Default is 10000.
    - num_circles(int): Maximum number of circles to generate. Default is 5000.
    - Dir(str): Directory to save the generated images and masks. If None or not specified, an path will be constructed based on the min_radi, max_radi values, and color setting, e.g., 'data/synthetic/synthetic_01_3000_bw'.
    - max_attempts(int): Maximum attempts to find a non-overlapping position for each circle. Default is 100.
    - overlap(bool): If True, circles can overlap. Default is False.
    - ran_color(bool): If True, circles will have random colors; otherwise, they will be white. Default is True.
    - generate_dem(bool): If True, generates a digital elevation model (DEM) based on the circles assumed to be semispheres. Default is True. Note that this is required for the shadow casting.
    Returns:
    - None: The function saves the generated images and masks to the specified directory.
    '''

    if not Dir:
        Dir_pre=os.path.join('data', 'synthetic', f'synthetic_{min_radi:02}_{max_radi}')
        if ran_color:
            Dir=os.path.join(Dir_pre, 'cl')
        else:
            Dir=os.path.join(Dir_pre, 'bw')
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint16)
    height_image = np.zeros((image_size, image_size), dtype=np.float32)
    circles = []
    for circle_id in range(1, num_circles + 1):
        for attempt in range(max_attempts):
            radius = random.randint(min_radi, max_radi)
            center_x = random.randint(radius, image_size - radius)
            center_y = random.randint(radius, image_size - radius)
            center = (center_x, center_y)
            overlap = False
            # Check for overlap with existing circles
            
            for (existing_center, existing_radius) in circles:
                dist = np.sqrt((center_x - existing_center[0]) ** 2 + (center_y - existing_center[1]) ** 2)
                if dist < radius + existing_radius + 1:#+1 so that no contact between circles
                    overlap = True
                    break
                    

            # If no overlap, add the circle and break out of the attempt loop
            if not overlap:
                circles.append((center, radius))
                
                if ran_color:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    color = (255,255,255)

                cv2.circle(image, center, radius, color, -1)
                
                cv2.circle(mask, center, radius, circle_id, -1)

                #DEM
                if generate_dem:
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
    np.save(os.path.join(Dir, 'msk'),mask) 
    np.save(os.path.join(Dir, 'img'),image)
    if generate_dem:
        np.save(os.path.join(Dir, 'dem'),height_image)
        
        
        
def shadow_image_generator(*, baseline_Dir=None, Dir=None, min_radi=1, max_radi=3000
                           , image_size=10000, num_circles=5000
                           , max_attempts = 100, overlap = False, ran_color=True
                           , generate_dem=True
                           , azimuth=0, inclination=45, shadow_strength=0.5):
    '''
    Generates a synthetic scene with circles of random radii and positions, and casts shadows based on the specified azimuth and inclination.
    Parameters:
    - baseline_Dir(str): Directory to containing the baseline synthetic scene. If None, a baseline scene will be generated with the specified/default parameters. Note that a path will first be generated based on the min_radi and max_radi values, e.g., 'data/synthetic/synthetic_01_3000'. The baseline scene will only be generated if the directory does not exist before the execution of this code.
    - Dir(str): Directory to save the generated images and masks. If None or not specified, an path will be constructed based on the min_radi, max_radi, azimuth, and inclination values, e.g., 'data/synthetic/synthetic_01_3000_shadow_0_45'.
    - min_radi(int): Minimum radius of the circles. Default is 1.
    - max_radi(int): Maximum radius of the circles. Default is 3000.
    - image_size(int): Size of the generated image (image_size x image_size). Default is 10000.
    - num_circles(int): Maximum number of circles to generate. Default is 5000.
    - max_attempts(int): Maximum attempts to find a non-overlapping position for each circle. Default is 100.
    - overlap(bool): If True, circles can overlap. Default is False.
    - ran_color(bool): If True, circles will have random colors; otherwise, they will be white. Default is True.
    - generate_dem(bool): If True, generates a digital elevation model (DEM) based on the circles assumed to be semispheres. Default is True. Note that this is required for the shadow casting.
    - azimuth(float): Azimuth angle for the shadow casting. Default is 0 degrees.
    - inclination(float): Inclination angle for the shadow casting. Default is 45 degrees.
    - shadow_strength(float): Strength of the shadow. This is a scaling factor applied to the image's 8 bit RGB channels to pixels that are in shadow. Default is 0.5.
    Returns:
    - None: The function saves the generated images and masks to the specified directory.
    '''
    if baseline_Dir is None:
        baseline_Dir=os.path.join('data', 'synthetic', f'synthetic_{min_radi:02}_{max_radi}')
        if ran_color:
            baseline_Dir=os.path.join(baseline_Dir, 'cl')
        else:
            baseline_Dir=os.path.join(baseline_Dir, 'bw')
        if not os.path.exists(baseline_Dir):  
            generator(min_radi=min_radi, max_radi=max_radi, image_size=image_size, num_circles=num_circles, Dir=baseline_Dir, ran_color=ran_color, generate_dem=generate_dem, max_attempts=max_attempts, overlap=overlap)
    if not Dir:
        Dir=os.path.join(baseline_Dir, f'_shadow_{azimuth}_{inclination}')
    if not os.path.exists(Dir):    
        os.makedirs(Dir)
    image=np.load(os.path.join(baseline_Dir, 'img.npy'))
    mask=np.load(os.path.join(baseline_Dir, 'msk.npy'))
    height_image=np.load(os.path.join(baseline_Dir, 'dem.npy'))

    shadow_image = cast_shadow(height_image, azimuth, inclination)
    shadow_image-=1
    shadow_image=np.abs(shadow_image)
    shadow_image=np.clip(shadow_image,shadow_strength, None)
    shadowed_rgb = (image * shadow_image[:, :, np.newaxis]).astype(np.uint8)

    np.save(os.path.join(Dir,'img'),shadowed_rgb)
    np.save(os.path.join(Dir,'shd'),shadow_image)
    np.save(os.path.join(Dir,'msk'),mask)

def noise_image_generator(*, baseline_Dir=None, Dir=None, min_radi=1, max_radi=3000
                           , image_size=10000, num_circles=5000
                           , max_attempts = 100, overlap = False, ran_color=True
                           , generate_dem=True, std=192):
    '''
    Generates a synthetic scene with circles of random radii and positions, and adds Gaussian noise to the image.
    Parameters:
    - baseline_Dir(str): Directory to containing the baseline synthetic scene. If None, a baseline scene will be generated with the specified/default parameters. Note that a path will first be generated based on the min_radi and max_radi values, e.g., 'data/synthetic/synthetic_01_3000'. The baseline scene will only be generated if the directory does not exist before the execution of this code.
    - Dir(str): Directory to save the generated images and masks. If None or not specified, an path will be constructed based on the min_radi, max_radi, and std values, e.g., 'data/synthetic/synthetic_01_3000_noise_192'.
    - min_radi(int): Minimum radius of the circles. Default is 1.
    - max_radi(int): Maximum radius of the circles. Default is 3000.
    - image_size(int): Size of the generated image (image_size x image_size). Default is 10000.
    - num_circles(int): Maximum number of circles to generate. Default is 5000.
    - max_attempts(int): Maximum attempts to find a non-overlapping position for each circle. Default is 100.
    - overlap(bool): If True, circles can overlap. Default is False.
    - ran_color(bool): If True, circles will have random colors; otherwise, they will be white. Default is True.
    - generate_dem(bool): If True, generates a digital elevation model (DEM) based on the circles assumed to be semispheres. Default is True. Note that this is required for the shadow casting.
    - std(int): Standard deviation of the Gaussian noise to be added to the image. Default is 192.
    Returns:
    '''
    
    if baseline_Dir is None:
        baseline_Dir=os.path.join('data', 'synthetic', f'synthetic_{min_radi:02}_{max_radi}')
        if ran_color:
            baseline_Dir=os.path.join(baseline_Dir, 'cl')
        else:
            baseline_Dir=os.path.join(baseline_Dir, 'bw')
        if not os.path.exists(baseline_Dir):  
            generator(min_radi=min_radi, max_radi=max_radi, image_size=image_size, num_circles=num_circles, Dir=baseline_Dir, ran_color=ran_color, generate_dem=generate_dem, max_attempts=max_attempts, overlap=overlap)
    if not Dir:
        Dir=os.path.join(baseline_Dir, f'_noise_{std}')
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    image=np.load(os.path.join(baseline_Dir, 'img.npy'))
    mask=np.load(os.path.join(baseline_Dir, 'msk.npy'))
    noisy_image = add_gaussian_noise_to_circle(image, 0, std, mask=mask, edge_std=std)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8) 
    np.save(os.path.join(Dir,'img'),noisy_image)
    np.save(os.path.join(Dir,'msk'),mask)
    try:
        height_image=np.load(os.path.join(baseline_Dir, 'dem.npy'))
        np.save(os.path.join(Dir,'dem'),height_image)
    except FileNotFoundError:
        print('No DEM found, skipping saving DEM')


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