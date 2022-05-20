# Author: Eunsub Lee

import cv2
import numpy as np
from matplotlib import pyplot as plt

def compute_energy_img(im):
    gray_im = cv2.cvtColor(np.uint8(im), cv2.COLOR_RGB2GRAY)

    x = np.array([[-1, 1]])
    y = np.array([[-1], [1]])
    
    dx = cv2.filter2D(gray_im, -1, x)
    dy = cv2.filter2D(gray_im, -1, y)
    energy = np.sqrt(dx ** 2 + dy ** 2)
    return energy

def cumulative_min_energy_map(energy, direction):
    cumulative = np.copy(energy)
    nrows = cumulative.shape[0]
    ncols = cumulative.shape[1]

    if direction == 'VERTICAL':
        for row in range(1, nrows):
            for col in range(ncols):
                min_val = cumulative[row - 1][col]
                
                if col - 1 >= 0:
                    min_val = min(min_val, cumulative[row - 1][col - 1])
                if col + 1 < ncols:
                    min_val = min(min_val, cumulative[row - 1][col + 1])
                    
                cumulative[row][col] += min_val
    else:
        for col in range(1, ncols):
            for row in range(nrows):
                min_val = cumulative[row][col - 1]
                
                if row - 1 >= 0:
                    min_val = min(min_val, cumulative[row - 1][col - 1])
                if row + 1 < nrows:
                    min_val = min(min_val, cumulative[row + 1][col - 1])
                    
                cumulative[row][col] += min_val
                
    return cumulative

def find_vertical_seam(cumulative):
    from collections import deque
    
    nrows, ncols = cumulative.shape
    column_indices = deque()
    column_indices.appendleft(np.argmin(cumulative[nrows - 1]))
    
    for i in reversed(range(nrows - 1)):
        row = cumulative[i]

        best_index, index = column_indices[0], column_indices[0]
        min_val = row[index]
        
        if index - 1 >= 0 and row[index - 1] < min_val:
            best_index = index - 1
            min_val = row[best_index]
        if index + 1 < ncols and row[index + 1] < min_val:
            best_index = index + 1
            min_val = row[best_index]
        
        column_indices.appendleft(best_index)
    
    return column_indices

def find_horizontal_seam(cumulative):
    from collections import deque

    nrows, ncols = cumulative.shape
    row_indices = deque()
    row_indices.appendleft(np.argmin(cumulative[:,ncols - 1]))
    
    for i in reversed(range(ncols - 1)):
        col = cumulative[:,i]

        best_index, index = row_indices[0], row_indices[0]
        min_val = col[index]
        
        if index - 1 >= 0 and col[index - 1] < min_val:
            best_index = index - 1
            min_val = col[best_index]
        if index + 1 < nrows and col[index + 1] < min_val:
            best_index = index + 1
            min_val = col[best_index]
        
        row_indices.appendleft(best_index)
    
    return row_indices

def view_seam(im, seam, direction):
    im = np.copy(im)
    if direction == 'VERTICAL':
        for i in range(len(seam)):
            im[i][seam[i]][:] = [0,0,255]
    else:
        for i in range(len(seam)):
            im[seam[i]][i][:] = [0,0,255]

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.pause(0.0000001)

def decrease_width(im, energy):
    nrows, ncols, ndepths = im.shape
    reduced_im = np.uint8(np.zeros((nrows, ncols - 1, ndepths)))
    cumulative = cumulative_min_energy_map(energy, 'VERTICAL')
    vertical_seam = find_vertical_seam(cumulative)

    view_seam(im, vertical_seam, 'VERTICAL')

    for row in range(nrows):
        skip_col = vertical_seam[row]
        reduced_im[row][:skip_col][:] = im[row][:skip_col][:]
        reduced_im[row][skip_col:][:] = im[row][skip_col + 1:][:]        

    return reduced_im, compute_energy_img(reduced_im)

def decrease_height(im, energy):
    nrows, ncols, ndepths = im.shape
    reduced_im = np.uint8(np.zeros((nrows - 1, ncols, ndepths)))
    cumulative = cumulative_min_energy_map(energy, 'HORIZONTAL')
    hori_seam = find_horizontal_seam(cumulative)

    view_seam(im, hori_seam, 'HORIZONTAL')

    for col in range(ncols):
        skip_row = hori_seam[col]
        reduced_im[:skip_row,col,:] = im[:skip_row,col,:]
        reduced_im[skip_row:,col,:] = im[skip_row + 1:,col,:]

    return reduced_im, compute_energy_img(reduced_im)

def compute_average(im, row, col):
    rgb = [0, 0, 0]
    count = 0
    nrows, ncols = im.shape[0], im.shape[1]
    
    for dr in [-1, 0, 1]:
        if row + dr < 0 or row + dr >= nrows:
            continue
        
        for dc in [-1, 0, 1]:
            if col + dc < 0 or col + dc >= ncols:
                continue
            for i in range(len(rgb)):
                rgb[i] += im[row + dr][col + dc][i]
            
            count += 1
    
    return [color // count for color in rgb]

def increase_width(im, energy):
    nrows, ncols, ndepths = im.shape
    reduced_im = np.uint8(np.zeros((nrows, ncols + 1, ndepths)))
    cumulative = cumulative_min_energy_map(energy, 'VERTICAL')
    vertical_seam = find_vertical_seam(cumulative)

    view_seam(im, vertical_seam, 'VERTICAL')
    
    for row in range(nrows):
        additional_col = vertical_seam[row]
        
        reduced_im[row][:additional_col][:] = im[row][:additional_col][:]
        reduced_im[row][additional_col][:] = compute_average(im, row, additional_col)
        reduced_im[row][additional_col+1:][:] = im[row][additional_col:][:]
        
    return reduced_im, compute_energy_img(reduced_im)

def increase_height(im, energy):
    nrows, ncols, ndepths = im.shape
    reduced_im = np.uint8(np.zeros((nrows + 1, ncols, ndepths)))
    cumulative = cumulative_min_energy_map(energy, 'HORIZONTAL')
    hori_seam = find_horizontal_seam(cumulative)

    view_seam(im, hori_seam, 'HORIZONTAL')
    
    for col in range(ncols):
        additional_row = hori_seam[col]
        
        reduced_im[:additional_row, col, :] = im[:additional_row, col, :]
        reduced_im[additional_row, col, :] = compute_average(im, additional_row, col)
        reduced_im[additional_row + 1:, col, :] = im[additional_row:, col, :]
        
    return reduced_im, compute_energy_img(reduced_im)
    
def main():
    # parse arguments
    im_name = 'imgs/inputSeamCarvingPrague.jpg'
    im = cv2.imread(im_name)
    energy = compute_energy_img(im)

    for i in range(30):
        print(i)
        im, energy = increase_height(im, energy)
    
    
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    fig, _ = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.axis('off')
    
    main()
    
    plt.pause(1)
    plt.show()
