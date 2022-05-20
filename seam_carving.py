# Author: Eunsub Lee

import cv2
import numpy as np
from matplotlib import pyplot as plt

def compute_energy_img(im):
    gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

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
        
        row_indices.appendleft(best_index)
    
    return row_indices

def view_seam(im, seam, direction):
    plt.clf()
    plt.axis('off')

    if direction == 'VERTICAL':
        plt.plot(seam, range(len(seam)), 'r')
    else:
        plt.plot(range(len(seam)), seam, 'r')

    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.pause(0.00001)

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

def compute_average(im, row, col, direction):
    rgb = [0, 0, 0]
    nrows, ncols = im.shape[0], im.shape[1]
    count = 0
    
    if direction == 'VERTICAL':
        for dc in [-1, 0, 1]:
            if col + dc >= 0 and col + dc < ncols:
                for i in range(len(rgb)):
                    rgb[i] += im[row][col + dc][i]
                count += 1
    else:
        for dr in [-1, 0, 1]:
            if row + dr >= 0 and row + dr < nrows:
                for i in range(len(rgb)):
                    rgb[i] += im[row + dr][col][i]
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
        reduced_im[row][additional_col][:] = compute_average(im, row, additional_col, 'VERTICAL')
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
        reduced_im[additional_row, col, :] = compute_average(im, additional_row, col, 'HORIZONTAL')
        reduced_im[additional_row + 1:, col, :] = im[additional_row:, col, :]
        
    return reduced_im, compute_energy_img(reduced_im)
    
def main():
    import argparse
    fig, _ = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-im', help='Path to image', required=True)
    parser.add_argument('-out', help='Name of output file', required=True)
    parser.add_argument('-dw', help='Change in width (in pixels)', type=int, default=0)
    parser.add_argument('-dh', help='Change in height (in pixels)', type=int, default=0)
    args = parser.parse_args()

    im = cv2.imread(args.im)
    energy = compute_energy_img(im)

    if args.dw > 0:
        for _ in range(args.dw):
            im, energy = increase_width(im, energy)
    else:
        for _ in range(-args.dw):
            im, energy = decrease_width(im, energy)
            
    if args.dh > 0:
        for _ in range(args.dh):
            im, energy = increase_height(im, energy)
    else:
        for _ in range(-args.dh):
            im, energy = decrease_height(im, energy)
    
    plt.clf()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.savefig(args.out, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':    
    main()

