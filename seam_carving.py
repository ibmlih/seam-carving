import cv2
import numpy as np
from matplotlib import pyplot as plt

def display_img(im, title='image'):
    plt.imshow(im, cmap='gray')
    plt.show()
    # cv2.imshow(title, im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def compute_energy_img(im_name):
    im = cv2.imread(im_name)
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    x = np.array([[-1, 1]])
    y = np.array([[-1], [1]])
    
    dx = cv2.filter2D(gray_im, cv2.CV_64F, x)
    dy = cv2.filter2D(gray_im, cv2.CV_64F, y)
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
    pass

def view_seam(im, seam, direction):
    if direction == 'VERTICAL':
        for i in range(len(seam)):
            im[i][seam[i]][0] = 0
            im[i][seam[i]][1] = 0
            im[i][seam[i]][2] = 255

    else:
        pass

    cv2.imshow('image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def decrease_width(im, energy):
    pass
    
    return reduced_im, reduced_energy

def main():
    im_name = 'imgs/inputSeamCarvingPrague.jpg'
    im = cv2.imread(im_name)
    
    energy = compute_energy_img(im_name)
    cumulative_energy_map = cumulative_min_energy_map(energy, 'VERTICAL')
    vertical_seam = find_vertical_seam(cumulative_energy_map)
    view_seam(im, vertical_seam, 'VERTICAL')

    
    
    
if __name__ == '__main__':
    main()
