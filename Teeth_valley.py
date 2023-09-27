import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage import exposure

def find_gap_valleys(img_path):
    # Load image and convert to grayscale
    img = cv2.imread(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply median filter to remove salt-and-pepper noise
    img_median = cv2.medianBlur(img, 3)
    
    # Apply contrast stretching to improve contrast
    p2, p98 = np.percentile(img_median, (2, 98))
    img_contrast = exposure.rescale_intensity(img_median, in_range=(p2, p98))
    
    # Apply histogram equalization to further improve contrast
    img_eq = cv2.equalizeHist(img_contrast)

    # Initialize empty list to store gap valley coordinates
    gap_coords = []

    # Iterate over each column in the image
    # Defining the image for final output
    img_display1 = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
    for col_idx in range(8, img_eq.shape[1]-5):
        img_display = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)

        # Extract window of 11 pixels centered at the current column and compute intensity profile
        intensity_profile = np.mean(img_eq[:, col_idx-5:col_idx+6], axis=1)

        # Find local minima (valley locations) in the smoothed intensity profile
        valley_locs, _ = find_peaks(-intensity_profile, prominence=0.02)

       
        
        #contains the absolute distances of each index in the intensity_profile array from the center index
        distance_from_center = np.abs(np.arange(len(intensity_profile)) - len(intensity_profile) // 2)
        
        
        #extracts the intensity values of all the valleys in the intensity_profile array and stores them in a 1D numpy array
        valley_depths = intensity_profile[valley_locs]
        #epresent the probability that each valley corresponds to a gap between objects in the image.
        peak_valley = (np.exp(-(distance_from_center*distance_from_center) / ((90*90))))[valley_locs] * (1 - valley_depths / np.max(valley_depths))

        # Find the index of valley location with maximum PVI, which corresponds to gap valley
        gap_valley_idx = np.argmax(peak_valley)

        # Compute the coordinate of gap valley and add it to the list
        gap_valley_loc = valley_locs[gap_valley_idx]
        gap_coords.append((col_idx, gap_valley_loc))

        # Draw rectangle around window
        for coord in gap_coords:
            cv2.drawMarker(img_display, (coord[0], coord[1]), (43, 0, 255), cv2.MARKER_TILTED_CROSS, thickness=1, markerSize=10)

        # Display image
#         # Draw filled rectangle around window using background color
#         cv2.rectangle(img_display, (col_idx-5, 0), (col_idx+5, img_eq.shape[0]), (0, 0, 0), thickness=-1)

        # Draw rectangle around window
        cv2.rectangle(img_display, (col_idx-5, 0), (col_idx+5, img_eq.shape[0]), (0, 255, 0), thickness=2)

        cv2.imshow('image', img_display)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
        
        

    # Draw markers on image at gap valley coordinates
    cv2.rectangle(img_display1, (col_idx-5, 0), (col_idx+5, img_eq.shape[0]), (0, 255, 0), thickness=1)
    for coord in gap_coords:
        cv2.drawMarker(img_display1, (coord[0], coord[1]), (250, 0, 255), cv2.MARKER_TILTED_CROSS, thickness=1, markerSize=10)

    return img_display, img_display1, gap_coords

img_path = 'teeth_sample.png'
img_input = cv2.imread(img_path)

# Find gap valleys
img_display, img_display1, gap_coords = find_gap_valleys(img_path)

# Create new image with double width
h, w = img_input.shape[:2]
img_combined = np.zeros((h, w*2, 3), dtype=np.uint8)

# Copy input image to left half of new image
img_combined[:, :w, :] = img_input

# Copy output image to right half of new image
img_combined[:, w:, :] = img_display1

# Display combined image
cv2.imshow('Left is Input Image,      Right is Output Image with GAP valley detection', img_combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
