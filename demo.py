from __future__ import division
import cv2
#to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin



def find_biggest_contour(image):
    # Copy
    image = image.copy()
    #input, gives all the contours, contour approximation compresses horizontal, 
    #vertical, and diagonal segments and leaves only their end points. For example, 
    #an up-right rectangular contour is encoded with 4 points.
    #Optional output vector, containing information about the image topology. 
    #It has as many elements as the number of contours.
    #we dont need it
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    print len(contours)
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, (255,255,0), -1)

    cv2.imshow('contour',mask)
   

    return biggest_contour, mask

def find_suit(image):
    #RGB stands for Red Green Blue. Most often, an RGB color is stored 
    #in a structure or unsigned integer with Blue occupying the least 
    #significant area (a byte in 32-bit and 24-bit formats), Green the 
    #second least, and Red the third least. BGR is the same, except the 
    #order of areas is reversed. Red occupies the least significant area,
    # Green the second (still), and Blue the third.
    # we'll be manipulating pixels directly
    #most compatible for the transofrmations we're about to do
    print image.shape
    cv2.imshow('image', image);
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make a consistent size
    #get largest dimension
    max_dimension = max(image.shape)
    #The maximum window size is 700 by 660 pixels. make it fit in that
    scale = 700/max_dimension
    # #resize it. same width and hieght none since output is 'image'.
    image = cv2.resize(image, None, fx=scale, fy=scale)
    cv2.imshow('resized image', image)
    #we want to eliminate noise from our image. clean. smooth colors without
    #dots
    # Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imshow('blured_image', image_blur)
    #t unlike RGB, HSV separates luma, or the image intensity, from
    # chroma or the color information.
    #just want to focus on color, segmentation
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    cv2.imshow('blured_image hsv', image_blur_hsv);
    # Filter by colour
    # 0-10 hue
    #minimum red amount, max red amount
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    
    cv2.imshow('mask1', mask1)

    #birghtness of a color is hue
    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
    cv2.imshow('mask2', mask2);
    #looking for what is in both ranges
    # Combine masks
    mask = mask1 + mask2
    array = np.asarray(mask)

    #Check if is not a black suit
    if np.count_nonzero(array) < 5000:
        min_red = np.array([0, 0, 0])
        max_red = np.array([180, 256, 50])
        #layer
        mask = cv2.inRange(image_blur_hsv, min_red, max_red)
    cv2.imshow('result mask', mask)


    # Clean up
    #we want to circle our strawberry so we'll circle it with an ellipse
    #with a shape of 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #morph the image. closing operation Dilation followed by Erosion. 
    #It is useful in closing small holes inside the foreground objects, 
    #or small black points on the object.
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('dilation', mask_closed)
    #erosion followed by dilation. It is useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    cv2.imshow('erosion', mask_clean)
        # Find biggest strawberry
    #get back list of segmented strawberries and an outline for the biggest one
    suit_contour, mask_suit = find_biggest_contour(mask_clean)

    
    return suit_contour, mask_suit

#read the image
image = cv2.imread('cornerkq1.png')
#detect it
result = find_suit(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#write the new image
#cv2.imwrite('yo2.jpg', result)