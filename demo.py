import cv2
#to show the image
import numpy as np
from math import cos, sin
import os



def find_biggest_contour(image):
    # Copy
    image = image.copy()
   
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, (255,255,0), -1)
   
    return biggest_contour, mask

def find_suit_in_a_card(image, verbose=False):
    
    if verbose:
        cv2.imshow('image', image);
        cv2.imwrite('corner_used.png', image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    is_red = 1
    
    # Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    if verbose:
        cv2.imshow('corner_blured', image_blur)
        cv2.imwrite('corner_blured.png',image_blur)
    #t unlike RGB, HSV separates luma, or the image intensity, from
    # chroma or the color information.
    #just want to focus on color, segmentation
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    
    # Filter by colour
    # 0-10 hue
    #minimum red amount, max red amount
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    if verbose:
        cv2.imshow('mask1_red', mask1)
        cv2.imwrite('mask1.png',mask1)

    #birghtness of a color is hue
    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    if verbose:
        cv2.imshow('mask2_red', mask2);
        cv2.imwrite('mask2_red.png',mask2)

    #looking for what is in both ranges
    # Combine masks
    mask = mask1 + mask2
    array = np.asarray(mask)

    #Check if is not a black suit
    if np.count_nonzero(array) < 5000:
        is_red = 0
        min_black = np.array([0, 0, 0])
        max_black = np.array([180, 256, 150])
        #layer
        mask = cv2.inRange(image_blur_hsv, min_black, max_black)

    if verbose:
        cv2.imshow('result mask', mask)
        cv2.imwrite('result_mask.png',mask)


    # Clean up
    #we want to circle our strawberry so we'll circle it with an ellipse
    #with a shape of 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #morph the image. closing operation Dilation followed by Erosion. 
    #It is useful in closing small holes inside the foreground objects, 
    #or small black points on the object.
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if verbose:
        cv2.imshow('dilation', mask_closed)
        cv2.imwrite('dilated_corner.png',mask_closed)
    #erosion followed by dilation. It is useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    if verbose:
        cv2.imshow('erosion', mask_clean)
        cv2.imwrite('dilated_corner_with_erosion.png',mask_clean)
    
    # Find biggest strawberry
    #get back list of segmented strawberries and an outline for the biggest one
    suit_contour, mask_suit = find_biggest_contour(mask_clean)
    
    if verbose:
        cv2.imshow('isolated suit', mask_suit)
        cv2.imwrite('isolated_suit.png', mask_suit)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return suit_contour, mask_suit, is_red

def redSuitContours(back_corner):
    if back_corner == 0:
        ouros = cv2.imread('corner_ouros_up.png')
        ouros_contour, mask_ouros, is_red = find_suit_in_a_card(ouros, False)

        hearts = cv2.imread('corner_copas_up.png')
        hearts_contour, mask_hearts, is_red = find_suit_in_a_card(hearts, False)
    else:
        ouros = cv2.imread('corner_ouros_back.png')
        ouros_contour, mask_ouros, is_red = find_suit_in_a_card(ouros, False)

        hearts = cv2.imread('corner_copas_back.png')
        hearts_contour, mask_hearts, is_red = find_suit_in_a_card(hearts, False)

    return ouros_contour, hearts_contour

def blackSuitContours(back_corner):
    if back_corner == 0:
        espadas = cv2.imread('corner_espadas_up.png')
        espadas_contour, mask_espadas, is_red = find_suit_in_a_card(espadas, False)

        paus = cv2.imread('corner_paus_up.png')
        paus_contour, mask_paus, is_red = find_suit_in_a_card(paus, True)
    else:
        espadas = cv2.imread('corner_espadas_back.png')
        espadas_contour, mask_espadas, is_red = find_suit_in_a_card(espadas, False)

        paus = cv2.imread('corner_paus_back.png')
        paus_contour, mask_paus, is_red = find_suit_in_a_card(paus, False)


    return espadas_contour, paus_contour

def cornerHasRed(im):
    image_blur = cv2.GaussianBlur(im, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
    # Filter by colour
    # 0-10 hue
    #minimum red amount, max red amount
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    #birghtness of a color is hue
    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
    
    #looking for what is in both ranges
    # Combine masks
    mask = mask1 + mask2
    array = np.asarray(mask)
    print 'Numero de pixeis vermelhos: ' + str(np.count_nonzero(array))

    return np.count_nonzero(array)

def cornerHasBlack(im):
    image_blur = cv2.GaussianBlur(im, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
    min_black = np.array([0, 0, 0])
    max_black = np.array([180, 256, 150])
    #layer
    mask = cv2.inRange(image_blur_hsv, min_black, max_black)
    array = np.asarray(mask)
    print 'Numero de pixeis pretos: ' + str(np.count_nonzero(array))

    return np.count_nonzero(array)

def compareCountours(cnt1,is_red, back_corner):

    if is_red == 1:
        ouros_contour, hearts_contour = redSuitContours(back_corner)
        is_ouros = cv2.matchShapes(cnt1,ouros_contour,1,0.0)
        is_hearts = cv2.matchShapes(cnt1,hearts_contour,1,0.0)
        print 'Ouros' + str(is_ouros)
        print 'Copas' + str(is_hearts)

        return is_ouros, is_hearts
    else:
        espadas_contour, paus_contour = blackSuitContours(back_corner)
        is_espadas = cv2.matchShapes(cnt1,espadas_contour,1,0.0)
        is_paus = cv2.matchShapes(cnt1,paus_contour,1,0.0)

        print 'Espadas' + str(is_espadas)
        print 'Paus' + str(is_paus)

        return is_espadas, is_paus

#read the image
#image = cv2.imread('cornerkq0.png')
#detect it

def getSuitString(image, back_corner, verbose):
    
    suit_contour, mask_suit, is_red = find_suit_in_a_card(image, verbose)
    
    if is_red == 1:
        is_ouros, is_hearts = compareCountours(suit_contour, is_red, back_corner)
        if is_ouros > is_hearts:
            print 'Copas'
            return 'Copas'
        else:
            print 'Ouros'
            return 'Ouros'
    else:
        is_espadas, is_paus = compareCountours(suit_contour, is_red, back_corner)
        if is_espadas > is_paus:
            print 'Paus'
            return 'Paus'
        else:
            print 'Espadas'
            return 'Espadas'

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #write the new image
    #cv2.imwrite('yo2.jpg', result)
