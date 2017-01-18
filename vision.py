import cv2
import numpy as np
import demo


def rectify(h):
  h = h.reshape((4,2))
  hnew = np.zeros((4,2),dtype = np.float32)

  add = h.sum(1)
  hnew[0] = h[np.argmin(add)]
  hnew[2] = h[np.argmax(add)]
   
  diff = np.diff(h,axis = 1)
  hnew[1] = h[np.argmin(diff)]
  hnew[3] = h[np.argmax(diff)]

  return hnew

def getCards(im, numcards=4):
  gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(1,1),1000)
  flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY) 
     
  image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:4]  

  for card in contours:
    peri = cv2.arcLength(card,True)
    approx = cv2.approxPolyDP(card,0.02*peri,True)
    if (len(approx) == 4) & (cv2.contourArea(approx) > 9000):
      approx = rectify(approx)
      
      #Show the image
      box = np.int0(approx)
      cv2.drawContours(im,[box],0,(255,255,0),6)
      imx = cv2.resize(im,(1000,600))
      cv2.imshow('cards_frame',imx)
      cv2.imwrite('cards_frame.png',imx)      
  
      h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

      transform = cv2.getPerspectiveTransform(approx,h)
      warp = cv2.warpPerspective(im,transform,(450,450))

   
      width = approx[1][0] - approx[0][0]
      height = approx[3][1] - approx[0][1]

      if width > height:
        warp = cv2.transpose(warp)
      

      cv2.imshow('card',warp)
      cv2.imwrite('card.png',warp)
  
      yield warp

def getGoodCorner(img):

  corner1 = img[55:150, 5:75]
  corner2 = img[275:400, 5:75]
  corner1_r = corner1.copy()
  corner2_r = corner2.copy()
  back_corner = 0

  corner1_red = demo.cornerHasRed(corner1_r)
  corner2_red = demo.cornerHasRed(corner2_r)
 
  cv2.imshow('corner1',corner1)
  cv2.imshow('corner2',corner2)

  cv2.waitKey(0) 

  if (corner1_red > 100) | (corner2_red > 100):
    if corner1_red > corner2_red:
    
      return corner1, back_corner

      
    else:
      back_corner = 1
      return corner2, back_corner
  
  corner1_black = demo.cornerHasBlack(corner1_r)
  corner2_black = demo.cornerHasBlack(corner2_r)

  if (corner1_black > 100) | (corner2_black > 100):
    if corner1_black > corner2_black:
      return corner1, back_corner
    else:
      back_corner = 1
      return corner2, back_corner
 


def getInnerContours(im):

  corner, back_corner = getGoodCorner(im)
  print back_corner

  big_corner = cv2.resize(corner,(400,400))
  #cv2.imwrite('corner_copas_up.png',big_corner)
  suit = demo.getSuitString(big_corner, back_corner, True)

  return suit


if __name__ == '__main__':
  
  cap = cv2.VideoCapture(1)
  
  while(True):
    # Capture frame-by-frame
    ret, im = cap.read()

    # # Our operations on the frame come herels
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  while(True):
  #  im = cv2.imread('espadas.png')
    width = im.shape[0]
    height = im.shape[1]

    for i,c in enumerate(getCards(im, 4)): 
      cv2.imshow(getInnerContours(c),c)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
      
       
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break 
        
        
        
      
