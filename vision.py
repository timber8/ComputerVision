import cv2
import numpy as np


def rectify(h):
  h = h.reshape((4,2))
  print h
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
       
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:4]  

    for card in contours:
        peri = cv2.arcLength(card,True)
        approx = cv2.approxPolyDP(card,0.02*peri,True)

        if len(approx) == 4:
            approx = rectify(approx)
            box = np.int0(approx)
            cv2.drawContours(im,[box],0,(255,255,0),6)
            imx = cv2.resize(im,(1000,600))
            cv2.imshow('a',imx)      
        
            h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

            transform = cv2.getPerspectiveTransform(approx,h)
            warp = cv2.warpPerspective(im,transform,(450,450))
        
            yield warp



if __name__ == '__main__':
  
    # cap = cv2.VideoCapture(1)
    
    # while(True):
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()

    #     # Our operations on the frame come here
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     # Display the resulting frame
    #     cv2.imshow('frame',gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    while(True):
        # ret, im = cap.read()
        im = cv2.imread('cards.jpg')
        width = im.shape[0]
        height = im.shape[1]

        if width < height:
            im = cv2.transpose(im)
            im = cv2.flip(im,1)

        for i,c in enumerate(getCards(im,4)):
            cv2.imwrite(str(i) + '.png',c)
            cv2.imshow(str(i),c)
        
       
        cv2.waitKey(0)
        break 
        
        
        
      
