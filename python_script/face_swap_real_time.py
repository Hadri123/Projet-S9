# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 02:04:47 2021

@author: pcmaroc
"""

import dlib 
import cv2
import numpy as np 


def extract_index(nparray):
    index = None 
    for num in nparray[0]:
        index = num
        break
    return index 


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0) # ouvrir la caméra 
while cap.isOpened():    # tant que la caméa est ouverte  

    img = cv2.imread("D:\python_code\Images\ot.png")
    _, img2 = cap.read()
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    img2_new_face = np.zeros_like(img2) 
    faces = detector(img_gray)
    #Détction des points caractéristiques 
    for face in faces:
        landmarks=predictor(img_gray, face)
        landmarks_point = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_point.append((x, y))
            points = np.array(landmarks_point, np.int32)  
            hull=cv2.convexHull(points)
            
            cv2.fillConvexPoly(mask,hull,255)
      
            face_image_1 = cv2.bitwise_and(img, img, mask=mask)
            rect = cv2.boundingRect(hull)
            (x,y,w,h) = rect
          
            #Delaunay triangulation
              
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmarks_point)
        
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)
        indexes_triangles = []
        for t in triangles :
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            
            #cv2.circle(img, pt2, 3,(255,255,255),-1)
            
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index(index_pt1)
            
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index(index_pt2)
            
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index(index_pt3)
           
            
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)
    
    faces2 = detector(img_gray2)

    for face in faces2:
        landmarks = predictor(img_gray2, face)
        landmarks_point2 = []
        for n in range(0,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_point2.append((x,y))
             
    #triangulisation of the second face, from the first face 
    
        for triangle_index in indexes_triangles:
            tr1_pt1 = landmarks_point[triangle_index[0]]
            tr1_pt2 = landmarks_point[triangle_index[1]]
            tr1_pt3 = landmarks_point[triangle_index[2]]
            triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
            
            rect1 = cv2.boundingRect(triangle1)
            (x,y,w,h) = rect1
            
            cropped_triangle = img[y:y+h, x:x+w]
            cropped_mask1 = np.zeros((h,w), np.uint8)
            points = np.array([[tr1_pt1[0]-x, tr1_pt1[1]-y],
                              [tr1_pt2[0]-x, tr1_pt2[1]-y],
                              [tr1_pt3[0]-x, tr1_pt3[1]-y]])
            
            cv2.fillConvexPoly(cropped_mask1,points, 255)
            cropped_triangle = cv2.bitwise_and(cropped_triangle,cropped_triangle, mask=cropped_mask1)
            #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            
            
            
            #triangulisation of the second face
            tr2_pt1 = landmarks_point2[triangle_index[0]]
            tr2_pt2 = landmarks_point2[triangle_index[1]]
            tr2_pt3 = landmarks_point2[triangle_index[2]]
            triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
            rect2 = cv2.boundingRect(triangle2)
            (x,y,w,h) = rect2
        
            cropped_triangle2 = img2[y:y+h, x:x+w]
            cropped_mask2 = np.zeros((h,w), np.uint8)
            points2 = np.array([[tr2_pt1[0]-x, tr2_pt1[1]-y],
                              [tr2_pt2[0]-x, tr2_pt2[1]-y],
                              [tr2_pt3[0]-x, tr2_pt3[1]-y]])
            
            cv2.fillConvexPoly(cropped_mask2, points2, 255)
            cropped_triangle2 = cv2.bitwise_and(cropped_triangle2,cropped_triangle2, mask=cropped_mask2)
            
           
            
            
            #warp triangles 
            points = np.float32(points)
            points2 = np.float32(points2)
            M = cv2.getAffineTransform(points, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle, M, (w,h))
            
            
            #reconstruct destination face  
            triangle_area = img2_new_face[y:y+h, x:x+w]
            triangle_area = cv2.add(triangle_area, warped_triangle)
            img2_new_face[y:y+h, x:x+w] = triangle_area 
         
           
    
    #face swapped (putting 1st face in the second )
    img2_new_face_gray = cv2.cvtColor(img2_new_face, cv2.COLOR_BGR2GRAY)
    _,background =  cv2.threshold(img2_new_face_gray, 1,255,cv2.THRESH_BINARY_INV)
    background = cv2.bitwise_and(img2, img2, mask = background )
    result = cv2.add(background, img2_new_face)

    cv2.imshow("res",result)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break      

cap.release()
cv2.destroyAllWindows()
                
                
                
                
                
            