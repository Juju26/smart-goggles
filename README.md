<h1 align="center">Smart Goggles codebase</h1>


## Overview

This Repository is a code base for the project which 
aims to help visually challenged people for recognizing people in front to give them better understanding of their surroundings.


## Modules 
-	FACE DETECTION AND RECOGNITION :
	
      Face Detection : 
      The face detection is generally considered as finding the faces (location and size) in an image and probably extract them to be used by the
 	face detection algorithm. 
  
      Face Recognition: 
       The face recognition algorithm is used in finding features that are uniquely described in the image. The facial image is already 
      extracted, cropped, resized, and usually converted in the grayscale.(Pytorch)
	 
       MODULES -  Imutils , Sklearn-encoder ,  SVC, open-cv


- TEXT TO SPEECH RECOGNITION :

     Text-To-Speech is a process in which input text is first analyzed, then processed and understood, and then the text is converted to digital audio and                             then Spoken. 

	 MODULES -  Pyttsx3 , Gtts , Speech recognition


-	GENERAL MODULES :	

	These modules support for the image processing like converting the input into several frames and these frame processing with the already trained images in the dataset and recognition of the image with the database.

     MODULES - numpy , pickle , OS
     
 ## Code Workflow
<img src="https://github.com/Juju26/smart-goggles/blob/smart-goggles/documentation/code%20work%20flow.jpeg" alt="Image went missing" height="550px" width="750px" align="center">
