import os
import pyttsx3
import speech_recognition as sr
from cv2 import cv2
'''using this script to create a folder for a person imags with his name using audio input 

2 module usage

1. pyttsx3 --- for converting text to speech ( useful in saying commands like listening)
2. speech recognition --- for speech to text and it uses google speech recognition  
3. creates a folder for the listned name 
4. using webcam captures 30 images of unknown person for trainning
'''
r=sr.Recognizer()
face_status='unknown'

def savecmd():
    eng=pyttsx3.init()
    print("Listening name....")
    eng.say('Listening name')    #(1)
    eng.runAndWait()
    
    with sr.Microphone() as source2:
        try:    
            r.adjust_for_ambient_noise(source2,duration=0.5)
            user_said_person_name=r.listen(source2,2,4)
            print(user_said_person_name)
            
            person_name=r.recognize_google(user_said_person_name)
            print(person_name)
            person_name=person_name.lower()
        except (sr.UnknownValueError,TypeError):
            print("Error")
        
        

            
    dir=str(person_name)  #name for a directory with user name
    current_directory=os.getcwd()
    #pare="C:/Users/USER/Desktop/juju/smrt goggles/codes/using web cam/database/"
    if dir!='quit' or dir!='stop':
        pare=current_directory+'/database'
        path=os.path.join(pare,dir)
        os.mkdir(path)              
        print("Directory '% s' created" %dir)    #folder created
        
        #to record face samples
        cap = cv2.VideoCapture(0)
        i = 1
        os.chdir(path)

        while(i<31):
            ret, frame = cap.read()
            if ret == False:
                break     
            # Save Frame by Frame into specified path using imwrite method and name file like juju(1).jpg
            cv2.imwrite(dir+str(i)+'.jpg', frame)   
            i += 1 
        cap.release()
        cv2.destroyAllWindows()

        
        print('_|-- Person added --|_')
    else:
        print('-|stopped|-')
        eng.say('Stopped')
#cutt
if face_status=='unknown':
    savecmd()