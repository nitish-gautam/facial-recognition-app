"""
Detetcing face landmark using face_recognition library
File: realtime_face_recognition.py
"""

# How does recognise and display the name known and point unknown face in real time.  
# STEP1: Import libararies
# STEP2: Load the sample images to get 128 face embeddings and save the encoding 
# with the corresponding labels
# STEP3: Start the webcam to get the current frame from the video stream as an image
# STEP4: Resize the by 1/4 for faster execution and run the 'hog' algorithm to detect the faces in the image
# and then apply encoding for all the faces detected.
# STEP5: Now resize the image back to the original size i.e multiply by 4 which will be actual size of the video frame
# STEP6: Now compare the faces stored as sample images to the all found faces in the image and the face does not
# matches with the known faces named that face as "unknown face"
# STEP7: Create a rectangle around the face and display the name as text in the image
#
# Result:
# All known faces as per the sample image will be identifies and any unknown face will be tagged as 'unknown face'


# STEP1: Importing needed libraries
import cv2
import face_recognition

# STEP2: load the sample image to extract 128 face enconding value
Donald_Trump_img = face_recognition.load_image_file('images/sample/Donald_Trump.jpg')
Donald_Trump_face_encodings = face_recognition.face_encodings(Donald_Trump_img)[0]

# load the sample image2 to extract 128 face enconding value
Tom_Cruise_img = face_recognition.load_image_file('images/sample/Tom_Cruise.jpg')
Tom_Cruise_face_encodings = face_recognition.face_encodings(Tom_Cruise_img)[0]

# load the sample image2 to extract 128 face enconding value
Nitish_img = face_recognition.load_image_file('images/sample/Nitish.jpg')
Nitish_face_encodings = face_recognition.face_encodings(Nitish_img)[0]

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [Donald_Trump_face_encodings, Tom_Cruise_face_encodings, Nitish_face_encodings]
known_face_names = ["Donald Trump", "Tom Cruise", "Nitish"]

# STEP3: capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)

# initailse the array to hold the face locations, encoding and names
all_face_locations = []
all_face_encodings = []
all_face_names = []

# Loop every farme of the video
while True:
    #get the current frame from the video stream as an image
    _,frame = webcam_video_stream.read()
    #STEP4: resize the current frame to 1/4 size for faster execution
    frame_small = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    
    #detect all faces in the image with arguments as image,no_of_times_to_upsample, model
    # note of you increse the upsample then you may have more accuracy in mutiimage system but need more compuation, hence 
    # balance need to be made between aacuracy and computation
    all_face_locations = face_recognition.face_locations(frame_small,number_of_times_to_upsample=1,model='hog')
    
    #detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(frame_small,all_face_locations)

    #looping through the face locations and the face embeddings
    for frame_face_location,frame_face_encoding in zip(all_face_locations,all_face_encodings):
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = frame_face_location
        
        # STEP5: change the position maginitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
         # STEP6: find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, frame_face_encoding)
       
        #string to hold the label
        name_of_person = 'Unknown face'
        
        #check if the all_matches have at least one item
        #if yes, get the index number of face that is located in the first index of all_matches
        #get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        # STEP7: draw rectangle around the face    
        cv2.rectangle(frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    
    #display the video
    cv2.imshow("Webcam Video",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the stream and close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()    
