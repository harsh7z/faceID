import cv2 as cv
import os 

def register_user():
    name = input("Enter your name: ")
    root = os.getcwd()
    save_path = os.path.join(root, "data/users", name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print("User already exists, please try again.")
        return

    cap = cv.VideoCapture(1)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    img_count = 0
    max_images = 300  # Number of face images to collect
    
    while True:
        ret, frame = cap.read()
        # frame = frame[240:240+500,400:400+500]
        if not ret:
            break 
        
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)


        for (x, y, w, h) in faces:
            
            if cv.waitKey(1) & 0xFF == ord('s'):
                face = frame[y:y+h, x:x+w]
                face = cv.resize(face, (105, 105))
                file_path = os.path.join(save_path, f"{img_count}.jpg")
                cv.imwrite(file_path, face)
                img_count += 1
            
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, f"Saved: {img_count}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv.imshow("Register User", frame)
        
        if cv.waitKey(1) & 0xFF == ord('q') or img_count >= max_images:
            break
        
if __name__ == '__main__':
    register_user()