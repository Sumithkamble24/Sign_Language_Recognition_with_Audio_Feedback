#dataset collection


import cv2
import os

# List of 15 ISL signs
sign_labels = [
    "hello", "thank_you",  "yes", "no", "i_love_you","help","sorry"
]

# Create dataset folder if it doesn't exist
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)


# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()
#Before capturing each gesture, it waits for you to press ENTER.
for label in sign_labels:
    print(f"\n▶ Ready to record: {label.upper()}")
    input("Press ENTER to start capturing...")
    
#Creates a subfolder like: dataset/hello/  to store all images for that specific sign.

    label_dir = os.path.join(dataset_path, label)
    os.makedirs(label_dir, exist_ok=True)

    img_count = 0
    total_imgs = 32 # You can increase/decrease this

    while img_count < total_imgs:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        # Display Live Feed with Instructions Recording: HELLO (12/32)
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Recording: {label} ({img_count}/{total_imgs})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Dataset Collector", display_frame)

        # Save frame
        img_path = os.path.join(label_dir, f"{label}_{img_count}.jpg") #dataset/hello/hello_0.jpg
        cv2.imwrite(img_path, frame)
        img_count += 1

        # Wait 2 Seconds Between Captures
        if cv2.waitKey(2000) & 0xFF == 27:
            print("Recording interrupted.")
            break

    print(f"✅ Collected {img_count} images for: {label}")

print("\n🎉 Dataset collection completed!")
cap.release()
cv2.destroyAllWindows()

