import cv2

from ultralytics import solutions
from ultralytics import YOLO
model = YOLO('number_plate.pt')


cap = cv2.VideoCapture(r'/content/runs/detect/predict/Untitled video - Made with Clipchamp (6).avi')
assert cap.isOpened(), "Error reading video file"


w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

speed_region = [(20, 100), (1080, 104), (1080, 500), (20, 500)]


speed = solutions.SpeedEstimator(
    # show=True,
    model="yolo11n.pt",

    region=speed_region,
    classes=[2, 5,7],
    line_width=2,
)


# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if success:
        out = speed.estimate_speed(im0)
        video_writer.write(im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    print("Video frame is empty or video processing has been successfully completed.")
    break

cap.release()
cv2.destroyAllWindows()


python predictWithOCR.py model='/content/number_plate.pt' source='/content/speed_management.avi' save=True
