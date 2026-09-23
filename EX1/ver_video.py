import cv2
from pathlib import Path


ROOT = Path(__file__).resolve().parent

video = (
    ROOT
    / "saidas"
    / "video_ar.avi"
)

cap = cv2.VideoCapture(
    str(video)
)

while True:
    ok, frame = cap.read()

    if not ok:
        break

    cv2.imshow(
        "Video - Realidade Aumentada",
        frame
    )

    if cv2.waitKey(67) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()