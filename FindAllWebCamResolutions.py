import cv2
import csv
import codecs

cap = cv2.VideoCapture(0)
resolutions = csv.reader(codecs.open('resolutions.csv', encoding='utf-8'),delimiter="\t")
all_resolutions = {}

for row in resolutions:
    print(str(row[1]).split(str('×'))[0])
    width = int(row[1].split("×")[0])
    height = int(row[1].split("×")[1])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    all_resolutions[str(width) + "x" + str(height)] = "Supported"

print(all_resolutions)
cv2.destroyAllWindows()
cap.release()
