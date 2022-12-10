import cv2
import os
import csv

csv_rows = []
csv_filename = "skin_csv_info.csv"
for filename in os.listdir("scraped_skins"):
    f = os.path.join("scraped_skins", filename)
    img = cv2.imread(f)
    width = img.shape[0]
    height = img.shape[1] 
    if width == 512 and height == 512:
        csv_rows.append([f, 0])
    if width == 1024 and height == 1024:
        csv_rows.append([f, 1])
with open(csv_filename, "w") as g:
    csvwriter = csv.writer(g)
    csvwriter.writerow(["Filename", "Label (0 is Close and 1 is Full)"])
    csvwriter.writerows(csv_rows)