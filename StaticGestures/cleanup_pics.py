import os
import os.path
import shutil
import cv2
from scipy import misc
import imageio

# ### organize the photos
# folder_path = "fingers/test"

# images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# for image in images:
#     folder_name = image.split('_')[1][0]

#     new_path = os.path.join(folder_path, folder_name)
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)

#     old_image_path = os.path.join(folder_path, image)
#     new_image_path = os.path.join(new_path, image)
#     shutil.move(old_image_path, new_image_path)


### binarize the photos
outPath = "fingers/test_binary/0"
path = "fingers/test/0"

# iterate through the names of contents of the folder
for image_path in os.listdir(path):

    # create the full input path and read the file
    input_path = os.path.join(path, image_path)
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img,5)

    ret,th1 = cv2.threshold(img,75,255,cv2.THRESH_BINARY)

    # create full output path, 'example.jpg' 
    # becomes 'rotate_example.jpg', save the file to disk
    fullpath = os.path.join(outPath, image_path)
    imageio.imwrite(fullpath, th1)

