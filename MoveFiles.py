#this is a simple script to order the image files into correct folders so that we can use Keras.

import shutil
import os

exec(open("Locations.py").read())


#where are the images located?
files = os.listdir(source)

#if the image names begin with "dog" or "cat" we move the images to their corresponding image folder
for f in files:
    newfile = os.path.join(source, f)
    if (f.startswith("dog")):
        shutil.move(newfile, dest1)
    elif (f.startswith("cat")):
        shutil.move(newfile, dest2)


#furthermore, we subset the "cat" and "dog" folders by "training" and "validation" folders
#subset the "cat folder" (first 4000 images)
files1 = os.listdir(dest1)

for f1 in files1:
    if files1.index(f1) < 4000:
        newfile1 = os.path.join(dest1, f1)
        shutil.move(newfile1, dest3)
    else:
        break


#subset the "cat folder" (first 4000 images)
files2 = os.listdir(dest2)

for f2 in files2:
    if files2.index(f2) < 4000:
        newfile2 = os.path.join(dest2, f2)
        shutil.move(newfile2, dest4)
    else:
        break
