import os


path = "images/fruits/Test"
i = 0

filenames = os.listdir(path)

for dir, subdir, listfilename in os.walk(path):
    for filename in listfilename:
        i += 1
        new_filename = 'img_' + str(i) + ".jpg"
        src = os.path.join(dir, filename)  # NOTE CHANGE HERE
        dst = os.path.join(dir, new_filename)  # AND HERE
        os.rename(src, dst)
