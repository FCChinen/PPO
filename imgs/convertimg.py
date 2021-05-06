from PIL import Image, ImageDraw
from os import walk
f = []
for _, _, filenames in walk("."):
    f.extend(filenames)
    break

print(f)

for item in f:
    name, ext = item.split('.')
    if ext == 'png':
        img = Image.open(item)
        new_img = img.resize((40, 40))
        draw = ImageDraw.Draw(new_img)
        rect = [(40, 40), (new_img.size[0] - 10, new_img.size[1] - 10)] 
        draw.rectangle(rect)
        new_img.save(name + 'converted.' + ext)