import skvideo.io
from PIL import Image

images = skvideo.io.vread("/home/nico/Downloads/MyRoom1.mp4")

print(images.shape)

for i, image in enumerate(images):
    if i%3==0:
        im = Image.fromarray(image)
        im.save(f"Images/MyRoom1-{i}.jpeg")
        print(i)
    
    #if i>500:
    #    break
    
print("Done!")