import sys
from PIL import Image
import numpy as np
import scipy.ndimage

def preprocess_image(img):
    # invert the image
    img = Image.eval(img, lambda x: 255 - x)

    # Find the bounding box of the digit and crop the image
    inverted_img = np.array(img)
    rows = np.any(inverted_img, axis=1)
    cols = np.any(inverted_img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # keep the aspect ratio 1:1
    width = rmax - rmin
    height = cmax - cmin
    if width > height:
        cmin -= (width - height) // 2
        cmax += (width - height) // 2
    else:
        rmin -= (height - width) // 2
        rmax += (height - width) // 2
    img.crop((rmin, cmin, rmax, cmax))
    
    # Resize the image to fit in a 20x20 pixel box while preserving aspect ratio
    img.thumbnail((20, 20), Image.LANCZOS)
    
    # Create a new 28x28 image and paste the resized image at the center
    new_img = Image.new('L', (28, 28), (0))  # Create a black 28x28 image
    img_size = img.size
    new_img.paste(img, ((28 - img_size[0]) // 2, (28 - img_size[1]) // 2))
    
    # Convert image to numpy array
    img_array = np.array(new_img)
    
    # Compute the center of mass
    cy, cx = scipy.ndimage.center_of_mass(img_array)
    
    # Translate the image to center the mass
    shiftx = np.round(14 - cx).astype(int)
    shifty = np.round(14 - cy).astype(int)
    shifted_img = scipy.ndimage.shift(img_array, shift=[shifty, shiftx], mode='constant', cval=0.0)

    # Convert the numpy array back to an image
    shifted_img = Image.fromarray(shifted_img)
    
    return shifted_img

# Example usage
image_path = sys.argv[1]
img = Image.open(image_path).convert('L')
if img.size != (28, 28):
    img = preprocess_image(img)
img.save('image.ppm')
