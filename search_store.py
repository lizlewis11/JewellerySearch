from Jewelry_Necklace.index_images import ImageIndex
import pandas as pd
from Jewelry_Necklace import config
from PIL import Image

# Handle different image file formats jpg,jpeg,bmp,png
def load_image(image_input):

    original_image = Image.open(image_input)
    # getpalettemode() and mode available only after image load
    original_image.load()
    if original_image.mode == 'P':
        # PIL loses palette attribute during crop. So we use low-level api.
        alpha = 'A' in original_image.im.getpalettemode()
        original_image = original_image.convert('RGBA' if alpha else 'RGB')


    # not elif...after P this should be executed
    if original_image.mode in ('RGBA','LA'):
        background = Image.new(original_image.mode[:-1], original_image.size, (255, 255, 255))
        background.paste(original_image,original_image.split()[-1])
        original_image = background

    img_resize_dim = 1400
    old_size = original_image.size  # old_size is in (height, width) format
    if (old_size[0] > 1400) or (old_size[1] > 1400):  # aspect ratio is not 1
        ratio = float(img_resize_dim) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        new_im = original_image.resize(new_size, Image.ANTIALIAS)
    else:
        new_im = original_image

    # image = image.resize((self.img_resize_dim,self.img_resize_dim), Image.ANTIALIAS)
    return new_im

# take image paths from csv
input_csv = pd.read_csv("C:\\Users\\wielj\\Work\\VisualSearch\\Jewellery\\jewelcsv.csv")

# total number of images in image set
no_of_images = len(input_csv.index)

# store all image paths in a list
for counter in range(0, no_of_images):
    print(counter)
    new_image = load_image(input_csv.iloc[counter][0])
    new_path = input_csv.iloc[counter][0].replace("Jewellery","Jewellery\\Resize2")
    new_image.save(new_path, quality = 95, optimize = True)

