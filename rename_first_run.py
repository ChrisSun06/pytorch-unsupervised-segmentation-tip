import os 

for filename in os.listdir("sh_1k_data/segmented_images/train/"):
    my_source ="sh_1k_data/segmented_images/train/" + filename
    my_dest ="sh_1k_data/segmented_images/train/" + filename[5:]
    os.rename(my_source, my_dest)


for filename in os.listdir("sh_1k_data/segmented_plants/train/"):
    my_source ="sh_1k_data/segmented_plants/train/" + filename
    my_dest ="sh_1k_data/segmented_plants/train/" + filename[5:]
    os.rename(my_source, my_dest)