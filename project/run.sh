mkdir -p model_weights/
wget -O model_weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O model_weights/best_unet_model.pth "https://huggingface.co/Venbiess/best_unet_model_car_segmentation/resolve/main/best_unet_model.pth?download=true"
# docker compose up --build