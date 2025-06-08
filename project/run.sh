mkdir -p /app/model_weights
wget -O model_weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
docker compose up --build