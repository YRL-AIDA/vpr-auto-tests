#!/bin/bash

declare file_name;
declare download_link;


# ViTB
file_name="pairvpr-vitB.pth";
download_link="https://cloud.icc.ru/index.php/s/MrNBD23jML9QRfk/download/pairvpr-vitB.pth";

# Download models
echo Downloading $file_name...;
wget "$download_link" -O $file_name;

echo "--- --- ---";

# ViTB 500 epochs
file_name="pairvpr-pretrained-500epochs-vitB.pth";
download_link="https://cloud.icc.ru/index.php/s/6pbrXNNjjjNFwqx/download/pairvpr-pretrained-500epochs-vitB.pth";

# Download models
echo Downloading $file_name...;
wget "$download_link" -O $file_name;

echo "--- --- ---";

# ViTG
file_name="pairvpr-vitG.pth";
download_link="https://cloud.icc.ru/index.php/s/7g3JTmJWPnRmYka/download/pairvpr-vitG.pth";

# Download models
echo Downloading $file_name...;
wget "$download_link" -O $file_name;
