#!/bin/bash

# ************************
# * LINKS DOESNT WORK!!! *
# * DOWNLOAD MANUALLY!!! *
# ************************

declare file_name;
declare download_link;

file_name="dnv2_NV_AB_wpca8192_last.ckpt";
download_link="https://universityofadelaide.app.box.com/index.php?rm=box_download_shared_file&shared_name=xykdjfh7wuwvpy9ft58izqeqe30nkvw1&file_id=f_1658938720721";

# Download models
echo Downloading $file_name...;
curl "$download_link" > $file_name;

# ----------------

file_name="dnv2_NV_192PCA_AB_wpca4096_last.ckpt";
download_link="https://universityofadelaide.app.box.com/index.php?rm=box_download_shared_file&shared_name=xykdjfh7wuwvpy9ft58izqeqe30nkvw1&file_id=f_1658931124063";

# Download models
echo Downloading $file_name...;
curl "$download_link" > $file_name;
