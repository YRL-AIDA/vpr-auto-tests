#!/bin/bash

declare file_name;
declare download_link;

# MSLS-Moscow
file_name="msls_moscow.zip";
download_link="https://cloud.icc.ru/index.php/s/J6H3wQe55T2jjAj/download/msls_moscow.zip";

mkdir -p train_val;
cd train_val;

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;
