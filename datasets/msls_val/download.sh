#!/bin/bash

declare file_name;
declare download_link;

file_name="checksums.txt";
download_link="https://cloud.icc.ru/index.php/s/X5ZPynZ8wmCT83i";

# Download checksums
echo Downloading $file_name...;
wget "$download_link" -O $file_name;

# Metadata
file_name="msls_metadata.zip";
download_link="https://cloud.icc.ru/index.php/s/cXyY2yRJKpfE3J8";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-1
file_name="msls1.zip";
download_link="https://cloud.icc.ru/index.php/s/YSn6yk34qLMtwEk";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-2
file_name="msls2.zip";
download_link="https://cloud.icc.ru/index.php/s/Kid5Gd8dJ2kXZB8";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-3
file_name="msls3.zip";
download_link="https://cloud.icc.ru/index.php/s/DZP8P7goqdXocLz";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-4
file_name="msls4.zip";
download_link="https://cloud.icc.ru/index.php/s/2iiridsbT6JMBj2";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-5
file_name="msls5.zip";
download_link="https://cloud.icc.ru/index.php/s/dTaaRTdNoRoN4Ma";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-6
file_name="msls6.zip";
download_link="https://cloud.icc.ru/index.php/s/9ockwoGsKacHjNf";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Patch
file_name="msls_patch.zip";
download_link="https://cloud.icc.ru/index.php/s/zX8d2cjSRwNraFF";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;
