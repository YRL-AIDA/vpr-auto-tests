#!/bin/bash

declare file_name;
declare download_link;

file_name="checksums.txt";
download_link="https://cloud.icc.ru/index.php/s/X5ZPynZ8wmCT83i/download/checksums.txt";

# Download checksums
echo Downloading $file_name...;
wget "$download_link" -O $file_name;

# Metadata
file_name="msls_metadata.zip";
download_link="https://cloud.icc.ru/index.php/s/cXyY2yRJKpfE3J8/download/msls_metadata.zip";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-1
file_name="msls1.zip";
download_link="https://cloud.icc.ru/index.php/s/YSn6yk34qLMtwEk/download/msls1.zip";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-2
file_name="msls2.zip";
download_link="https://cloud.icc.ru/index.php/s/Kid5Gd8dJ2kXZB8/download/msls2.zip";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-3
file_name="msls3.zip";
download_link="https://cloud.icc.ru/index.php/s/DZP8P7goqdXocLz/download/msls3.zip";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-4
file_name="msls4.zip";
download_link="https://cloud.icc.ru/index.php/s/2iiridsbT6JMBj2/download/msls4.zip";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-5
file_name="msls5.zip";
download_link="https://cloud.icc.ru/index.php/s/dTaaRTdNoRoN4Ma/download/msls5.zip";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Msls-6
file_name="msls6.zip";
download_link="https://cloud.icc.ru/index.php/s/9ockwoGsKacHjNf/download/msls6.zip";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;

# Patch
file_name="msls_patch.zip";
download_link="https://cloud.icc.ru/index.php/s/zX8d2cjSRwNraFF/download/msls_patch.zip";

echo Downloading $file_name...;
wget "$download_link" -O $file_name;
echo Exctracting $file_name...;
unzip $file_name;
