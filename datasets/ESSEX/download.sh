#!/bin/bash

declare file_name;
declare download_link;

file_name="ESSEX3IN1_dataset.zip";
download_link="https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W/download?path=%2F&files=ESSEX3IN1_dataset.zip";

# Download models
echo Downloading $file_name...;
curl "$download_link" > $file_name;
unzip $file_name;
