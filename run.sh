#!/bin/bash 
# Include argument parsing
# while getopts u:a:f: flag
# do
#     case "${flag}" in
#         u) username=${OPTARG};;
#         a) age=${OPTARG};;
#         f) fullname=${OPTARG};;
#     esac
# done
# echo "Username: $username";
# echo "Age: $age";
# echo "Full Name: $fullname";a

docker build -t instadeep:latest . 
docker run -dit -p 0.0.0.0:6006:6006 --name main -it --rm instadeep:latest  -d 'random_split' train