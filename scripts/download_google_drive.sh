#!/bin/bash
#==============================================================================
# 
#         FILE : download_google_drive.sh
#
#         USAGE: ./download_google_drive.sh file_id output_filename 
#
#        AUTHOR: Jae Yong Lee
#  ORGANIZATION: University of Illinois at Urbana-Champaign
#   DESCRIPTION: Downloads file from google drive using file_id
#       CREATED: 08/21/2018 09:08:10 AM
#         NOTES: only works for publically available files
#
#==============================================================================

FILE_ID=$1
FILE_NAME=$2
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=$FILE_ID" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > $FILE_NAME
