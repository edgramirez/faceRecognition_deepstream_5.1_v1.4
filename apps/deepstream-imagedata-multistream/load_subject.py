#!/usr/bin/python3
import sys
import os
from pathlib import Path
import lib.common as com

param_length = len(sys.argv)
base_input_dir = com.INPUT_DB_DIRECTORY

msg = 'Usage: ' + sys.argv[0] + ' newBlackList | newWhiteList | addToBlackList | addToWhiteList | remoteBlackList | removeWhiteList '

if param_length < 2:
    com.log_error(msg)

if sys.argv[1] == 'newBlackList':
    if param_length == 2:
        blacklist_face_images = base_input_dir + '/blacklist_faces'
        blacklist_results_dir = base_input_dir + '/blacklist_db'
        com.create_data_dir(blacklist_results_dir)
        com.create_data_dir(blacklist_face_images)
        #blacklist_data = blacklist_results_dir + '/' + 'BlackList.dat'
        try:
            blacklist_data = blacklist_results_dir + '/' + com.BLACKLIST_DB_NAME
        except AttributeError:
            com.log_error("Configuration error - environment variable 'BLACKLIST_DB_NAME' not set")
    else:
        com.log_error(msg)

    com.create_data_dir(com.RESULTS_DIRECTORY)
    com.log_debug("Saving data in directory: {}".format(blacklist_results_dir))

    import lib.biblioteca as biblio 
    biblio.encode_known_faces_from_images_in_dir(blacklist_face_images, blacklist_data, 'blacklist')
elif sys.argv[1] == 'newWhiteList':
    if param_length == 2:
        whitelist_face_images = base_input_dir + '/whitelist_faces'
        whitelist_results_dir = base_input_dir + '/whitelist_db'
        com.create_data_dir(whitelist_results_dir)
        com.create_data_dir(whitelist_face_images)
        #whitelist_data = whitelist_results_dir + '/WhiteList.dat'
        try:
            whitelist_data = whitelist_results_dir + '/' + com.WHITELIST_DB_NAME
        except AttributeError:
            com.log_error("Configuration error - environment variable 'WHITELIST_DB_NAME' not set")
    else:
        com.log_error(msg)

    com.create_data_dir(com.RESULTS_DIRECTORY)
    com.log_debug("Saving data in directory: {}".format(whitelist_results_dir))

    import lib.biblioteca as biblio 
    biblio.encode_known_faces_from_images_in_dir(whitelist_face_images, whitelist_data, 'whitelist')
else:
    com.log_error(msg)
