# coding: utf-8

import mimetypes
import os
import sys
import traceback
from time import sleep
from urllib.request import urlopen, Request


PARENT_DIR = 'downloaded_imgs'
MY_EMAIL_ADDR = ''
INTERVAL = 0.1


def fetch_and_save(fname):
    with open(fname) as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue

            name, img_url = fields
            sleep(INTERVAL)
            img, ext = download(img_url)
            if img is None or ext is None:
                print('Skipped ' + img_url)
            else:
                save(name, img, ext)
                print('Fetched ' + img_url)


def download(img_url):
    req = Request(img_url, headers={'User-Agent': MY_EMAIL_ADDR})
    try:
        with urlopen(req, timeout=3) as p:
            byte_content = p.read()
            content_type = p.getheader('Content-Type')
            if not content_type:
                return None, None
            ext = mimetypes.guess_extension(content_type.split(';')[0])
            if not ext:
                return None, None
            return byte_content, ext
    except:
        print('Error in downloading ' + img_url)
        print(traceback.format_exc())
        return None, None


def save(name, byte_content, extention):
    dir_to_save = os.path.join(PARENT_DIR, name)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    new_id = len(os.listdir(dir_to_save)) + 1
    with open(os.path.join(dir_to_save, str(new_id) + extention), mode='wb') as fp:
        fp.write(byte_content)


if __name__ == '__main__':
    fetch_and_save(sys.argv[1])
