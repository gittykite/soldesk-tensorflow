# (machinegpu) C:\WINDOWS\system32>pip uninstall pyopenssl (생략 가능)
# (machinegpu) C:\WINDOWS\system32>pip install pyopenssl (생략 가능)
▷ / ws_python / notebook / machine / cnn_actor / google_crawler.ipynb
-------------------------------------------------------------------------------------
from icrawler.builtin import GoogleImageCrawler
import requests
import urllib.request
import datetime

from scrapy.selector import Selector

count = 0
count_max = 0

inputSearch = input('Enter name to crawl.')

# 150~50: Train, 10: validation
count_max = int(input('Enter num of images to save.'))

base_url = "https://www.google.co.kr/search?biw=1597&bih=925&" \
           "tbm=isch&sa=1&btnG=%EA%B2%80%EC%83%89&q=" + inputSearch

# create folders
# full_name = "C:/ai_201912/ws_python/notebook/machine/cnn_actor/src/"+inputSearch+"/"+inputSearch+"_"+str(count)+"_"+nowDatetime+".jpg"
full_name = "C:/ai_201912/ws_python/notebook/machine/cnn_actor/src/" + inputSearch
google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4,
                                    storage={'root_dir': full_name})

# download img larger than 200 * 200
google_crawler.crawl(keyword=inputSearch,
                     max_num=count_max,
                     min_size=(200, 200),
                     max_size=None)