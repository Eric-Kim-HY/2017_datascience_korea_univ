import sys
import os
from collections import defaultdict
from scrapy import Selector

# Create HTML dictionary
html_dic = defaultdict(list)            # {"attraction_id": [html1, html2, ...]}

# city = sys.argv[1]          # 실행창에 입력한 도시 이름을 city에 넣는다
city = "kyoto"
"""
cities = ["paris","tokyo","new york", "kyoto", "seoul"]
"""

inner1 = os.listdir(city)[0] # city폴더 내 첫번째 폴더 이름을 inner1이라고 한다. inner1 = com
inner2 = os.listdir('./' + city + '/' + inner1)[0] # com폴더 내 첫번째 폴더 이름을 inner2라고 한다. inner2 = city_id
folders = os.listdir('./' + city + '/' + inner1 + '/' + inner2)[:3] # 그 city에 해당되는 모든 attraction_id가 folders에 저장된다.

for fol in folders:         # fol은 하나의 attraction_id이다
    html_files = os.listdir('./' + city + '/' + inner1 + '/' + inner2 + '/' + fol)
    # html_files는 여러개의 리뷰 페이지에 대한 각각의 html파일이다
    for h in html_files:
        # with open(h, encoding='utf-8') as fp:
        with open('./' + city + '/' + inner1 + '/' + inner2 + '/' + fol + '/' + h, encoding='utf-8') as fp:
            html_txt = fp.read()
            html_dic[fol].append(html_txt) # 각 attraction의 value값에 html을 추가한다


result = []             # will become a list of review_data



for attraction_id in html_dic:
    attraction_htmls = html_dic[attraction_id]


    for h in attraction_htmls:
        sel = Selector(text=h)
        users = sel.xpath('//div[contains(@id,"review_4")]/@id').extract()
        # print(users)
        for user in users:
            user_num = str(user)[7:]
            review_data = {}

            # Define xpath address
            xpath_country = '//*[@id="taplc_breadcrumb_desktop_0"]/div/div/ul/li[2]/a/span/text()'
            # xpath_city = '//*[@id="HEADING_GROUP"]/div/div[2]/div[2]/span/div/a/text()'
            xpath_city = '//*[@id="HEADING_GROUP"]/div/div[2]/div[2]/span/div/a/text()'
            xpath_attractionid = '//*[@id="HEADING"]'
            xpath_userlevel = './/div[contains(@class,"memberBadging")]//div[contains(@class, "levelBadge")]/@class'
            xpath_reviewtitle = '//*[@id="rn'+user_num+'"]/span/text()'
            # xpath_reviewtext = '//*[@id='+user+']/div/div[2]/div/div/div[3]/p/text()'
            xpath_reviewtext = '//*[@id='+str(user)+']/div/div[2]/div/div/div[3]/p/text()'


            review_data['country'] = sel.xpath(xpath_country).extract()[:1]
            review_data['city'] = sel.xpath(xpath_city).extract()
            review_data['attraction_id'] = fol
            review_data['user_id'] = user
            # review_data['user_level'] = sel.xpath(xpath_userlevel).extract()[0].split(' ')[-1][4:]
            review_data['review_title'] = sel.xpath(xpath_reviewtitle).extract()
            review_data['review_text'] = sel.xpath(xpath_reviewtext).extract()
            result.append(review_data)

    print(result)




