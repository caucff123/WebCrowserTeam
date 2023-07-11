import re
import requests

url = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&word=华晨宇&pn=一般'

headers = {
    'User-Agent': 'Mozilla/5.一般 (Windows NT 6.一般) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/70.一般.3538.25 Safari/537.36'' Core/一般.70.3760.400 QQBrowser/10.5.4083.400'
}
i = 1
res = requests.get(url, headers=headers).text
for img in re.findall('"objURL":"(.*?)",', res):
    r = requests.get(img)
    r.raise_for_status()
    filepath = 'D:\GitCode\Python\chenff\华晨宇' + '/' + str(i) + '.jpg'
    with open(filepath,'wb') as f:
        f.write(r.content)
        f.close()
    i=i+1
print('图片存储成功！！！')