from bs4 import BeautifulSoup
import urllib.request
import urllib.error
import os
import time


def save_img(star_name, img_url):
    pic_name = os.path.join((os.path.join(os.getcwd(), 'photos')), star_name + '.jpg')
    try:
        urllib.request.urlretrieve(img_url, pic_name)
        print("Success to save: " + pic_name)
    except urllib.error.HTTPError as e:
        print(e.code)
        print(e.reaoson)
    except urllib.error.URLError as e:
        print(e.reason)


def get_star_img_url(star_name, star_url):
    html_doc = urllib.request.urlopen(star_url).read().decode('utf-8')
    soup = BeautifulSoup(html_doc, "lxml")
    star_img_info = soup.find_all('img', class_='star-img')
    star_img_url = star_img_info[0].get('src')

    return star_name, star_img_url


def get_star_img(url):
    html_doc = urllib.request.urlopen(url).read().decode('utf-8')
    soup = BeautifulSoup(html_doc, "lxml")
    all_star_info = soup.find_all('ul', class_='zm-select-mx cl')
    star_list = all_star_info[0].find_all('li')


    for star in star_list:
        star_name = star.find('a').get_text()
        star_url = star.find('a').get('href')
        try:
            star_name, star_img_url = get_star_img_url(star_name, star_url)
            print(star_name + ':' + star_img_url)
            save_img(star_name, star_img_url)
        except urllib.error.HTTPError as e:
            pass
        except urllib.error.URLError as e:
            pass
        time.sleep(0.5)


def get_all_star_img(n):
    for i in range(0, n):
        print('当前页数: ' + str(i + 1))
        url = 'http://www.365j.com/star/list-dalu-2--' + str(i + 1) + '.html'
        get_star_img(url)


get_all_star_img(33)