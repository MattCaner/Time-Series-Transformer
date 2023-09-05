# load libraries
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pysenti

date_start = 1, 1, 2020
date_end = 11, 1, 2021


def transform_date(date_day, date_month, date_year):
    date_day_as_text = str(int(date_day)) if int(date_day) > 10 else '0' + str(int(date_day))
    date_month_as_text = str(int(date_month)) if int(date_month) > 10 else '0' + str(int(date_month))
    date_year_as_text = str(int(date_year))
    return date_day_as_text, date_month_as_text, date_year_as_text

def get_tfidf_fit(day_start, day_end, month_start, month_end, year_start, year_end):

    date_start = transform_date(day_start, month_start, year_start)
    date_end = transform_date(day_end, month_end, year_end)

    url = 'https://www.upstreamonline.com/archive?sort=reld&publishdate='+date_start[0]+'.'+date_start[1]+'.'+date_start[2]+'-'+date_end[0]+'.'+date_end[1]+'.'+date_end[2]

    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')

    corpus = [s.text for s in soup.select('.card-link.text-reset')]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

    return vectorizer

def get_tfidf_transform(vectorizer, day_start, day_end, month_start, month_end, year_start, year_end):

    date_start = transform_date(day_start, month_start, year_start)
    date_end = transform_date(day_end, month_end, year_end)

    url = 'https://www.upstreamonline.com/archive?sort=reld&publishdate='+date_start[0]+'.'+date_start[1]+'.'+date_start[2]+'-'+date_end[0]+'.'+date_end[1]+'.'+date_end[2]

    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')

    corpus = [s.text for s in soup.select('.card-link.text-reset')]
    return vectorizer.transform(corpus)


vectorizer = get_tfidf_fit(date_start[0], date_end[0], date_start[1], date_end[1], date_start[2], date_start[2])

transform = get_tfidf_transform(vectorizer, date_end[0], date_end[0],date_end[1], date_end[1], date_end[2], date_end[2])

print(transform)