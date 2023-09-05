# load libraries
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pysenti
import statistics

date_start = 1, 1, 2020
date_end = 1, 1, 2020



def transform_date(date_day, date_month, date_year):
    date_day_as_text = str(int(date_day)) if int(date_day) > 10 else '0' + str(int(date_day))
    date_month_as_text = str(int(date_month)) if int(date_month) > 10 else '0' + str(int(date_month))
    date_year_as_text = str(int(date_year))
    return date_day_as_text, date_month_as_text, date_year_as_text


def get_period_score(day_start, day_end, month_start, month_end, year_start, year_end):

    date_start = transform_date(day_start, month_start, year_start)
    date_end = transform_date(day_end, month_end, year_end)

    url = 'https://www.upstreamonline.com/archive?sort=reld&publishdate='+date_start[0]+'.'+date_start[1]+'.'+date_start[2]+'-'+date_end[0]+'.'+date_end[1]+'.'+date_end[2]

    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    
    if len(soup.select('.card-link.text-reset')) == 0:
        return {
            'negative_mean': 0,
            'positive_mean': 0,
            'neutral_mean': 0,
            'negative_std': 0,
            'positive_std': 0,
            'neutral_std': 0,
        }
    
    corpus = [s.text for s in soup.select('.card-link.text-reset')]
    strengths_negative = []
    strengths_positive = []
    strengths_neutral = []
    for c in corpus:
        try:
            sentiment = pysenti.get_senti(c)
            strengths_positive.append(sentiment[0])
            strengths_negative.append(sentiment[1])
            strengths_neutral.append(sentiment[2])
        except:
              continue


    if len(corpus) == 1:
            return {
            'negative_mean': statistics.mean(strengths_negative),
            'positive_mean': statistics.mean(strengths_positive),
            'neutral_mean': statistics.mean(strengths_neutral),
            'negative_std': 0,
            'positive_std': 0,
            'neutral_std': 0,
        }

    return {
        'negative_mean': statistics.mean(strengths_negative),
        'positive_mean': statistics.mean(strengths_positive),
        'neutral_mean': statistics.mean(strengths_neutral),
        'negative_std': statistics.stdev(strengths_negative),
        'positive_std': statistics.stdev(strengths_positive),
        'neutral_std': statistics.stdev(strengths_neutral),
    }


def prepare_scores(input_file: str, output_file: str):
    file = open(input_file,'r')
    df = pd.read_csv(file)
    df[["day", "month", "year"]] = df["Date"].str.split("/", expand = True)
    df['day'] = df['day'].astype(float)
    df['month'] = df['month'].astype(float)
    df['year'] = df['year'].astype(float)

    df_scores = pd.DataFrame(columns=['positive_mean', 'positive_std', 'negative_mean', 'negative_std', 'neutral_mean', 'neutral_std'])


    for index, row in df.iterrows():
        print('processing ', row['day'], '-', row['month'], '-', row['year'])
        scores = get_period_score(row['day'],row['day'],row['month'],row['month'],row['year'],row['year'])
        df_scores[index] = scores
    
    file.close()
    save_file = open(output_file,'w')
    df_scores.to_csv(save_file)
    save_file.close()

print(get_period_score(date_start[0], date_end[0], date_start[1], date_end[1], date_start[2], date_end[2]))

prepare_scores('Avg_data.csv', 'Sentiment_data.csv')