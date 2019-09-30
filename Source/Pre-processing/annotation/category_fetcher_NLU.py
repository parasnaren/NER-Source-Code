from __future__ import print_function
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions
import pandas as pd
import time
from newspaper import Article
import newspaper
from googlesearch import search
import requests
from sklearn.model_selection import train_test_split as tts

"""service = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api',
    iam_apikey='CiZe9FtA-2B8MagnyvbFK6ashdfCJZOBpH-VHqUM4IGQ')

response = service.analyze(
    text='Status quo will not be disturbed at Ayodhya; says Vajpayee',
    features=Features(categories = CategoriesOptions(
            model='8134b0aa-f11b-4697-acf9-bc8e48f36ae2',
            limit=3))).get_result()

print(json.dumps(response, indent=2))"""



def conll_to_text(filename):
    """
    return the list of text present in the conll file
    """
    df = pd.read_csv(filename).iloc[:, [3, 4]]
    text = []
    sent = ""
    for i in range(len(df)):        
        if(i != 0 and df.at[i, 'col2'] == 0):
            text.append(sent)
            sent = ""
        if(i == len(df)-1):
            text.append(sent)
            sent = ""
            
        sent += df.at[i, 'col3']
        if(i < len(df)-1 and "\'" in df.at[i+1, 'col3']):
            sent += ""
        elif(i < len(df)-1 and "." in df.at[i+1, 'col3']):
            sent += ""
        else:
            sent += " "
        
    return text
        
def annotate_category_for_text(text):
    """
    text: list of strings
    yield label for each string annotated
    """
    service = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api',
    iam_apikey='CiZe9FtA-2B8MagnyvbFK6ashdfCJZOBpH-VHqUM4IGQ')
        
    for i, sentence in enumerate(text):
        if(len(sentence.split()) > 5):
            labels = []
            response = service.analyze(
                text = sentence,
                features=Features(categories = CategoriesOptions(
                        model='1edd4142-422a-47d2-918d-de51929ff607',
                        limit=3))).get_result()
            
            for i in range(len(response['categories'])):
                labels.append((response['categories'][i]['label'],
                               response['categories'][i]['score']))
        
            yield labels
            
        else:
            yield [('unknown', 0)]
        
        print(i)
        
        
def get_changed_category(category):
    """
    convert the category into the type required in dataset
    """
    
    if any(book in category for book in ['books','book']):
        return 'books'
    elif 'south-asia' in category:
        return 'south-asia'
    elif 'video' in category:
        return 'video'
    elif any(tech in category for tech in ['tech','gadgets']):
        return 'tech'    
    elif 'science' in category:
        return 'science'
    elif 'health' in category:
        return 'health'
    elif any(sport in category for sport in ['sports','cricket']):
        return 'sports'
    elif 'environment' in category:
        return 'environment'
    elif 'woman' in category:
        return 'woman'
    elif 'election' in category:
        return 'politics, government'
    elif any(film in category for film in ['movie', 'bollywood', 'hollywood', 'film']):
        return 'film'
    elif 'education' in category:
        return 'education'
    elif 'tv.news' in category:
        return 'media'
    elif 'world' in category:
        return 'world'
    elif 'business' in category:
        return 'business, economy, trade'
    else:
        return 'none'
    
    
def create_dataset(filename):
    """
    takes the india-news-headlines.csv as input and creates
    a dataset with the categories that are required
    """
    
    cols = ['id', 'category', 'headline']
    news = pd.read_csv('data/news-files/india-news-headlines.csv')
    news.columns = cols
    
    
    dummy = news.copy()
    dummy['category'] = dummy['category'].apply(lambda x: get_changed_category(x))
    dummy = dummy.where(dummy['category'] != 'none').dropna()
    
    #category = pd.get_dummies(dummy['category'])
    dummy['id'] = dummy.index
    #dummy.drop('category', axis=1, inplace=True)
    
    #dummy = pd.concat([dummy, category], axis=1)
    
    categories = dummy['category'].value_counts()
    df = pd.DataFrame()        
    
    for group, frame in dummy.groupby('category'):        
        df = pd.concat([df, frame.sample(500)], axis=0)
    
        
    cat = pd.get_dummies(df['category'])
    df.drop('category', axis=1, inplace=True)
    df = pd.concat([df, cat], axis=1)
    df.to_csv('./data/temp-files/dummy-mini.csv', index=False)    
    #dummy.to_csv('./data/temp-files/dummy.csv', index=False)


dummy.columns


def a(hello):
    
    def b(what, you, mean):
        print(hello)
    
    return b

a('paras')

import os
train_file = os.path.join('./working', '/train.tf_record')


category = 'space'
url = ('https://newsapi.org/v2/everything?'
           'language=en&'
           'q=space&'
           'pageSize=50&'
           'page=1&'
           'apiKey=e24807bf80314286b613c54d3f8fb214')


def scrapping_news_headlines(category):
    titles = []
    i = 10
    while i < 31:
        url = "https://newsapi.org/v2/everything?q={}&pageSize=100&from=2019-07-{}&to=2019-07-{}&apiKey=e24807bf80314286b613c54d3f8fb214".format(category, i, i+1)
        response = requests.get(url).json()
        news = response['articles']
        for row in news:
            titles.append(row['title'])
        yield titles
        i+=1
        print(i, ": {}".format(url))
        
                    
s = []
for titles in scrapping_news_headlines('space'):
    s += titles
    
    
    
    link = "https://www.nbcnews.com/science/space"
    paper = newspaper.build(link)
    len(paper.articles)
    titles = []
    for article in paper.articles:
        print(article.url)
        article.download()
        article.parse()
        titles.append(article.title.strip())
        #titles.append(article.title.strip().replace('- SpaceNews.com', ''))
        
        
    #space = []
    #space += titles
    #pd.DataFrame(space).to_csv('data/category-news/space.csv', index=False)
    
    titles = []
    category = "news space"
    count = 0
    for url in search(category, lang='en', num=5):
        link = url        
        print(link)
        try:
            article = Article(link, language="en")
            article.download()
            article.parse()
            count+=1            
            titles.append(article.title.strip())
            print(count)
            if count == 3:
                break
        except Exception as e:
            continue
    
    
    
    
    
    
    
    
    
    
    
    
    


dummy = news[1000:10000:2].copy()
dummy['category'].value_counts()
start_time = time.time()
dummy['category'] = dummy['category'].apply(lambda x: get_changed_category(x))
print("--- %s seconds ---" % (time.time() - start_time))


dummy = news.copy()
dummy['category'] = dummy['category'].apply(lambda x: get_changed_category(x))
dummy['category'].value_counts()

dummy[dummy['category'] == 'film']

###############################################################################
cols = ['data','category','headline']
news = pd.read_csv('data/india-news-headlines.csv')
news.columns = cols
categories = pd.DataFrame(news['category'].value_counts())

category = 'life-style.relationships.work'
news[news['category'] == category]

actual = news[:100]
headlines = list(news['headline'][:100])

labels = []
for label in annotate_category_for_text(headlines):
    labels.append(label)
    
    
