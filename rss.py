import feedparser

NewsFeed = feedparser.parse("https://energetyka24.com/_rss")

entry = NewsFeed.entries[1]

print(entry.published)
print("******")
print(entry.summary)
print("------News Link--------")
print(entry.link)