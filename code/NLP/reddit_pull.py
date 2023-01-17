import praw
import csv
import pandas as pd
reddit = praw.Reddit(
    client_id=",
    client_secret="",
    password="",
    user_agent="redditdev scraper by u/VirtualFlow",
    username="VirtualFlow",
)
submission = reddit.submission("yevn6s")
print(submission.title)
#all_comments = submission.comments.body.list()
from praw.models import MoreComments
all_coms = []
for top_level_comment in submission.comments:
    if isinstance(top_level_comment, MoreComments):
        continue
    all_coms.append(top_level_comment.body)

#with open('glossier_sub.csv','w',encoding='UTF8', newline='') as f:
#	writer = csv.writer(f)
#	writer.writerow(all_comments)
print(all_coms)
df = pd.DataFrame(all_coms) 
    
# saving the dataframe 
df.to_csv('glossier_sub1.csv') 
