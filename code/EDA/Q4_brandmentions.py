# Databricks notebook source
#Question 4: How often are cosmetic providers mentioned in the general makeup channel and how does Glossier compare?
#Scope: Glossier Subreddit (Competitor_Comments and Competitor_Submissions)
#Cleaning Steps: Regular (above steps) 
#Visualization(s): Heat Map Or Tree Map 



# COMMAND ----------

# Step 1: read in the data 
comments = spark.read.parquet("/FileStore/glossier/competitor_comments")
submissions= spark.read.parquet("/FileStore/glossier/competitor_submissions")
#check that data is read in correctly 
#comments.show(5)         
#submissions.show(5)     
comments_count=comments.count()
submissions_count=submissions.count()
print("comments_count:",comments_count)
print("submissions_count:",submissions_count)

# COMMAND ----------

#Step 2: Data cleaning (specific subredit channels + cleaning )
#a) We are only interested in general makeup channels 
#in this case we are interested in 2 subreddits specifically "Makeup", "MakeupAddiction"
#So lets filter based on those subredits 
competitor_list = ["Makeup", "MakeupAddiction"]
competitor_comments = comments.filter(comments.subreddit.isin(competitor_list))
competitor_submissions = submissions.filter(submissions.subreddit.isin(competitor_list))
#b) Now that we have the correct subreddits we are going to use the body column to get our counts for glossier commpetitors including glossier 
#so keep only body column for comments 
#and title comlumn for submissions
competitor_comments_body=competitor_comments.select("body")
competitor_comments_body.show(1)
competitor_submissions_title=competitor_submissions.select("title")
competitor_submissions_title.show(1)
#Lets get a total count for these variables 
competitor_comments_body_count=competitor_comments_body.count()
print("competitor_comments_body_count:",competitor_comments_body_count)
competitor_submissions_title_count=competitor_submissions_title.count()
print("competitor_submissions_title_count",competitor_submissions_title_count)

# COMMAND ----------

#c) #check shape of the data frame and null values !
print("competitor_comments_body:")
(competitor_comments_body.count(), len(competitor_comments_body.columns))
print("competitor_submissions_title:")
(competitor_submissions_title.count(), len(competitor_submissions_title.columns))
#------
## Convert the data frame into a view to run SQL on 
competitor_comments_body.createOrReplaceTempView("c_c_body")
competitor_submissions_title.createOrReplaceTempView("c_s_title")


missing_competitor_comm = spark.sql("select \
                    sum(case when body is null then 1 else 0 end) as body_count \
                    from c_c_body")


missing_competitor_sub = spark.sql("select \
                    sum(case when title is null then 1 else 0 end) as title_count \
                    from c_s_title")

## Show and view the results 
missing_competitor_comm.show()
missing_competitor_sub.show()
#I have no missing values which is great !

# COMMAND ----------

from pyspark.sql.functions import col
#now we need to find how much their competitors are mentioned 
#we consider main competitors, Sephora, Ulta, Fenty, Milk 
main_competitors = ["Sephora", "Ulta","Fenty"] 
#Now we need to finf the rows that contain Sephora and keep them 
#competitor_comments_body_Sephora=competitor_comments_body.filter(col("body").rlike("Sephora"))
competitor_comments_body_brand=competitor_comments_body.withColumn("Sephora",col("body").rlike("Sephora|sephora"))
#Ulta
competitor_comments_body_brand2=competitor_comments_body_brand.withColumn("Ulta",col("body").rlike("Ulta|ulta"))
#competitor_comments_body_brand2.show()
competitor_comments_body_brand3=competitor_comments_body_brand2.withColumn("Fenty",col("body").rlike("Fenty|fenty"))
#competitor_comments_body_brand3.show()
competitor_comments_body_brand4=competitor_comments_body_brand3.withColumn("Glossier",col("body").rlike("Glossier|glossier"))
#competitor_comments_body_brand4.show()
count_Sephora=competitor_comments_body_brand4.groupBy('Sephora').count()
count_Sephora.show()
count_Ulta=competitor_comments_body_brand4.groupBy('Ulta').count()
count_Ulta.show()
count_Fenty=competitor_comments_body_brand4.groupBy('Fenty').count()
count_Fenty.show()
count_Glossier=competitor_comments_body_brand4.groupBy('Glossier').count()
count_Glossier.show()



# COMMAND ----------

#Now we need to do the same for submissions 
from pyspark.sql.functions import col
competitor_submissions_title2=competitor_submissions_title.withColumn("Sephora",col("title").rlike("Sephora|sephora"))
#Ulta
competitor_submissions_title3=competitor_submissions_title2.withColumn("Ulta",col("title").rlike("Ulta|ulta"))
competitor_submissions_title4=competitor_submissions_title3.withColumn("Fenty",col("title").rlike("Fenty|fenty"))
competitor_submissions_title5=competitor_submissions_title4.withColumn("Glossier",col("title").rlike("Glossier|glossier"))
count_Sephora_title=competitor_submissions_title5.groupBy('Sephora').count()
count_Sephora_title.show()
count_Ulta_title=competitor_submissions_title5.groupBy('Ulta').count()
count_Ulta_title.show()
count_Fenty_title=competitor_submissions_title5.groupBy('Fenty').count()
count_Fenty_title.show()
count_Glossier_title=competitor_submissions_title5.groupBy('Glossier').count()
count_Glossier_title.show()

# COMMAND ----------

#final transformation step: make into one dataframe 
#We can use pandas now 
import pandas as pd
#submissions
competitor_submissions_title6=competitor_submissions_title5.select('Sephora','Ulta','Fenty','Glossier')
#competitor_submissions_title6.show()
#comments
competitor_comments_body_brand5=competitor_comments_body_brand4.select('Sephora','Ulta','Fenty','Glossier')
#competitor_comments_body_brand5.show()
#Now I am going to group by columns 
group_cols = ["Sephora", "Ulta","Fenty","Glossier"]
competitor_submissions_title7=competitor_submissions_title6.groupBy(group_cols).count()
competitor_submissions_title7.show()
competitor_comments_body_brand6=competitor_comments_body_brand5.groupBy(group_cols).count()
competitor_comments_body_brand6.show()
#this is interesting but perhpas a bit complicated for the average consumer 
#however I am keeping these as 2 tables that could be interesting for some further analysis down the road 
#Save these dataframes 
import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
fpath = os.path.join(CSV_DIR, "summary_table_3.csv")
summary_3 =competitor_submissions_title7.toPandas()
summary_3.to_csv(fpath)
summary_4 =competitor_comments_body_brand6.toPandas()
fpath = os.path.join(CSV_DIR, "summary_table_4.csv")
summary_4.to_csv(fpath)


# COMMAND ----------

##Step 3: Prepping data for viz 
count_Sephora_True=count_Sephora.filter("Sephora==True")
count_Sephora_True2=count_Sephora_True.select("count")
#count_Sephora_True.show()
count_Ulta_True=count_Ulta.filter("Ulta==True")
count_Ulta_True2=count_Ulta_True.select("count")
#count_Ulta_True.show()
count_Fenty_True=count_Fenty.filter("Fenty==True")
count_Fenty_True2=count_Fenty_True.select("count")
#count_Fenty_True.show()
count_Glossier_True=count_Glossier.filter("Glossier==True")
count_Glossier_True2=count_Glossier_True.select("count")
#count_Glossier_True.show()

result1 = count_Sephora_True2.union(count_Ulta_True2)
result2 = result1.union(count_Fenty_True2)
result3 = result2.union(count_Glossier_True2)
result3 = result3.toPandas()

brand_names=["Sephora","Ulta","Fenty","Glossier"]
result3['brands']= brand_names
result3

# COMMAND ----------

#now for submissions 
count_Sephora_title_True=count_Sephora_title.filter("Sephora==True")
count_Sephora_title_True2=count_Sephora_title_True.select("count")
#count_Sephora_title_True.show()
count_Ulta_title_True=count_Ulta_title.filter("Ulta==True")
count_Ulta_title_True2=count_Ulta_title_True.select("count")
#count_Ulta_title_True.show()
count_Fenty_title_True=count_Fenty_title.filter("Fenty==True")
count_Fenty_title_True2=count_Fenty_title_True.select("count")
#count_Fenty_title_True.show()
count_Glossier_title_True=count_Glossier_title.filter("Glossier==True")
count_Glossier_title_True2=count_Glossier_title_True.select("count")
#count_Glossier_title_True.show()

result1_title = count_Sephora_title_True2.union(count_Ulta_title_True2)
result2_title = result1_title.union(count_Fenty_title_True2)
result3_title = result2_title.union(count_Glossier_title_True2)
result3_title= result3_title.toPandas()

brand_names=["Sephora","Ulta","Fenty","Glossier"]
result3_title['brands_title']= brand_names
result3_title['count_title']= result3_title['count']
result3_title=result3_title.drop(['count'], axis=1)
result3_title

# COMMAND ----------

#Now make these 2 data frames into one 
#result3
#result3_title
final_df_temp=pd.concat([result3, result3_title], axis=1)
#final_df_temp
final_df_temp['Total_Count'] = final_df_temp['count'] + final_df_temp['count_title']
#final_df_temp
final_df=final_df_temp.drop(['count','brands_title','count_title'], axis=1)
final_df
#save dataframe to cvs
fpath = os.path.join(CSV_DIR, "viz_3.csv")
final_df.to_csv(fpath)

# COMMAND ----------

!pip install pywaffle

# COMMAND ----------

import matplotlib.pyplot as plt
from pywaffle import Waffle

df=final_df
#make the valus from the dataframe into a list 
value_list = df["Total_Count"].tolist()
#print(value_list)

#get the total count of  values, meaning comment that contain, sephora, fenty, ulta or glossier 
sum_value_list=sum(value_list)
value_list2 = [round(x / sum_value_list*100) for x in value_list]
print(value_list2)




# COMMAND ----------

import matplotlib.pyplot as plt
from pywaffle import Waffle
data = {'Sephora': 34, 'Ulta': 25, 'Fenty': 28, 'Glossier':13}
fig = plt.figure(
    FigureClass=Waffle, 
    rows=10, 
    values=data, 
    colors=("#00008B", "#0000FF","#5C5CFF","#FFB6C1"),
    title={'label': 'Percentage of Brand Mentions in the Makeup \nand MakeupAddiction subreddits from Jan 21 - Aug 22', 'loc': 'left',
            'fontdict': {
            'fontsize': 14
        }},
    labels=["{0} ({1}%)".format(k, v) for k, v in data.items()],
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1),
           'fontsize': 12},
    icons='person', icon_size=18, 
    icon_legend=True
)
fig.gca().set_facecolor('white')
fig.set_facecolor('white')
plot_fpath = os.path.join(PLOT_DIR, 'viz_3.png')
plt.savefig(plot_fpath)
plt.show()
