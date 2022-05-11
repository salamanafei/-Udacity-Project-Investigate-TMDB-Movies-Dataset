#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate The [TMDb movie data]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# >This movie database contains information about approximately 10,000 movies including genres, ratings, revenue, budget, and more. It contains movies which are released over 56 years between 1960 and 2015, it also has two columns for budget and revenue in terms of 2010 dollars accounting for inflation over time which will be used in any comparisons in my analysis instead of unadjusted ones. 
# ##### contain:
# <ul>
#     <li>Total Rows = 10866</li>
#     <li>Total Columns = 21</li>
#     <li>After Seeing the dataset we can say that some columns is contain null values</li>
# </ul>
# 
# 
# ### Question(s) for Analysis
# <ol>
#     <li>Which movie title had the longest run time?</li>
#     <li>Is there a relation between popularity and revenue ?</li>
#     <li>Which Genre Has The Highest Release Of Movies?</li>
#     <li>Which year has the heighest release of movies?</li>
#     <li>Top ten movies in terms of revenues.</li>
#     <li>Average Revenue of the movies</li>
#     <li>Top ten movies in terms of budget.</li>
#     <li>Average Budget of the movies</li>
#     <li>Top ten movies in terms of popularity.</li>
#     <li>highest profit movies</li>
#     <li>Most Frequent Cast</li>
# </ol>
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


# Upgrade pandas to use dataframe.explode() function. 
 


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# ### General Properties

# In[2]:


#Loading Data
df = pd.read_csv('tmdb-movies.csv' , index_col="id")
df.head()


# In[3]:


#Exploring the shape of data
df.shape


# #### this data consist of (10866) rows, and (21) columns
# #### We won't need all these columns in our investigation so we are going to drop some of them in later steps.The columns we need are: [popularity, budget, revenue, original_title, cast, director, keywords, runtime, genres, production_companies, release_date, vote_count, vote_average, release_year.]</p>

# In[4]:


#check for duplications
df.duplicated().sum()


# #### there's onley 1 duplicated row

# ### Let's see some summary statistics about the data

# In[5]:


df.describe()


# In[6]:


#inspecting for missing values to solve it
df.info()


# #### Most columns are represented by appropriate data types except release date column so, i'll change it from String to DateTime later during the cleaning process

# ### Let's check if there's null values or not

# In[7]:


#checking for null values
df.isnull().sum()


# #### there are null values in some columns so, we will investigate those null values and look for ways to elimenate them.

# 
# ### Data Cleaning
# #### in this process we need to:
# >
# <ul>
#     <li>Remove duplicate rows from the dataset.</li>
#     <li>Change format of release date to datetime format.</li>
#     <li>Remove the unused columns that we don't need in the analysis process.</li>
#     <li>Remove the movies which are having zero valuse of budget and revenue columns.</li>
# </ul>
# 
# 
# 

# #### 1: Removing duplicate rows

# In[9]:


#removing duplicated rows
df.drop_duplicates(inplace = True)


# In[10]:


#one more check for duplicates rows
df.duplicated().sum()


# #### 2: Removing unused columns

# In[11]:


#removing unneccesesary data
df.drop(['budget_adj','revenue_adj','overview','imdb_id','homepage','tagline'],axis = 1 , inplace = True)


# In[12]:


df.shape


# #### After dropping unneccesesary columns, now we have (10865) rows and (14) columns

# #### 3: Drop theses rows which contain incorrect or inappropriate values. 

# In[13]:


#drop NAN values
df.dropna(inplace = True)


# In[14]:


#Checking for zero values in revenue and budget columns
len(df.query('revenue == "0"')),len(df.query('budget == "0"'))


# #### Row with zero values in revenue column : (4130) and in revenue column : (3940)

# #### Since there is a lot of zero values in revenue and budget column, calculating the profits of these movies would lead to inappropriate results, So i think the best option here is to drop them all 

# In[15]:


#Droping zero values from revenue and budget column
zero_values_rev = df[df['revenue'] == 0].index
df.drop(zero_values_rev , inplace = True , axis = 0)


# In[16]:


#get rows with zero values in budget column and drop them all
zero_values_bud = df[df['budget'] == 0].index
df.drop(zero_values_bud , inplace = True , axis = 0)


# In[17]:


df.info()


# >now there is no NAN values in this dataset

# #### 4: Change format of release date column to datetime format

# In[18]:


df['release_date'] = pd.to_datetime(df['release_date'])
# Verifying successful type_change 
print(df['release_date'].dtypes)


# #### 5: splitting geners column into multiple rows in a seperate data frame so that we don't duplicate all values with it in the original data frame.
# #### Then create a seperate dataframe from unique geners records and rotate it

# In[19]:


genres_df = df['genres'].str.split("|" , expand = True)
genres_df = genres_df.stack()
genres_df = pd.DataFrame(genres_df)


# In[258]:


# # Verifying successful separation 
genres_df.head()


# In[259]:


#Renaming the genres column and verifying the genres value count
genres_df.rename(columns = {0:'genres_adj'} , inplace = True)
genres_df.head()


# #### merging geners_df with the original datafeame

# In[260]:


merged_df = df.merge(genres_df, left_index = True, right_index = True)


# In[261]:


#check the merged dataframe
merged_df.head()


# #### droping the original genres column

# In[262]:


merged_df.drop('genres' , inplace = True , axis = 1)


# In[263]:


merged_df.head()


# In[ ]:


new_df


# # <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 

# ### General look

# In[264]:


merged_df.hist(figsize = (20,10.5));


# #### Let's make some functions to help us during the visualization process

# In[265]:


def get_max(column_name):
    """This function will help me get the name and its maximum value"""
    column_value = merged_df[column_name].max()
    movie_name = merged_df[merged_df[column_name] == column_value]['original_title'][0]
    return movie_name , column_value


# In[266]:


def figuer_labels(title , x ,y):
    """used for the figuer title and labels"""
    plt.title(title,fontsize=15)
    plt.xlabel(x,fontsize=15)
    plt.ylabel(y,fontsize= 15);
    return title,x,y


# In[267]:


def top_ten_plot_fun(column1 , column2 ,title , xlabel, ylabel):
    """used to get top 10 movie names in a specefic category in a plot figure"""
    info = pd.DataFrame(merged_df[column1])
    info[column2] = merged_df[column2]
    info_group = info.groupby(column2)[column1].sum().sort_values(ascending = False)[:10]
    info_group.plot.barh(color = 'green' , figsize = (10,5) , fontsize = 10)
    figuer_labels(title,xlabel,ylabel)
    return column1, column2, title, xlabel, ylabel


# ## Q1: Which movie title had the longest run time?

# In[268]:


#get movie that has the longest run time
get_max('runtime')


# #### Carlos is the longest runtime movie with (338 h)

# ## Q2: Is there a relation between popularity and revenue ?

# In[269]:


def box_polt(column_name):
    """This function will help the destripuation between revenue and popularity"""
    merged_df.boxplot(column_name , vert = False , showfliers = False)


# In[270]:


box_polt('revenue')


# In[271]:


box_polt('popularity')


# #### Let's see the relation between Popularity and Revenue

# In[272]:


#plot a 'scatter' plot to get the relation between popularity and revenue.
plt.scatter(data = df, x = 'popularity', y = 'revenue')
plt.xlim(-.5, 15)
plt.ylim(-119075292.35, 1.5e9)
figuer_labels('Popularity VS Revenue', 'Popularity', 'Revenue');


# #### here we can find that Revenue increases with increase in Popularity. So we can conclude that there's a positive corelation between Popularity and Revenue

# ## Q3: Which Genre Has The Highest Release Of Movies?

# In[273]:


#count each of the gener
info = merged_df.genres_adj.value_counts()
#plot a 'barh' plot using plot function for 'genre vs number of movies'.
info.plot.barh(color = 'green' , figsize = (15,10) , fontsize = 10)
figuer_labels("Genre With Highest Release", "Number Of Movies" , "Genres");


# #### According to the plot Drama has the highest release of movies followed by Comedy and Thriller.

# ## Q4: Which year has the heighest release of movies?

# In[274]:


info = merged_df.groupby('release_year').popularity.count()
info.tail()


# In[275]:


# make group for each year and count the number of movies in each year 
info = merged_df.groupby('release_year').count()
info.plot(color = 'green' , figsize = (25,10) , fontsize = 15,xticks = np.arange(1960,2016 , 3))
figuer_labels("Number Of Movies every 5 Years", "Release Year" , "Number Of Movies");


# #### here we can conclude that year 2011 has the highest release of movies (485) then 2013 (399) and year 2015 (391).

# ## Q5: Top ten movies in terms of revenues.

# In[54]:


get_name_value('revenue')


# In[82]:


#top 10 movies which made highest revenue.
                # column1    #column2               # title                   xlabel     ylabel
top_ten_plot_fun('revenue', 'original_title', 'Top 10 High Revenue Movies', 'Revenue', 'Movie Names');


# #### As we can see that 'Avatar' movie has the highest profit in all, making over 2.7B in profit in this dataset.

# ## Q6: Top ten movies in terms of budget.

# In[277]:


get_name_value('budget')


# In[85]:


#top 10 movies in terms of budget.
                # column1    #column2               # title               xlabel     ylabel
top_ten_plot_fun('budget', 'original_title', 'Top 10 High Budget Movies', 'Budget', 'Movie Names');


# #### As we can see that 'The Warrior's Way' movie has the highest budget in all, over 4.2B in this dataset.

# ## Q7: Top ten movies in terms of popularity.

# In[66]:


get_name_value('popularity')


# In[97]:


# top 10 movies in terms of popularity.
top_ten_plot_fun('popularity', 'original_title', "Top 10 Popular Movies", "popularity" , "Movie Name");


# #### As we can see that 'Jurassic World' movie is the most popular movie in this dataset.

# #### Function to get the average

# In[178]:


def get_avg(column_name):
    """will return the average of a specefic column"""
    return merged_df[column_name].mean().round()


# ## Q8: Average Budget of the movies

# In[179]:


#get the average of the budget
get_avg('budget')


# ## Q9: Average Revenue of the movies

# In[180]:


#get the average of the revenue
get_avg('revenue')


# ## Q10: Average Runtime movies

# In[181]:


#get the average runtime
get_avg('runtime')


# ## Q11: Most Frequent Cast

# ####  splitting cast column into multiple rows in a seperate data frame then create a seperate dataframe from unique geners records and rotate it

# In[237]:


cast = merged_df['cast'].str.split("|" , expand = True)
cast = cast.stack()
cast = pd.Series(cast)


# In[239]:


cast.value_counts().head()


# <a id='conclusions'></a>
# ## Conclusions
# > In this project, we started our analysis by examining the heighst release of movies regarding the gener, we notice the Drama movies are the most popular movies gener, the examined the movie popularty year by year, we notice that 2011 has the heighst release of movies. finally there were a positive corelation between popularity and revenue
# <ul>
#     <li>Drama is the most popular genre, following by action, comedy and thriller. so, my recomended Gener is: Drama, Thriller, Action, Comedy</li>
#     <li>Average duration of the movie should be around 110 min.</li>
#     <li>Average budget should be around 43M or above.</li>
#     <li>Recomnded cast: "Bruce Willis", "Nicolas Cage", "Samuel L. Jackson", "Robert De Niro", "Eddie Murphy".</li>
#     <li>Higher popularity leads to higher profits</li>
# </ul>
# 
# ## Limitations
# 
# <ul>
#     <li>During the analysis process the columns (revenue and budget) contain many missing values which've been dropped. This seems not the best way to fix those columns but was the best way to deal with these missing values at least in my prespective of view</li>
# </ul>
# 
# ## Submitting your Project 
# 
# > **Tip**: Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > **Tip**: Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > **Tip**: Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[279]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




