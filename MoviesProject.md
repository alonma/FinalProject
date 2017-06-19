
# Movies Data Project

## Miri Hazanov, Alon Maharshak

### First step-collect and preapre the data



#### During this project we will work with the "Cornell Movie-Dialogs Corpus" created by the Cornell University

Our first step is to understand the data.

* Import the relevent Packeges 
* Open the movie titles files and the movie lines file.


```python
import pandas as pd
import nltk
import re
import string 

fileName='C:/Users/Alon/Desktop/Alon/School/4th year/SemesterB/Data science/Project/Data/cornell movie-dialogs corpus/movie_lines.txt'
with open(fileName) as f:
    lines = f.readlines()
fileName='C:/Users/Alon/Desktop/Alon/School/4th year/SemesterB/Data science/Project/Data/cornell movie-dialogs corpus/movie_titles_metadata.txt'
with open(fileName) as f:
    titles = f.readlines()
```

Now lets print an exapmle so we can see how our meta data and raw data looks like


```python
print(titles[0])
print(lines[0])
```

    m0 +++$+++ 10 things i hate about you +++$+++ 1999 +++$+++ 6.90 +++$+++ 62847 +++$+++ ['comedy', 'romance']
    
    L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
    
    

As we can see, the data has a '+++$+++' seperator, so now we can start work on the data.

Our stages we'll be:
* Create a list of each genre using the generes list in the titles file
* Get the title with the most lines in each genre so well have a lot of lines to work with
* Clean the data, getting only the movie line after cleaning of lower case and other unnecessary marks.
* Create a csv file with the Line, movie name and movie genre.

### Create list function:
returns a list of all the movie id's that are relevent to a selected genre.


```python
def createList(genre):
    ans=[]
    for title in titles:
        att=title.split('+++$+++')
        if genre in att[5]:
            ans.append(att[0])
    return ans
```

### Get the movie with the most lines function


```python
def getMaxLines(movielist):
    x=0
    maxMovie=''
    for movie in movielist:
        movieline='+++$+++ '+movie+'+++$+++'
        movieLines=[line for line in lines if movieline in line]
        if len(movieLines)>x:
            x=len(movieLines)
            maxMovie=movie
    return maxMovie
```

### Clean the movie data function

* takes the raw line, strips only the movie line.
* remove from the movie line unnecessary marks and swtich to lower case 
* strip the movie name 

Returns a tuple of the movie name + a list of all the movie lines(clean)


```python
def cleanMovie(movie):
    num=int(movie[1:])
    movieLinesRaw=[line for line in lines if '+++$+++ '+ movie+'+++$+++' in line]
    MovieLinesClean=[]
    results=[]
    for line in movieLinesRaw:
         att=line.split('+++$+++')
         MovieLinesClean.append(att[4])
    returnLines=[]
    att=titles[num].split('+++$+++')
    results.append(att[1])
    for line in MovieLinesClean:
        line=line.lower()
        for char in string.punctuation:
            if char!="'":
                line = line.replace(char, ' ')
        line=' '.join(line.split())
        returnLines.append(line)
    
    results.append(returnLines)
    return results
```

### Create CSV files function:


```python
def makeCsv(movie,name,genre):
    df=pd.DataFrame()
    df['lines']=movie
    df['name']=name
    df['type']=genre
    df.to_csv('C:/Users/Alon/Desktop/Alon/School/4th year/SemesterB/Data science/Project/Data/'+genre+'Lines.csv')
    return df
```

Now we can activate all of the stages for all the 3 genres and concatene them into one csv file:


```python
adventureList=createList('adventure') 
dramaList=createList('drama') 
comedyList=createList('comedy') 

maxAdventure=getMaxLines(adventureList)
maxDrama=getMaxLines(dramaList)
maxComedy=getMaxLines(comedyList)

movieDrama=cleanMovie(maxDrama)
movieComedy=cleanMovie(maxComedy)
movieAdventure=cleanMovie(maxAdventure)

dramaDf=makeCsv(movieDrama[1],movieDrama[0],'Drama')
comedyDf=makeCsv(movieComedy[1],movieComedy[0],'Comedy')
AdventureDf=makeCsv(movieAdventure[1],movieAdventure[0],'Adventure')

result = pd.concat([dramaDf,comedyDf,AdventureDf])
result.to_csv('C:/Users/Alon/Desktop/Alon/School/4th year/SemesterB/Data science/Project/Data/MoviesLines.csv')
```
