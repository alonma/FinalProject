# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import nltk
import re
import string 

fileName='C:/Users/amaharsh/Desktop/Project/Data/cornell movie-dialogs corpus/movie_lines.txt'
with open(fileName) as f:
    lines = f.readlines()
fileName='C:/Users/amaharsh/Desktop/Project/Data/cornell movie-dialogs corpus/movie_titles_metadata.txt'
with open(fileName) as f:
    titles = f.readlines()
    
    
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

def makeCsv(movie,name,genre):
    df=pd.DataFrame()
    df['lines']=movie
    df['name']=name
    df['type']=genre
    df.to_csv('C:/Users/amaharsh/Desktop/Project/Data/'+genre+'Lines.csv')
    return df

def createList(genre):
    ans=[]
    for title in titles:
        att=title.split('+++$+++')
        if genre in att[5]:
            ans.append(att[0])
    return ans
    
         
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
result.to_csv('C:/Users/amaharsh/Desktop/Project/Data/Lines.csv')
