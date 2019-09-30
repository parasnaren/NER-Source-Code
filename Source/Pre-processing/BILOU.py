# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 01:37:38 2019

@author: SHREYAS 
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 01:11:38 2019

@author: SHREYAS 
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 23:36:37 2019

@author: SHREYAS 
"""

import pandas as pd
import numpy as np
df=pd.read_csv('ontonotes_processed.csv',engine='python')
d=df['10'].astype(str)
for i in range(0,len(d)):
    k=0
    if d[i][k]=='(':
        s=""
        while(d[i][k]!='*'):
            
            s=s+d[i][k]
            if(k==len(d[i])-1):
                
                d[i]='U'+'-'+s[1:-1]
             
            
                break
            k=k+1
for i in range(0,len(d)):
    k=0            
    if d[i][k]=='(':
        s1=d[i][1:-1]
        i=i+1
        
        while(d[i][k]!=')'):
            if(d[i]=='*'):
                
                d[i]='I'+'-'+s1
                
                i=i+1
                k=0
                continue
            else:
                
                d[i]='L'+'-'+s1
                
                i=i+1
                break
            k=k+1
            
for i in range(0,len(d)):
    k=0            
    if d[i][k]=='(':
        s=""
        while(d[i][k]!='*'):
            
            s=s+d[i][k]
            k=k+1
            
        d[i]='B'+'-'+s[1:]
        
for i in range(0,len(d)):
    k=0            
    if d[i]=='*':
        d[i]='O'
        
df.insert(12, "11", d , True)
df['11']
df.to_csv('Ontonotes_final.csv',encoding='utf-8')

                