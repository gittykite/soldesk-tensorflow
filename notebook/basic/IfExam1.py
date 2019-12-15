
# coding: utf-8

# ## [실습 1] 하나의 수를 입력받아 2, 3, 4, 5의 배수인지 판단하는 프로그램을 제작하세요. 
# 
# ### console에서의 실행

# In[1]:


import sys

filename = sys.argv[0]
print('filen ame:' + filename)  # python IfExam1.py

var0 = sys.argv[0] # 입력한 파일명
var1 = sys.argv[1] # 입력한 수
var2 = int(var1)


# In[3]:


if var2 % 2 == 0:
    print(var1 + '은/는 2의 배수입니다')
elif var2 % 3 == 0:
    print(var1 + '은/는 3의 배수입니다')    
elif var2 % 4 == 0:
    print(var1 + '은/는 4의 배수입니다')    
elif var2 % 5 == 0:
    print(var1 + '은/는 5의 배수입니다')        
else:
    print(var1 + '은/는 2,3,4,5의 배수가 아닙니다')
    

