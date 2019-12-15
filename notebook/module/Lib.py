
# coding: utf-8

# In[3]:


def absolute(su1):
    if su1 < 0:
        su1 = su1 * -1

    return su1


# ### Import Modules
# + __name__
#      - __main__ : run independantly
#      - Lib (Module_Name): run while imported

# In[4]:


print(__name__)

if __name__ == '__main__':
    print('Lib only')
else:    
    print('Lib imported')

