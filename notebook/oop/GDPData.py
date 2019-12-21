
# coding: utf-8

# In[1]:


class GDPData: 
    
    def __init__(self): 
        self.count = 0
        print('instance on memory')

    def __del__(self):  
        print('instnace off memory')

    def getNation(self, code):
        self.count = self.count + 1
        str1 = ""  
        
        if code == "KO":
            str1 = "Korean"
        elif code == "JA":
            str1 = "Japanese"
        elif code == "EN":
            str1 = "English"
        
        return str1     
    
    def getGDP(self, code):
        self.count = self.count + 1
        gdp = 0
        if code == "KO":
            gdp = 1630
        elif code == "JP":
            gdp = 5150
        elif code == "EN":
            gdp = 21440
        
        return gdp

