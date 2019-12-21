
# coding: utf-8

# In[3]:


class Class1:
    # class variables
    year = 0
    product = ''
    price = 0
    dc = 0
    service = False
    


# In[4]:


if __name__ == '__main__':
    # ref by class name (static)
    Class1.year = 2019
    Class1.product = 'SSD-2TB'
    Class1.price = 40 * 10000
    Class1.dc = 3.5
    Class1.service = False
 
    print(Class1.year)
    print(Class1.product)
    print(Class1.price)
    print(Class1.dc)
    print(Class1.service)


# In[19]:


prod1 = Class1()
print(prod1.year)

prod2 = Class1()
print(prod2.year)


# In[21]:


prod1.year = 2020
prod2.year = 2021

print(prod1.year)
print(prod2.year)
print(Class1.year)


# In[14]:


class Product:

    # req "self" argument => connect func & instance   
    def setData1(self): 
        print('type of self:', type(self))
        
    def setData2(self, year, product): # self 는 전달 받지 않음.
        self.year = year  # instance 변수, field, property, attribute, 멤버 변수, 속성...
        self.name = name
        self.price = 0
        self.dc = 0
        
    def printData(self):
        print('----------------------')
        print("Manufactured:", self.year)
        print("Product name:", self.name)
        print("Price:", self.price)
        print("Sale Price:", self.dc)


# In[22]:


product1 = Product()
product2 = Product()

product1.setData1()
product1.setData2(2019, "GalaxxyS10")
product2.setData2(2020, "GalaxxyS11")

product1.printData()
product2.printData()


# In[31]:


class Nation:

    # Constructor
    def __init__(self, code='KOR'): 
        self.count = 0
        self.code = code
        print('instance on memory')

    # Destructor    
    def __del__(self):  
        print('instnace off memory')

    def getNation(self, code):
        self.count = self.count + 1
        
        # local variable
        str1 = ""  
        
        if code == "KO":
            str1 = "Korean"
        elif code == "JA":
            str1 = "Japanese"
        elif code == "EN":
            str1 = "English"
        
        return str1


# In[32]:


# init
nation1 = Nation()
# del -> init
nation1 = Nation() 

print(nation1.code)
print(nation1.count)
print(nation1.getNation("KO"))
print(nation1.getNation(nation1.code))

nation2 = Nation("JA")
nation3 = Nation("EN")

print(nation2.getNation(nation2.code))
print(nation3.getNation(nation3.code))

