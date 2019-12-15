
# coding: utf-8

# In[1]:


if __name__ == '__main__':
    print(maxsu(1000, 2000))
    print(minsu(1000, 2000))
    print(swap(1000, 2000))

def maxsu(su1, su2):
    if su1 > su2:
        return su1
    else:
        return su2


def minsu(su1, su2):
    if su1 < su2:
        return su1
    else:
        return su2


def swap(su1, su2):
    temp = su1
    su1 = su2
    su2 = temp

    return su1, su2

