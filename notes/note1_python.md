# 1. Python

## 1-1. Python is
+ Interpreter Language
    - can run by line 
    - while compile lang must run whole code (ex. Java)
    - relatively slower 
+ consists Google's SW by 50% 
+ OOP(obejct oriented) 
+ Functional
+ Big Data Analysis
    - while R results in fluctuant output
+ but, 모바일/브라우저 부재, 버전 호환성, 하드웨어 접근속도 상대적 느림
  
## 1-2. Anaconda install 
+ https://repo.continuum.io/archive/  (ver. 3.5.1.0)
+ Install as **Administrator** (to fix path) 
	- C:\programData 
+ create working folder
	- C:\ai_201912\ws_python\notebook
+ check System Environment Variables (Path)
    - C:\ProgramData\Anaconda3;
    - C:\ProgramData\Anaconda3\Library\mingw-w64\bin;
    - C:\ProgramData\Anaconda3\Library\usr\bin;
    - C:\ProgramData\Anaconda3\Library\bin;
    - C:\ProgramData\Anaconda3\Scripts;
    - C:\cuda\bin;
    - C:\cuda\include;
    - C:\cuda\lib;
    - C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common
	- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;
	- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp;
+ 가상환경 설정
	- conda info --envs 
	- conda create -n machine python=3.6 numpy scipy matplotlib spyder pandas seaborn scikit-learn h5py
	- conda activate machine

## 1-3. Jupyter Notebook Interpreter의 사용 및 설정 
+ 가상환경에 Jupyter 커널 연동
	- C:\Windows\system32>python -m ipykernel install --user --name=machine
+ Jupyter 폴더 지정 커맨드  
 	- C:/ai_201912/jupyter.cmd

## 1-4. 컴파일, 데이터 형(data type), 연산자(Operator) 

## 1-5. 시퀀스 자료형, 제어문, 함수 Python 

### 시퀀스 Type
+ mutable(can change elements) 
    - list(*)
    - bytes
    - bytearray
+ immutable(cannot change elements)
    - str(*)
    - tuple(*)
    - range
+ operator

    |operator|func|
    |---|---|
    | + | concatenate seqs |
    | * | repeat seqs |

+ format keyword  

    |operator|represents|
    |---|---|
    | %s | string  |
    | %c | character |
    | %f | float |
    | %d | decimal |
    | %% | '%'  |

    ex) print('Type %s %d times' % ('hey', 5))
 
+ escape words

    | operator | represents |
    |---|---|
    | \t | Tab |
    | \\ |  \  |
    | \' |   ' |
    | \" |   " | 
    | \n | line break |
    | \ + Enter | line continue |

### 배열 Type
+ List 
    - 원본 값 변경 가능 
    - format: [1, 'a', 2.5]
+ Tuple
    - 원본 값 변경 불가
    - format: (1, 'a', 2.5)
    - 데이터전처리 lib 에서 주로 사용
+ Dictionary
    - 키와 값의 쌍 
    - JAVA의 Map과 유사
    - format: {'year': 1, 'season': '여름'}

#### 배열원소 참조
- index
    * starts with 0
    * reverse ref starts with -1 
- [from_index: to_index]
- [: to_index] : from the start
- [from_index:] : to the end
- [::step_by]: slice by 

### 배열 관련 Func
+ len()
    - len(matrix)
    - len(matrix[0])
+ insert()
+ append()
+ remove()
    - remove(data[0])
    - remove(150)
+ sort()
    - sort(data, reverse=True)
    - sort(data, key=str.lower)
+ dict()
    - dict(key1=val1, key2=val2 ...)

 ### IF 제어문
+ If keywords
    - if
    - elif
    - else
    - or
    - and
    - not
+ Indent in block
    - block requires indent (2~4 spaces)
    - all indent must have same length 
    - [TAB] => indent 
    - [Shift + TAB] => delete indent
  
### Loop
+ while
+ for
    - for i in list
    - for i in range(10)
+ func
  - join()

### Defining Functions
+ parameter -> argument
+ no "Method Overriding" -> utilize variable-len-arg instead
+ variable length arguments
    - *actors: pass args in tuple
    - **actors: pass args in dictionary 
+ returns
    - multiple return value
    - returns in tuple
+ scope
    - local
    - unlocal
    - global

## 1-6. 모듈과 패키지의 사용, import의 사용 
+ module  
: func/class in other python file
   - import MODULE_NAME                                
   - import PKG_NAME.MODULE_NAME  
   - import PKG_NAME.MODULE_NAME as ALIS_NAME 
   - from MODULE_NAME import FUNC_NAME        
   - from PKG_NAME import MODULE_NAME 
   - from PKG_NAME.MODULE_NAME import FUNC_NAME
+ package  
: modules in folder
    - use '__init__.py' to mark package (ver < 3.3)
    - every folder is package (ver >= 3.3)
    - may refer children 
    - no "main" method like Java
+ reset memory before test import
   - kernal restart  
   : delete modules on memory
   - %reset  
   : delete variables only (modules on memory still exist)   

## 1-7. Class 선언, 클래스 멤버, 메소드, 인스턴스 멤버 

+ Class definition
    - class name: not related to python file name
    - naming convetion: starts with upper-case
+ class variable
    - ref by class name
    - use static memory 
+ instance variable
    -  self.field_name 
+ constructor (__init__)
    - 객체 생성 시 자동실행
    - 인스턴스 변수 초기화에 이용
    - 생략가능 
+ destructor (__del__)
    - 객체 소멸 시 자동실행
    - 생략가능 
+ class import 
    - import PKG_NAME.PY_NAME
    - from PKG_NAME import PY_NAME
    - from PKG_NAME.PY_NAME import *
    - from PKG_NAME.PY_NAME import CLASS_NAME

## 1-8. Matplotlib library  
+ Overview
    - data visulaization library 
    - https://matplotlib.org
+ Links
    - pyplot API: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot
    - Color: https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    - Color map: https://matplotlib.org/tutorials/colors/colormaps.html
    - Line: https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html
    - Marker: https://matplotlib.org/api/markers_api.html

+ Funcs
    - figure()
        * figure(figsize=(10, 2))
        * inch 
    - plot()
        * plot(y_list)
        * plot(x_list, y_list) 
        * plot(x_list, y_list, color='' ...) 
    - scatter()
    - axis()
        * axis([x_start, y_start_, x_end, y_end]) 
    - gird()
        * gird(True) 
        * gird(False) 
    - title()
    - xlabel()
    - ylabel()
    - legend() 
    - show() 
+ plot() args
    - color
    - label
    - linestyle
    - marker
    - markerfacecolor
    - markersize

## 1-9. Numpy Library 
+ Overview
    -  library to handle math & matrix data
    -  vecotrized operation => calculate matrix like R
+ tensor
    - vector
    - matrix
    - 3-dimension
    - 4-dimension 
+ broadcasting
    - allows to perform operations on arrays of compatible shapes 
    - create arrays bigger than either of the starting ones
    - rules: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    - ref: https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch01.html
+ ndarray
    - multi-dimensional array
    - itme type must be same 
    - fast access & loop operation
    - many numpy func returns ndarray as result

### numpy functions
+ NDArray function
    - np.array()
        * np.array(list)
    - np.arange()
        * np.arange(to) 
        * np.arange(from, to) 
        * np.arange(from, to, step_by) 
    - np.dot()
        * can operate even row/col does not match 
        * array1.dot(array2) 
    - np.matmul()
    - np.zeros()
        * np.zeros(rows) 
        * np.zeros((rows, cols)) 
+ Random Function
    - random.normal()
        * random.normal(avg, sigma, num_of_data)
    - random.rand()
        * random.rand()
        * random.rand(rows, cols) 
    - random.randint()
        * random.randint(max_int, size=(rows))
        * random.randint(max_int, size=(rows, cols))
    - random.randn()
        * random num in normal distribution
        * random.rand(from, to)
    - random.uniform()
        * random num in uniform distribution
        * random.uniform(from, to, num_of_data)
    - random.seed()
        * random.seed(seed_num) 
+ Statistics Func
    - amin()
    - amax()
    - quantile()
    - median()
    - average()
    - mean()
    - std()
    - var()
    - corrcoef()
    - correlate()
    - cov()
    - histogram()
    - histogram2d()
    - bincount()



## 1-10. 데이터셋 생성 및 분석 package(library) Pandas 

## ETC
+ Built-in Function
    - https://docs.python.org/ko/3/library/functions.html
+ inspect library  
: library for extract class/method info   
    - api: https://docs.python.org/3/library/inspect.html  
    - example: https://www.journaldev.com/19946/python-inspect-module 
