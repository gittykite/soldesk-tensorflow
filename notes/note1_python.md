# 1. Python

## 1-1. Python is
+  인터프리터 언어: 라인별로 실행 가능 (java 등 compile 언어는 전체 실행 필수)
+ 구글SW에 50% 이상 
+ 객체지향 & 함수언어 기능
+ 데이터분석 용에 강력 (R은 대규모 데이터 이용 시 결과 달라지는 증상)
+ but, 모바일/브라우저 부재, 버전 호환성, 하드웨어 접근속도 상대적 느림
  
## 1-2. Anaconda install 
+ https://repo.continuum.io/archive/  에서 3.5.1.0 설치
+  관리자 권한으로 설치! 
	- C:\programData 에 설치됨
+ 작업폴더 생성
	- C:\ai_201912\ws_python\notebook
+ 시스템 환경변수 확인 (Path)
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

### Tensorflow 2.0 설치
+ CPU의 AVX를 지원여부 확인 
	- https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX 
+ AVX available -> Tensorflow 2.0 설치
	- pip install tensorflow==2.0.0
+ AVX unavailable -> Tensorflow 1.6 + Keras 2.2 설치
	- pip install tensorflow==1.6.0
	- pip install keras==2.2.2

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
    - %s: string 
    - %c: character
    - %f: float
    - %d: decimal
    - %%: '%' 
 
+ escape words
    - \n: line break
    - \t: Tab
    - \ + Enter: line continue
    - \\: \ 
    - \':  '
    - \":  " 

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
- 시작 인덱스는 0부터 시작하며 - 인덱스는 요소의 끝부터 -1을 시작으로 지정함.
- [시작 인덱스: 마지막 인덱스]: 시작 index부터 마지막 index-1 부분까지 요소 추출 
- [: 마지막 인덱스]: 처음부터 마지막 index-1 부분까지 요소 추출 
- [시작 인덱스:] : 시작 index부터 마지막까지 요소 추출
- [::2]: step을 2로해서 요소 슬라이싱. 

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
    - if block requires indent(2~4spaces)
    - all indent in if block must have same spaces 
    - shortcut: 'TAB' <-> 'Shift+TAB'
  
### Loop
+ while
+ for
    - for i in list
    - for i in range(10)
+ func
  - join()

### Defining Functions
+ parameter -> argument
+ no "Method Overriding" -> 가변인수 사용해 구현 가능
+ 가변인수
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

## 1-8. 메소드의 실습, 생성자, 소멸자, 모듈 분리 

## 1-8. 대용량 데이터 연산 package(library) Numpy 실습 

## 1-9. 데이터셋 생성 및 분석 package(library) Pandas 

## 1-10. 데이터 시각화 library Matplotlib(맷플롯립) 
