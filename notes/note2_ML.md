
# 2. 머신러닝 

## 2-1. 머신러닝 개론
+ normalization
    - $${x_{new} = (x-x_{min}) / (x_{max} - x_{min})}$$
+ standardization
    - $${x_{new} = (x- \mu) / \sigma}$$
+ learning rate
    - error updating rate
    - corr with learning speed (+)
    - if too large => cannot find min error 
    - error * (1 - learning_rate) = fianl_error
+ error diff
    - error diff ~= 0 => stop learning

## 2-2. Install Conda & Tensorflow
https://www.tensorflow.org/

### Install Tensorflow 2.0 
+ check CPU supports AVX 
	- https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX 
+ AVX available -> Tensorflow 2.0 설치
	- pip install tensorflow==2.0.0
+ AVX unavailable -> Tensorflow 1.6 + Keras 2.2 설치
	- pip install tensorflow==1.6.0
	- pip install keras==2.2.2

### Install Conda 

### INstall Jupyter Notebook  

## 2-3. Regression Analysis

### Linear Regression 
+ equation
    - y = aX + b

### 평균 제곱근 오차(RMSE: Root Mean Square Error) 

### 텐서 구조, 텐서와 그래프 실행 절차, 텐서의 데이터 타입 기반 

### 정규 분포 난수의 생성, 균등 분포 난수의 생성 

### Tensorflow에서의 경사 하강법(gradient decent) 머신러닝 
+ gradient decent
  -  
+ delta rule
  - update weight  
  - w <- w + αδx
+ auto 

### 다중 선형 회귀(Multiple Linear Regression) 모델 
+ equation
    - y = a1X1 + a2X2 + a3X3 + ... + b


### 로지스틱 회귀(Logistic Regression) 모델의 구현, 

## 2-4. 신경망 학습

### 퍼셉트론(perceptron), 오차 역전파(Back Propagation) 
+ perceptron
    - single layer ANN(Artificial Neuron Network)
+ back propagation 
    - update weight involving error 

### 기울기 소실 문제와 활성화 함수, 손실 함수 
+ MSE(Mean Squared Error)
+ Activation Function (=Transfer Function)
  - $${y=f\left(\sum _{i=1}^{n}x_{n}\cdot w_{n}-\theta \cdot w_{0}\right)}$$ 
  - Sigmoid function
  - softmax function
  - step function


### 1차원 데이터의 사용, Keras를 이용한 2차원 데이터의 사용 

## 2-5. 분류 모델

### 이항 분류(Binary Classification) 모델 개발 

### 다중 분류(Multi Classification) 모델 개발 

## 2-6. 신경망 모델

### 컨볼루션 신경망 레이어 CNN 모델 개발 

### 미국 국립 표준 기술원(NIST)의 MNIST 이용 모델 제작 

### CIFAR-10, OpenCV를 이용한 이미지 인식 모델 개발 

### VGG 학습모델 재사용 

### 순환 신경망 레이어 RNN 모델 개발 

## 2.7 함수형 API 사용과 Parameter 최적화 


## ETC
+ numpy 미분 https://pinkwink.kr/1233
+ 자연어 처리 https://wikidocs.net/21667
+ Dive into Deep Learning https://www.d2l.ai/chapter_preface/index.html
+ 큐스터디 권태원 교수님 선형대수 강의
+ dot() vs matmul() https://m.blog.naver.com/PostView.nhn?blogId=cjh226&logNo=221356884894
