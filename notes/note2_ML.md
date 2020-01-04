
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

### Install Keras
+ https://keras.io/
+ consists high-level layer of deep learning
    - user friendly (consistent, simple)
    - optimization 
+ need backend deep learning engine 
    - tensorflow
    - PyTorch
    - CNTK
+ sequential class => add layers 
    - model.add(Dense(100, input_dim=17, activation='relu')) 
        * output node
        * input node
        * activation function
+ output layer
    - model.add(Dense(1))  
    - model.add(Dense(1, activation='sigmoid')) 
      * 수치예측 
      * 이항분류 : sigmoid => 미분최대치가 0.3 => 은닉층에 이용시 가중치 0에 수렴가능  
      * 다항분류 : softmax
+ model setting
    - model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    - metrics
        * 수치예측: mse
        * 분류: accuracy
    - loss 
        * 수치예측: mean squared error, mean absolute error ...
        * 이항분류: binary_crossentropy
        * 다항분류: categorical_crossentropy  
+ model learning
    - model.fit(X, Y, epochs=10, batch_size=10) 
        * data
        * actual_data
        * epochs
        * batch_size: corr speed(+)/accuracy(-)/outlier_effect(-)
+ model charting
+ model evaluation
    - model.evaluate(X, Y)[1]) 
+ model save/load    
    - model.save('CancerSurvival.h5')
    - load_model('CancerSurvival.h5') 
### Install Conda 

### INstall Jupyter Notebook  

## 2-3. Regression Analysis

### Linear Regression 
+ equation
    - y = aX + b

### 평균 제곱근 오차(RMSE: Root Mean Square Error) 

### 텐서 구조, 텐서와 그래프 실행 절차, 텐서의 데이터 타입 기반 

### 정규/균등 분포 난수 생성 
+ Normal Distribution
    + tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=0)
    + tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
    + tensorflow.model.add(Dense(60, kernel_initializer='normal', input_shape=(2, ), activation='linear'))  
+ Uniform Distribution
    + tensorflow.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=0)
    + tensorflow.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=None)
    + model.add(Dense(60, kernel_initializer='uniform', input_shape=(2, ), activation='linear'))  

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
    - multiple perceptrons => deep learning
+ back propagation 
    - update weight involving error 

### 기울기 소실 문제와 활성화 함수, 손실 함수 
+ MSE(Mean Squared Error)
+ Activation Function (=Transfer Function)
  - $${y=f\left(\sum _{i=1}^{n}x_{n}\cdot w_{n}-\theta \cdot w_{0}\right)}$$ 
  - Sigmoid function
    * y = 1 / (1 + np.exp(-x))
    * binary classification
    * vanishing gradient problem: max deviation 0.3 => earlier layers' weight become 0 
  - relu function
    * np.where(x <= 0, 0, x)
    * x <= 0 -> 0
    * x > 0  -> x 
    * binary classification
  - softmax function
    * y = np.exp(x) / np.sum(np.exp(x)) 
  - step function
    * np.where(x < 0, 0, 1) 
    * x <= 0 -> 0
    * x > 0  -> 1
  - Tanh(hyperbolic tangent) function
    * np.tanh(x)
    * usage: RNN(Recurrent Neural Network), time-series data 

![](https://miro.medium.com/max/1192/1*4ZEDRpFuCIpUjNgjDdT2Lg.png)

### 경사하강법
+ SGD (Stochastic Gradient Descent)
    - increase speed
+ Momentum
    - increase accuracy 
+ NAG
    - decrease reluctant move 
+ Adagrad
    - adjust learning rate by variable update interval
+ RMSProp
+ Adam
    - Momentum + RMSProp

### 1차원 데이터의 사용, Keras를 이용한 2차원 데이터의 사용 

### Overfitting Problem
+ Methods https://kolikim.tistory.com/50
+ L1/L2 
    - https://greatjoy.tistory.com/58
    - https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
+ Drop Out
+ Reduce Network
+ Weight Decay Regularization 
    - L1 (Lasso Regression): regulate by absolute value of weight  
    - L2 (Ridge Regression) : rebulate by square value of weight


## 2-5. 분류 모델

### 이항 분류(Binary Classification)
+ LabelEncoder()  
: convert string category to number group
    ```
    from sklearn.preprocessing import LabelEncoder

    e = LabelEncoder() 
    e.fit(Y_obj)  

    # transform to integer: 0, 1, 2
    Y = e.transform(Y_obj)  
    print(Y)
    ```
+ train_test_split()  
: split test & validation data randomly
    ```
    from sklearn.model_selection import train_test_split

    seed = 0

    # split data => test 85%, validation 15%
    x_train_all, x_test, y_train_all, y_test = train_test_split(X, Y,
                                                                stratify=Y,
                                                                test_size=0.15,
                                                                random_state=seed)
    # split test data(85%) => test 90%, validation 10%
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all,
                                                    stratify=y_train_all,
                                                    test_size=0.1,
                                                    random_state=seed)
    ```
    - stratify  
    : makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify.  

    For example, if variable y is a binary categorical variable with values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will make sure that your random split has 25% of 0's and 75% of 1's.

### 다중 분류(Multi Classification)

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
