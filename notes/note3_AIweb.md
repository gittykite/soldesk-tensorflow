# 3. AI Web Project 

## 3-1 Django
+ create app
  - $ pip install django
  - $ django-admin startproject config .
  - $ python manage.py migrate
  - $ python manage.py startapp app_default 
  - $ python manage.py createsuperuser
  - $ python manage.py runserver
  - $ python manage.py startapp ais
+ add views & logic
  - urls.py: set urlpatterns
  - views.py: add request & response process
  - models.py: add class models & logic
+ add templates & rsc
  - templates: add html files to templates folder
  - statics: add rsc files to rsc folder 
+ run app
  - $ python manage.py runserver  
  - $ python manage.py runserver 0.0.0.0:8000 

## 3-2 Google Cloud Colab
+ free cloud svc for GPU
+ resembles jupyter notebook
+ pkg: Tensorflow, keras, matplotlib, scikit-learn, panda
  
### Add Colab App
+ access google drive
+ new > add more apps > colab > connect
+ access https://colab.research.google.com/
+ create new notebook
  - need mounting to google drive to save  
  - save to Colab Notebooks folder automatically
+ machine may reset after a while
