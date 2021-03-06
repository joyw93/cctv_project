# :pushpin: 고속도로 통행량 탐지
>YOLO를 이용한 실시간 객체탐지 프로젝트  

</br>

## 1. 제작 기간 & 참여 인원
- 2021.11.5 ~ 2021.11.12
- 개인 프로젝트

</br>

## 2. 사용 기술
  - Python 3
  - Flask
  - YOLO
  
</br>

## 3. UI
![cover](https://user-images.githubusercontent.com/48177285/141255585-cae6c8b4-a1f9-4e40-85da-4e186479b5e8.JPG)



## 4. 시연영상
https://youtu.be/ufnI6nOHfJ8


## 5. 개발과정
### 5.1. 실시간 고속도로 CCTV API 요청
![image](https://user-images.githubusercontent.com/48177285/151138768-ff5c7a00-12f8-49cf-886d-bc1989bb94e2.png)
![image](https://user-images.githubusercontent.com/48177285/151139684-645fbbd3-0763-41d5-97af-a1c69bb91eee.png)
~~~python
<li><a href="{{url_for('cctv', src='WmRWBzX6SFFPBvbFmAiCLhXBI4b/+H61JldCRpyvHfuP/l6fxlDkWQyLDFRU2FfSjBPBPUJwns2dUERd40H4lA==')}}" class="link-dark rounded">송파</a></li>
<li><a href="{{url_for('cctv', src='QUTyqK9d00rhb4kslLJZ37ARoGZUzu20y61LHqqGPdJxsS1arEVVw8AmgHJwV4ofxvNqeo9Oxqli8Knftyu9aw==')}}" class="link-dark rounded">성남</a></li>
~~~
</br>

### 5.2. 딥러닝 모델(YOLO) 전이학습
~~~python
!cd /content/yolov5; python train.py --img 320 --batch 8 --epochs 200 --data /content/dataset_v1/data.yaml  \
                                     --project=/content/data_v1_320 --name cctv_v1_320 --exist-ok 
~~~
![image](https://user-images.githubusercontent.com/48177285/151142218-0f2568ea-1cb7-4c34-95f1-e4f362a3f0e4.png)

### 5.3. 실시간 그래프 생성
~~~python
@app.route('/live-data')
def live_data():
    # Create a PHP array and echo it as JSON
    data = [time() * 1000, car.num]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response
~~~
